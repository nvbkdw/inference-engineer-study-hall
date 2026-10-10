"""Execution boundary, job lifecycle, and actual compiler integration for the editor."""

import importlib.util
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cuteviz.model import Capture
from cuteviz.server import create_app
from cuteviz.workbench import MAX_LOG, TERMINAL

ROOT = Path(__file__).resolve().parents[1]
COMPILER = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTeDSL required"
)


def wait_job(client, job_id):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        result = client.get(f"/api/compilations/{job_id}").json()
        if result["status"] in TERMINAL:
            return result
        time.sleep(0.05)
    pytest.fail("Worker did not finish")


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(output_dir=tmp_path), base_url="http://127.0.0.1") as client:
        info = client.get("/api/workbench").json()
        client.headers["X-CuteViz-Token"] = info["token"]
        yield client


def submit(client, source, **kwargs):
    response = client.post(
        "/api/compilations", json={"source": source, "target": "sm_80", **kwargs}
    )
    assert response.status_code == 202, response.text
    return response.json()["id"]


def test_execution_boundary_and_read_only(tmp_path):
    with TestClient(create_app(output_dir=tmp_path), base_url="http://127.0.0.1") as c:
        token = c.get("/api/workbench").json()["token"]
        assert c.post("/api/compilations", json={"source": "pass"}).status_code == 403
        assert c.get("/api/workbench", headers={"Host": "attacker.example"}).status_code == 403
        assert c.get("/api/workbench", headers={"Origin": "https://example.com"}).status_code == 403
        assert c.get("/api/workbench", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
        c.headers["X-CuteViz-Token"] = token
        assert (
            c.post("/api/compilations", json={"source": "pass", "entry": "entry()"}).status_code
            == 422
        )
        assert c.post("/api/compilations", json={"source": "x" * 256001}).status_code == 422
        assert c.get("/api/capture").status_code == 404
        assert c.get("/api/compilations/unknown").status_code == 404
        assert not list(tmp_path.iterdir())
    with TestClient(create_app(enabled=False), base_url="http://127.0.0.1") as c:
        assert not c.get("/api/workbench").json()["can_compile"]
        assert c.post("/api/compilations", json={"source": "pass"}).status_code == 403


def test_missing_compiler_still_serves_portable_capture(tmp_path):
    app = create_app(ROOT / "examples/captures/copy.cuteviz.json", output_dir=tmp_path)
    app.state.workbench.compiler = None
    with TestClient(app, base_url="http://127.0.0.1") as c:
        info = c.get("/api/workbench").json()
        assert not info["can_compile"] and "cuteviz[capture]" in info["reason"]
        assert c.get("/api/capture").status_code == 200
        c.headers["X-CuteViz-Token"] = info["token"]
        assert c.post("/api/compilations", json={"source": "pass"}).status_code == 409


@COMPILER
@pytest.mark.compiler
@pytest.mark.parametrize(
    "example,target",
    [("copy", "sm_80"), ("mma", "sm_90a"), ("mma", "sm_100a"), ("tiles", "sm_80")],
)
def test_editor_templates_compile_and_map(client, example, target):
    source = (ROOT / f"src/cuteviz/templates/{example}.py").read_text()
    job_id = submit(client, source, target=target)
    job = wait_job(client, job_id)
    assert job["status"] == "succeeded", job
    capture = client.get("/api/capture", params={"capture_id": job_id}).json()
    assert capture["objects"] and any(s["text"] == source for s in capture["sources"])
    obj = capture["objects"][0]
    mapped = client.post(f"/api/mapping/query?capture_id={job_id}", json={"object_id": obj["id"]})
    assert mapped.status_code == 200 and mapped.json()["cells"]
    downloaded = client.get(f"/api/compilations/{job_id}/capture")
    assert ".cuteviz.json" in downloaded.headers["content-disposition"]
    assert Capture.model_validate(downloaded.json()).outcome == "complete"
    assert Path(job["capture_path"]).is_file()
    assert client.get(f"/api/compilations/{job_id}/source").json()["source"] == source


@COMPILER
@pytest.mark.compiler
def test_arbitrary_entry_args_local_import_and_immutable_runs(client, tmp_path):
    # Dynamic shape changes must not silently reuse an earlier compile or capture.
    project = tmp_path / "project"
    project.mkdir()
    (project / "local_shape.py").write_text("COLUMNS = 5\n")
    client.app.state.workbench.workdir = project
    source = """import cutlass
import cutlass.cute as cute
import cuteviz
from local_shape import COLUMNS
@cute.jit
def my_entry(rows: cutlass.Constexpr, columns: cutlass.Constexpr):
    cuteviz.inspect("user_layout", cute.make_layout((rows, columns), stride=(columns, 1)))
def arguments():
    return {"args": (2,), "kwargs": {"columns": COLUMNS}}
"""
    ids = []
    for rows in [2, 3]:
        job_id = submit(
            client, source.replace("(2,)", f"({rows},)"), entry="my_entry", args_factory="arguments"
        )
        assert wait_job(client, job_id)["status"] == "succeeded"
        ids.append(job_id)
    for rows, job_id in zip([2, 3], ids):
        c = client.get(f"/api/capture?capture_id={job_id}").json()
        query = client.post(
            f"/api/mapping/query?capture_id={job_id}", json={"object_id": c["objects"][0]["id"]}
        )
        assert query.json()["total_elements"] == rows * 5


@COMPILER
@pytest.mark.compiler
def test_errors_empty_and_partial_captures(client):
    source = """import cutlass.cute as cute
import cuteviz
@cute.jit
def entry():
    cuteviz.inspect("before_error", cute.make_layout((2, 2)))
    raise RuntimeError("intentional compilation failure")
"""
    job = wait_job(client, submit(client, source))
    assert job["status"] == "failed" and job["outcome"] == "partial"
    assert job["object_count"] == 1 and "intentional compilation failure" in job["log"]
    job = wait_job(client, submit(client, "def broken(:\n    pass\n"))
    assert job["status"] == "failed" and "SyntaxError" in job["log"]
    job = wait_job(
        client, submit(client, "import cutlass.cute as cute\n@cute.jit\ndef entry():\n    pass\n")
    )
    assert job["status"] == "failed" and "No probes were observed" in job["log"]


@COMPILER
@pytest.mark.compiler
def test_cancel_timeout_log_bounds_and_shutdown(client):
    source = 'import time\nprint("x" * 100000, flush=True)\ntime.sleep(30)\n'
    job_id = submit(client, source)
    busy = client.post("/api/compilations", json={"source": "pass"})
    assert busy.status_code == 409
    deadline = time.monotonic() + 15
    while len(client.get(f"/api/compilations/{job_id}").json()["log"]) < MAX_LOG:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert client.post(f"/api/compilations/{job_id}/cancel").status_code == 200
    job = wait_job(client, job_id)
    assert job["status"] == "cancelled" and len(job["log"]) <= MAX_LOG
    assert client.app.state.workbench.get(job_id).process.poll() is not None
    client.app.state.workbench.timeout = 0.2
    job = wait_job(client, submit(client, source))
    assert job["status"] == "timed_out"
    assert "limit" in job["error"]
    client.app.state.workbench.timeout = 30
    job_id = submit(client, source)
    client.app.state.workbench.close()
    assert wait_job(client, job_id)["status"] == "cancelled"
