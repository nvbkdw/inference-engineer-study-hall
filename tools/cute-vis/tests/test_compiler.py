import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cuteviz.algebra import compile_layout
from cuteviz.mapping import MappingQuery, query_mapping
from cuteviz.model import Capture

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.compiler,
    pytest.mark.skipif(
        importlib.util.find_spec("cutlass") is None,
        reason="Install cuteviz[capture] for compiler validation",
    ),
]


@pytest.mark.parametrize("target", ["sm_80", "sm_90a", "sm_100a"])
def test_actual_compiler_partition_oracle(target, tmp_path):
    output, oracle = tmp_path / "capture.json", tmp_path / "oracle.json"
    process = subprocess.run(
        [sys.executable, "tests/compiler_oracle.py", str(output), str(oracle)],
        cwd=ROOT,
        env=os.environ | {"CUTE_DSL_ARCH": target, "CUTE_DSL_NO_CACHE": "1"},
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    capture = Capture.load(output)
    assert capture.outcome == "complete", [
        (o.label, o.diagnostics) for o in capture.objects if o.diagnostics
    ]
    by_label = {obj.label: obj for obj in capture.objects}
    cases = json.loads(oracle.read_text())
    assert len(cases) > 1000
    # Compare descriptor evaluation to actual NVIDIA partition output, not to itself.
    for case in cases:
        obj = by_label[case["label"]]
        view = next(v for v in obj.views if v.role == case["role"])
        flat = int(compile_layout(view.tv_layout)((case["thread"], case["value"])))
        expected = case["coord"][0] + case["coord"][1] * view.shape[0]
        assert flat == expected, case
    if target == "sm_100a":
        for obj in capture.objects:
            if obj.kind != "mma":
                continue
            # Independent NVIDIA reference: mma_traits_sm100_frag.hpp, tmem_frg::make.
            # M=64 uses half subpartitions; C occupies full 32-bit columns even for FP16.
            m = obj.views[2].shape[0]
            row = 37 if m == 128 else 37 % 16 + 32 * (37 // 16)
            cell = query_mapping(
                capture, MappingQuery(object_id=obj.id, role="C", start=[37, 3], extent=[1, 1])
            )["cells"][0]
            assert cell["tmem"]["value"] == {"data_path": row, "column": 3, "bit_offset": 0}
            if obj.views[0].storage.value == "tmem":
                cell = query_mapping(
                    capture, MappingQuery(object_id=obj.id, role="A", start=[37, 3], extent=[1, 1])
                )["cells"][0]
                assert cell["tmem"]["value"] == {"data_path": row, "column": 1, "bit_offset": 16}
            assert not query_mapping(
                capture, MappingQuery(object_id=obj.id, role="A", extent=[1, 1])
            )["cells"][0]["owners"]
    for label in ("ldmatrix/False", "ldmatrix/True"):
        obj = by_label[label]
        cells = query_mapping(capture, MappingQuery(object_id=obj.id, role="S"))["cells"]
        assert any(
            c["links"]
            and {(o["thread"], o["value"]) for o in c["owners"]}
            != {(o["thread"], o["value"]) for o in c["links"]}
            for c in cells
        )
        assert all(link["coordinate"] == c["coordinate"] for c in cells for link in c["links"])


def test_example_capture_source_and_refresh(tmp_path):
    # The real example checks probes embedded in a @cute.kernel, not only native helpers.
    code = """
from examples.inspect_copy import compile_example
from cuteviz.model import Capture
import sys
for _ in range(2):
    compile_example(sys.argv[1])
    c=Capture.load(sys.argv[1])
    assert c.outcome == 'complete'
    assert len(c.objects) == 8
    obj=c.objects[0]
    source=next(s for s in c.sources if s.id == obj.source.source_id)
    assert 'cuteviz.inspect("gA", gA)' in source.text.splitlines()[obj.source.line-1]
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "copy.json")],
        cwd=ROOT,
        env=os.environ | {"CUTE_DSL_ARCH": "sm_80"},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_probe_ir_and_ptx_neutrality_and_partial_capture(tmp_path):
    process = subprocess.run(
        [sys.executable, "tests/compiler_safety.py", str(tmp_path / "safety.json")],
        cwd=ROOT,
        env=os.environ
        | {"CUTE_DSL_ARCH": "sm_80", "CUTE_DSL_KEEP": "ptx", "CUTE_DSL_DUMP_DIR": str(tmp_path)},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_actual_cache_hit_diagnostic(tmp_path):
    process = subprocess.run(
        [sys.executable, "tests/compiler_safety.py", str(tmp_path / "cache.json"), "cache"],
        cwd=ROOT,
        env=os.environ | {"CUTE_DSL_ARCH": "sm_80", "CUTE_DSL_KEEP": "", "CUTE_DSL_NO_CACHE": "0"},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("CUTEVIZ_GPU_TEST") != "1",
    reason="Set CUTEVIZ_GPU_TEST=1 to execute the optional GPU test",
)
def test_gpu_copy_neutrality(tmp_path):
    process = subprocess.run(
        [sys.executable, "-m", "tests.gpu_smoke", str(tmp_path / "gpu.json")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
