import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from tensor_layouts import Layout as CPULayout

import cuteviz
from cuteviz.algebra import evaluate
from cuteviz.banks import Access, BankQuery, analyze_banks
from cuteviz.mapping import MappingQuery, Selection, query_mapping
from cuteviz.model import Capture, CapturedObject, Layout, View, known, unknown
from cuteviz.server import create_app

ROOT = Path(__file__).resolve().parents[1]


def simple_capture(shape=(4, 8), stride=(8, 1), bits=16):
    layout = Layout(shape=list(shape), stride=list(stride))
    return Capture(
        objects=[
            CapturedObject(
                id="tensor",
                label="tensor",
                kind="tensor",
                views=[
                    View(
                        role="tensor",
                        shape=list(shape),
                        layout=layout,
                        storage=known("smem"),
                        element_bits=known(bits),
                        dtype=known("Float16"),
                        layout_status=known("relative"),
                    )
                ],
            )
        ]
    )


@pytest.mark.parametrize(
    "shape,stride,coord,expected",
    [
        ([4, 8], [8, 1], [3, 5], 29),
        ([4, 8], [1, 4], [3, 5], 23),
        ([[2, 3], 4], [[1, 8], 2], [1, 2, 3], 23),
        ([4, 8], [0, 1], [3, 5], 5),
        ([4, 8], [-8, 1], [3, 5], -19),
    ],
)
def test_independent_affine_examples(shape, stride, coord, expected):
    assert evaluate(Layout(shape=shape, stride=stride), coord) == expected


@pytest.mark.parametrize("bits,base,shift", [(3, 0, 3), (2, 2, 3), (2, 1, -3)])
def test_swizzle_matches_independent_bit_definition(bits, base, shift):
    layout = Layout(
        kind="compose",
        left=Layout(kind="swizzle", bits=bits, base=base, shift=shift),
        right=Layout(shape=[16, 16], stride=[16, 1]),
    )
    for index in range(256):
        y_mask = ((1 << bits) - 1) << (base + max(0, shift))
        masked = index & y_mask
        expected = index ^ (masked >> shift if shift >= 0 else masked << -shift)
        assert evaluate(layout, [index // 16, index % 16]) == expected


def test_composition_offset():
    layout = Layout(
        kind="compose",
        left=Layout(shape=[4, 4], stride=[4, 1]),
        offset=1,
        right=Layout(shape=[2, 3], stride=[1, 2]),
    )
    # right(1,2)=5; left(6) uses CuTe first-mode-fast unflattening: (2,1) -> 9.
    assert evaluate(layout, [1, 2]) == 9


def test_bytes_slices_and_bounds():
    capture = simple_capture(shape=(4, 8, 2), stride=(16, 2, 1))
    result = query_mapping(
        capture,
        MappingQuery(object_id="tensor", axes=[0, 1], fixed={2: 1}, start=[2, 3], extent=[1, 1]),
    )
    cell = result["cells"][0]
    assert cell["coordinate"] == [2, 3, 1]
    assert cell["element_offset"]["value"] == 39
    assert cell["byte_offset"]["value"] == 78
    assert "2 × 16 + 3 × 2 + 1 × 1" in cell["explanation"]
    for kwargs in [
        {"extent": [65, 64]},
        {"axes": [0, 0]},
        {"start": [-1, 0]},
        {"axes": [4]},
        {"fixed": {2: 2}},
        {"fixed": {0: 0}},
        {"start": [4, 0]},
    ]:
        with pytest.raises(ValueError):
            query_mapping(capture, MappingQuery(object_id="tensor", **kwargs))


def test_large_tensor_is_bounded():
    capture = simple_capture(shape=(10**9, 10**9), stride=(10**9, 1))
    result = query_mapping(capture, MappingQuery(object_id="tensor", extent=[64, 64]))
    assert result["total_elements"] == 10**18
    assert len(result["cells"]) == 4096


def test_symbolic_stride_retains_coordinate():
    capture = simple_capture()
    view = capture.objects[0].views[0]
    view.layout.stride[0] = unknown("runtime stride", "symbolic", "stride_m").model_dump()
    result = query_mapping(capture, MappingQuery(object_id="tensor", extent=[1, 1]))
    assert result["cells"][0]["element_offset"]["status"] == "symbolic"


def test_no_false_thread_ownership_and_alias_preservation():
    capture = simple_capture(stride=(0, 1))
    result = query_mapping(capture, MappingQuery(object_id="tensor", extent=[2, 1]))
    assert [c["element_offset"]["value"] for c in result["cells"]] == [0, 0]
    assert not result["cells"][0]["owners"]
    view = capture.objects[0].views[0]
    view.tv_layout = Layout(shape=[2, 32], stride=[0, 1])
    view.ownership = known("explicit test mapping with replicated owners")
    result = query_mapping(capture, MappingQuery(object_id="tensor", extent=[1, 1]))
    assert [o["thread"] for o in result["cells"][0]["owners"]] == [0, 1]


def test_scalar_bank_models():
    def analyze(offsets, operation="read", active=None):
        return analyze_banks(
            BankQuery(
                mode="scalar-warp",
                operation=operation,
                accesses=[
                    Access(lane=i, byte_offset=o, active=active is None or i in active)
                    for i, o in enumerate(offsets)
                ],
            )
        )

    assert analyze(list(range(0, 128, 4)))["serialization_rounds"] == 1
    assert analyze([i * 128 for i in range(32)])["serialization_rounds"] == 32
    broadcast = analyze([0] * 32)
    assert broadcast["serialization_rounds"] == 1 and len(broadcast["broadcasts"][0]) == 32
    assert analyze([0] * 32, "write")["status"] == "unsupported"
    assert analyze([0, 128], active=[0])["serialization_rounds"] == 1
    assert analyze([0, 128], active=[])["serialization_rounds"] == 0
    assert "serialization_rounds" not in analyze_banks(BankQuery(byte_offsets=[0, 128]))
    assert analyze_banks(BankQuery(byte_offsets=[3], access_width_bytes=4))["placement"][0][
        "banks"
    ] == [0, 1]
    for kwargs in [
        {"access_width_bytes": 16},
        {"accesses": [Access(lane=0, byte_offset=1)]},
        {"accesses": [Access(lane=0, byte_offset=0), Access(lane=0, byte_offset=4)]},
    ]:
        with pytest.raises(ValueError):
            analyze_banks(BankQuery(mode="scalar-warp", **kwargs))


def test_capture_nested_labels_ancestry_and_failure(tmp_path):
    with cuteviz.capture(tmp_path / "outer.json") as outer:
        assert cuteviz.inspect("a", CPULayout((4, 8), (8, 1))) is None
        cuteviz.inspect("slice", CPULayout((2, 8), (8, 1)), parent="a", transform="first two rows")
        with cuteviz.capture(tmp_path / "inner.json"):
            cuteviz.inspect("inner", CPULayout(8))
        cuteviz.inspect("a", CPULayout(8))
        cuteviz.inspect("bad", object())
    loaded = Capture.load(tmp_path / "outer.json")
    assert len(loaded.objects) == 4
    assert loaded.objects[0].id != loaded.objects[2].id
    assert loaded.objects[1].relationships[0].kind == "declared_parent"
    assert loaded.objects[0].source.line > 0
    assert loaded.outcome == "partial"
    assert len(Capture.load(tmp_path / "inner.json").objects) == 1
    assert outer.data.objects[0].views[0].layout.shape == [4, 8]
    with pytest.raises(RuntimeError, match="compile failed"):
        with cuteviz.capture(tmp_path / "failure.json"):
            cuteviz.inspect("kept", CPULayout(8))
            raise RuntimeError("compile failed")
    assert Capture.load(tmp_path / "failure.json").outcome == "partial"
    with pytest.raises(cuteviz.EmptyCaptureError, match="CUTE_DSL_NO_CACHE=1"):
        with cuteviz.capture(tmp_path / "empty.json"):
            pass
    assert Capture.load(tmp_path / "empty.json").outcome == "empty"


def test_malformed_captures(tmp_path):
    for change in [{"schema_version": "9.0"}, {"observation": "runtime"}, {"executable": "bad()"}]:
        with pytest.raises(ValidationError):
            Capture.model_validate(simple_capture().model_dump() | change)
    for shape, stride in [([0, 8], [1, 2]), ([2, 3], [1]), ([True], [1])]:
        with pytest.raises(ValidationError):
            Layout(shape=shape, stride=stride)
    duplicate = simple_capture().model_dump()
    duplicate["objects"] *= 2
    with pytest.raises(ValidationError):
        Capture.model_validate(duplicate)
    path = tmp_path / "malformed.json"
    path.write_text('{"schema_version":')
    with pytest.raises(ValueError):
        Capture.load(path)


def test_api_and_static_assets():
    client = TestClient(create_app(simple_capture()), base_url="http://127.0.0.1")
    assert client.get("/api/capture").json()["observation"] == "compile-time"
    assert client.post("/api/mapping/query", json={"object_id": "missing"}).status_code == 404
    assert (
        client.post(
            "/api/mapping/query", json={"object_id": "tensor", "extent": [100, 100]}
        ).status_code
        == 422
    )
    assert client.post("/api/mapping/query", json={"object_id": "tensor"}).status_code == 200
    assert client.post("/api/analysis/banks", json={"byte_offsets": [0, 128]}).status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/../../pyproject.toml").status_code == 404


def test_imported_capture_without_cuda():
    code = """
import sys
class NoCUDA:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('cutlass', 'cuda'):
            raise RuntimeError('Playback tried to import CUDA')
sys.meta_path.insert(0, NoCUDA())
from cuteviz.model import Capture
from cuteviz.mapping import query_mapping, MappingQuery
from cuteviz.server import create_app
c = Capture.load('examples/captures/sm_100a.cuteviz.json')
o = next(o for o in c.objects if o.kind == 'mma')
r = query_mapping(c, MappingQuery(object_id=o.id, role='C', extent=[1,1]))
assert r['cells'][0]['tmem']['status'] == 'known'
create_app(c)
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


def test_fixture_copy_links_and_mma_storage():
    capture = Capture.load(ROOT / "examples/captures/copy.cuteviz.json")
    obj = next(o for o in capture.objects if o.kind == "copy")
    query = MappingQuery(object_id=obj.id, role="S", extent=[1, 1])
    cell = query_mapping(capture, query)["cells"][0]
    assert cell["links"][0]["coordinate"] == [0, 0]
    query.role = "D"
    query.selection = Selection(role="S", coordinate=[0, 0])
    assert query_mapping(capture, query)["cells"][0]["selected"]
    for arch, stores in [
        ("sm_80", ["rmem"] * 3),
        ("sm_90a", ["smem", "smem", "rmem"]),
        ("sm_100a", ["smem", "smem", "tmem"]),
    ]:
        cap = Capture.load(ROOT / f"examples/captures/{arch}.cuteviz.json")
        mma = next(o for o in cap.objects if o.kind == "mma")
        assert [v.storage.value for v in mma.views] == stores
        for role in "ABC":
            result = query_mapping(cap, MappingQuery(object_id=mma.id, role=role, extent=[2, 2]))
            assert len(result["cells"]) == 4


def test_register_fragment_indices_are_not_physical_addresses():
    cap = Capture.load(ROOT / "examples/captures/gemm.cuteviz.json")
    acc = next(o for o in cap.objects if o.label == "warp_gemm.C")
    cell = query_mapping(cap, MappingQuery(object_id=acc.id, extent=[1, 1]))["cells"][0]
    assert cell["fragment_index"]["value"] == 0
    assert cell["byte_offset"]["status"] == "unavailable"
    assert not cell["owners"]


def test_canvas_cross_object_selection_respects_explicit_copy_domains():
    capture = Capture.load(ROOT / "examples/captures/copy.cuteviz.json")
    copy = next(o for o in capture.objects if o.kind == "copy")
    src, dst = copy.views
    standalone = next(o for o in capture.objects if o.label == "gA")
    selected = Selection(object_id=copy.id, role="S", coordinate=[1, 2])
    for target in [src.observed_object, dst.observed_object]:
        result = query_mapping(capture, MappingQuery(object_id=target, selection=selected))
        assert [c["coordinate"] for c in result["cells"] if c["selected"]] == [[1, 2]]
        assert not any(c["owners"] for c in result["cells"])
    # Equal coordinates in an unrelated snapshot do not establish identity.
    assert not any(
        c["selected"]
        for c in query_mapping(capture, MappingQuery(object_id=standalone.id, selection=selected))[
            "cells"
        ]
    )
    # Selection also travels from the explicit tensor snapshot back to the copy.
    selected = Selection(object_id=src.observed_object, role="tensor", coordinate=[1, 2])
    result = query_mapping(capture, MappingQuery(object_id=copy.id, role="D", selection=selected))
    assert [c["coordinate"] for c in result["cells"] if c["selected"]] == [[1, 2]]
    selected = Selection(thread=1, value=0)
    result = query_mapping(capture, MappingQuery(object_id=dst.observed_object, selection=selected))
    assert any(c["selected"] for c in result["cells"])


def test_canvas_does_not_reinterpret_mma_fragments_as_matrix_coordinates():
    capture = Capture.load(ROOT / "examples/captures/sm_100a.cuteviz.json")
    mma = next(o for o in capture.objects if o.kind == "mma")
    selected = Selection(object_id=mma.id, role="C", coordinate=[1, 2])
    fragment = mma.views[2].observed_object
    result = query_mapping(capture, MappingQuery(object_id=fragment, selection=selected))
    assert not any(c["selected"] for c in result["cells"])
    result = query_mapping(capture, MappingQuery(object_id=mma.id, role="A", selection=selected))
    assert {c["coordinate"][0] for c in result["cells"] if c["selected"]} == {1}


def test_full_window_at_large_tensor_boundary():
    capture = simple_capture(shape=(10**9, 10**9), stride=(10**9, 1))
    result = query_mapping(
        capture, MappingQuery(object_id="tensor", start=[10**9 - 64, 10**9 - 64], extent=[64, 64])
    )
    assert len(result["cells"]) == 4096
    assert result["cells"][-1]["coordinate"] == [10**9 - 1, 10**9 - 1]
