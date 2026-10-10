import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from cuteviz.mapping import MappingQuery, Selection, query_mapping
from cuteviz.model import Capture, CoordinateMap, Relationship

ROOT = Path(__file__).resolve().parents[1]


def tile_capture():
    return Capture.load(ROOT / "examples/captures/tiles.cuteviz.json")


def one(capture, obj, coord, **options):
    axes = list(range(min(2, len(coord))))
    return query_mapping(
        capture,
        MappingQuery(
            object_id=obj.id,
            axes=axes,
            start=coord[:2] if len(coord) > 1 else [coord[0], 0],
            extent=[1, 1],
            fixed={i: c for i, c in enumerate(coord) if i >= 2},
            **options,
        ),
    )["cells"][0]


def test_backing_coordinates_selection_and_footprint():
    c = tile_capture()
    tile, backing, column = c.objects
    bindings = {"blockIdx.x": 2, "blockIdx.y": 1}
    cell = one(c, tile, [3, 5], bindings=bindings)
    link = cell["coordinate_mappings"][0]
    assert link["coordinate"] == [19, 21]
    assert link["element_offset"]["value"] == 19 * 64 + 21
    assert link["byte_offset"]["value"] == (19 * 64 + 21) * 4
    assert link["in_bounds"]
    result = query_mapping(
        c,
        MappingQuery(
            object_id=backing.id,
            extent=[32, 64],
            focus_object_id=tile.id,
            bindings=bindings,
            selection=Selection(object_id=tile.id, coordinate=[3, 5]),
        ),
    )
    assert [x["coordinate"] for x in result["cells"] if x["selected"]] == [[19, 21]]
    assert {tuple(x["coordinate"]) for x in result["cells"] if x["in_focus"]} == {
        (i, j) for i in range(16, 24) for j in range(16, 32)
    }
    assert result["footprints"][0]["start"] == [16, 16]
    assert result["footprints"][0]["extent"] == [8, 16]
    assert one(
        c,
        tile,
        [3, 5],
        bindings=bindings,
        selection=Selection(object_id=backing.id, coordinate=[19, 21]),
    )["selected"]
    assert not one(
        c,
        tile,
        [2, 5],
        bindings=bindings,
        selection=Selection(object_id=backing.id, coordinate=[19, 21]),
    )["selected"]
    trail = one(c, column, [3], bindings=bindings)["coordinate_mappings"]
    assert [link["coordinate"] for link in trail] == [[3, 3], [19, 19]]
    assert query_mapping(
        c, MappingQuery(object_id=backing.id, bindings=bindings, focus_object_id=column.id)
    )["footprints"][0]["extent"] == [8, 1]


def test_missing_bindings_and_textual_parents_do_not_invent_coordinates():
    c = tile_capture()
    tile, backing, _ = c.objects
    result = query_mapping(c, MappingQuery(object_id=tile.id, extent=[1, 1]))
    assert not result["cells"][0]["coordinate_mappings"]
    assert "blockIdx.x" in result["mapping_diagnostics"][0]
    tile.relationships = [
        Relationship(
            target=backing.id, kind="declared_parent", transform="local_tile((8,16),(2,1))"
        )
    ]
    assert not one(c, tile, [3, 5], selection=Selection(object_id=backing.id, coordinate=[19, 21]))[
        "selected"
    ]


def test_ancestry_import_validation_and_backward_compatibility():
    c = tile_capture()
    assert Capture.model_validate_json(c.model_dump_json()).schema_version == "1.1"
    old = Capture.load(ROOT / "examples/captures/sm_80.cuteviz.json")
    assert old.schema_version == "1.0"
    c.objects[1].relationships = [Relationship(target=c.objects[0].id, kind="derived_parent")]
    with pytest.raises(ValidationError, match="cyclic"):
        Capture.model_validate_json(c.model_dump_json())
    c = tile_capture()
    c.objects[0].relationships[0].mapping.matrix = [[1], [0]]
    with pytest.raises(ValidationError, match="does not match"):
        Capture.model_validate_json(c.model_dump_json())


def test_aliases_and_out_of_bounds_are_explicit():
    c = tile_capture()
    tile, backing, _ = c.objects
    tile.relationships[0].mapping = CoordinateMap(
        operation="cute.local_tile", origin=[30, 60], matrix=[[1, 0], [0, 1]]
    )
    out = one(c, tile, [3, 5])["coordinate_mappings"][0]
    assert out["coordinate"] == [33, 65] and not out["in_bounds"]
    result = query_mapping(
        c, MappingQuery(object_id=backing.id, extent=[32, 64], focus_object_id=tile.id)
    )
    assert sum(x["in_focus"] for x in result["cells"]) == 8
    assert result["footprints"][0]["extent"] == [2, 4]
    tile.relationships[0].mapping = CoordinateMap(
        operation="cute.slice", origin=[0, 0], matrix=[[0, 0], [0, 1]]
    )
    selected = query_mapping(
        c,
        MappingQuery(
            object_id=tile.id, selection=Selection(object_id=backing.id, coordinate=[0, 3])
        ),
    )
    assert [x["coordinate"] for x in selected["cells"] if x["selected"]] == [
        [i, 3] for i in range(8)
    ]
    # A diagonal affine image is not its enclosing rectangle.
    tile.relationships[0].mapping = CoordinateMap(
        operation="cute.slice", origin=[0, 0], matrix=[[1, 0], [1, 0]]
    )
    result = query_mapping(
        c, MappingQuery(object_id=backing.id, extent=[1, 1], focus_object_id=tile.id)
    )
    assert not result["footprints"][0]["exact"]


@pytest.mark.compiler
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe compiler required")
def test_real_compiler_backing_transform_and_probe_neutrality(tmp_path):
    path = tmp_path / "ancestry.json"
    result = subprocess.run(
        [sys.executable, "tests/compiler_ancestry.py", str(path)],
        cwd=ROOT,
        env=os.environ | {"CUTE_DSL_ARCH": "sm_80", "CUTE_DSL_NO_CACHE": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    c = Capture.load(path)
    by_label = {obj.label: obj for obj in c.objects}
    cases = [
        ("static", [3, 5], {}, [11, 37]),
        ("dynamic", [3, 5], {"blockIdx.x": 2, "blockIdx.y": 1}, [19, 21]),
        ("projected", [3, 5], {"blockIdx.x": 2, "blockIdx.y": 1}, [19, 21]),
        ("rest", [3, 5, 2], {}, [19, 21]),
        ("nested", [1, 3], {}, [11, 43]),
        ("column", [3], {"blockIdx.x": 2, "blockIdx.y": 1}, [19, 19]),
        ("shifted", [2, 3], {"blockIdx.x": 1, "blockIdx.y": 2}, [14, 19]),
        ("edge", [2, 3], {}, [10, 11]),
    ]
    for label, coordinate, bindings, expected in cases:
        obj = by_label[label]
        assert not obj.diagnostics, (label, obj.diagnostics)
        trail = one(c, obj, coordinate, bindings=bindings)["coordinate_mappings"]
        assert trail[-1]["coordinate"] == expected, (label, trail)
    unsupported = by_label["unsupported_gather"]
    assert unsupported.views and unsupported.relationships
    assert unsupported.relationships[0].mapping is None
    assert "unit-stride" in unsupported.relationships[0].mapping_status.reason
    assert len([obj for obj in c.objects if obj.label.endswith(".backing")]) == 2
