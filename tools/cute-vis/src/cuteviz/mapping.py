"""Bounded CPU queries. All tensor, ownership, and cross-operand math lives here."""

from functools import lru_cache
from math import prod
from typing import Annotated

from pydantic import Field

from .algebra import compile_layout, evaluate, explain, leaves, size
from .coordinates import CoordinateGraph, contains
from .model import MAX_MAPPING_ENTRIES, Capture, Layout, Model, known, unknown


class Selection(Model):
    object_id: str | None = None
    role: str | None = None
    coordinate: list[int] | None = Field(default=None, max_length=32)
    thread: int | None = Field(default=None, ge=0)
    value: int | None = Field(default=None, ge=0)
    warp: int | None = Field(default=None, ge=0)
    warpgroup: int | None = Field(default=None, ge=0)


class MappingQuery(Model):
    object_id: str
    role: str | None = None
    axes: list[int] | None = Field(default=None, min_length=1, max_length=2)
    fixed: dict[int, int] = Field(default_factory=dict)
    start: list[int] = Field(default=[0, 0], min_length=2, max_length=2)
    extent: list[int] = Field(default=[16, 16], min_length=2, max_length=2)
    selection: Selection | None = None
    focus_object_id: str | None = None
    bindings: dict[str, Annotated[int, Field(strict=True, ge=-(2**53 - 1), le=2**53 - 1)]] = Field(
        default_factory=dict, max_length=64
    )


@lru_cache(maxsize=8)
def inverse_tv(serialized):
    layout = Layout.model_validate_json(serialized)
    if not isinstance(layout.shape, list) or len(layout.shape) != 2:
        raise ValueError("Expected a (thread, value) layout")
    threads, values = map(size, layout.shape)
    if threads * values > MAX_MAPPING_ENTRIES:
        raise ValueError(
            f"Ownership mapping exceeds {MAX_MAPPING_ENTRIES} entries; probe a smaller tile"
        )
    mapping = compile_layout(layout)
    inverse = {}
    for t in range(threads):
        for v in range(values):
            index = int(mapping((t, v)))
            inverse.setdefault(index, []).append((t, v))
    return inverse


def index_of(coord, shape):
    index, stride = 0, 1
    for c, (_, n) in zip(coord, leaves(shape)):
        index += c * stride
        stride *= n
    return index


def owners_at(view, coord):
    if view.tv_layout is None:
        return []
    return inverse_tv(view.tv_layout.model_dump_json()).get(index_of(coord, view.shape), [])


def find_object(capture, object_id):
    try:
        return next(o for o in capture.objects if o.id == object_id)
    except StopIteration:
        raise KeyError(f"Unknown object {object_id!r}") from None


def find_view(obj, role):
    if not obj.views:
        raise ValueError("No supported view was captured for this object")
    try:
        return next(v for v in obj.views if role is None or v.role == role)
    except StopIteration:
        raise ValueError(f"Unknown operand {role!r}") from None


def matches(obj, view, coord, owners, selection):
    if selection is None:
        return False
    if any(getattr(selection, f) is not None for f in ["thread", "value", "warp", "warpgroup"]):
        return any(
            (selection.thread is None or t == selection.thread)
            and (selection.value is None or v == selection.value)
            and (selection.warp is None or t // 32 == selection.warp)
            and (selection.warpgroup is None or t // 128 == selection.warpgroup)
            for t, v in owners
        )
    sc = selection.coordinate
    if sc is None:
        return False
    if selection.role in (None, view.role):
        return coord == sc
    if obj.kind == "copy":
        return coord == sc
    if obj.kind == "mma" and len(sc) == len(coord) == 2:
        axes = {"A": ("M", "K"), "B": ("N", "K"), "C": ("M", "N")}
        a, b = axes[selection.role], axes[view.role]
        return all(coord[b.index(axis)] == sc[a.index(axis)] for axis in set(a) & set(b))
    return False


def selection_contexts(capture, obj, view, selection):
    """Resolve explicit copy operand links without inferring tensor aliasing.

    A copy's observed full tensor has the same coordinate domain only when its
    shape matches. MMA fragment snapshots have different domains and are not
    treated as matrix coordinates. Declared ancestry is explanatory, not algebra.
    """
    if selection is None:
        return []
    source = find_object(capture, selection.object_id) if selection.object_id else obj
    coordinate = selection.coordinate is not None
    contexts = []
    if not coordinate or source.id == obj.id:
        contexts.append((obj, view, selection))
    source_view = find_view(source, selection.role) if coordinate else None
    for operation in capture.objects:
        if operation.kind != "copy":
            continue
        target_roles = [
            v
            for v in operation.views
            if (operation.id == obj.id and v.role == view.role)
            or (v.observed_object == obj.id and v.shape == view.shape)
        ]
        if not target_roles:
            continue
        if not coordinate:
            contexts.extend((operation, v, selection) for v in target_roles)
        elif source.id == operation.id:
            contexts.extend((operation, v, selection) for v in target_roles)
        else:
            for operand in operation.views:
                if operand.observed_object == source.id and operand.shape == source_view.shape:
                    linked = selection.model_copy(update={"role": operand.role})
                    contexts.extend((operation, v, linked) for v in target_roles)
    return contexts


def cell_at(obj, view, coord, selection):
    element = unknown(view.layout_status.reason or "No element address mapping")
    byte = unknown("Byte offset requires a known address layout and byte-sized dtype")
    tmem = unknown("This view does not use tensor memory")
    fragment_index = unknown("This view is not a thread-local register fragment")
    explanation = f"Logical coordinate ({', '.join(map(str, coord))}) in operand {view.role}. "
    internal_owners = (
        owners_at(view, coord)
        if view.ownership.status == "known" or view.address_unit == "tmem"
        else []
    )
    owners = internal_owners if view.ownership.status == "known" else []

    def tensor_memory(offset):
        bits = view.element_bits.value
        if type(bits) is not int:
            return unknown("Tensor-memory coordinates require the element width")
        packed, bit_offset = divmod(offset * bits, 32)
        return known(
            {"data_path": packed >> 16, "column": packed & 65535, "bit_offset": bit_offset}
        )

    if view.layout is not None:
        try:
            offset = evaluate(view.layout, coord)
            if view.address_unit == "tmem":
                tmem = tensor_memory(offset)
                explanation += (
                    f"Tensor-memory element offset {offset} → {tmem.value} (32-bit columns). "
                )
            elif view.address_unit == "none" and view.storage.value == "rmem":
                fragment_index = known(offset)
                explanation += f"Thread-local fragment index: {explain(view.layout, coord)}. Physical register allocation is unavailable. "
            elif view.address_unit == "element":
                element = known(offset)
                explanation += f"Element offset: {explain(view.layout, coord)}. "
                bits = view.element_bits.value
                if view.pointer_swizzle:
                    byte = unknown(view.layout_status.reason)
                    explanation += "Offset is before the position-dependent pointer swizzle; byte placement needs the runtime base phase. "
                elif type(bits) is int and bits % 8 == 0:
                    byte = known(offset * (bits // 8))
                    explanation += f"{offset} elements × {bits // 8} bytes = {byte.value} bytes from the view base. "
        except ValueError as exc:
            element = unknown(str(exc), "symbolic")
    if view.fragment_layout and view.address_unit == "tmem" and internal_owners:
        # tcgen05 logical value indices enumerate the compiler-created fragment.
        value = internal_owners[0][1]
        packed = int(compile_layout(view.fragment_layout)(value))
        tmem = tensor_memory(packed)
        explanation += (
            f"{view.layout_status.value}. Value {value} → tensor-memory element offset {packed}; "
            f"{tmem.value} in 32-bit column units, relative to the allocation. "
        )
    if owners:
        explanation += (
            "; ".join(f"thread {t} (warp {t // 32}), value index {v}" for t, v in owners) + ". "
        )
        explanation += "Value indices do not specify physical registers."
    elif view.ownership.reason:
        explanation += view.ownership.reason + "."
    links = []
    if obj.kind == "copy":
        for other in obj.views:
            if other.role != view.role and other.tv_layout:
                for t, v in owners_at(other, coord):
                    links.append(
                        {
                            "role": other.role,
                            "coordinate": coord,
                            "thread": t,
                            "value": v,
                        }
                    )
    return {
        "coordinate": coord,
        "element_offset": element.model_dump(),
        "byte_offset": byte.model_dump(),
        "tmem": tmem.model_dump(),
        "fragment_index": fragment_index.model_dump(),
        "owners": [
            {"thread": t, "value": v, "warp": t // 32, "warpgroup": t // 128} for t, v in owners
        ],
        "selected": matches(
            obj, view, coord, owners if view.ownership.status == "known" else [], selection
        ),
        "links": links,
        "explanation": explanation,
        "unswizzled_offset": (
            evaluate(view.layout.right, coord)
            if view.layout and view.layout.kind == "compose" and view.layout.left.kind == "swizzle"
            else None
        ),
    }


def query_mapping(capture: Capture, query: MappingQuery):
    obj = find_object(capture, query.object_id)
    view = find_view(obj, query.role)
    dimensions = leaves(view.shape)
    axes = query.axes if query.axes is not None else list(range(min(2, len(dimensions))))
    if len(set(axes)) != len(axes) or any(a < 0 or a >= len(dimensions) for a in axes):
        raise ValueError("Choose one or two distinct valid axes")
    if any(n < 1 for n in query.extent) or prod(query.extent) > 4096:
        raise ValueError("A query must contain between 1 and 4096 visible cells")
    if any(s < 0 for s in query.start):
        raise ValueError("Page starts cannot be negative")
    if any(a < 0 or a >= len(dimensions) or a in axes for a in query.fixed):
        raise ValueError("Fixed coordinates must address non-visible axes")
    fixed = [query.fixed.get(i, 0) for i in range(len(dimensions))]
    for i, (_, n) in enumerate(dimensions):
        if type(n) is not int:
            raise ValueError(
                f"Mode {dimensions[i][0]} has a symbolic extent; inspect a static slice"
            )
        if fixed[i] < 0 or fixed[i] >= n:
            raise ValueError(f"Fixed coordinate for mode {i} is out of bounds")
    shape = [dimensions[a][1] for a in axes]
    if len(axes) == 1:
        shape.append(1)
    if any(start >= n for start, n in zip(query.start, shape)):
        raise ValueError("Page starts outside the view")
    extent = [min(e, n - start) for e, n, start in zip(query.extent, shape, query.start)]
    if query.selection and query.selection.coordinate is not None:
        selected_object = (
            find_object(capture, query.selection.object_id) if query.selection.object_id else obj
        )
        selected_view = find_view(selected_object, query.selection.role)
        sdims = leaves(selected_view.shape)
        sc = query.selection.coordinate
        if len(sc) != len(sdims) or any(
            type(n) is not int or not 0 <= c < n for c, (_, n) in zip(sc, sdims)
        ):
            raise ValueError("Selection coordinate is outside its operand")
    contexts = selection_contexts(capture, obj, view, query.selection)
    graph = CoordinateGraph(capture, query.bindings)
    key = (obj.id, view.role)
    selected_anchors = set()
    if graph.has_ancestry and query.selection and query.selection.coordinate is not None:
        selected_anchors = graph.anchors(
            (selected_object.id, selected_view.role), query.selection.coordinate
        )
    footprints = graph.footprints(query.focus_object_id, key)
    cells = []
    for y in range(extent[0]):
        for x in range(extent[1]):
            coord = fixed.copy()
            coord[axes[0]] = y + query.start[0]
            if len(axes) == 2:
                coord[axes[1]] = x + query.start[1]
            cell = cell_at(obj, view, coord, None)
            cell["selected"] = any(
                matches(
                    context_obj,
                    context_view,
                    coord,
                    owners_at(context_view, coord)
                    if context_view.ownership.status == "known"
                    else [],
                    selection,
                )
                for context_obj, context_view, selection in contexts
            )
            if selected_anchors and not cell["selected"]:
                cell["selected"] = bool(selected_anchors & graph.anchors(key, coord))
            cell["in_focus"] = any(
                contains(f["origin"], f["matrix"], f["shape"], coord) for f in footprints
            )
            cell["coordinate_mappings"] = graph.trail(key, coord) if graph.has_ancestry else []
            for link in cell["coordinate_mappings"]:
                cell["explanation"] += (
                    f" Backing tensor {link['label']}: ({', '.join(map(str, link['coordinate']))}); "
                    + "; ".join(link["calculation"])
                    + "."
                )
                if not link["in_bounds"]:
                    cell["explanation"] += (
                        " This coordinate is outside the backing tensor; this access needs a kernel predicate."
                    )
            cell.update(x=x, y=y)
            cells.append(cell)
    return {
        "object_id": obj.id,
        "footprints": [
            {
                "label": f["label"],
                "source_id": f["source_id"],
                "exact": f["exact"],
                "start": [max(0, f["origin"][a]) for a in axes] + ([0] if len(axes) == 1 else []),
                "extent": [
                    max(0, min(dimensions[a][1] - 1, f["maximum"][a]) - max(0, f["origin"][a]) + 1)
                    for a in axes
                ]
                + ([1] if len(axes) == 1 else []),
            }
            for f in footprints
            if all(
                f["origin"][a] <= fixed[a] <= f["maximum"][a]
                for a in range(len(dimensions))
                if a not in axes
            )
        ],
        "mapping_diagnostics": graph.diagnostics,
        "role": view.role,
        "axes": axes,
        "start": query.start,
        "extent": extent,
        "dimensions": [{"path": p, "size": n} for p, n in dimensions],
        "cells": cells,
        "total_elements": size(view.shape),
        "visible_cells": len(cells),
        "storage": view.storage.model_dump(),
        "ownership": view.ownership.model_dump(),
        "address_unit": view.address_unit,
        "layout_status": view.layout_status.model_dump(),
    }
