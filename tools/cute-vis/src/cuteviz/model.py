"""Versioned data contract. Captures contain data, never executable expressions."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

MAX_CAPTURE_BYTES = 32 * 1024 * 1024
MAX_MAPPING_ENTRIES = 262144


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Observation(Model):
    status: Literal["known", "symbolic", "unsupported", "unavailable"] = "known"
    value: JsonValue = None
    reason: str | None = None

    @model_validator(mode="after")
    def explain_unknown(self):
        if self.status != "known" and not self.reason:
            raise ValueError("Unknown fields require a reason")
        return self


def known(value) -> Observation:
    return Observation(value=value)


def unknown(reason: str, status="unavailable", value=None) -> Observation:
    return Observation(status=status, reason=reason, value=value)


def validate_tree(tree, *, shape=False, depth=0):
    if depth > 16:
        raise ValueError("Layout nesting exceeds 16 levels")
    if type(tree) is int:
        if abs(tree) > 2**63 - 1 or (shape and tree < 1):
            raise ValueError("Layout integers must be signed 64-bit; extents must be positive")
    elif isinstance(tree, list) and 0 < len(tree) <= 32:
        for child in tree:
            validate_tree(child, shape=shape, depth=depth + 1)
    elif isinstance(tree, dict):
        obs = Observation.model_validate(tree)
        if obs.status == "known":
            raise ValueError("Static tree leaves must be integers")
    else:
        raise ValueError("Expected an integer, nested modes, or an unresolved field")


class Layout(Model):
    kind: Literal["affine", "compose", "swizzle"] = "affine"
    shape: JsonValue = None
    stride: JsonValue = None
    left: Layout | None = None
    right: Layout | None = None
    offset: int = 0
    bits: int = Field(default=0, ge=0, le=16)
    base: int = Field(default=0, ge=0, le=32)
    shift: int = Field(default=0, ge=-32, le=32)

    @model_validator(mode="after")
    def valid_expression(self):
        if self.kind == "affine":
            validate_tree(self.shape, shape=True)
            validate_tree(self.stride)

            def congruent(a, b):
                if isinstance(a, list) or isinstance(b, list):
                    return (
                        isinstance(a, list)
                        and isinstance(b, list)
                        and len(a) == len(b)
                        and all(congruent(x, y) for x, y in zip(a, b))
                    )
                return True

            if not congruent(self.shape, self.stride):
                raise ValueError("Shape and stride mode trees must match")
        elif self.kind == "compose":
            if self.left is None or self.right is None:
                raise ValueError("Composition needs left and right expressions")
            self.shape = self.right.shape
        elif abs(self.shift) < self.bits:
            raise ValueError("Swizzle bit fields must not overlap")
        return self


class Source(Model):
    id: str
    path: str
    text: str = Field(max_length=1_000_000)


class Span(Model):
    source_id: str
    line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class CoordinateParameter(Model):
    label: str = Field(max_length=200)
    coefficients: list[int] = Field(min_length=1, max_length=32)


class CoordinateMap(Model):
    """Parent logical coordinates = origin + matrix @ child + parameter terms."""

    operation: Literal["cute.local_tile", "cute.slice"]
    origin: list[int] = Field(min_length=1, max_length=32)
    matrix: list[list[int]] = Field(min_length=1, max_length=32)
    parameters: dict[str, CoordinateParameter] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def valid_dimensions(self):
        n = len(self.origin)
        if len(self.matrix) != n or not self.matrix[0] or len(self.matrix[0]) > 32:
            raise ValueError("Coordinate map dimensions do not match")
        if any(len(row) != len(self.matrix[0]) for row in self.matrix):
            raise ValueError("Coordinate matrix must be rectangular")
        if any(len(p.coefficients) != n for p in self.parameters.values()):
            raise ValueError("Coordinate parameter dimension does not match")
        values = self.origin + [v for row in self.matrix for v in row]
        values += [v for p in self.parameters.values() for v in p.coefficients]
        if any(abs(v) > 2**63 - 1 for v in values):
            raise ValueError("Coordinate coefficients must be signed 64-bit integers")
        return self


class Relationship(Model):
    target: str
    kind: Literal["declared_parent", "operand", "derived_parent"]
    transform: str | None = None
    mapping: CoordinateMap | None = None
    mapping_status: Observation = Field(
        default_factory=lambda: unknown("This relationship has no captured coordinate transform")
    )


class View(Model):
    """A logical operand or tensor view, with independently known mapping fields."""

    role: str
    shape: JsonValue
    storage: Observation
    dtype: Observation = Field(default_factory=lambda: unknown("No tensor dtype"))
    element_bits: Observation = Field(default_factory=lambda: unknown("No tensor dtype width"))
    layout: Layout | None = None
    layout_status: Observation = Field(default_factory=lambda: unknown("No address mapping"))
    tv_layout: Layout | None = None
    ownership: Observation = Field(
        default_factory=lambda: unknown("No producing operation was probed")
    )
    address_unit: Literal["element", "tmem", "none"] = "element"
    base_address: Observation = Field(
        default_factory=lambda: unknown("Runtime base pointer is unresolved")
    )
    # For tcgen05: compiler-created fragment indexed by a TV value index.
    fragment_layout: Layout | None = None
    pointer_swizzle: Layout | None = None
    pointer_alignment: Observation = Field(
        default_factory=lambda: unknown("Pointer alignment unavailable")
    )
    observed_object: str | None = None

    @model_validator(mode="after")
    def valid_shape(self):
        validate_tree(self.shape, shape=True)
        return self


class CapturedObject(Model):
    id: str
    label: str = Field(max_length=1000)
    kind: Literal["layout", "tensor", "copy", "mma", "unsupported"]
    source: Span | None = None
    fields: dict[str, Observation] = Field(default_factory=dict)
    views: list[View] = Field(default_factory=list, max_length=8)
    relationships: list[Relationship] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


class Capture(Model):
    schema_version: Literal["1.0", "1.1"] = "1.1"
    producer_version: str = "0.1.0"
    observation: Literal["compile-time"] = "compile-time"
    compiler: Observation = Field(
        default_factory=lambda: unknown("Imported or CPU-authored capture")
    )
    target: Observation = Field(default_factory=lambda: unknown("Target was not supplied"))
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    sources: list[Source] = Field(default_factory=list, max_length=100)
    objects: list[CapturedObject] = Field(default_factory=list, max_length=5000)
    diagnostics: list[str] = Field(default_factory=list)
    outcome: Literal["complete", "partial", "empty"] = "complete"

    @model_validator(mode="after")
    def check_links(self):
        ids = {o.id for o in self.objects}
        by_id = {o.id: o for o in self.objects}
        sources = {s.id: s for s in self.sources}
        if len(ids) != len(self.objects) or len(sources) != len(self.sources):
            raise ValueError("Duplicate object or source IDs")
        for obj in self.objects:
            if obj.source:
                span = obj.source
                if span.source_id not in sources:
                    raise ValueError("Unknown source ID")
                if not span.line <= span.end_line <= len(sources[span.source_id].text.splitlines()):
                    raise ValueError("Source span is outside the embedded snapshot")
            for rel in obj.relationships:
                if rel.target not in ids:
                    raise ValueError("Dangling object relationship")
                if rel.mapping:
                    from .algebra import leaves

                    parent = by_id[rel.target]
                    if rel.kind != "derived_parent" or not obj.views or not parent.views:
                        raise ValueError("Coordinate maps require derived tensor parents")
                    if len(rel.mapping.origin) != len(leaves(parent.views[0].shape)) or len(
                        rel.mapping.matrix[0]
                    ) != len(leaves(obj.views[0].shape)):
                        raise ValueError("Coordinate map does not match its tensor shapes")
            for view in obj.views:
                if view.observed_object and view.observed_object not in ids:
                    raise ValueError("Dangling operand reference")
        # Derived ancestry must be finite even in untrusted imported captures.
        depths = {}

        def visit(oid, path):
            if oid in path or len(path) > 16:
                raise ValueError("Derived tensor ancestry is cyclic or exceeds 16 levels")
            if oid in depths:
                return depths[oid]
            depth = 0
            for relation in by_id[oid].relationships:
                if relation.kind == "derived_parent":
                    depth = max(depth, 1 + visit(relation.target, path | {oid}))
            if depth > 16:
                raise ValueError("Derived tensor ancestry is cyclic or exceeds 16 levels")
            depths[oid] = depth
            return depth

        for oid in ids:
            visit(oid, set())
        return self

    def save(self, path: str | Path):
        path = Path(path)
        data = self.model_dump_json(indent=2)
        # Validate references added incrementally during capture, before replacing a file.
        self.model_validate_json(data)
        if len(data.encode()) > MAX_CAPTURE_BYTES:
            raise ValueError("Capture exceeds the 32 MiB import limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".cuteviz-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(data)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open("rb") as stream:
            data = stream.read(MAX_CAPTURE_BYTES + 1)
        if len(data) > MAX_CAPTURE_BYTES:
            raise ValueError("Capture exceeds 32 MiB")
        return cls.model_validate(json.loads(data))
