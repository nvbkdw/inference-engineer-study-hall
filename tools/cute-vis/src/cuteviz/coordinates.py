"""Portable, bounded coordinate ancestry. No compiler or GPU dependency."""

from collections import deque

from .algebra import evaluate, leaves
from .model import known, unknown


def dims(view):
    return [n for _, n in leaves(view.shape)]


def matvec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def in_bounds(coordinate, shape):
    return len(coordinate) == len(shape) and all(
        type(n) is int and 0 <= c < n for c, n in zip(coordinate, shape)
    )


def contains(origin, matrix, shape, coordinate):
    """Membership in rectangular/mixed-radix CuTe tiles, including alias modes."""
    used = set()
    for r, (o, row, c) in enumerate(zip(origin, matrix, coordinate)):
        modes = sorted((stride, shape[i], i) for i, stride in enumerate(row) if stride)
        covered = 1
        for stride, extent, i in modes:
            if type(extent) is not int or stride < covered or i in used:
                return False
            covered = stride * extent
            used.add(i)
        remaining = c - o
        if remaining < 0:
            return False
        for stride, extent, _ in reversed(modes):
            digit, remaining = divmod(remaining, stride)
            if digit >= extent:
                return False
        if remaining:
            return False
    return True


class CoordinateGraph:
    def __init__(self, capture, bindings):
        self.objects = {obj.id: obj for obj in capture.objects}
        self.views = {(obj.id, v.role): v for obj in capture.objects for v in obj.views}
        self.edges = {}
        self.diagnostics = []
        self.has_ancestry = False
        for obj in capture.objects:
            if not obj.views:
                continue
            source = (obj.id, obj.views[0].role)
            for rel in obj.relationships:
                if rel.kind != "derived_parent":
                    continue
                self.has_ancestry = True
                mapping = rel.mapping
                if mapping is None:
                    self.diagnostics.append(f"{obj.label}: {rel.mapping_status.reason}")
                    continue
                missing = mapping.parameters.keys() - bindings.keys()
                if missing:
                    self.diagnostics.append(
                        f"{obj.label}: choose inspection values for {', '.join(sorted(missing))}; these are runtime coordinates."
                    )
                    continue
                origin = mapping.origin.copy()
                for name, parameter in mapping.parameters.items():
                    origin = [
                        o + bindings[name] * a for o, a in zip(origin, parameter.coefficients)
                    ]
                parent = self.objects[rel.target]
                self.edges.setdefault(source, []).append(
                    ((parent.id, parent.views[0].role), mapping.matrix, origin, mapping.operation)
                )
        # Copy operands have an explicit same-logical-element correspondence.
        for obj in capture.objects:
            if obj.kind != "copy":
                continue
            for view in obj.views:
                key = (obj.id, view.role)
                for other in obj.views:
                    if other.role != view.role and other.shape == view.shape:
                        self.edges.setdefault(key, []).append(
                            ((obj.id, other.role), None, None, "copy")
                        )
                observed = self.objects.get(view.observed_object)
                if observed and observed.views and observed.views[0].shape == view.shape:
                    target = (observed.id, observed.views[0].role)
                    self.edges.setdefault(key, []).append((target, None, None, "operand"))
                    self.edges.setdefault(target, []).append((key, None, None, "operand"))

    def anchors(self, key, coordinate):
        found = set()
        queue = deque([(key, tuple(coordinate))])
        while queue and len(found) < 128:
            key, coordinate = queue.popleft()
            marker = (*key, coordinate)
            if marker in found:
                continue
            found.add(marker)
            for target, matrix, origin, _ in self.edges.get(key, []):
                mapped = (
                    coordinate
                    if matrix is None
                    else tuple(o + d for o, d in zip(origin, matvec(matrix, coordinate)))
                )
                queue.append((target, mapped))
        return found

    def trail(self, key, coordinate):
        """Only real backing ancestry goes in the coordinate explanation trail."""
        links, seen = [], {key}
        queue = deque([(key, list(coordinate), [])])
        while queue and len(links) < 16:
            source, coord, path = queue.popleft()
            for target, matrix, origin, operation in self.edges.get(source, []):
                if target in seen or operation == "copy":
                    continue
                seen.add(target)
                if matrix is None:
                    queue.append((target, coord, path))
                    continue
                mapped = [o + d for o, d in zip(origin, matvec(matrix, coord))]
                view, obj = self.views[target], self.objects[target[0]]
                element = unknown("Backing layout is unavailable")
                byte = unknown("Backing byte offset is unavailable")
                if view.layout and view.address_unit == "element":
                    try:
                        element = known(evaluate(view.layout, mapped))
                        bits = view.element_bits.value
                        if type(bits) is int and bits % 8 == 0 and not view.pointer_swizzle:
                            byte = known(element.value * bits // 8)
                    except ValueError:
                        pass
                formula = [
                    f"{o}"
                    + "".join(f" + {a} × {c}" for a, c in zip(row, coord) if a)
                    + f" = {value}"
                    for o, row, value in zip(origin, matrix, mapped)
                ]
                links.append(
                    {
                        "object_id": obj.id,
                        "label": obj.label,
                        "role": view.role,
                        "coordinate": mapped,
                        "in_bounds": in_bounds(mapped, dims(view)),
                        "element_offset": element.model_dump(),
                        "byte_offset": byte.model_dump(),
                        "operation": operation,
                        "calculation": formula,
                        "path": path + [operation],
                    }
                )
                queue.append((target, mapped, path + [operation]))
        return links

    def footprints(self, focus_id, target_key):
        if focus_id is None or focus_id not in self.objects:
            return []
        focus = self.objects[focus_id]
        results = []
        for source_view in focus.views:
            shape = dims(source_view)
            if not shape or not all(type(n) is int for n in shape):
                continue
            identity = [[int(i == j) for j in range(len(shape))] for i in range(len(shape))]
            queue = deque([((focus.id, source_view.role), identity, [0] * len(shape), False)])
            seen = set()
            while queue and len(seen) < 128:
                key, matrix, origin, derived = queue.popleft()
                if key in seen:
                    continue
                seen.add(key)
                if key == target_key and derived:
                    maximum = [
                        o + sum(a * (n - 1) for a, n in zip(row, shape))
                        for o, row in zip(origin, matrix)
                    ]
                    exact = True
                    used = set()
                    for row in matrix:
                        running = 1
                        for stride, extent, i in sorted(
                            (a, n, i) for i, (a, n) in enumerate(zip(row, shape)) if a
                        ):
                            exact &= stride == running and i not in used
                            used.add(i)
                            running = stride * extent
                    results.append(
                        {
                            "label": focus.label,
                            "source_id": focus.id,
                            "shape": shape,
                            "origin": origin,
                            "matrix": matrix,
                            "maximum": maximum,
                            "exact": exact,
                        }
                    )
                for target, edge_matrix, edge_origin, _ in self.edges.get(key, []):
                    if edge_matrix is None:
                        queue.append((target, matrix, origin, derived))
                    else:
                        queue.append(
                            (
                                target,
                                matmul(edge_matrix, matrix),
                                [a + b for a, b in zip(edge_origin, matvec(edge_matrix, origin))],
                                True,
                            )
                        )
        return results
