"""Small boundary around tensor-layouts; CuTe and this library name composition differently."""

from functools import lru_cache
from math import prod

from tensor_layouts import ComposedLayout, Swizzle
from tensor_layouts import Layout as CPULayout

from .model import Layout


def leaves(tree, prefix=()):
    if isinstance(tree, list):
        return [item for i, child in enumerate(tree) for item in leaves(child, prefix + (i,))]
    return [(".".join(map(str, prefix)) or "0", tree)]


def size(tree):
    values = [v for _, v in leaves(tree)]
    if any(type(v) is not int for v in values):
        raise ValueError("This shape is symbolic; a static extent is required")
    return prod(values)


def tuples(tree):
    if isinstance(tree, list):
        return tuple(tuples(t) for t in tree)
    if type(tree) is not int:
        raise ValueError("This mapping contains an unresolved dynamic field")
    return tree


def unflatten(values, tree):
    it = iter(values)

    def visit(t):
        return tuple(visit(c) for c in t) if isinstance(t, list) else next(it)

    return visit(tree)


def flat_coord(index, shape):
    coords = []
    for _, n in leaves(shape):
        if type(n) is not int:
            raise ValueError("Cannot unflatten a symbolic shape")
        coords.append(index % n)
        index //= n
    return coords


@lru_cache(maxsize=512)
def _compile(serialized):
    layout = Layout.model_validate_json(serialized)
    if layout.kind == "affine":
        return CPULayout(tuples(layout.shape), tuples(layout.stride))
    if layout.kind == "swizzle":
        return Swizzle(layout.bits, layout.base, layout.shift)
    # Contract: left(offset + right(c)). tensor-layouts calls these outer / inner.
    return ComposedLayout(
        compile_layout(layout.left), compile_layout(layout.right), offset=layout.offset
    )


def compile_layout(layout):
    return _compile(layout.model_dump_json())


def evaluate(layout, coord):
    return int(compile_layout(layout)(unflatten(coord, layout.shape)))


def explain(layout, coord):
    if layout.kind == "affine":
        terms = [f"{c} × {s}" for c, (_, s) in zip(coord, leaves(layout.stride))]
        return " + ".join(terms) + f" = {evaluate(layout, coord)}"
    right = evaluate(layout.right, coord)
    if layout.left.kind == "swizzle":
        s = layout.left
        return (
            f"{explain(layout.right, coord)}; add {layout.offset}; "
            f"Swizzle<{s.bits},{s.base},{s.shift}>({right + layout.offset}) "
            f"= {evaluate(layout, coord)}"
        )
    return (
        f"{explain(layout.right, coord)}; left({layout.offset} + {right}) "
        f"= {evaluate(layout, coord)}"
    )
