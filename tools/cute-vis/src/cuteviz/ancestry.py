"""Snapshot supported view ancestry from CuTe 4.8 IR in the detached probe context."""

from .algebra import leaves, unflatten
from .model import CoordinateMap, CoordinateParameter, Relationship, known, unknown


def value_key(tensor):
    value = getattr(tensor, "value", None)
    if value is None:
        return None
    try:
        # Opaque identity tokens only: never retain live compiler values in a capture.
        return hash(value), hash(value.owner.operation.parent)
    except (AttributeError, TypeError):
        return None


def flat(tree):
    return (
        [item for child in tree for item in flat(child)]
        if isinstance(tree, (tuple, list))
        else [tree]
    )


def scalar_expression(value, depth=0):
    """Recognize CTA indices and affine scalar arithmetic, not executable strings."""
    if depth > 8:
        return None
    try:
        op = value.owner.operation
        name = op.name
        if name.startswith("nvvm.read.ptx.sreg.ctaid."):
            return 0, {"blockIdx." + name.rsplit(".", 1)[1]: 1}
        if name == "arith.constant":
            return int(op.attributes["value"].value), {}
        if name in {"arith.index_cast", "arith.extsi", "arith.extui"}:
            return scalar_expression(op.operands[0], depth + 1)
        if name in {"arith.addi", "arith.subi", "arith.muli"}:
            lhs, rhs = [scalar_expression(v, depth + 1) for v in op.operands]
            if lhs is None or rhs is None:
                return None
            if name == "arith.muli":
                if lhs[1] and rhs[1]:
                    return None
                if lhs[1]:
                    lhs, rhs = rhs, lhs
                return lhs[0] * rhs[0], {k: lhs[0] * v for k, v in rhs[1].items()}
            sign = -1 if name == "arith.subi" else 1
            terms = lhs[1].copy()
            for k, v in rhs[1].items():
                terms[k] = terms.get(k, 0) + sign * v
            return lhs[0] + sign * rhs[0], terms
    except (AttributeError, TypeError, ValueError):
        pass
    return None


def coordinate_map(tensor, operation, record_id):
    import cutlass.cute as cute
    from cutlass.cute import core
    from cutlass.cute.tensor import _Tensor

    parent = operation.input
    parent_shape = list(parent.shape) if isinstance(parent.shape, tuple) else [parent.shape]
    if not all(type(n) is int for n in parent_shape):
        raise ValueError("Backing coordinates currently require a flat, static parent shape")
    child_shape = tensor.shape
    child_tree = [*child_shape] if isinstance(child_shape, tuple) else child_shape
    child_dims = [n for _, n in leaves(child_tree)]
    if not all(type(n) is int for n in child_dims) or len(child_dims) > 32:
        raise ValueError("Backing coordinates require at most 32 static child modes")
    name = operation.operation.name
    if name == "cute.local_tile":
        tiler = core._unpack_x_tuple(operation.tile)
        for mode in flat(tiler):
            if type(mode.shape) is not int or mode.stride != 1:
                raise ValueError("Only rectangular unit-stride local_tile tilers are supported")
    coord = core._unpack_x_tuple(operation.coord)
    raw = (
        list(operation.coord.owner.operation.operands)
        if operation.coord.owner.operation.name == "cute.make_coord"
        else []
    )
    dynamic = iter(raw)
    index = 0
    expressions = []

    def constant_tree(item):
        nonlocal index
        if isinstance(item, tuple):
            return tuple(constant_tree(c) for c in item)
        if item is None or type(item) is int:
            return item
        value = next(dynamic, None)
        expr = scalar_expression(value) if value is not None else None
        if expr is None:
            expr = 0, {f"{record_id}.tile_index_{index}": 1}
        expressions.append((index, expr))
        index += 1
        return expr[0]

    constants = constant_tree(coord)

    def replace_dynamic(tree, deltas):
        cursor = 0

        def visit(original, constant):
            nonlocal cursor
            if isinstance(original, tuple):
                return tuple(visit(a, b) for a, b in zip(original, constant))
            if original is None or type(original) is int:
                return constant
            pos = cursor
            cursor += 1
            return constant + deltas.get(pos, 0)

        return visit(tree, constants)

    identity = cute.make_identity_tensor(parent.shape)

    def apply(coord_value):
        packed = core._pack_coord(coord_value)
        if name == "cute.local_tile":
            value = core._cute_ir.local_tile(
                identity.value, operation.tile, packed, proj=operation.proj
            )
        else:
            value = core._cute_ir.slice(identity.value, packed)
        return _Tensor(value)

    mapped = apply(constants)
    zero = [0] * len(child_dims)

    def point(view, position):
        result = flat(view[unflatten(position, child_tree)])
        if not all(type(n) is int for n in result):
            raise ValueError("CuTe could not resolve the parent logical coordinates")
        return result

    origin = point(mapped, zero)
    matrix = [[0] * len(child_dims) for _ in origin]
    for axis, extent in enumerate(child_dims):
        if extent == 1:
            continue
        coordinate = zero.copy()
        coordinate[axis] = 1
        step = point(mapped, coordinate)
        for r in range(len(origin)):
            matrix[r][axis] = step[r] - origin[r]
    # Restrict to affine coordinate layouts and validate corners against CuTe.
    for coordinate in [[n - 1 for n in child_dims], [n // 2 for n in child_dims]]:
        expected = [
            o + sum(a * c for a, c in zip(row, coordinate)) for o, row in zip(origin, matrix)
        ]
        if point(mapped, coordinate) != expected:
            raise ValueError("This view needs a non-affine parent coordinate mapping")
    parameters = {}
    for position, (_, terms) in expressions:
        shifted = point(apply(replace_dynamic(coord, {position: 1})), zero)
        delta = [v - o for v, o in zip(shifted, origin)]
        for key, coefficient in terms.items():
            if key not in parameters:
                parameters[key] = CoordinateParameter(
                    label=key if key.startswith("blockIdx.") else f"Runtime tile index {position}",
                    coefficients=[0] * len(origin),
                )
            parameters[key].coefficients = [
                a + coefficient * b for a, b in zip(parameters[key].coefficients, delta)
            ]
    return CoordinateMap(operation=name, origin=origin, matrix=matrix, parameters=parameters)


def snapshot_ancestry(session, record, tensor, depth=0):
    from .adapters import basic

    key = value_key(tensor)
    if key is not None:
        session._tensor_ids.setdefault(key, record.id)
    try:
        operation = tensor.value.owner
        name = operation.operation.name
    except AttributeError:
        return
    if name not in {"cute.local_tile", "cute.slice"}:
        return
    if depth >= 16:
        record.diagnostics.append("Backing ancestry exceeds 16 levels")
        return
    parent_tensor = operation.input
    parent_id = session._tensor_ids.get(value_key(parent_tensor))
    if parent_id is None:
        parent = session.new_object(f"{record.label}.backing", "tensor", record.source)
        parent.fields["provenance"] = known(
            f"Input tensor recovered from {name}; not a separate user probe"
        )
        basic(session, parent, parent_tensor, depth=depth + 1)
        parent_id = parent.id
    relation = Relationship(target=parent_id, kind="derived_parent", transform=name)
    record.relationships.append(relation)
    try:
        relation.mapping = coordinate_map(tensor, operation, record.id)
        relation.mapping_status = known("Compiler-derived parent logical coordinate transform")
    except Exception as exc:
        relation.mapping_status = unknown(
            f"Backing tensor is available; coordinate mapping is unsupported: {str(exc)[:300]}",
            "unsupported",
        )
        record.diagnostics.append(relation.mapping_status.reason)
