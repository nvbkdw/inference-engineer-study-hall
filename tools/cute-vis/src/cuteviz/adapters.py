"""CuTeDSL 4.8.0 snapshots. Called only while compiler objects are valid.

No parsing of pretty-printed layouts or inferred thread ownership from tensor shapes.
All compiler algebra here executes in a detached module (see probes.py).
"""

from importlib.metadata import version

from .algebra import leaves, size
from .model import Layout, Relationship, View, known, unknown

PINNED_COMPILER = "4.8.0"


def tree(value):
    if isinstance(value, (tuple, list)):
        return [tree(v) for v in value]
    if type(value) is int:
        return value
    return unknown(
        "Runtime-dependent or non-scalar CuTe field", "symbolic", str(value)[:200]
    ).model_dump()


def snapshot_layout(value):
    # Both CuTe and tensor-layouts expose shape/stride, but reverse composition names.
    import tensor_layouts as cpu

    if isinstance(value, cpu.ComposedLayout):
        return Layout(
            kind="compose",
            left=snapshot_layout(value.outer),
            right=snapshot_layout(value.inner),
            offset=value.offset,
        )
    if isinstance(value, cpu.Swizzle):
        return Layout(kind="swizzle", bits=value.bits, base=value.base, shift=value.shift)
    if hasattr(value, "num_bits"):
        return Layout(
            kind="swizzle", bits=value.num_bits, base=value.num_base, shift=value.num_shift
        )
    if hasattr(value, "inner") and hasattr(value, "outer"):
        offset = value.offset
        if type(offset) is not int:
            raise ValueError("Tuple or symbolic composition offsets are not supported")
        return Layout(
            kind="compose",
            left=snapshot_layout(value.inner),
            right=snapshot_layout(value.outer),
            offset=offset,
        )
    return Layout(shape=tree(value.shape), stride=tree(value.stride))


def observe(fn, reason):
    try:
        return known(fn())
    except Exception as exc:
        return unknown(f"{reason}: {type(exc).__name__}: {str(exc)[:300]}")


def snapshot_tensor(obj, role="tensor"):
    layout = None
    try:
        layout = snapshot_layout(obj.layout if hasattr(obj, "layout") else obj)
        layout_status = known("Relative to this view's base")
    except Exception as exc:
        layout_status = unknown(f"Address layout unavailable: {str(exc)[:300]}", "unsupported")
    shape = tree(obj.shape) if hasattr(obj, "shape") else unknown("Shape unavailable").model_dump()
    is_tensor = hasattr(obj, "dtype")
    storage = (
        observe(lambda: str(obj.memspace), "Memory space unavailable")
        if is_tensor
        else known("layout")
    )
    view = View(role=role, shape=shape, layout=layout, layout_status=layout_status, storage=storage)
    if is_tensor:
        view.dtype = observe(lambda: str(obj.dtype), "Dtype unavailable")
        view.element_bits = observe(lambda: obj.dtype.width, "Element width unavailable")
        if type(obj).__module__.startswith("cutlass"):
            pointer = obj.iterator
            if hasattr(pointer, "type") and hasattr(pointer.type, "is_swizzled"):
                view.pointer_alignment = known(pointer.type.alignment)
                if pointer.type.is_swizzled:
                    swizzle = pointer.type.swizzle_type
                    view.pointer_swizzle = Layout(
                        kind="swizzle",
                        bits=swizzle.num_bits,
                        base=swizzle.num_base,
                        shift=swizzle.num_shift,
                    )
                    view.layout_status = unknown(
                        "Position-dependent pointer swizzle: logical offsets are available, "
                        "but physical byte placement requires the runtime base phase."
                    )
    if storage.value == "tmem":
        view.address_unit = "tmem"
    elif storage.value == "rmem":
        view.address_unit = "none"
        view.layout_status = known(
            "Thread-local fragment indices; physical register allocation is unavailable"
        )
    return view


def basic(session, record, obj, *, depth=0):
    if not hasattr(obj, "shape"):
        record.kind = "unsupported"
        raise ValueError(f"No layout/tensor adapter for {type(obj).__name__}")
    record.views.append(snapshot_tensor(obj))
    record.kind = "tensor" if hasattr(obj, "dtype") else "layout"
    if type(obj).__module__.startswith("cutlass") and record.kind == "tensor":
        from .ancestry import snapshot_ancestry

        snapshot_ancestry(session, record, obj, depth=depth)


def operand(session, record, role, tensor):
    if tensor is None:
        return None, None
    child = session.new_object(f"{record.label}.{role}", "tensor", record.source)
    try:
        basic(session, child, tensor)
    except Exception as exc:
        child.diagnostics.append(str(exc)[:500])
    record.relationships.append(Relationship(target=child.id, kind="operand", transform=role))
    return child.id, child.views[0] if child.views else None


def copy(session, record, obj, src, dst):
    import cutlass.cute as cute

    record.fields["operation"] = known(type(obj.op).__module__ + "." + type(obj.op).__name__)
    record.fields["execution_group"] = known({"kind": "threads", "size": obj.size})
    shape = [int(cute.size(t)) for t in obj.tiler_mn]
    record.fields["tile_shape"] = known(shape)
    record.fields["correspondence"] = known(
        "Same logical element in the compiler's reference tile; source and destination owners can differ"
    )
    for role, tensor, name in [
        ("S", src, "layout_src_tv_tiled"),
        ("D", dst, "layout_dst_tv_tiled"),
    ]:
        oid, observed = operand(session, record, role, tensor)
        view = View(
            role=role, shape=shape, storage=unknown("Operand not supplied"), observed_object=oid
        )
        view.tv_layout = snapshot_layout(getattr(obj, name))
        view.ownership = known("Compiler TiledCopy thread/value mapping within one tile")
        attach_address(view, observed, shape)
        record.views.append(view)


def attach_address(view, observed, shape):
    if observed is None:
        return
    view.storage = observed.storage
    view.dtype = observed.dtype
    view.element_bits = observed.element_bits
    view.base_address = observed.base_address
    view.address_unit = observed.address_unit
    view.pointer_swizzle = observed.pointer_swizzle
    view.pointer_alignment = observed.pointer_alignment
    # Whole tensors may be larger than the tile. Their leading origin tile is addressable.
    dims = [n for _, n in leaves(observed.shape)]
    if len(dims) == len(shape) and all(type(n) is int and n >= s for n, s in zip(dims, shape)):
        view.layout = observed.layout
        view.layout_status = observed.layout_status
    else:
        view.layout_status = unknown(
            "Supplied tensor is a partition or fragment. Its storage is shown separately; "
            "a matrix-to-address relationship was not supplied."
        )


def mma(session, record, obj, a, b, c):
    from cutlass.cute.nvgpu import tcgen05, warp, warpgroup

    op = obj.op
    record.fields["operation"] = known(type(op).__module__ + "." + type(op).__name__)
    for role, tensor in [("A", a), ("B", b), ("C", c)]:
        operand(session, record, role, tensor)
    if version("nvidia-cutlass-dsl") != PINNED_COMPILER:
        raise ValueError(f"MMA adapters require CuTeDSL {PINNED_COMPILER}")
    if type(op) is warp.MmaF16BF16Op:
        family, group_size, sources = "Ampere mma.sync", 32, ["rmem", "rmem", "rmem"]
    elif type(op) is warpgroup.MmaF16BF16Op:
        family, group_size = "Hopper wgmma", 128
        sources = ["smem" if op.a_src == warpgroup.OperandSource.SMEM else "rmem", "smem", "rmem"]
    elif type(op) is tcgen05.MmaF16BF16Op:
        if op.cta_group != tcgen05.CtaGroup.ONE:
            raise ValueError("Multi-CTA tcgen05 is outside the supported single-CTA model")
        family, group_size = "Blackwell tcgen05", 1
        sources = ["smem" if op.a_src == tcgen05.OperandSource.SMEM else "tmem", "smem", "tmem"]
    else:
        raise ValueError(
            "Only dense FP16/BF16 warp, warpgroup, and single-CTA tcgen05 MMA are supported"
        )
    tile = [int(obj.get_tile_size(i)) for i in range(3)]
    record.fields["family"] = known(family)
    record.fields["instruction_shape"] = known(tree(obj.shape_mnk))
    record.fields["tile_shape"] = known(tile)
    record.fields["execution_group"] = known(
        {
            "kind": "single-CTA, single issuing thread"
            if group_size == 1
            else "warpgroup"
            if group_size == 128
            else "warp",
            "size": group_size,
            "partition_count": obj.size,
            "note": "Issue participation is separate from operand storage and ownership",
        }
    )
    for attr in ["a_major_mode", "b_major_mode", "cta_group"]:
        if hasattr(op, attr):
            record.fields[attr] = known(str(getattr(op, attr)))
    for role, shape, storage in zip(
        "ABC", [[tile[0], tile[2]], [tile[1], tile[2]], tile[:2]], sources
    ):
        rel = next((r for r in record.relationships if r.transform == role), None)
        observed = next((o for o in session.data.objects if rel and o.id == rel.target), None)
        view = View(
            role=role,
            shape=shape,
            storage=known(storage),
            observed_object=rel.target if rel else None,
        )
        dtype = (
            op.acc_dtype
            if role == "C"
            else getattr(op, "ab_dtype", None) or getattr(op, role.lower() + "_dtype")
        )
        view.dtype, view.element_bits = known(str(dtype)), known(dtype.width)
        view.tv_layout = snapshot_layout(getattr(obj, f"tv_layout_{role}_tiled"))
        if storage == "rmem":
            view.ownership = known(
                "Compiler TiledMMA logical thread/value indices; not physical registers"
            )
            view.address_unit = "none"
        else:
            view.ownership = unknown(
                "Collective operand storage; TV partition indices are not owning GPU threads",
                "unavailable",
            )
        if storage == "smem":
            if observed and observed.views:
                attach_address(view, observed.views[0], shape)
                view.storage = known(storage)
            if not view.layout:
                view.layout_status = unknown(
                    "Descriptor operand: inspect the backing shared-memory tensor for byte placement"
                )
        if storage == "tmem":
            # Derive actual packed tensor-memory addresses from CuTe's fragment allocator.
            fragment_shape = getattr(obj, f"partition_shape_{role}")(tuple(shape))
            fragment = getattr(obj, f"make_fragment_{role}")(fragment_shape)
            view.fragment_layout = snapshot_layout(fragment.layout)
            view.address_unit = "tmem"
            view.layout_status = known(
                "Canonical compiler fragment: relative tensor-memory data-path/column coordinates"
            )
            if observed and observed.views:
                supplied = observed.views[0]
                if supplied.storage.value == "tmem" and supplied.layout:
                    if size(supplied.shape) == size(view.fragment_layout.shape):
                        view.fragment_layout = supplied.layout
                        view.layout_status = known(
                            "Supplied fragment: relative tensor-memory data-path/column coordinates"
                        )
                    else:
                        view.fragment_layout = None
                        view.layout_status = unknown(
                            "Supplied tensor-memory fragment has a different shape; inspect a matching MMA tile"
                        )
        record.views.append(view)
