"""Explicit observation probes; these functions always return None."""

from contextlib import nullcontext
from importlib.metadata import version

from . import adapters
from .capture import ACTIVE
from .model import Relationship, known, unknown

try:
    from cutlass.cutlass_dsl import dsl_user_op
except ImportError:

    def dsl_user_op(fn):
        return fn


def _probe(label, obj, kind, *, parent=None, transform=None, **operands):
    session = ACTIVE.get()
    if session is None:
        return
    record = session.new_object(label, kind, session.source_span())
    try:
        context = nullcontext()
        if type(obj).__module__.startswith("cutlass"):
            from cutlass._mlir import ir
            from cutlass.base_dsl.dsl import BaseDSL

            session.data.compiler = known(
                {"name": "nvidia-cutlass-dsl", "version": version("nvidia-cutlass-dsl")}
            )
            if session.data.target.status != "known":
                session.data.target = known(str(BaseDSL._get_dsl().get_arch_enum()))
            scratch = ir.Module.create()
            context = ir.InsertionPoint(scratch.body)
        with context:
            if kind == "copy":
                adapters.copy(session, record, obj, **operands)
            elif kind == "mma":
                adapters.mma(session, record, obj, **operands)
            else:
                adapters.basic(session, record, obj)
        if parent is not None:
            record.relationships.append(
                Relationship(
                    target=session.resolve(parent), kind="declared_parent", transform=transform
                )
            )
    except Exception as exc:
        record.fields["adapter"] = unknown(
            f"{type(exc).__name__}: {str(exc)[:1000]}", "unsupported"
        )
        record.diagnostics.append(record.fields["adapter"].reason)
        session.data.outcome = "partial"


@dsl_user_op
def inspect(label, obj, *, parent=None, transform=None, loc=None):
    """Snapshot a layout or tensor. parent is a unique earlier probe label or ID."""
    _probe(label, obj, "tensor", parent=parent, transform=transform)


@dsl_user_op
def inspect_copy(label, tiled_copy, *, src=None, dst=None, loc=None):
    """Inspect one TiledCopy tile and its source/destination correspondence."""
    _probe(label, tiled_copy, "copy", src=src, dst=dst)


@dsl_user_op
def inspect_mma(label, tiled_mma, *, a=None, b=None, c=None, loc=None):
    """Inspect dense MMA logical operands, participation, and storage."""
    _probe(label, tiled_mma, "mma", a=a, b=b, c=c)
