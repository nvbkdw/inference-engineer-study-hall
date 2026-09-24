"""Compile-time probes and portable, CPU-only capture playback."""

__version__ = "0.1.0"

from .capture import EmptyCaptureError, capture

__all__ = ["capture", "inspect", "inspect_copy", "inspect_mma", "EmptyCaptureError"]


def __getattr__(name):
    if name in {"inspect", "inspect_copy", "inspect_mma"}:
        from . import probes

        value = getattr(probes, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
