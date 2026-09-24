"""Context-local capture sessions; no compiler objects survive a probe."""

from __future__ import annotations

import ast
import hashlib
import inspect as pyinspect
import warnings
from contextvars import ContextVar
from pathlib import Path

from .model import Capture, CapturedObject, Source, Span, known

ACTIVE: ContextVar[CaptureSession | None] = ContextVar("cuteviz_capture", default=None)


class EmptyCaptureError(RuntimeError):
    pass


class CaptureSession:
    def __init__(self, path, *, target=None, parameters=None):
        self.path = Path(path)
        self.data = Capture(parameters=parameters or {})
        if target:
            self.data.target = known(target)
        self._token = None
        self._counts = {}
        self._tensor_ids = {}

    def __enter__(self):
        if self._token is not None:
            raise RuntimeError("A capture session cannot be re-entered")
        self._token = ACTIVE.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._tensor_ids.clear()
        ACTIVE.reset(self._token)
        self._token = None
        if exc is not None:
            self.data.outcome = "partial"
            self.data.diagnostics.append(
                f"Compilation failed: {type(exc).__name__}: {str(exc)[:2000]}"
            )
        if not self.data.objects:
            self.data.outcome = "empty"
            message = (
                "No probes were observed. Add cuteviz.inspect calls inside the compiled "
                "function and compile again. A cached callable does not run probes. "
                "Use a fresh process with CUTE_DSL_NO_CACHE=1 and cute.compile(entrypoint, "
                "*args); CuTeDSL 4.8.0 cute.compile bypasses the compilation cache."
            )
            self.data.diagnostics.append(message)
        try:
            self.data.save(self.path)
        except Exception as save_error:
            if exc is None:
                raise
            warnings.warn(f"Could not save partial capture: {save_error}", stacklevel=2)
        if not self.data.objects and exc is None:
            raise EmptyCaptureError(message)
        return False

    def source_span(self):
        frame = pyinspect.currentframe()
        try:
            while frame:
                filename = frame.f_code.co_filename
                if (
                    "/cuteviz/" not in filename
                    and "/nvidia_cutlass_dsl/" not in filename
                    and not filename.startswith("<")
                ):
                    path = Path(filename)
                    if path.is_file():
                        source = path.read_text()[:1_000_000]
                        line = frame.f_lineno
                        if line > len(source.splitlines()):
                            return None
                        sid = (
                            "source-"
                            + hashlib.sha256((str(path.resolve()) + source).encode()).hexdigest()[
                                :16
                            ]
                        )
                        if not any(s.id == sid for s in self.data.sources):
                            self.data.sources.append(
                                Source(id=sid, path=str(path.resolve()), text=source)
                            )
                        end = line
                        try:
                            calls = [
                                n
                                for n in ast.walk(ast.parse(source))
                                if isinstance(n, ast.Call) and n.lineno <= line <= n.end_lineno
                            ]
                            if calls:
                                call = min(calls, key=lambda n: n.end_lineno - n.lineno)
                                line, end = call.lineno, call.end_lineno
                        except SyntaxError:
                            pass
                        return Span(source_id=sid, line=line, end_line=end)
                frame = frame.f_back
        finally:
            del frame
        return None

    def new_object(self, label, kind, source=None):
        if len(self.data.objects) >= 5000:
            raise ValueError("Capture is limited to 5000 objects; reduce probe repetition")
        key = f"{label}:{kind}:{source.source_id if source else ''}:{source.line if source else 0}"
        count = self._counts.get(key, 0)
        self._counts[key] = count + 1
        oid = "probe-" + hashlib.sha256(f"{key}:{count}".encode()).hexdigest()[:16]
        obj = CapturedObject(id=oid, label=label, kind=kind, source=source)
        self.data.objects.append(obj)
        return obj

    def resolve(self, label_or_id):
        matches = [o for o in self.data.objects if o.id == label_or_id or o.label == label_or_id]
        if len(matches) != 1:
            raise ValueError(f"Parent {label_or_id!r} must name exactly one earlier probe")
        return matches[0].id


def capture(path, *, target=None, parameters=None):
    """Capture explicit probes during compilation, writing partial data even on failure."""
    return CaptureSession(path, target=target, parameters=parameters)
