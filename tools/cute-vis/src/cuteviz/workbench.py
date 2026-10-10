"""Bounded local compilation jobs. Subprocesses isolate compiler state, not privileges."""

from __future__ import annotations

import importlib.metadata
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .model import Capture

TERMINAL = {"succeeded", "failed", "cancelled", "timed_out"}
MAX_LOG = 64 * 1024


class CompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1, max_length=256_000)
    entry: str = Field(default="entry", max_length=200, pattern=r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
    args_factory: str = Field(
        default="make_args", max_length=200, pattern=r"^([A-Za-z_]\w*(\.[A-Za-z_]\w*)*)?$"
    )
    target: str = Field(default="", max_length=32, pattern=r"^(sm_\d{2,3}[af]?)?$")


@dataclass
class Job:
    id: str
    request: CompileRequest
    directory: Path
    status: str = "queued"
    log: str = ""
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: float | None = None
    returncode: int | None = None
    data: Capture | None = None
    process: subprocess.Popen | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    @property
    def capture_path(self):
        return self.directory / "kernel.cuteviz.json"

    def public(self):
        return {
            "id": self.id,
            "status": self.status,
            "log": self.log,
            "error": self.error,
            "started": self.started,
            "finished": self.finished,
            "returncode": self.returncode,
            "entry": self.request.entry,
            "target": self.request.target,
            "capture_available": self.data is not None,
            "outcome": self.data.outcome if self.data else None,
            "object_count": len(self.data.objects) if self.data else 0,
            "capture_path": str(self.capture_path) if self.data else None,
        }


class Workbench:
    def __init__(self, *, enabled=True, workdir=None, output_dir=None, timeout=120):
        self.enabled = enabled
        self.workdir = Path(workdir or Path.cwd()).resolve()
        self.output_dir = Path(output_dir or self.workdir / ".cuteviz").resolve()
        self.timeout = timeout
        self.token = secrets.token_urlsafe(32)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.closed = False
        try:
            self.compiler = importlib.metadata.version("nvidia-cutlass-dsl")
        except importlib.metadata.PackageNotFoundError:
            self.compiler = None

    def info(self):
        return {
            "enabled": self.enabled,
            "compiler": self.compiler,
            "can_compile": self.enabled and self.compiler == "4.8.0",
            "reason": (
                "Server started with --read-only."
                if not self.enabled
                else "Install cuteviz[capture] in this Python environment (CuTeDSL 4.8.0)."
                if self.compiler != "4.8.0"
                else ""
            ),
            "python": sys.executable,
            "workdir": str(self.workdir),
            "output_dir": str(self.output_dir),
            "timeout_seconds": self.timeout,
            "token": self.token if self.enabled else "",
            "templates": [
                {"id": path.stem, "source": path.read_text()}
                for path in sorted((Path(__file__).parent / "templates").glob("*.py"))
            ],
        }

    def submit(self, request: CompileRequest):
        with self.lock:
            if self.closed:
                raise RuntimeError("Workbench is shutting down")
            if not self.info()["can_compile"]:
                raise RuntimeError(self.info()["reason"])
            if any(job.status not in TERMINAL for job in self.jobs.values()):
                raise RuntimeError("A compilation is already running. Wait or cancel it first.")
            if len(self.jobs) >= 100:
                raise RuntimeError(
                    "Session limit of 100 runs reached. Restart the server to continue."
                )
            job_id = uuid.uuid4().hex
            directory = self.output_dir / job_id
            directory.mkdir(parents=True, mode=0o700)
            (directory / "kernel.py").write_text(request.source)
            job = Job(job_id, request, directory)
            config = request.model_dump(exclude={"source"}) | {
                "source_path": str(directory / "kernel.py"),
                "capture_path": str(job.capture_path),
            }
            (directory / "config.json").write_text(json.dumps(config))
            self.jobs[job_id] = job
            job.thread = threading.Thread(target=self._run, args=(job,), daemon=True)
            job.thread.start()
            return job.public()

    @staticmethod
    def _kill(process):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        except ProcessLookupError:
            pass

    def _drain(self, job, process):
        # Always drain stdout to avoid deadlocking a verbose compiler. Retain only a tail.
        import codecs

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        with process.stdout:
            while chunk := process.stdout.read1(4096):
                with self.lock:
                    job.log = (job.log + decoder.decode(chunk))[-MAX_LOG:]

    def _run(self, job):
        reader = None
        final_status = "failed"
        try:
            env = os.environ.copy()
            env.update(CUTE_DSL_NO_CACHE="1", PYTHONUNBUFFERED="1")
            if job.request.target:
                env["CUTE_DSL_ARCH"] = job.request.target
            # Also supports editable installations while cwd is the user's project.
            env["PYTHONPATH"] = os.pathsep.join(
                [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]
            )
            process = subprocess.Popen(
                [sys.executable, "-m", "cuteviz.worker", str(job.directory / "config.json")],
                cwd=self.workdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            with self.lock:
                job.process = process
                job.status = "running"
            reader = threading.Thread(target=self._drain, args=(job, process), daemon=True)
            reader.start()
            deadline = time.monotonic() + self.timeout
            while process.poll() is None:
                if job.cancelled.is_set():
                    final_status = "cancelled"
                    break
                if time.monotonic() >= deadline:
                    final_status = "timed_out"
                    job.error = f"Compilation exceeded the {self.timeout:g} second limit."
                    break
                job.cancelled.wait(0.05)
            else:
                final_status = "succeeded" if process.returncode == 0 else "failed"
            if job.cancelled.is_set():
                final_status = "cancelled"
            # Reap the worker and its descendants, including ones that inherited stdout.
            self._kill(process)
            job.returncode = process.wait()
            reader.join(timeout=2)
            if job.capture_path.is_file():
                try:
                    job.data = Capture.load(job.capture_path)
                except (ValueError, OSError) as exc:
                    job.error = f"Invalid generated capture: {exc}"
                    final_status = "failed"
            if final_status == "succeeded" and (job.data is None or not job.data.objects):
                final_status = "failed"
                job.error = "No observations captured. Add cuteviz.inspect probes inside the entry."
            if final_status == "failed" and not job.error:
                job.error = "Compilation failed. See the compiler output below."
        except Exception as exc:
            job.error = f"Could not compile: {exc}"
        finally:
            with self.lock:
                job.status = final_status
                job.finished = time.time()

    def get(self, job_id):
        with self.lock:
            return self.jobs[job_id]

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job.status not in TERMINAL:
                job.cancelled.set()
            return job.public()

    def close(self):
        with self.lock:
            self.closed = True
            jobs = list(self.jobs.values())
            for job in jobs:
                if job.status not in TERMINAL:
                    job.cancelled.set()
        for job in jobs:
            if job.thread:
                job.thread.join(timeout=5)
