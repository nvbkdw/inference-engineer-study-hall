"""Local playback and a fresh-process workbench. Never imports CuTe/CUDA itself."""

import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .banks import BankQuery, analyze_banks
from .mapping import MappingQuery, query_mapping
from .model import Capture
from .workbench import CompileRequest, Workbench


def create_app(capture: Capture | str | Path | None = None, **workbench_options):
    data = capture if isinstance(capture, Capture) or capture is None else Capture.load(capture)
    workbench = Workbench(**workbench_options)

    @asynccontextmanager
    async def lifespan(app):
        yield
        workbench.close()

    app = FastAPI(title="CuteViz", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.workbench = workbench

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        # Loopback binding alone does not stop DNS rebinding or cross-site requests.
        host = urlsplit("//" + request.headers.get("host", "")).hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            return JSONResponse({"detail": "A loopback Host is required."}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers['host']}":
            return JSONResponse({"detail": "Same-origin requests are required."}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-site requests are not allowed."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def authorize(request: Request):
        if not workbench.enabled:
            raise HTTPException(403, "Compilation is disabled (--read-only).")
        if not secrets.compare_digest(request.headers.get("x-cuteviz-token", ""), workbench.token):
            raise HTTPException(403, "Missing workbench token. Reload the app and try again.")

    def job_for(job_id):
        try:
            return workbench.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown compilation run") from exc

    def capture_for(capture_id):
        selected = job_for(capture_id).data if capture_id else data
        if selected is None:
            raise HTTPException(
                404, "No capture is available yet. Compile a probed entry function."
            )
        return selected

    @app.get("/api/capture")
    def get_capture(capture_id: str | None = None):
        return capture_for(capture_id)

    @app.get("/api/workbench")
    def get_workbench():
        return workbench.info() | {"has_capture": data is not None}

    @app.get("/api/compilations")
    def list_compilations():
        with workbench.lock:
            return [job.public() for job in reversed(workbench.jobs.values())]

    @app.post("/api/compilations", dependencies=[Depends(authorize)], status_code=202)
    def compile_source(request: CompileRequest):
        try:
            return workbench.submit(request)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(422, f"Could not create compilation files: {exc}") from exc

    @app.get("/api/compilations/{job_id}")
    def get_compilation(job_id: str):
        with workbench.lock:
            return job_for(job_id).public()

    @app.post("/api/compilations/{job_id}/cancel", dependencies=[Depends(authorize)])
    def cancel_compilation(job_id: str):
        job_for(job_id)
        return workbench.cancel(job_id)

    @app.get("/api/compilations/{job_id}/source")
    def get_source(job_id: str):
        return job_for(job_id).request

    @app.get("/api/compilations/{job_id}/capture")
    def download_capture(job_id: str):
        job = job_for(job_id)
        if job.data is None:
            raise HTTPException(404, "This run has no capture.")
        return FileResponse(
            job.capture_path, media_type="application/json", filename=f"{job.id}.cuteviz.json"
        )

    @app.post("/api/mapping/query")
    def mapping(query: MappingQuery, capture_id: str | None = None):
        try:
            return query_mapping(capture_for(capture_id), query)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ValueError, TypeError, IndexError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/analysis/banks")
    def banks(query: BankQuery):
        try:
            return analyze_banks(query)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    static = Path(__file__).parent / "static"
    if static.is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="frontend")
    return app
