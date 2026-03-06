from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __project__, __version__
from .jobs import JobManager
from .steps_registry import get_step, list_steps

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "fim_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FIM Pipeline")
job_mgr = JobManager()


# ── Static files & UI ───────────────────────────────────────────────


@app.get("/")
async def root():
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── Health ───────────────────────────────────────────────────────────


@app.get("/healthz")
async def healthz():
    return {"message": f"Running '{__project__}' ver: '{__version__}'"}


# ── Step metadata ────────────────────────────────────────────────────


@app.get("/api/steps")
async def api_list_steps():
    return {
        "steps": [
            {
                "id": s.step_id,
                "title": s.title,
                "script": s.script,
            }
            for s in list_steps()
        ]
    }


@app.get("/api/steps/{step_id}")
async def api_get_step(step_id: str):
    s = get_step(step_id)
    payload: dict[str, Any] = {
        "step_id": s.step_id,
        "title": s.title,
        "script": s.script,
        "essential": s.essential,
        "advanced": s.advanced,
    }
    if s.common is not None:
        payload["common"] = s.common
    if s.methods is not None:
        payload["methods"] = s.methods
    return payload


# ── File upload ──────────────────────────────────────────────────────

# ── Filesystem browsing ──────────────────────────────────────────────


@app.get("/api/fs/list")
async def api_fs_list(path: str | None = None):
    target = Path(path).resolve() if path else Path.home()
    if not target.is_dir():
        target = target.parent
    dirs = []
    try:
        for entry in sorted(target.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                dirs.append({"name": entry.name, "path": str(entry)})
    except PermissionError:
        pass
    return {"path": str(target), "root": str(target.parent), "dirs": dirs}


# ── File upload ──────────────────────────────────────────────────────


@app.post("/api/upload")
async def api_upload(file: UploadFile):
    dest = _UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": str(dest), "filename": file.filename}


# ── Single-step async run ───────────────────────────────────────────


class RunRequest(BaseModel):
    params: dict[str, Any] = {}


@app.post("/api/run_async/{step_id}")
async def api_run_async(step_id: str, req: RunRequest):
    step = get_step(step_id)
    job_id = uuid.uuid4().hex[:12]
    job_mgr.create(job_id)
    job_mgr.start_step(job_id, step, req.params)
    return {"job_id": job_id}


# ── Multi-step pipeline async run ───────────────────────────────────


class PipelineRequest(BaseModel):
    start_step_id: str
    params_by_step: dict[str, dict[str, Any]] = {}


@app.post("/api/run_pipeline_async")
async def api_run_pipeline_async(req: PipelineRequest):
    all_steps = list_steps()
    step_ids = [s.step_id for s in all_steps]

    if req.start_step_id not in step_ids:
        return {"error": f"Unknown step_id: {req.start_step_id}"}

    start_idx = step_ids.index(req.start_step_id)
    steps_to_run = all_steps[start_idx:]

    job_id = uuid.uuid4().hex[:12]
    job_mgr.create(job_id)
    job_mgr.start_pipeline(job_id, steps_to_run, req.params_by_step)
    return {"job_id": job_id}


# ── Job status polling ──────────────────────────────────────────────


@app.get("/api/jobs/{job_id}")
async def api_get_job(job_id: str):
    job = job_mgr.get(job_id)
    if job is None:
        return {"error": "not found"}
    return {
        "job_id": job.job_id,
        "status": job.status,
        "step_id": job.step_id,
        "progress": job.progress,
        "log": job.log,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }
