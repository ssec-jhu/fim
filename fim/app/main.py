from __future__ import annotations

import os
import shutil
import subprocess
import sys
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


def _fs_browse_root() -> Path | None:
    """Optional root directory for /api/fs/list (e.g. Docker volume mount at /data)."""
    raw = os.environ.get("FIM_FS_LIST_ROOT", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _fs_clamp_under_root(target: Path, root: Path | None) -> Path:
    if root is None:
        return target
    try:
        target.relative_to(root)
        return target
    except ValueError:
        return root


def _resolve_fs_list_target(path: str | None) -> Path:
    root = _fs_browse_root()
    default = root if root is not None else Path.home()
    if path:
        target = Path(path).expanduser().resolve()
    else:
        target = default
    if not target.is_dir():
        target = target.parent
    target = _fs_clamp_under_root(target, root)
    if not target.is_dir():
        target = default
    return target


def _fs_browse_hint() -> str:
    root = _fs_browse_root()
    if root is not None:
        return (
            "Folders are on the server running this app, starting at "
            f"{root}. In Docker, map a host directory to that path so outputs are saved on your machine."
        )
    return (
        "Folders are on the machine running this app (starting from your home directory). "
        "For Docker, set FIM_FS_LIST_ROOT to a mounted path (see README)."
    )


def _native_folder_picker_env_enabled() -> bool:
    raw = os.environ.get("FIM_NATIVE_FOLDER_PICKER", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def native_folder_picker_available() -> bool:
    """True when the server can show a real OS folder dialog (macOS + osascript)."""
    if not _native_folder_picker_env_enabled():
        return False
    if sys.platform != "darwin":
        return False
    return shutil.which("osascript") is not None


def _pick_folder_macos(initial: Path | None) -> str | None:
    """Open the macOS system folder chooser via AppleScript (same style as Finder)."""
    if shutil.which("osascript") is None:
        return None
    start = initial if initial and initial.is_dir() else Path.home()
    if not start.is_dir():
        start = Path.home()
    pstr = str(start.resolve())
    esc = pstr.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        f'set defLoc to POSIX file "{esc}"\n'
        "try\n"
        '  set f to choose folder with prompt "Choose output folder" default location defLoc\n'
        "on error\n"
        '  set f to choose folder with prompt "Choose output folder"\n'
        "end try\n"
        "return POSIX path of f"
    )
    r = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if r.returncode != 0:
        return None
    out = (r.stdout or "").strip()
    return out if out else None


@app.get("/api/fs/config")
async def api_fs_config():
    root = _fs_browse_root()
    native = native_folder_picker_available()
    return {
        "browse_root": str(root) if root else None,
        "browse_hint": _fs_browse_hint(),
        "native_folder_picker": native,
        "native_folder_hint": (
            "Opens the macOS folder dialog (server must run on this Mac, not inside Linux Docker)." if native else None
        ),
    }


@app.post("/api/fs/pick_folder_native")
def api_pick_folder_native():
    """Open the OS-native folder dialog on the machine running the API (macOS only)."""
    if not native_folder_picker_available():
        return {"ok": False, "error": "unavailable"}
    root = _fs_browse_root()
    initial = root if root and root.is_dir() else None
    try:
        picked = _pick_folder_macos(initial)
    except OSError:
        return {"ok": False, "error": "failed"}
    if not picked:
        return {"ok": False, "error": "cancelled"}
    p = Path(picked).expanduser().resolve()
    p = _fs_clamp_under_root(p, root)
    if not p.is_dir():
        return {"ok": False, "error": "invalid"}
    return {"ok": True, "path": str(p)}


@app.get("/api/fs/list")
async def api_fs_list(path: str | None = None):
    root = _fs_browse_root()
    target = _resolve_fs_list_target(path)
    hint = _fs_browse_hint()
    dirs = []
    try:
        for entry in sorted(target.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                dirs.append({"name": entry.name, "path": str(entry)})
    except PermissionError:
        pass
    return {
        "path": str(target),
        "root": str(target.parent),
        "dirs": dirs,
        "browse_root": str(root) if root else None,
        "browse_hint": hint,
    }


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


@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel_job(job_id: str):
    """Request cancellation of a running job (terminates subprocess if still running)."""
    ok = job_mgr.cancel(job_id)
    return {"ok": ok}
