from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from .steps_registry import StepSpec, normalize_params


@dataclass(frozen=True)
class RunResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    command: list[str]


@dataclass(frozen=True)
class PipelineResult:
    ok: bool
    results: list[RunResult]


def _as_cli_args(params: dict[str, Any]) -> list[str]:
    """Convert a params dict into CLI args.

    Rules:
    - bool True -> --key
    - bool False / None -> omit
    - other -> --key value
    """
    out: list[str] = []
    for k, v in params.items():
        if v is None:
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                out.append(flag)
            continue
        out.extend([flag, str(v)])
    return out


def run_step(step: StepSpec, params: dict[str, Any], *, extra_cli_args: list[str] | None = None) -> RunResult:
    """Run a pipeline step via its Python module, using the current interpreter.

    This keeps CLI-mode and UI-mode consistent and avoids importing scripts that parse args at import time.
    extra_cli_args are appended raw (bypassing normalize_params) for internal pipeline flags.
    """
    params = normalize_params(step, params)

    if step.step_id == "tracking":
        method = params.pop("method", "physics")
        if method != "physics":
            return RunResult(
                ok=False,
                returncode=2,
                stdout="",
                stderr=f"Tracking method '{method}' is not implemented yet.",
                command=[],
            )
        cmd = [sys.executable, "-m", "fim.refactor.deformation_tracking", *_as_cli_args(params)]
    elif step.step_id == "inverse":
        cmd = [sys.executable, "-m", "fim.refactor.main_VFM", *_as_cli_args(params)]
    elif step.step_id == "distortion":
        return RunResult(
            ok=False,
            returncode=2,
            stdout="",
            stderr="Distortion step runner not implemented yet (no runnable script found).",
            command=[],
        )
    else:
        return RunResult(ok=False, returncode=2, stdout="", stderr=f"Unknown step_id: {step.step_id}", command=[])

    if extra_cli_args:
        cmd.extend(extra_cli_args)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    return RunResult(
        ok=(proc.returncode == 0),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        command=cmd,
    )


def run_step_streaming(
    step: StepSpec,
    params: dict[str, Any],
    *,
    on_stdout=None,
    on_stderr=None,
    env_overrides: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
) -> RunResult:
    """Run a step and stream output line-by-line via callbacks."""
    params = normalize_params(step, params)

    if step.step_id == "tracking":
        method = params.pop("method", "physics")
        if method != "physics":
            return RunResult(
                ok=False,
                returncode=2,
                stdout="",
                stderr=f"Tracking method '{method}' is not implemented yet.",
                command=[],
            )
        cmd = [sys.executable, "-m", "fim.refactor.deformation_tracking", *_as_cli_args(params)]
    elif step.step_id == "inverse":
        cmd = [sys.executable, "-m", "fim.refactor.main_VFM", *_as_cli_args(params)]
    elif step.step_id == "distortion":
        return RunResult(
            ok=False,
            returncode=2,
            stdout="",
            stderr="Distortion step runner not implemented yet (no runnable script found).",
            command=[],
        )
    else:
        return RunResult(ok=False, returncode=2, stdout="", stderr=f"Unknown step_id: {step.step_id}", command=[])

    if extra_cli_args:
        cmd.extend(extra_cli_args)

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)

    out_lines: list[str] = []
    err_lines: list[str] = []

    def _reader(stream, sink: list[str], cb):
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            sink.append(line)
            if cb:
                cb(line)
        stream.close()

    import threading

    t1 = threading.Thread(target=_reader, args=(p.stdout, out_lines, on_stdout), daemon=True)
    t2 = threading.Thread(target=_reader, args=(p.stderr, err_lines, on_stderr), daemon=True)
    t1.start()
    t2.start()
    rc = p.wait()
    t1.join(timeout=1)
    t2.join(timeout=1)

    stdout = "".join(out_lines)
    stderr = "".join(err_lines)
    return RunResult(ok=(rc == 0), returncode=rc, stdout=stdout, stderr=stderr, command=cmd)


def _should_skip_grids(steps: list[StepSpec]) -> bool:
    """When tracking and inverse run together, tracking can skip writing grid files."""
    step_ids = {s.step_id for s in steps}
    return "tracking" in step_ids and "inverse" in step_ids


def _propagate_tracking_output(params_by_step: dict[str, dict[str, Any]]) -> None:
    """Auto-set inverse data_path to tracking out_dir when both run together."""
    tracking = params_by_step.get("tracking", {})
    inverse = params_by_step.get("inverse", {})
    out_dir = tracking.get("out_dir")
    if out_dir and not inverse.get("data_path"):
        resolved = str(os.path.abspath(out_dir))
        params_by_step.setdefault("inverse", {})["data_path"] = resolved


def run_pipeline(steps: list[StepSpec], params_by_step: dict[str, dict[str, Any]] | None = None) -> PipelineResult:
    """Run multiple steps in sequence (stop on first failure)."""
    params_by_step = params_by_step or {}
    skip_grids = _should_skip_grids(steps)
    if skip_grids:
        _propagate_tracking_output(params_by_step)
    results: list[RunResult] = []

    for step in steps:
        params = params_by_step.get(step.step_id, {})
        extra = ["--skip_grids"] if skip_grids and step.step_id == "tracking" else None
        res = run_step(step, params, extra_cli_args=extra)
        results.append(res)
        if not res.ok:
            return PipelineResult(ok=False, results=results)

    return PipelineResult(ok=True, results=results)


def run_pipeline_streaming(
    steps: list[StepSpec],
    params_by_step: dict[str, dict[str, Any]] | None = None,
    *,
    on_stdout=None,
    on_stderr=None,
    env_overrides_by_step: dict[str, dict[str, str]] | None = None,
) -> PipelineResult:
    """Run steps sequentially and stream output with step_id."""
    params_by_step = params_by_step or {}
    skip_grids = _should_skip_grids(steps)
    if skip_grids:
        _propagate_tracking_output(params_by_step)
    results: list[RunResult] = []

    for step in steps:
        params = params_by_step.get(step.step_id, {})
        env_overrides = (env_overrides_by_step or {}).get(step.step_id)
        extra = ["--skip_grids"] if skip_grids and step.step_id == "tracking" else None
        res = run_step_streaming(
            step,
            params,
            on_stdout=(lambda ln, sid=step.step_id: on_stdout(sid, ln)) if on_stdout else None,
            on_stderr=(lambda ln, sid=step.step_id: on_stderr(sid, ln)) if on_stderr else None,
            env_overrides=env_overrides,
            extra_cli_args=extra,
        )
        results.append(res)
        if not res.ok:
            return PipelineResult(ok=False, results=results)
    return PipelineResult(ok=True, results=results)
