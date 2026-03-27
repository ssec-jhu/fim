from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .pipeline_runner import RunResult, run_pipeline_streaming, run_step_streaming
from .steps_registry import StepSpec, normalize_params

_PROG_RE = re.compile(r"^FIM_PROGRESS\s+iter=(\d+)\s+total=(\d+)\s*$")


def _step_banner(step_id: str, title: str | None = None) -> str:
    """Build a prominent UI log marker for major step transitions."""
    label = (title or step_id).strip()
    return f"\n========== START {label.upper()} ==========\n"


@dataclass
class JobState:
    job_id: str
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    status: str = "queued"  # queued|running|done|failed|cancelled
    step_id: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    log: str = ""
    results: list[RunResult] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def append_log(self, text: str, limit: int = 200_000) -> None:
        self.log += text
        if len(self.log) > limit:
            self.log = self.log[-limit:]


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}

    def create(self, job_id: str) -> JobState:
        with self._lock:
            job = JobState(job_id=job_id)
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _update(self, job_id: str, fn: Callable[[JobState], None]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            fn(job)

    def cancel(self, job_id: str) -> bool:
        """Request cancellation of a running job (terminates subprocess if any)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != "running":
                return False
            job.cancel_event.set()
            job.append_log("\n[Stop requested — terminating process if still running]\n")
            return True

    def start_step(self, job_id: str, step: StepSpec, params: dict[str, Any]) -> None:
        params = normalize_params(step, params)

        def worker():
            self._update(job_id, lambda j: setattr(j, "status", "running"))
            self._update(job_id, lambda j: setattr(j, "started_at", time.time()))
            self._update(job_id, lambda j: setattr(j, "step_id", step.step_id))
            self._update(job_id, lambda j: j.append_log(_step_banner(step.step_id, step.title)))

            cancel_ev = self.get(job_id)
            cancel_event = cancel_ev.cancel_event if cancel_ev else threading.Event()

            def on_line(src: str, line: str) -> None:
                # append to log
                self._update(job_id, lambda j: j.append_log(f"[{src}] {line}"))
                # parse progress markers
                m = _PROG_RE.match(line.strip())
                if m:
                    it = int(m.group(1))
                    total = int(m.group(2))
                    self._update(
                        job_id,
                        lambda j: j.progress.update({"iter": it, "total": total, "step_id": step.step_id}),
                    )

            res = run_step_streaming(
                step,
                params,
                on_stdout=lambda ln: on_line("stdout", ln),
                on_stderr=lambda ln: on_line("stderr", ln),
                env_overrides={"FIM_UI_NO_TQDM": "1"},
                cancel_event=cancel_event,
            )
            # If runner returns early (e.g. "not implemented"), ensure message is visible in log.
            # For normal runs, output is already streamed line-by-line via callbacks.
            if not res.command:
                if res.stdout:
                    self._update(job_id, lambda j: j.append_log(res.stdout))
                if res.stderr:
                    self._update(job_id, lambda j: j.append_log(res.stderr))
            self._update(job_id, lambda j: j.results.append(res))
            self._update(job_id, lambda j: setattr(j, "finished_at", time.time()))

            def _set_final_status(j: JobState) -> None:
                if j.cancel_event.is_set() and not res.ok:
                    j.status = "cancelled"
                else:
                    j.status = "done" if res.ok else "failed"

            self._update(job_id, _set_final_status)

        threading.Thread(target=worker, daemon=True).start()

    def start_pipeline(self, job_id: str, steps: list[StepSpec], params_by_step: dict[str, dict[str, Any]]) -> None:
        normalized = {s.step_id: normalize_params(s, params_by_step.get(s.step_id, {})) for s in steps}

        def worker():
            self._update(job_id, lambda j: setattr(j, "status", "running"))
            self._update(job_id, lambda j: setattr(j, "started_at", time.time()))

            cancel_ev = self.get(job_id)
            cancel_event = cancel_ev.cancel_event if cancel_ev else threading.Event()
            step_titles = {s.step_id: s.title for s in steps}
            current_step_id: str | None = None

            def on_line(step_id: str, src: str, line: str) -> None:
                nonlocal current_step_id
                if step_id != current_step_id:
                    current_step_id = step_id
                    title = step_titles.get(step_id, step_id)
                    self._update(job_id, lambda j, t=title: j.append_log(_step_banner(step_id, t)))
                self._update(job_id, lambda j: setattr(j, "step_id", step_id))
                self._update(job_id, lambda j: j.append_log(f"[{step_id} {src}] {line}"))
                m = _PROG_RE.match(line.strip())
                if m:
                    it = int(m.group(1))
                    total = int(m.group(2))
                    self._update(
                        job_id,
                        lambda j: j.progress.update({"iter": it, "total": total, "step_id": step_id}),
                    )

            pres = run_pipeline_streaming(
                steps,
                normalized,
                on_stdout=lambda step_id, ln: on_line(step_id, "stdout", ln),
                on_stderr=lambda step_id, ln: on_line(step_id, "stderr", ln),
                env_overrides_by_step={"tracking": {"FIM_UI_NO_TQDM": "1"}},
                cancel_event=cancel_event,
            )
            # Ensure any non-streamed stderr/stdout is captured (e.g., early "not implemented").
            for r in pres.results:
                if not r.command:
                    if r.stdout:
                        self._update(job_id, lambda j, t=r.stdout: j.append_log(t))
                    if r.stderr:
                        self._update(job_id, lambda j, t=r.stderr: j.append_log(t))
            self._update(job_id, lambda j: j.results.extend(pres.results))
            self._update(job_id, lambda j: setattr(j, "finished_at", time.time()))

            def _set_pipeline_final(j: JobState) -> None:
                if j.cancel_event.is_set() and not pres.ok:
                    j.status = "cancelled"
                else:
                    j.status = "done" if pres.ok else "failed"

            self._update(job_id, _set_pipeline_final)

        threading.Thread(target=worker, daemon=True).start()
