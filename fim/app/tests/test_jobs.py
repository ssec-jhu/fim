from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from fim.app.jobs import _PROG_RE, JobManager, JobState
from fim.app.pipeline_runner import PipelineResult, RunResult
from fim.app.steps_registry import StepSpec


def _make_step(step_id: str = "inverse", **kw) -> StepSpec:
    return StepSpec(
        step_id=step_id,
        title=kw.get("title", step_id.capitalize()),
        script=None,
        essential=kw.get("essential", []),
        advanced=kw.get("advanced", []),
        common=kw.get("common"),
        methods=kw.get("methods"),
    )


class TestJobState:
    def test_defaults(self):
        js = JobState(job_id="abc")
        assert js.status == "queued"
        assert js.log == ""
        assert js.results == []
        assert js.progress == {}

    def test_append_log(self):
        js = JobState(job_id="abc")
        js.append_log("hello ")
        js.append_log("world")
        assert js.log == "hello world"

    def test_append_log_truncation(self):
        js = JobState(job_id="abc")
        js.append_log("x" * 300_000, limit=200_000)
        assert len(js.log) == 200_000


class TestProgRE:
    def test_matches_progress_line(self):
        m = _PROG_RE.match("FIM_PROGRESS iter=5 total=100")
        assert m
        assert m.group(1) == "5"
        assert m.group(2) == "100"

    def test_no_match_for_other_lines(self):
        assert _PROG_RE.match("Some other output") is None


class TestJobManager:
    def test_create_and_get(self):
        mgr = JobManager()
        job = mgr.create("j1")
        assert job.job_id == "j1"
        assert mgr.get("j1") is job

    def test_get_missing(self):
        mgr = JobManager()
        assert mgr.get("nonexistent") is None

    def test_update(self):
        mgr = JobManager()
        mgr.create("j1")
        mgr._update("j1", lambda j: setattr(j, "status", "running"))
        assert mgr.get("j1").status == "running"

    def test_update_missing_job(self):
        mgr = JobManager()
        mgr._update("nonexistent", lambda j: setattr(j, "status", "running"))  # should not raise

    @patch("fim.app.jobs.run_step_streaming")
    def test_start_step(self, mock_stream):
        mock_stream.return_value = RunResult(
            ok=True, returncode=0, stdout="done\n", stderr="", command=["python", "-m", "test"]
        )
        mgr = JobManager()
        mgr.create("j1")
        step = _make_step("inverse")
        mgr.start_step("j1", step, {})

        # Wait for the background thread to finish
        for _ in range(50):
            time.sleep(0.05)
            job = mgr.get("j1")
            if job and job.status in ("done", "failed"):
                break

        job = mgr.get("j1")
        assert job.status == "done"
        assert len(job.results) == 1
        assert job.results[0].ok
        assert job.started_at is not None
        assert job.finished_at is not None

    @patch("fim.app.jobs.run_step_streaming")
    def test_start_step_failure(self, mock_stream):
        mock_stream.return_value = RunResult(ok=False, returncode=1, stdout="", stderr="error\n", command=["python"])
        mgr = JobManager()
        mgr.create("j2")
        step = _make_step("inverse")
        mgr.start_step("j2", step, {})

        for _ in range(50):
            time.sleep(0.05)
            job = mgr.get("j2")
            if job and job.status in ("done", "failed"):
                break

        assert mgr.get("j2").status == "failed"

    @patch("fim.app.jobs.run_step_streaming")
    def test_start_step_with_progress(self, mock_stream):
        def streaming_with_callback(
            step, params, *, on_stdout=None, on_stderr=None, env_overrides=None, extra_cli_args=None
        ):
            if on_stdout:
                on_stdout("FIM_PROGRESS iter=3 total=10\n")
            return RunResult(ok=True, returncode=0, stdout="", stderr="", command=["cmd"])

        mock_stream.side_effect = streaming_with_callback
        mgr = JobManager()
        mgr.create("j3")
        step = _make_step("inverse")
        mgr.start_step("j3", step, {})

        for _ in range(50):
            time.sleep(0.05)
            job = mgr.get("j3")
            if job and job.status in ("done", "failed"):
                break

        job = mgr.get("j3")
        assert job.progress.get("iter") == 3
        assert job.progress.get("total") == 10

    @patch("fim.app.jobs.run_step_streaming")
    def test_start_step_early_return_captures_output(self, mock_stream):
        mock_stream.return_value = RunResult(ok=False, returncode=2, stdout="info\n", stderr="warn\n", command=[])
        mgr = JobManager()
        mgr.create("j4")
        step = _make_step("inverse")
        mgr.start_step("j4", step, {})

        for _ in range(50):
            time.sleep(0.05)
            job = mgr.get("j4")
            if job and job.status in ("done", "failed"):
                break

        job = mgr.get("j4")
        assert "info" in job.log
        assert "warn" in job.log

    @patch("fim.app.jobs.run_pipeline_streaming")
    def test_start_pipeline(self, mock_pipeline):
        mock_pipeline.return_value = PipelineResult(
            ok=True,
            results=[
                RunResult(ok=True, returncode=0, stdout="s1\n", stderr="", command=["cmd1"]),
                RunResult(ok=True, returncode=0, stdout="s2\n", stderr="", command=["cmd2"]),
            ],
        )
        mgr = JobManager()
        mgr.create("p1")
        steps = [_make_step("tracking"), _make_step("inverse")]
        mgr.start_pipeline("p1", steps, {"tracking": {}, "inverse": {}})

        for _ in range(50):
            time.sleep(0.05)
            job = mgr.get("p1")
            if job and job.status in ("done", "failed"):
                break

        job = mgr.get("p1")
        assert job.status == "done"
        assert len(job.results) == 2

    @patch("fim.app.jobs.run_pipeline_streaming")
    def test_start_pipeline_failure(self, mock_pipeline):
        mock_pipeline.return_value = PipelineResult(
            ok=False,
            results=[RunResult(ok=False, returncode=1, stdout="", stderr="fail\n", command=["cmd"])],
        )
        mgr = JobManager()
        mgr.create("p2")
        steps = [_make_step("tracking")]
        mgr.start_pipeline("p2", steps, {})

        for _ in range(50):
            time.sleep(0.05)
            job = mgr.get("p2")
            if job and job.status in ("done", "failed"):
                break

        assert mgr.get("p2").status == "failed"

    @patch("fim.app.jobs.run_pipeline_streaming")
    def test_start_pipeline_early_return(self, mock_pipeline):
        mock_pipeline.return_value = PipelineResult(
            ok=False,
            results=[RunResult(ok=False, returncode=2, stdout="early_out\n", stderr="early_err\n", command=[])],
        )
        mgr = JobManager()
        mgr.create("p3")
        mgr.start_pipeline("p3", [_make_step("tracking")], {})

        for _ in range(50):
            time.sleep(0.05)
            job = mgr.get("p3")
            if job and job.status in ("done", "failed"):
                break

        job = mgr.get("p3")
        assert "early_out" in job.log
        assert "early_err" in job.log
