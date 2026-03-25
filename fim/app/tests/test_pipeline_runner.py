from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fim.app.pipeline_runner import (
    RunResult,
    _as_cli_args,
    _propagate_tracking_output,
    _should_skip_grids,
    run_pipeline,
    run_pipeline_streaming,
    run_step,
    run_step_streaming,
)
from fim.app.steps_registry import StepSpec


def _make_step(step_id: str, **kw) -> StepSpec:
    return StepSpec(
        step_id=step_id,
        title=kw.get("title", step_id.capitalize()),
        script=kw.get("script"),
        essential=kw.get("essential", []),
        advanced=kw.get("advanced", []),
        common=kw.get("common"),
        methods=kw.get("methods"),
    )


class TestAsCliArgs:
    def test_basic_key_value(self):
        assert _as_cli_args({"alpha": 3.0, "name": "test"}) == ["--alpha", "3.0", "--name", "test"]

    def test_bool_true_becomes_flag(self):
        assert _as_cli_args({"verbose": True}) == ["--verbose"]

    def test_bool_false_omitted(self):
        assert _as_cli_args({"verbose": False}) == []

    def test_none_omitted(self):
        assert _as_cli_args({"x": None}) == []

    def test_empty_dict(self):
        assert _as_cli_args({}) == []

    def test_mixed(self):
        result = _as_cli_args({"a": 1, "b": True, "c": None, "d": "hi"})
        assert "--a" in result
        assert "--b" in result
        assert "--c" not in result
        assert "--d" in result


class TestRunStep:
    def test_distortion_not_implemented(self):
        step = _make_step("distortion")
        result = run_step(step, {})
        assert not result.ok
        assert result.returncode == 2
        assert "not implemented" in result.stderr.lower()

    def test_unknown_step(self):
        step = _make_step("unknown_step")
        result = run_step(step, {})
        assert not result.ok
        assert "Unknown step_id" in result.stderr

    def test_tracking_non_physics_method(self):
        step = _make_step(
            "tracking",
            common={
                "essential": [
                    {"key": "method", "type": "select", "default": "physics", "options": ["physics", "feature"]},
                ],
                "advanced": [],
            },
            methods={"physics": {"essential": [], "advanced": []}, "feature": {"essential": [], "advanced": []}},
        )
        result = run_step(step, {"method": "feature"})
        assert not result.ok
        assert "not implemented" in result.stderr.lower()

    @patch("fim.app.pipeline_runner.subprocess.run")
    def test_tracking_physics_runs_subprocess(self, mock_run):
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        step = _make_step(
            "tracking",
            common={
                "essential": [
                    {"key": "method", "type": "select", "default": "physics", "options": ["physics"]},
                ],
                "advanced": [],
            },
            methods={"physics": {"essential": [], "advanced": []}},
        )
        result = run_step(step, {"method": "physics"})
        assert result.ok
        assert mock_run.called

    @patch("fim.app.pipeline_runner.subprocess.run")
    def test_inverse_runs_subprocess(self, mock_run):
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="done\n", stderr="")
        step = _make_step("inverse")
        result = run_step(step, {})
        assert result.ok

    @patch("fim.app.pipeline_runner.subprocess.run")
    def test_extra_cli_args_appended(self, mock_run):
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        step = _make_step("inverse")
        run_step(step, {}, extra_cli_args=["--skip_grids"])
        cmd = mock_run.call_args[0][0]
        assert "--skip_grids" in cmd


class TestRunStepStreaming:
    def test_distortion_not_implemented(self):
        step = _make_step("distortion")
        result = run_step_streaming(step, {})
        assert not result.ok
        assert "not implemented" in result.stderr.lower()

    def test_unknown_step(self):
        step = _make_step("unknown_step")
        result = run_step_streaming(step, {})
        assert not result.ok

    def test_tracking_non_physics(self):
        step = _make_step(
            "tracking",
            common={
                "essential": [
                    {"key": "method", "type": "select", "default": "physics", "options": ["physics"]},
                ],
                "advanced": [],
            },
            methods={"physics": {"essential": [], "advanced": []}},
        )
        result = run_step_streaming(step, {"method": "feature"})
        assert not result.ok

    @patch("fim.app.pipeline_runner.subprocess.Popen")
    def test_inverse_streaming(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = ["line1\n", ""]
        mock_proc.stderr.readline.side_effect = ["err1\n", ""]
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        step = _make_step("inverse")
        stdout_lines = []
        stderr_lines = []
        result = run_step_streaming(
            step,
            {},
            on_stdout=lambda ln: stdout_lines.append(ln),
            on_stderr=lambda ln: stderr_lines.append(ln),
            env_overrides={"MY_VAR": "1"},
        )
        assert result.ok
        assert result.returncode == 0

    @patch("fim.app.pipeline_runner.subprocess.Popen")
    def test_extra_cli_args_streaming(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = [""]
        mock_proc.stderr.readline.side_effect = [""]
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        step = _make_step("inverse")
        run_step_streaming(step, {}, extra_cli_args=["--flag"])
        cmd = mock_popen.call_args[0][0]
        assert "--flag" in cmd

    def test_cancel_event_already_set_returns_early(self):
        step = _make_step("inverse")
        ev = threading.Event()
        ev.set()
        res = run_step_streaming(step, {}, cancel_event=ev)
        assert not res.ok
        assert res.returncode == -1
        assert res.command == []

    @patch("fim.app.pipeline_runner.subprocess.Popen")
    def test_reader_skips_none_streams(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = None
        mock_proc.stderr = None
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        step = _make_step("inverse")
        res = run_step_streaming(step, {})
        assert res.ok

    @patch("fim.app.pipeline_runner.subprocess.Popen")
    def test_cancel_watcher_terminates_process(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = ["", ""]
        mock_proc.stderr.readline.side_effect = ["", ""]
        mock_proc.poll.return_value = None
        cancel_event = threading.Event()
        step = _make_step("inverse")

        def wait_side():
            cancel_event.set()
            time.sleep(0.05)
            return -15

        mock_proc.wait.side_effect = wait_side
        mock_popen.return_value = mock_proc
        run_step_streaming(step, {}, cancel_event=cancel_event)
        assert mock_proc.terminate.called

    @patch("fim.app.pipeline_runner.subprocess.Popen")
    def test_cancel_watcher_poll_oserror_swallowed(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = ["", ""]
        mock_proc.stderr.readline.side_effect = ["", ""]
        mock_proc.poll.side_effect = OSError("gone")
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        cancel_event = threading.Event()
        step = _make_step("inverse")
        cancel_event.set()
        run_step_streaming(step, {}, cancel_event=cancel_event)

    @patch("fim.app.pipeline_runner.subprocess.Popen")
    def test_cancel_watcher_terminate_oserror_swallowed(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = ["", ""]
        mock_proc.stderr.readline.side_effect = ["", ""]
        mock_proc.poll.return_value = None
        mock_proc.terminate.side_effect = OSError("gone")
        cancel_event = threading.Event()

        def wait_side():
            cancel_event.set()
            time.sleep(0.05)
            return 0

        mock_proc.wait.side_effect = wait_side
        mock_popen.return_value = mock_proc
        step = _make_step("inverse")
        run_step_streaming(step, {}, cancel_event=cancel_event)

    @patch("fim.app.pipeline_runner.subprocess.Popen")
    def test_tracking_physics_streaming_invokes_module(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = [""]
        mock_proc.stderr.readline.side_effect = [""]
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        step = _make_step(
            "tracking",
            common={
                "essential": [
                    {"key": "method", "type": "select", "default": "physics", "options": ["physics", "feature"]},
                ],
                "advanced": [],
            },
            methods={
                "physics": {"essential": [], "advanced": []},
                "feature": {"essential": [], "advanced": []},
            },
        )
        run_step_streaming(step, {"method": "physics"})
        cmd = mock_popen.call_args[0][0]
        assert "fim.refactor.deformation_tracking" in cmd


class TestShouldSkipGrids:
    def test_both_tracking_and_inverse(self):
        steps = [_make_step("tracking"), _make_step("inverse")]
        assert _should_skip_grids(steps) is True

    def test_only_tracking(self):
        steps = [_make_step("tracking")]
        assert _should_skip_grids(steps) is False

    def test_only_inverse(self):
        steps = [_make_step("inverse")]
        assert _should_skip_grids(steps) is False

    def test_empty(self):
        assert _should_skip_grids([]) is False


class TestPropagateTrackingOutput:
    def test_sets_inverse_data_path(self):
        params = {"tracking": {"out_dir": "/tmp/out"}, "inverse": {}}
        _propagate_tracking_output(params)
        assert params["inverse"]["data_path"] == "/tmp/out"

    def test_does_not_overwrite_existing_data_path(self):
        params = {"tracking": {"out_dir": "/tmp/out"}, "inverse": {"data_path": "/my/path"}}
        _propagate_tracking_output(params)
        assert params["inverse"]["data_path"] == "/my/path"

    def test_no_tracking_out_dir(self):
        params = {"tracking": {}, "inverse": {}}
        _propagate_tracking_output(params)
        assert "data_path" not in params["inverse"]

    def test_missing_steps(self):
        params = {}
        _propagate_tracking_output(params)  # should not raise


class TestRunPipeline:
    @patch("fim.app.pipeline_runner.run_step")
    def test_runs_all_steps(self, mock_run_step):
        mock_run_step.return_value = RunResult(ok=True, returncode=0, stdout="ok", stderr="", command=["cmd"])
        steps = [_make_step("distortion"), _make_step("inverse")]
        result = run_pipeline(steps, {})
        assert mock_run_step.call_count == 2
        assert result.ok

    @patch("fim.app.pipeline_runner.run_step")
    def test_stops_on_failure(self, mock_run_step):
        mock_run_step.return_value = RunResult(ok=False, returncode=1, stdout="", stderr="fail", command=["cmd"])
        steps = [_make_step("tracking"), _make_step("inverse")]
        result = run_pipeline(steps, {})
        assert not result.ok
        assert len(result.results) == 1

    @patch("fim.app.pipeline_runner.run_step")
    def test_skip_grids_injected(self, mock_run_step):
        mock_run_step.return_value = RunResult(ok=True, returncode=0, stdout="", stderr="", command=["cmd"])
        steps = [_make_step("tracking"), _make_step("inverse")]
        run_pipeline(steps, {"tracking": {"out_dir": "/tmp"}, "inverse": {}})
        first_call = mock_run_step.call_args_list[0]
        assert first_call.kwargs.get("extra_cli_args") == ["--skip_grids"]


class TestRunPipelineStreaming:
    @patch("fim.app.pipeline_runner.run_step_streaming")
    def test_runs_all_steps(self, mock_stream):
        mock_stream.return_value = RunResult(ok=True, returncode=0, stdout="ok", stderr="", command=["cmd"])
        steps = [_make_step("inverse")]
        result = run_pipeline_streaming(steps, {})
        assert result.ok

    @patch("fim.app.pipeline_runner.run_step_streaming")
    def test_stops_on_failure(self, mock_stream):
        mock_stream.return_value = RunResult(ok=False, returncode=1, stdout="", stderr="err", command=["cmd"])
        steps = [_make_step("tracking"), _make_step("inverse")]
        result = run_pipeline_streaming(steps, {})
        assert not result.ok
        assert len(result.results) == 1

    @patch("fim.app.pipeline_runner.run_step_streaming")
    def test_with_callbacks_and_env(self, mock_stream):
        mock_stream.return_value = RunResult(ok=True, returncode=0, stdout="", stderr="", command=["cmd"])
        steps = [_make_step("tracking"), _make_step("inverse")]
        run_pipeline_streaming(
            steps,
            {"tracking": {"out_dir": "/tmp"}, "inverse": {}},
            on_stdout=lambda sid, ln: None,
            on_stderr=lambda sid, ln: None,
            env_overrides_by_step={"tracking": {"FIM_UI_NO_TQDM": "1"}},
        )
        assert mock_stream.call_count == 2

    @patch("fim.app.pipeline_runner.run_step_streaming")
    def test_mid_pipeline_cancel_returns_early(self, mock_stream):
        cancel = threading.Event()

        def stream_side_effect(step, params, **kwargs):
            _ = (step, params)
            if kwargs.get("cancel_event") is not None:
                kwargs["cancel_event"].set()
            return RunResult(ok=True, returncode=0, stdout="", stderr="", command=["c"])

        mock_stream.side_effect = stream_side_effect
        steps = [_make_step("tracking"), _make_step("inverse")]
        result = run_pipeline_streaming(steps, {}, cancel_event=cancel)
        assert not result.ok
        assert len(result.results) == 1
