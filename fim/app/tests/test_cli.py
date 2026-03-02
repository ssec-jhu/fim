from __future__ import annotations

from unittest.mock import patch

import pytest

from fim.app.cli import _parse_kv, main
from fim.app.pipeline_runner import RunResult


class TestParseKV:
    def test_string_value(self):
        assert _parse_kv("name=hello") == ("name", "hello")

    def test_int_value(self):
        assert _parse_kv("count=42") == ("count", 42)

    def test_float_value(self):
        assert _parse_kv("rate=3.14") == ("rate", 3.14)

    def test_scientific_notation(self):
        k, v = _parse_kv("force=1.5e-3")
        assert k == "force"
        assert abs(v - 1.5e-3) < 1e-10

    def test_bool_true(self):
        assert _parse_kv("flag=true") == ("flag", True)

    def test_bool_false(self):
        assert _parse_kv("flag=false") == ("flag", False)

    def test_bool_case_insensitive(self):
        assert _parse_kv("flag=True") == ("flag", True)
        assert _parse_kv("flag=FALSE") == ("flag", False)

    def test_no_equals_raises(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="key=value"):
            _parse_kv("badparam")

    def test_value_with_equals(self):
        k, v = _parse_kv("path=/a/b=c")
        assert k == "path"
        assert v == "/a/b=c"

    def test_whitespace_stripped(self):
        assert _parse_kv(" key = value ") == ("key", "value")


class TestMainListSteps:
    def test_list_steps(self, capsys):
        rc = main(["list-steps"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "tracking" in captured.out.lower() or "distortion" in captured.out.lower()


class TestMainShowStep:
    def test_show_step(self, capsys):
        rc = main(["show-step", "tracking"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "tracking" in captured.out

    def test_show_step_unknown(self):
        with pytest.raises(KeyError):
            main(["show-step", "nonexistent"])


class TestMainRun:
    @patch("fim.app.cli.run_step")
    def test_run_step(self, mock_run, capsys):
        mock_run.return_value = RunResult(
            ok=True, returncode=0, stdout="output\n", stderr="", command=["python", "-m", "test"]
        )
        rc = main(["run", "inverse", "--set", "model=linear"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "ok: True" in captured.out
        assert "output" in captured.out

    @patch("fim.app.cli.run_step")
    def test_run_step_failure(self, mock_run, capsys):
        mock_run.return_value = RunResult(
            ok=False, returncode=1, stdout="", stderr="error msg\n", command=["python"]
        )
        rc = main(["run", "inverse"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "error msg" in captured.out

    @patch("fim.app.cli.run_step")
    def test_run_with_multiple_kvs(self, mock_run, capsys):
        mock_run.return_value = RunResult(ok=True, returncode=0, stdout="", stderr="", command=[])
        rc = main(["run", "inverse", "--set", "E1_init=15000", "--set", "v12=0.49"])
        assert rc == 0
