from __future__ import annotations

import runpy
import sys
from unittest.mock import patch

import pytest

from fim.app.run_ui import main


class TestRunUiMain:
    @patch("uvicorn.run")
    def test_defaults(self, mock_run: object) -> None:
        with patch.object(sys, "argv", ["fim-ui"]):
            main()
        mock_run.assert_called_once_with(
            "fim.app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
        )

    @patch("uvicorn.run")
    def test_host_port(self, mock_run: object) -> None:
        with patch.object(sys, "argv", ["fim-ui", "--host", "0.0.0.0", "--port", "9000"]):
            main()
        mock_run.assert_called_once_with(
            "fim.app.main:app",
            host="0.0.0.0",
            port=9000,
            reload=True,
        )

    @patch("uvicorn.run")
    def test_no_reload(self, mock_run: object) -> None:
        with patch.object(sys, "argv", ["fim-ui", "--no-reload"]):
            main()
        mock_run.assert_called_once_with(
            "fim.app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
        )

    def test_help_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(sys, "argv", ["fim-ui", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code in (0, None)
        out = capsys.readouterr().out
        assert "fim-ui" in out
        assert "host" in out.lower()

    def test_name_main_guard_invokes_main_for_help(self) -> None:
        """Cover the ``__main__`` entry by running the module source with ``run_name='__main__'``."""
        with patch.object(sys, "argv", ["fim-ui", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module("fim.app.run_ui", run_name="__main__", alter_sys=False)
        assert exc_info.value.code in (0, None)
