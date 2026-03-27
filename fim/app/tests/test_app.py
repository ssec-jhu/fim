from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import fim.app.main as app_main

from ..main import __version__


class TestLifespan:
    def test_lifespan_prints_browser_hint(self, monkeypatch, capsys):
        monkeypatch.delenv("FIM_SUPPRESS_BROWSER_HINT", raising=False)

        async def _run():
            async with app_main._lifespan(app_main.app):
                return None

        asyncio.run(_run())
        err = capsys.readouterr().err
        assert "http://localhost:8000" in err

    def test_lifespan_suppressed(self, monkeypatch, capsys):
        monkeypatch.setenv("FIM_SUPPRESS_BROWSER_HINT", "1")

        async def _run():
            async with app_main._lifespan(app_main.app):
                return None

        asyncio.run(_run())
        err = capsys.readouterr().err
        assert "localhost:8000" not in err


class TestApp:
    def test_root(self, app_client):
        response = app_client.get("/")
        assert response.status_code == 200

    def test_healthz(self, app_client):
        response = app_client.get("/healthz")
        assert response.status_code == 200
        if __version__:
            assert __version__ in response.json()["message"], response.json()

    def test_api_list_steps(self, app_client):
        response = app_client.get("/api/steps")
        assert response.status_code == 200
        data = response.json()
        assert "steps" in data
        assert isinstance(data["steps"], list)
        assert len(data["steps"]) > 0
        for step in data["steps"]:
            assert "id" in step
            assert "title" in step

    def test_api_get_step(self, app_client):
        response = app_client.get("/api/steps/tracking")
        assert response.status_code == 200
        data = response.json()
        assert data["step_id"] == "tracking"
        assert "title" in data
        assert "essential" in data

    def test_api_get_step_with_common_methods(self, app_client):
        response = app_client.get("/api/steps/inverse")
        assert response.status_code == 200
        data = response.json()
        if data.get("common") is not None:
            assert isinstance(data["common"], dict)
        if data.get("methods") is not None:
            assert isinstance(data["methods"], dict)

    def test_api_fs_config(self, app_client):
        response = app_client.get("/api/fs/config")
        assert response.status_code == 200
        data = response.json()
        assert "browse_root" in data
        assert "browse_hint" in data
        assert isinstance(data["browse_hint"], str)
        assert "native_folder_picker" in data
        assert isinstance(data["native_folder_picker"], bool)

    def test_api_pick_folder_native_unavailable(self, app_client, monkeypatch):
        monkeypatch.setattr("fim.app.main.native_folder_picker_available", lambda: False)
        response = app_client.post("/api/fs/pick_folder_native")
        assert response.status_code == 200
        assert response.json() == {"ok": False, "error": "unavailable"}

    def test_api_fs_list_home(self, app_client):
        response = app_client.get("/api/fs/list")
        assert response.status_code == 200
        data = response.json()
        assert "path" in data
        assert "root" in data
        assert "dirs" in data
        assert isinstance(data["dirs"], list)
        assert "browse_hint" in data

    def test_api_fs_list_with_path(self, app_client, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        response = app_client.get("/api/fs/list", params={"path": str(tmp_path)})
        assert response.status_code == 200
        data = response.json()
        assert data["path"] == str(tmp_path)
        names = [d["name"] for d in data["dirs"]]
        assert "subdir" in names

    def test_api_fs_list_hides_dotfiles(self, app_client, tmp_path):
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "visible").mkdir()
        response = app_client.get("/api/fs/list", params={"path": str(tmp_path)})
        data = response.json()
        names = [d["name"] for d in data["dirs"]]
        assert ".hidden" not in names
        assert "visible" in names

    def test_api_fs_list_non_dir_falls_back_to_parent(self, app_client, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        response = app_client.get("/api/fs/list", params={"path": str(f)})
        data = response.json()
        assert data["path"] == str(tmp_path)

    def test_api_fs_list_browse_root_defaults_and_clamps(self, app_client, tmp_path, monkeypatch):
        monkeypatch.setenv("FIM_FS_LIST_ROOT", str(tmp_path))
        (tmp_path / "inside").mkdir()
        r0 = app_client.get("/api/fs/list")
        assert r0.status_code == 200
        d0 = r0.json()
        assert d0["path"] == str(tmp_path)
        assert d0["browse_root"] == str(tmp_path)
        r1 = app_client.get("/api/fs/list", params={"path": "/etc"})
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["path"] == str(tmp_path)

    def test_api_upload(self, app_client):
        response = app_client.post(
            "/api/upload",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "path" in data
        assert data["filename"] == "test.txt"

    @patch("fim.app.main.job_mgr")
    def test_api_run_async(self, mock_mgr, app_client):
        from fim.app.jobs import JobState

        mock_mgr.create.return_value = JobState(job_id="test123")
        response = app_client.post(
            "/api/run_async/inverse",
            json={"params": {"model": "linear"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

    def test_api_run_pipeline_async_unknown_step(self, app_client):
        response = app_client.post(
            "/api/run_pipeline_async",
            json={"start_step_id": "nonexistent", "params_by_step": {}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "error" in data

    @patch("fim.app.main.job_mgr")
    def test_api_run_pipeline_async_valid(self, mock_mgr, app_client):
        from fim.app.jobs import JobState

        mock_mgr.create.return_value = JobState(job_id="pipe123")
        response = app_client.post(
            "/api/run_pipeline_async",
            json={"start_step_id": "tracking", "params_by_step": {}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

    def test_api_get_job_not_found(self, app_client):
        response = app_client.get("/api/jobs/nonexistent_id")
        assert response.status_code == 200
        data = response.json()
        assert data.get("error") == "not found"

    @patch("fim.app.main.job_mgr")
    def test_api_get_job_found(self, mock_mgr, app_client):
        from fim.app.jobs import JobState

        job = JobState(job_id="j1")
        job.status = "running"
        job.step_id = "inverse"
        job.log = "some log"
        mock_mgr.get.return_value = job
        response = app_client.get("/api/jobs/j1")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "j1"
        assert data["status"] == "running"
        assert data["log"] == "some log"

    @patch("fim.app.main.job_mgr")
    def test_api_cancel_job(self, mock_mgr, app_client):
        mock_mgr.cancel.return_value = True
        response = app_client.post("/api/jobs/abc123/cancel")
        assert response.status_code == 200
        assert response.json() == {"ok": True}
        mock_mgr.cancel.assert_called_once_with("abc123")


class TestMainResolveAndNative:
    def test_resolve_fs_list_target_falls_back_when_clamped_not_dir(self, monkeypatch, tmp_path):
        f = tmp_path / "not_a_dir"
        f.write_text("x")

        monkeypatch.setattr(app_main, "_fs_browse_root", lambda: None)

        def _clamp(target, root):  # noqa: ARG001
            return f

        monkeypatch.setattr(app_main, "_fs_clamp_under_root", _clamp)
        out = app_main._resolve_fs_list_target(str(tmp_path))
        assert out.is_dir()
        assert out == app_main.Path.home()

    def test_native_folder_picker_env_disabled(self, monkeypatch):
        for val in ("0", "false", "no", "off", "FALSE"):
            monkeypatch.setenv("FIM_NATIVE_FOLDER_PICKER", val)
            assert app_main._native_folder_picker_env_enabled() is False

    def test_native_folder_picker_available_non_darwin(self, monkeypatch):
        monkeypatch.setenv("FIM_NATIVE_FOLDER_PICKER", "1")
        monkeypatch.setattr(app_main.sys, "platform", "linux")
        assert app_main.native_folder_picker_available() is False

    def test_native_folder_picker_available_env_off_on_darwin(self, monkeypatch):
        monkeypatch.setenv("FIM_NATIVE_FOLDER_PICKER", "0")
        monkeypatch.setattr(app_main.sys, "platform", "darwin")
        monkeypatch.setattr(app_main.shutil, "which", lambda x: "/usr/bin/osascript" if x == "osascript" else None)
        assert app_main.native_folder_picker_available() is False

    def test_pick_folder_macos_no_osascript(self, monkeypatch):
        monkeypatch.setattr(app_main.shutil, "which", lambda x: None if x == "osascript" else "/usr/bin/x")
        assert app_main._pick_folder_macos(None) is None

    def test_pick_folder_macos_subprocess_fails(self, monkeypatch, tmp_path):
        monkeypatch.setattr(app_main.shutil, "which", lambda x: "/usr/bin/osascript" if x == "osascript" else None)

        class R:
            returncode = 1
            stdout = ""

        monkeypatch.setattr(app_main.subprocess, "run", lambda *a, **k: R())
        assert app_main._pick_folder_macos(tmp_path) is None

    def test_pick_folder_macos_success(self, monkeypatch, tmp_path):
        monkeypatch.setattr(app_main.shutil, "which", lambda x: "/usr/bin/osascript" if x == "osascript" else None)
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        picked = str(chosen.resolve())

        class R:
            returncode = 0
            stdout = picked + "\n"

        monkeypatch.setattr(app_main.subprocess, "run", lambda *a, **k: R())
        out = app_main._pick_folder_macos(tmp_path)
        assert out == picked

    def test_pick_folder_macos_invalid_start_resets_to_home(self, monkeypatch):
        monkeypatch.setattr(app_main.shutil, "which", lambda x: "/usr/bin/osascript" if x == "osascript" else None)
        bad_start = MagicMock()
        bad_start.is_dir.side_effect = [True, False]

        class R:
            returncode = 0
            stdout = "/tmp/from_osascript\n"

        monkeypatch.setattr(app_main.subprocess, "run", lambda *a, **k: R())
        out = app_main._pick_folder_macos(bad_start)
        assert out == "/tmp/from_osascript"


class TestMainHttpEdges:
    def test_api_fs_list_permission_error_on_iterdir(self, app_client, monkeypatch):
        mock_target = MagicMock()
        mock_target.iterdir.side_effect = PermissionError("denied")
        mock_target.parent = mock_target
        mock_target.__str__ = lambda self=None: "/mock"
        monkeypatch.setattr(app_main, "_resolve_fs_list_target", lambda path=None: mock_target)
        monkeypatch.setattr(app_main, "_fs_browse_root", lambda: None)
        monkeypatch.setattr(app_main, "_fs_browse_hint", lambda: "hint")

        r = app_client.get("/api/fs/list")
        assert r.status_code == 200
        assert r.json()["dirs"] == []

    def test_api_fs_config_native_hint_when_true(self, app_client, monkeypatch):
        monkeypatch.setattr(app_main, "native_folder_picker_available", lambda: True)
        r = app_client.get("/api/fs/config")
        assert r.status_code == 200
        assert r.json()["native_folder_hint"]

    def test_api_pick_folder_native_success(self, app_client, monkeypatch, tmp_path):
        dest = tmp_path / "picked"
        dest.mkdir()

        monkeypatch.setattr(app_main, "native_folder_picker_available", lambda: True)
        monkeypatch.setattr(app_main, "_pick_folder_macos", lambda initial: str(dest))
        r = app_client.post("/api/fs/pick_folder_native")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["path"] == str(dest.resolve())

    def test_api_pick_folder_native_cancelled(self, app_client, monkeypatch):
        monkeypatch.setattr(app_main, "native_folder_picker_available", lambda: True)
        monkeypatch.setattr(app_main, "_pick_folder_macos", lambda initial: None)
        r = app_client.post("/api/fs/pick_folder_native")
        assert r.json() == {"ok": False, "error": "cancelled"}

    def test_api_pick_folder_native_pick_raises_oserror(self, app_client, monkeypatch):
        monkeypatch.setattr(app_main, "native_folder_picker_available", lambda: True)

        def boom(_initial):
            raise OSError("boom")

        monkeypatch.setattr(app_main, "_pick_folder_macos", boom)
        r = app_client.post("/api/fs/pick_folder_native")
        assert r.json() == {"ok": False, "error": "failed"}

    def test_api_pick_folder_native_invalid_path(self, app_client, monkeypatch, tmp_path):
        monkeypatch.setattr(app_main, "native_folder_picker_available", lambda: True)
        fp = tmp_path / "onlyfile.txt"
        fp.write_text("x")
        monkeypatch.setattr(app_main, "_pick_folder_macos", lambda initial: str(fp))
        r = app_client.post("/api/fs/pick_folder_native")
        assert r.json() == {"ok": False, "error": "invalid"}
