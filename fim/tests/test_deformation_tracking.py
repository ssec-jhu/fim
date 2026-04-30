"""End-to-end tests for the :mod:`fim.refactor.deformation_tracking` driver.

The reusable helpers live in dedicated modules and are unit-tested there:

- ``fim.refactor.tracking_io``     → :mod:`fim.tests.test_tracking_io`
- ``fim.refactor.tracking_optim``  → :mod:`fim.tests.test_tracking_optim`
- ``fim.refactor.tracking_remap``  → :mod:`fim.tests.test_tracking_remap`

This file exercises ``deformation_tracking.main`` itself: argv handling,
output files written, run-info content, and the various optional code paths
(remap, smoothing, output downsampling, device selection, etc.). The two TIFF
loaders and the phase-correlation seed are patched at their source modules
because ``main`` resolves them as ``tio.load_tiff_zyx_to_xyz`` /
``topt.estimate_initial_shift`` via attribute lookup.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("torch", reason="deformation_tracking requires PyTorch")
import torch  # noqa: E402

from fim.refactor import deformation_tracking as dt  # noqa: E402
from fim.refactor import tracking_io as tio  # noqa: E402
from fim.refactor import tracking_optim as topt  # noqa: E402


@pytest.fixture
def tiny_stack() -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.standard_normal((6, 6, 6)).astype(np.float32)


def _patch_load_and_shift(vol: np.ndarray) -> tuple[patch, patch]:
    """Patch the TIFF loader and initial-shift estimator at their source modules.

    ``dt.main`` resolves both as ``tio.load_tiff_zyx_to_xyz`` /
    ``topt.estimate_initial_shift`` via attribute lookup on the imported module,
    so patching the helper module attribute is what actually intercepts the
    call in the pipeline.
    """

    def fake_load(*_a, **_k) -> np.ndarray:
        return vol.copy()

    return (
        patch.object(tio, "load_tiff_zyx_to_xyz", side_effect=fake_load),
        patch.object(topt, "estimate_initial_shift", return_value=(np.zeros(2, dtype=np.float64), 0.0)),
    )


def _base_argv(out: Path, *, num_iter: str = "2", batch_size: str = "64") -> list[str]:
    return [
        "deformation_tracking",
        "--out_dir",
        str(out),
        "--with_sphere",
        "w.tif",
        "--without_sphere",
        "wo.tif",
        "--num_iter",
        num_iter,
        "--batch_size",
        batch_size,
        "--progress_every",
        "1",
        "--device",
        "cpu",
        "--deform_downsample_factor_xy",
        "2",
        "--deform_downsample_factor_z",
        "2",
    ]


@pytest.mark.unit
class TestMainPipeline:
    def test_main_writes_outputs_and_grids(
        self,
        tiny_stack: np.ndarray,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        with patch.object(sys, "argv", _base_argv(tmp_path)):
            with pl, ps:
                dt.main()
        assert (tmp_path / "Ux.npy").exists()
        assert (tmp_path / "Uy.npy").exists()
        assert (tmp_path / "Uz.npy").exists()
        assert (tmp_path / "grid_params.json").exists()
        meta = json.loads((tmp_path / "grid_params.json").read_text(encoding="utf-8"))
        assert "shape" in meta and "voxel_volume_m3" in meta
        assert (tmp_path / "X.npy").exists()
        assert (tmp_path / "run_info.txt").exists()
        captured = capsys.readouterr()
        assert "Tracking step runtime" in captured.err
        assert "Saved outputs" in captured.out

    def test_main_skip_grids(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        with patch.object(sys, "argv", _base_argv(tmp_path) + ["--skip_grids"]):
            with pl, ps:
                dt.main()
        assert not (tmp_path / "X.npy").exists()
        assert (tmp_path / "Ux.npy").exists()

    def test_main_save_comparison_figure_flag_calls_helper(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        with patch.object(
            tio, "_save_comparison_figure", return_value=tmp_path / "comparison_prediction_vs_data.png"
        ) as m:
            with patch.object(sys, "argv", _base_argv(tmp_path) + ["--save_comparison_figure"]):
                with pl, ps:
                    dt.main()
        assert m.call_count == 1

    def test_main_remap_to_reference_writes_outputs_and_run_info(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        argv = _base_argv(tmp_path) + [
            "--remap_to_reference",
            "--remap_interp",
            "nearest",
            "--remap_max_iter",
            "5",
        ]
        with patch.object(sys, "argv", argv):
            with pl, ps:
                dt.main()
        assert (tmp_path / "Ux.npy").exists()
        ri = (tmp_path / "run_info.txt").read_text(encoding="utf-8")
        assert "remap_to_reference=True" in ri
        assert "remap_stage=coarse_then_upsample" in ri
        assert "--remap_to_reference" in ri

    def test_main_repro_includes_z_end_when_set(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        with patch.object(sys, "argv", _base_argv(tmp_path) + ["--z_end", "6"]):
            with pl, ps:
                dt.main()
        ri = (tmp_path / "run_info.txt").read_text(encoding="utf-8")
        assert "z_end=6" in ri
        assert "--z_end 6" in ri

    def test_main_trace_mse_writes_npy_and_run_info(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        with patch.object(sys, "argv", _base_argv(tmp_path) + ["--trace_mse"]):
            with pl, ps:
                dt.main()
        mse_path = tmp_path / "optimization_mse.npy"
        assert mse_path.is_file()
        arr = np.load(mse_path)
        assert arr.dtype == np.float64
        assert arr.shape == (2,)  # num_iter from _base_argv
        assert np.all(np.isfinite(arr))
        ri = (tmp_path / "run_info.txt").read_text(encoding="utf-8")
        assert "trace_mse=True" in ri
        assert "optimization_mse_npy=" in ri and "mse_curve_png=" in ri
        pytest.importorskip("matplotlib", reason="mse curve requires matplotlib")
        assert (tmp_path / "mse_curve.png").is_file()

    def test_main_shape_mismatch_raises(self, tmp_path: Path) -> None:
        v1 = np.ones((4, 4, 4), dtype=np.float32)
        v2 = np.ones((3, 4, 4), dtype=np.float32)
        n = [0]

        def fake_load(*_a, **_k) -> np.ndarray:
            n[0] += 1
            return v1.copy() if n[0] == 1 else v2.copy()

        with patch.object(sys, "argv", _base_argv(tmp_path)):
            with patch.object(tio, "load_tiff_zyx_to_xyz", side_effect=fake_load):
                with pytest.raises(ValueError, match="Shape mismatch"):
                    dt.main()

    def test_main_empty_volume_raises(self, tmp_path: Path) -> None:
        pl = patch.object(tio, "load_tiff_zyx_to_xyz", return_value=np.empty((0, 0, 0), dtype=np.float32))
        with patch.object(sys, "argv", _base_argv(tmp_path)):
            with pl:
                with pytest.raises(ValueError, match="Empty volume"):
                    dt.main()

    def test_main_lock_global_shift(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        with patch.object(sys, "argv", _base_argv(tmp_path) + ["--lock_global_shift"]):
            with pl, ps:
                dt.main()
        assert "lock_global_shift=True" in (tmp_path / "run_info.txt").read_text(encoding="utf-8")

    def test_main_tv2_and_indentation_penalty(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        extras = ["--TV2_reg", "0.01", "--Uz_penalty_weight", "0.1"]
        with patch.object(sys, "argv", _base_argv(tmp_path) + extras):
            with pl, ps:
                dt.main()
        assert (tmp_path / "Uz.npy").exists()

    def test_main_tv2_zero_skips_regularizer(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        with patch.object(sys, "argv", _base_argv(tmp_path) + ["--TV2_reg", "0"]):
            with pl, ps:
                dt.main()
        assert (tmp_path / "Ux.npy").exists()

    def test_main_smooth_gaussian_and_stderr(
        self,
        tiny_stack: np.ndarray,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        argv_ex = _base_argv(tmp_path) + ["--smooth_method", "gaussian", "--smooth_sigma", "0.5"]
        with patch.object(sys, "argv", argv_ex):
            with pl, ps:
                dt.main()
        err = capsys.readouterr().err
        assert "Smoothing displacement field" in err

    def test_main_smooth_laplacian(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        argv_ex = _base_argv(tmp_path) + ["--smooth_method", "laplacian", "--smooth_sigma", "2.0"]
        with patch.object(sys, "argv", argv_ex):
            with pl, ps:
                dt.main()
        assert (tmp_path / "Ux.npy").exists()

    def test_main_output_downsample(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        argv = _base_argv(tmp_path) + ["--output_downsample_xy", "2", "--output_downsample_z", "2"]
        with patch.object(sys, "argv", argv):
            with pl, ps:
                dt.main()
        u = np.load(tmp_path / "Ux.npy")
        assert u.shape == (3, 3, 3)

    def test_main_downsample_clamped_to_one(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        argv = _base_argv(tmp_path) + ["--output_downsample_xy", "0", "--output_downsample_z", "0"]
        with patch.object(sys, "argv", argv):
            with pl, ps:
                dt.main()
        u = np.load(tmp_path / "Ux.npy")
        assert u.shape == tiny_stack.shape

    def test_main_upsample_order_linear_and_quadratic(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        for order in ("1", "2"):
            sub = tmp_path / f"o{order}"
            sub.mkdir()
            pl, ps = _patch_load_and_shift(tiny_stack)
            with patch.object(sys, "argv", _base_argv(sub) + ["--upsample_order", order]):
                with pl, ps:
                    dt.main()
            assert (sub / "Ux.npy").exists()

    def test_main_progress_every_resets_when_larger_than_num_iter(
        self, tiny_stack: np.ndarray, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        argv = [
            "deformation_tracking",
            "--out_dir",
            str(tmp_path),
            "--with_sphere",
            "w.tif",
            "--without_sphere",
            "wo.tif",
            "--num_iter",
            "3",
            "--batch_size",
            "32",
            "--progress_every",
            "500",
            "--device",
            "cpu",
            "--deform_downsample_factor_xy",
            "2",
            "--deform_downsample_factor_z",
            "2",
        ]
        with patch.object(sys, "argv", argv):
            with pl, ps:
                dt.main()
        err = capsys.readouterr().err
        assert "FIM_PROGRESS" in err

    def test_main_fim_ui_no_tqdm_prints_progress(
        self,
        tiny_stack: np.ndarray,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("FIM_UI_NO_TQDM", "1")
        pl, ps = _patch_load_and_shift(tiny_stack)
        with patch.object(sys, "argv", _base_argv(tmp_path)):
            with pl, ps:
                dt.main()
        err = capsys.readouterr().err
        assert "FIM_PROGRESS" in err
        monkeypatch.delenv("FIM_UI_NO_TQDM", raising=False)

    def test_main_auto_device_uses_cpu_when_no_cuda(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        argv = [a for a in _base_argv(tmp_path) if a not in ("--device", "cpu")]
        argv.extend(["--device", "auto"])
        with patch.object(sys, "argv", argv):
            with pl, ps:
                with patch("torch.cuda.is_available", return_value=False):
                    dt.main()
        run_info = (tmp_path / "run_info.txt").read_text(encoding="utf-8").lower()
        assert "cpu" in run_info

    def test_main_explicit_cuda_branch_uses_cuda_device(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        """Cover ``--device cuda`` without requiring a real GPU (patch ``torch.device``)."""
        device_calls: list[str] = []
        real_device = torch.device

        def _device_stub(name: str | torch.device, *args: object, **kwargs: object) -> torch.device:
            device_calls.append(str(name))
            return real_device("cpu")

        pl, ps = _patch_load_and_shift(tiny_stack)
        argv = [a for a in _base_argv(tmp_path) if a not in ("--device", "cpu")]
        argv.extend(["--device", "cuda"])
        with patch.object(sys, "argv", argv):
            with patch("fim.refactor.deformation_tracking.torch.device", side_effect=_device_stub):
                with pl, ps:
                    dt.main()
        assert any("cuda" in str(c) for c in device_calls)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_main_explicit_cuda_device_on_gpu_machine(self, tiny_stack: np.ndarray, tmp_path: Path) -> None:
        pl, ps = _patch_load_and_shift(tiny_stack)
        argv = [a for a in _base_argv(tmp_path) if a not in ("--device", "cpu")]
        argv.extend(["--device", "cuda"])
        with patch.object(sys, "argv", argv):
            with pl, ps:
                dt.main()
        assert "cuda" in (tmp_path / "run_info.txt").read_text(encoding="utf-8").lower()


@pytest.mark.unit
class TestBuildParser:
    """Smoke tests for the CLI parser surface (defaults, choices, presence).

    Mirrors ``test_main_VFM.test_build_parser_defaults``: catches accidental
    flag renames or default-value drift without launching the pipeline.
    """

    def test_defaults_match_documented_values(self) -> None:
        args = dt.build_parser().parse_args([])
        # Geometry / preprocessing
        assert args.dxy_um == pytest.approx(0.492)
        assert args.dz_um == pytest.approx(3.0)
        assert args.sphere_diameter_mm == pytest.approx(1.0)
        assert args.downsamp_xy == 2
        assert args.downsamp_z == 1
        assert args.z_start == 0
        assert args.z_end is None
        # Optimization
        assert args.num_iter == 1001
        assert args.batch_size == 750_000
        assert args.progress_every == 100
        assert args.lr_shift == pytest.approx(5e-2)
        assert args.lr_axis == pytest.approx(1e-3)
        assert args.lr_angle == pytest.approx(1e-5)
        assert args.lr_deform == pytest.approx(0.3)
        assert args.TV2_reg == pytest.approx(30.0)
        assert args.Uz_penalty_weight == pytest.approx(0.0)
        assert args.lock_global_shift is False
        # Coarse grid + output downsample
        assert args.deform_downsample_factor_xy == 10
        assert args.deform_downsample_factor_z == 2
        assert args.output_downsample_xy == 10
        assert args.output_downsample_z == 3
        assert args.upsample_order == 3
        # Switches default off
        assert args.skip_grids is False
        assert args.save_comparison_figure is False
        assert args.remap_to_reference is False
        assert args.trace_mse is False
        # Smoothing / remap defaults
        assert args.smooth_method == "none"
        assert args.smooth_sigma == pytest.approx(1.0)
        assert args.remap_interp == "linear"
        assert args.remap_max_iter == 25
        # Device default
        assert args.device == "auto"

    def test_choice_flags_reject_invalid_values(self) -> None:
        parser = dt.build_parser()
        for argv in (
            ["--device", "tpu"],
            ["--upsample_order", "5"],
            ["--smooth_method", "bilateral"],
            ["--remap_interp", "cubic"],
        ):
            with pytest.raises(SystemExit):
                parser.parse_args(argv)

    def test_overrides_propagate(self) -> None:
        args = dt.build_parser().parse_args(
            [
                "--with_sphere",
                "/tmp/w.tif",
                "--without_sphere",
                "/tmp/wo.tif",
                "--out_dir",
                "/tmp/out",
                "--num_iter",
                "7",
                "--device",
                "cpu",
                "--smooth_method",
                "gaussian",
                "--remap_to_reference",
            ]
        )
        assert args.with_sphere == "/tmp/w.tif"
        assert args.without_sphere == "/tmp/wo.tif"
        assert args.out_dir == "/tmp/out"
        assert args.num_iter == 7
        assert args.device == "cpu"
        assert args.smooth_method == "gaussian"
        assert args.remap_to_reference is True

    def test_module_exposes_build_parser_callable(self) -> None:
        assert callable(dt.build_parser)


@pytest.mark.unit
class TestModuleMainGuard:
    def test_name_main_guard_invokes_main_for_help(self) -> None:
        """Cover ``if __name__ == '__main__': main()`` via runpy (same process / coverage)."""
        with patch.object(sys, "argv", ["deformation_tracking", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module(
                    "fim.refactor.deformation_tracking",
                    run_name="__main__",
                    alter_sys=False,
                )
        assert exc_info.value.code in (0, None)
