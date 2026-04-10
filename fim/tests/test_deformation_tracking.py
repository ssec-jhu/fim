"""Unit tests for ``fim.refactor.deformation_tracking``."""

from __future__ import annotations

import builtins
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("torch", reason="deformation_tracking requires PyTorch")
import torch

from fim.refactor import deformation_tracking as dt


@pytest.fixture
def tiny_stack() -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.standard_normal((6, 6, 6)).astype(np.float32)


def _patch_load_and_shift(vol: np.ndarray) -> tuple[patch, patch]:
    def fake_load(*_a, **_k) -> np.ndarray:
        return vol.copy()

    return (
        patch.object(dt, "load_tiff_zyx_to_xyz", side_effect=fake_load),
        patch.object(dt, "estimate_initial_shift", return_value=(np.zeros(2, dtype=np.float64), 0.0)),
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
class TestDisplayInputName:
    def test_plain_filename_no_underscore(self) -> None:
        assert dt._display_input_name("plain.tif") == "plain.tif"

    def test_uuid_prefixed_upload_name_is_cleaned(self) -> None:
        path = "/tmp/fim_uploads/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_def_image.tif"
        assert dt._display_input_name(path) == "def_image.tif"

    def test_non_uuid_prefix_kept(self) -> None:
        path = "/tmp/fim_uploads/notauuid_def_image.tif"
        assert dt._display_input_name(path) == "notauuid_def_image.tif"


@pytest.mark.unit
class TestUnravelFlatIndices:
    def test_numpy_path_when_torch_api_absent(self, monkeypatch) -> None:
        pytest.importorskip("torch")
        import torch

        monkeypatch.setattr(torch, "unravel_index", None, raising=False)
        idx = torch.tensor([0, 5], dtype=torch.long)
        x, y, z = dt._unravel_flat_indices(idx, (2, 3, 4))
        assert x.tolist() == [0, 0]
        assert y.tolist() == [0, 1]
        assert z.tolist() == [0, 1]

    def test_torch_callable_branch(self, monkeypatch) -> None:
        pytest.importorskip("torch")
        import torch

        used: list[int] = []

        def fake_unravel(idx, shape):
            used.append(1)
            exp = np.unravel_index(idx.detach().cpu().numpy().astype(np.int64), shape)
            return tuple(torch.as_tensor(x, device=idx.device, dtype=torch.long) for x in exp)

        monkeypatch.setattr(torch, "unravel_index", fake_unravel, raising=False)
        idx = torch.tensor([5], dtype=torch.long)
        x, y, z = dt._unravel_flat_indices(idx, (2, 3, 4))
        assert used == [1]
        assert int(x[0]) == 0 and int(y[0]) == 1 and int(z[0]) == 1


@pytest.mark.unit
class TestAxisAngleRotmat:
    def test_zero_angle_is_identity(self) -> None:
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        angle = torch.tensor(0.0, dtype=torch.float32)
        r = dt.axis_angle_rotmat(axis, angle)
        assert r.shape == (3, 3)
        assert torch.allclose(r, torch.eye(3, dtype=r.dtype), atol=1e-6, rtol=1e-5)

    def test_z_axis_small_angle_orthogonal_rows(self) -> None:
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        angle = torch.tensor(0.3, dtype=torch.float64)
        r = dt.axis_angle_rotmat(axis, angle)
        assert torch.allclose(r @ r.T, torch.eye(3, dtype=r.dtype), atol=1e-5, rtol=1e-5)


@pytest.mark.unit
class TestTotalVariationLoss:
    def test_constant_field_zero(self) -> None:
        x = torch.ones(5, 5, 5, 3)
        assert dt.total_variation_loss(x).item() == pytest.approx(0.0, abs=1e-7)

    def test_random_field_positive(self) -> None:
        rng = torch.Generator().manual_seed(7)
        x = torch.randn(5, 5, 5, 3, generator=rng)
        v = dt.total_variation_loss(x).item()
        assert v > 0.0


@pytest.mark.unit
class TestInterpolatedPrediction:
    def test_uniform_volume_interior(self) -> None:
        vol = torch.ones(6, 6, 6, dtype=torch.float32)
        x = torch.tensor([2.5, 3.0])
        y = torch.tensor([2.5, 3.0])
        z = torch.tensor([2.5, 3.0])
        out = dt.interpolated_prediction(x, y, z, vol, trilinear_interp=True)
        assert out.shape == (2,)
        assert torch.allclose(out, torch.ones(2))

    def test_gaussian_weights_normalize(self) -> None:
        vol = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4)
        x = torch.tensor([1.25])
        y = torch.tensor([1.25])
        z = torch.tensor([1.25])
        out = dt.interpolated_prediction(x, y, z, vol, trilinear_interp=False, sig_proj=0.5)
        assert out.shape == (1,)
        assert torch.isfinite(out).all()


@pytest.mark.unit
class TestSmoothDisplacementField:
    def test_gaussian_zeros_stays_zero(self) -> None:
        u = np.zeros((8, 8, 8), dtype=np.float32)
        got = dt.smooth_displacement_field(u, "gaussian", sigma=1.0)
        assert got.shape == u.shape
        assert np.allclose(got, 0.0, atol=1e-6)

    def test_laplacian_runs(self) -> None:
        u = np.random.default_rng(1).standard_normal((5, 5, 5)).astype(np.float32)
        got = dt.smooth_displacement_field(u, "laplacian", sigma=2.0)
        assert got.shape == u.shape
        assert np.isfinite(got).all()

    def test_unknown_method_raises(self) -> None:
        u = np.zeros((4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown smoothing method"):
            dt.smooth_displacement_field(u, "not_a_method", sigma=1.0)


@pytest.mark.unit
class TestRemapDisplacementLagrangian:
    def test_zero_field_small_grid(self) -> None:
        nx, ny, nz = 6, 7, 8
        x = np.linspace(0.0, (nx - 1) * 1e-4, nx)
        y = np.linspace(0.0, (ny - 1) * 1e-4, ny)
        z = np.linspace(0.0, (nz - 1) * 1e-4, nz)
        z0 = np.zeros((nx, ny, nz), dtype=np.float64)
        rx, ry, rz = dt.remap_displacement_lagrangian_griddata(z0, z0, z0, x, y, z, method="linear")
        assert np.allclose(rx, 0.0, atol=1e-9)
        assert np.allclose(ry, 0.0, atol=1e-9)
        assert np.allclose(rz, 0.0, atol=1e-9)

    def test_uniform_translation_recovered(self) -> None:
        nx, ny, nz = 10, 10, 10
        d = 1e-5
        x = np.arange(nx, dtype=np.float64) * d
        y = np.arange(ny, dtype=np.float64) * d
        z = np.arange(nz, dtype=np.float64) * d
        ux = np.full((nx, ny, nz), 2e-6, dtype=np.float64)
        uy = np.full((nx, ny, nz), -1.5e-6, dtype=np.float64)
        uz = np.full((nx, ny, nz), 0.5e-6, dtype=np.float64)
        rx, ry, rz = dt.remap_displacement_lagrangian_griddata(ux, uy, uz, x, y, z, method="linear")
        assert np.allclose(rx, 2e-6, rtol=1e-5, atol=1e-8)
        assert np.allclose(ry, -1.5e-6, rtol=1e-5, atol=1e-8)
        assert np.allclose(rz, 0.5e-6, rtol=1e-5, atol=1e-8)

    def test_shape_mismatch_raises(self) -> None:
        u2 = np.zeros((2, 2, 2), dtype=np.float64)
        ax_ok = np.array([0.0, 1.0])
        ax_wrong = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="Axis lengths"):
            dt.remap_displacement_lagrangian_griddata(u2, u2, u2, ax_wrong, ax_ok, ax_ok, method="nearest")

    def test_u_components_shape_mismatch_raises(self) -> None:
        u2 = np.zeros((2, 2, 2), dtype=np.float64)
        u3 = np.zeros((3, 2, 2), dtype=np.float64)
        ax = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="Ux_m, Uy_m, Uz_m"):
            dt.remap_displacement_lagrangian_griddata(u2, u3, u2, ax, ax, ax, method="nearest")

    def test_remap_nearest_method_runs(self) -> None:
        nx, ny, nz = 4, 4, 4
        d = 1e-6
        x = np.arange(nx, dtype=np.float64) * d
        y = np.arange(ny, dtype=np.float64) * d
        z = np.arange(nz, dtype=np.float64) * d
        u = np.zeros((nx, ny, nz), dtype=np.float64)
        rx, ry, rz = dt.remap_displacement_lagrangian_griddata(u, u, u, x, y, z, method="nearest")
        assert rx.shape == (nx, ny, nz)
        assert np.allclose(rx, 0.0, atol=1e-12)

    def test_remap_warns_when_not_converged(self, capsys: pytest.CaptureFixture[str]) -> None:
        nx = ny = nz = 3
        d = 1e-6
        x = np.arange(nx, dtype=np.float64) * d
        y = np.arange(ny, dtype=np.float64) * d
        z = np.arange(nz, dtype=np.float64) * d
        u = np.zeros((nx, ny, nz), dtype=np.float64)
        dt.remap_displacement_lagrangian_griddata(u, u, u, x, y, z, method="linear", max_iter=0)
        err = capsys.readouterr().err
        assert "Warning: remap_to_reference fixed-point" in err

    def test_coarse_reference_axes_match_full_when_same_resolution(self) -> None:
        n = 7
        x = np.arange(n, dtype=np.float64) * 1e-6
        y = np.arange(n, dtype=np.float64) * 2e-6
        z = np.arange(n, dtype=np.float64) * 3e-6
        xc, yc, zc = dt.coarse_reference_axes_m(x, y, z, n, n, n)
        np.testing.assert_allclose(xc, x)
        np.testing.assert_allclose(yc, y)
        np.testing.assert_allclose(zc, z)

    def test_coarse_reference_axes_single_coarse_node(self) -> None:
        n = 5
        x = np.arange(n, dtype=np.float64) * 1e-6
        y = np.arange(n, dtype=np.float64) * 2e-6
        z = np.arange(n, dtype=np.float64) * 3e-6
        xc, yc, zc = dt.coarse_reference_axes_m(x, y, z, 1, n, n)
        assert xc.shape == (1,)
        assert xc[0] == pytest.approx(x[2])
        np.testing.assert_allclose(yc, y)
        np.testing.assert_allclose(zc, z)

    def test_coarse_reference_axes_invalid_n_raises(self) -> None:
        x = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="n_c and n_full must be positive"):
            dt.coarse_reference_axes_m(x, x, x, 0, 2, 2)


@pytest.mark.unit
class TestUnlinkIfExists:
    def test_removes_file(self, tmp_path: Path) -> None:
        p = tmp_path / "old.npy"
        p.write_bytes(b"x")
        dt._unlink_if_exists(p)
        assert not p.exists()

    def test_missing_path_no_op(self, tmp_path: Path) -> None:
        dt._unlink_if_exists(tmp_path / "nope.npy")

    def test_oserror_swallowed(self, tmp_path: Path) -> None:
        p = tmp_path / "blocked"
        p.write_bytes(b"x")

        def boom(*_a, **_k):
            raise OSError("no")

        with patch.object(Path, "is_file", return_value=True):
            with patch.object(Path, "unlink", boom):
                dt._unlink_if_exists(p)


@pytest.mark.unit
class TestWriteGridsAndVolume:
    def test_write_xyz_grids_m_shape_and_values(self, tmp_path: Path) -> None:
        nx, ny, nz = 3, 4, 5
        x_axis = np.arange(nx, dtype=np.float64)
        y_axis = np.arange(ny, dtype=np.float64)
        z_axis = np.arange(nz, dtype=np.float64)
        dt.write_xyz_grids_m(tmp_path, x_axis, y_axis, z_axis, shape=(nx, ny, nz), chunk_z=2)

        x_mm = np.load(tmp_path / "X.npy", mmap_mode="r")
        y_mm = np.load(tmp_path / "Y.npy", mmap_mode="r")
        z_mm = np.load(tmp_path / "Z.npy", mmap_mode="r")
        assert x_mm.shape == (nx, ny, nz)
        assert int(x_mm[2, 1, 3]) == 2
        assert int(y_mm[2, 3, 1]) == 3
        assert int(z_mm[1, 2, 4]) == 4

    def test_write_xyz_grids_m_length_mismatch_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Axis lengths"):
            dt.write_xyz_grids_m(
                tmp_path,
                np.zeros(2),
                np.zeros(2),
                np.zeros(2),
                shape=(3, 3, 3),
            )

    def test_write_volume_matrix_constant(self, tmp_path: Path) -> None:
        shape = (2, 3, 4)
        vox = 1.25e-12
        dt.write_volume_matrix_m3(tmp_path, shape=shape, voxel_volume_m3=vox)
        vm = np.load(tmp_path / "volume_matrix.npy")
        assert vm.shape == shape
        assert np.all(vm == vox)


@pytest.mark.unit
class TestLoadTiffZyxToXyz:
    @patch.object(dt.io, "imread")
    def test_transpose_and_downsample(self, mock_imread) -> None:
        z, y, x = 2, 2, 3
        mock_imread.return_value = np.arange(z * y * x, dtype=np.uint16).reshape(z, y, x)
        out = dt.load_tiff_zyx_to_xyz("fake.tif", z_start=0, z_end=None, downsamp_xy=1, downsamp_z=1)
        assert out.shape == (x, y, z)
        assert out[0, 0, 0] == 0
        assert out[2, 0, 0] == 2

    @patch.object(dt.io, "imread")
    def test_z_crop_and_stride(self, mock_imread) -> None:
        z, y, x = 4, 2, 3
        mock_imread.return_value = np.ones((z, y, x), dtype=np.float16)
        out = dt.load_tiff_zyx_to_xyz("fake.tif", z_start=1, z_end=3, downsamp_xy=2, downsamp_z=2)
        # z slice [1:3:2] -> one plane; y,x [::2] -> sizes 1,2
        assert out.shape == (2, 1, 1)


@pytest.mark.unit
class TestEstimateInitialShift:
    @patch.object(dt, "phase_cross_correlation")
    def test_combines_xy_and_yz(self, mock_pcc) -> None:
        mock_pcc.side_effect = [
            (np.array([1.5, -2.0]), None, None),
            (np.array([9.0, 3.25]), None, None),
        ]
        vol = np.ones((4, 4, 4), dtype=np.float32)
        xy, z_shift = dt.estimate_initial_shift(vol, vol)
        assert np.allclose(xy, np.array([1.5, -2.0]))
        assert z_shift == pytest.approx(3.25)


@pytest.mark.unit
class TestComparisonFigureHelpers:
    def test_corr_rmse_ssim_2d_varying(self) -> None:
        rng = np.random.default_rng(7)
        p2d = rng.random((12, 10))
        d2d = p2d + 0.05 * rng.standard_normal((12, 10))
        corr, rmse, ssim_val = dt._corr_rmse_ssim_2d(p2d, d2d)
        assert np.isfinite(corr)
        assert rmse > 0
        assert 0.0 <= ssim_val <= 1.0

    def test_corr_rmse_ssim_2d_constant_nan_corr(self) -> None:
        # SSIM default win_size is 7; need at least 7×7 patches.
        p2d = np.ones((10, 10), dtype=np.float64)
        d2d = np.full((10, 10), 2.0, dtype=np.float64)
        corr, rmse, ssim_val = dt._corr_rmse_ssim_2d(p2d, d2d)
        assert np.isnan(corr)
        assert rmse == pytest.approx(1.0)
        assert np.isfinite(ssim_val)

    def test_save_comparison_figure_writes_png(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib", reason="comparison figure requires matplotlib")
        rng = np.random.default_rng(0)
        nx, ny, nz = 14, 12, 10
        stack_with = (rng.random((nx, ny, nz)) * 200).astype(np.float32)
        stack_without = (rng.random((nx, ny, nz)) * 200).astype(np.float32)
        u0 = np.zeros((nx, ny, nz), dtype=np.float64)
        rot = np.eye(3, dtype=np.float64)
        shift = np.zeros(3, dtype=np.float64)
        out = dt._save_comparison_figure(
            tmp_path,
            stack_with,
            stack_without,
            u0,
            u0,
            u0,
            rot,
            shift,
            1.0,
            1.0,
        )
        assert out is not None
        assert out == tmp_path / "comparison_prediction_vs_data.png"
        assert out.is_file()
        assert out.stat().st_size > 1000

    def test_save_comparison_figure_returns_none_without_matplotlib(self, tmp_path: Path) -> None:
        stack = np.ones((4, 4, 4), dtype=np.float32)
        u0 = np.zeros((4, 4, 4), dtype=np.float64)
        rot = np.eye(3, dtype=np.float64)
        shift = np.zeros(3, dtype=np.float64)
        real_import = builtins.__import__

        def block_matplotlib(name: str, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("matplotlib blocked for test")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=block_matplotlib):
            out = dt._save_comparison_figure(tmp_path, stack, stack, u0, u0, u0, rot, shift, 1.0, 1.0)
        assert out is None


@pytest.mark.unit
class TestSaveMseCurvePng:
    def test_returns_none_without_matplotlib(self, tmp_path: Path) -> None:
        trace = np.array([1.0, 0.5], dtype=np.float64)
        real_import = builtins.__import__

        def block_matplotlib(name: str, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("matplotlib blocked for test")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=block_matplotlib):
            out = dt._save_mse_curve_png(tmp_path, trace)
        assert out is None

    def test_writes_png_when_matplotlib_available(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib", reason="mse curve requires matplotlib")
        trace = np.linspace(1.0, 0.1, 5, dtype=np.float64)
        out = dt._save_mse_curve_png(tmp_path, trace)
        assert out is not None
        assert out == tmp_path / "mse_curve.png"
        assert out.is_file()


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
            dt, "_save_comparison_figure", return_value=tmp_path / "comparison_prediction_vs_data.png"
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
            with patch.object(dt, "load_tiff_zyx_to_xyz", side_effect=fake_load):
                with pytest.raises(ValueError, match="Shape mismatch"):
                    dt.main()

    def test_main_empty_volume_raises(self, tmp_path: Path) -> None:
        pl = patch.object(dt, "load_tiff_zyx_to_xyz", return_value=np.empty((0, 0, 0), dtype=np.float32))
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
