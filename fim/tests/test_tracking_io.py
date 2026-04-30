"""Unit tests for :mod:`fim.refactor.tracking_io`.

Covers the disk and figure I/O helpers used by ``deformation_tracking.main``:

- ``_display_input_name``     — friendlier name for UUID-prefixed uploads.
- ``_unlink_if_exists``       — defensive ``Path.unlink`` wrapper.
- ``write_xyz_grids_m``       — coordinate grid serialization to ``X.npy``/``Y.npy``/``Z.npy``.
- ``write_volume_matrix_m3``  — voxel-volume scalar broadcast to ``volume_matrix.npy``.
- ``load_tiff_zyx_to_xyz``    — TIFF loader + crop/downsample + (X,Y,Z) transpose.
- ``_corr_rmse_ssim_2d``      — per-slice correlation/RMSE/SSIM diagnostic.
- ``_save_comparison_figure`` — prediction-vs-data PNG (matplotlib optional).
- ``_save_mse_curve_png``     — MSE-trace PNG (matplotlib optional).
"""

from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fim.refactor import tracking_io as tio


@pytest.mark.unit
class TestDisplayInputName:
    def test_plain_filename_no_underscore(self) -> None:
        assert tio._display_input_name("plain.tif") == "plain.tif"

    def test_uuid_prefixed_upload_name_is_cleaned(self) -> None:
        path = "/tmp/fim_uploads/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_def_image.tif"
        assert tio._display_input_name(path) == "def_image.tif"

    def test_non_uuid_prefix_kept(self) -> None:
        path = "/tmp/fim_uploads/notauuid_def_image.tif"
        assert tio._display_input_name(path) == "notauuid_def_image.tif"


@pytest.mark.unit
class TestUnlinkIfExists:
    def test_removes_file(self, tmp_path: Path) -> None:
        p = tmp_path / "old.npy"
        p.write_bytes(b"x")
        tio._unlink_if_exists(p)
        assert not p.exists()

    def test_missing_path_no_op(self, tmp_path: Path) -> None:
        tio._unlink_if_exists(tmp_path / "nope.npy")

    def test_oserror_swallowed(self, tmp_path: Path) -> None:
        p = tmp_path / "blocked"
        p.write_bytes(b"x")

        def boom(*_a, **_k):
            raise OSError("no")

        with patch.object(Path, "is_file", return_value=True):
            with patch.object(Path, "unlink", boom):
                tio._unlink_if_exists(p)


@pytest.mark.unit
class TestWriteGridsAndVolume:
    def test_write_xyz_grids_m_shape_and_values(self, tmp_path: Path) -> None:
        nx, ny, nz = 3, 4, 5
        x_axis = np.arange(nx, dtype=np.float64)
        y_axis = np.arange(ny, dtype=np.float64)
        z_axis = np.arange(nz, dtype=np.float64)
        tio.write_xyz_grids_m(tmp_path, x_axis, y_axis, z_axis, shape=(nx, ny, nz), chunk_z=2)

        x_mm = np.load(tmp_path / "X.npy", mmap_mode="r")
        y_mm = np.load(tmp_path / "Y.npy", mmap_mode="r")
        z_mm = np.load(tmp_path / "Z.npy", mmap_mode="r")
        assert x_mm.shape == (nx, ny, nz)
        assert int(x_mm[2, 1, 3]) == 2
        assert int(y_mm[2, 3, 1]) == 3
        assert int(z_mm[1, 2, 4]) == 4

    def test_write_xyz_grids_m_length_mismatch_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Axis lengths"):
            tio.write_xyz_grids_m(
                tmp_path,
                np.zeros(2),
                np.zeros(2),
                np.zeros(2),
                shape=(3, 3, 3),
            )

    def test_write_volume_matrix_constant(self, tmp_path: Path) -> None:
        shape = (2, 3, 4)
        vox = 1.25e-12
        tio.write_volume_matrix_m3(tmp_path, shape=shape, voxel_volume_m3=vox)
        vm = np.load(tmp_path / "volume_matrix.npy")
        assert vm.shape == shape
        assert np.all(vm == vox)


@pytest.mark.unit
class TestLoadTiffZyxToXyz:
    @patch.object(tio.io, "imread")
    def test_transpose_and_downsample(self, mock_imread) -> None:
        z, y, x = 2, 2, 3
        mock_imread.return_value = np.arange(z * y * x, dtype=np.uint16).reshape(z, y, x)
        out = tio.load_tiff_zyx_to_xyz("fake.tif", z_start=0, z_end=None, downsamp_xy=1, downsamp_z=1)
        assert out.shape == (x, y, z)
        assert out[0, 0, 0] == 0
        assert out[2, 0, 0] == 2

    @patch.object(tio.io, "imread")
    def test_z_crop_and_stride(self, mock_imread) -> None:
        z, y, x = 4, 2, 3
        mock_imread.return_value = np.ones((z, y, x), dtype=np.float16)
        out = tio.load_tiff_zyx_to_xyz("fake.tif", z_start=1, z_end=3, downsamp_xy=2, downsamp_z=2)
        # z slice [1:3:2] -> one plane; y,x [::2] -> sizes 1,2
        assert out.shape == (2, 1, 1)


@pytest.mark.unit
class TestComparisonFigureHelpers:
    def test_corr_rmse_ssim_2d_varying(self) -> None:
        rng = np.random.default_rng(7)
        p2d = rng.random((12, 10))
        d2d = p2d + 0.05 * rng.standard_normal((12, 10))
        corr, rmse, ssim_val = tio._corr_rmse_ssim_2d(p2d, d2d)
        assert np.isfinite(corr)
        assert rmse > 0
        assert 0.0 <= ssim_val <= 1.0

    def test_corr_rmse_ssim_2d_constant_nan_corr(self) -> None:
        # SSIM default win_size is 7; need at least 7×7 patches.
        p2d = np.ones((10, 10), dtype=np.float64)
        d2d = np.full((10, 10), 2.0, dtype=np.float64)
        corr, rmse, ssim_val = tio._corr_rmse_ssim_2d(p2d, d2d)
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
        out = tio._save_comparison_figure(
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
            out = tio._save_comparison_figure(tmp_path, stack, stack, u0, u0, u0, rot, shift, 1.0, 1.0)
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
            out = tio._save_mse_curve_png(tmp_path, trace)
        assert out is None

    def test_writes_png_when_matplotlib_available(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib", reason="mse curve requires matplotlib")
        trace = np.linspace(1.0, 0.1, 5, dtype=np.float64)
        out = tio._save_mse_curve_png(tmp_path, trace)
        assert out is not None
        assert out == tmp_path / "mse_curve.png"
        assert out.is_file()
