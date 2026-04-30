"""Unit tests for :mod:`fim.refactor.tracking_remap`.

Covers post-processing of displacement fields:

- ``smooth_displacement_field``            — Gaussian / Laplacian smoothing dispatch.
- ``coarse_reference_axes_m``              — coarsening of (x,y,z) reference axes.
- ``remap_displacement_lagrangian_griddata`` — fixed-point inverse map onto the reference grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from fim.refactor import tracking_remap as trem


@pytest.mark.unit
class TestSmoothDisplacementField:
    def test_gaussian_zeros_stays_zero(self) -> None:
        u = np.zeros((8, 8, 8), dtype=np.float32)
        got = trem.smooth_displacement_field(u, "gaussian", sigma=1.0)
        assert got.shape == u.shape
        assert np.allclose(got, 0.0, atol=1e-6)

    def test_laplacian_runs(self) -> None:
        u = np.random.default_rng(1).standard_normal((5, 5, 5)).astype(np.float32)
        got = trem.smooth_displacement_field(u, "laplacian", sigma=2.0)
        assert got.shape == u.shape
        assert np.isfinite(got).all()

    def test_unknown_method_raises(self) -> None:
        u = np.zeros((4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown smoothing method"):
            trem.smooth_displacement_field(u, "not_a_method", sigma=1.0)


@pytest.mark.unit
class TestRemapDisplacementLagrangian:
    def test_zero_field_small_grid(self) -> None:
        nx, ny, nz = 6, 7, 8
        x = np.linspace(0.0, (nx - 1) * 1e-4, nx)
        y = np.linspace(0.0, (ny - 1) * 1e-4, ny)
        z = np.linspace(0.0, (nz - 1) * 1e-4, nz)
        z0 = np.zeros((nx, ny, nz), dtype=np.float64)
        rx, ry, rz = trem.remap_displacement_lagrangian_griddata(z0, z0, z0, x, y, z, method="linear")
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
        rx, ry, rz = trem.remap_displacement_lagrangian_griddata(ux, uy, uz, x, y, z, method="linear")
        assert np.allclose(rx, 2e-6, rtol=1e-5, atol=1e-8)
        assert np.allclose(ry, -1.5e-6, rtol=1e-5, atol=1e-8)
        assert np.allclose(rz, 0.5e-6, rtol=1e-5, atol=1e-8)

    def test_shape_mismatch_raises(self) -> None:
        u2 = np.zeros((2, 2, 2), dtype=np.float64)
        ax_ok = np.array([0.0, 1.0])
        ax_wrong = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="Axis lengths"):
            trem.remap_displacement_lagrangian_griddata(u2, u2, u2, ax_wrong, ax_ok, ax_ok, method="nearest")

    def test_u_components_shape_mismatch_raises(self) -> None:
        u2 = np.zeros((2, 2, 2), dtype=np.float64)
        u3 = np.zeros((3, 2, 2), dtype=np.float64)
        ax = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="Ux_m, Uy_m, Uz_m"):
            trem.remap_displacement_lagrangian_griddata(u2, u3, u2, ax, ax, ax, method="nearest")

    def test_remap_nearest_method_runs(self) -> None:
        nx, ny, nz = 4, 4, 4
        d = 1e-6
        x = np.arange(nx, dtype=np.float64) * d
        y = np.arange(ny, dtype=np.float64) * d
        z = np.arange(nz, dtype=np.float64) * d
        u = np.zeros((nx, ny, nz), dtype=np.float64)
        rx, ry, rz = trem.remap_displacement_lagrangian_griddata(u, u, u, x, y, z, method="nearest")
        assert rx.shape == (nx, ny, nz)
        assert np.allclose(rx, 0.0, atol=1e-12)

    def test_remap_warns_when_not_converged(self, capsys: pytest.CaptureFixture[str]) -> None:
        nx = ny = nz = 3
        d = 1e-6
        x = np.arange(nx, dtype=np.float64) * d
        y = np.arange(ny, dtype=np.float64) * d
        z = np.arange(nz, dtype=np.float64) * d
        u = np.zeros((nx, ny, nz), dtype=np.float64)
        trem.remap_displacement_lagrangian_griddata(u, u, u, x, y, z, method="linear", max_iter=0)
        err = capsys.readouterr().err
        assert "Warning: remap_to_reference fixed-point" in err

    def test_coarse_reference_axes_match_full_when_same_resolution(self) -> None:
        n = 7
        x = np.arange(n, dtype=np.float64) * 1e-6
        y = np.arange(n, dtype=np.float64) * 2e-6
        z = np.arange(n, dtype=np.float64) * 3e-6
        xc, yc, zc = trem.coarse_reference_axes_m(x, y, z, n, n, n)
        np.testing.assert_allclose(xc, x)
        np.testing.assert_allclose(yc, y)
        np.testing.assert_allclose(zc, z)

    def test_coarse_reference_axes_single_coarse_node(self) -> None:
        n = 5
        x = np.arange(n, dtype=np.float64) * 1e-6
        y = np.arange(n, dtype=np.float64) * 2e-6
        z = np.arange(n, dtype=np.float64) * 3e-6
        xc, yc, zc = trem.coarse_reference_axes_m(x, y, z, 1, n, n)
        assert xc.shape == (1,)
        assert xc[0] == pytest.approx(x[2])
        np.testing.assert_allclose(yc, y)
        np.testing.assert_allclose(zc, z)

    def test_coarse_reference_axes_invalid_n_raises(self) -> None:
        x = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="n_c and n_full must be positive"):
            trem.coarse_reference_axes_m(x, x, x, 0, 2, 2)
