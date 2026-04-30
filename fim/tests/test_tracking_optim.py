"""Unit tests for :mod:`fim.refactor.tracking_optim`.

Covers the torch-heavy forward-model pieces and the initial-shift estimator:

- ``_unravel_flat_indices``    — fallback when ``torch.unravel_index`` is unavailable.
- ``axis_angle_rotmat``        — Rodrigues rotation matrix.
- ``total_variation_loss``     — anisotropic TV regularizer on a vector field.
- ``interpolated_prediction``  — trilinear / Gaussian-projected sampling.
- ``estimate_initial_shift``   — combined XY + YZ phase-correlation seed.

Tests are skipped if PyTorch is not installed.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("torch", reason="tracking_optim requires PyTorch")
import torch  # noqa: E402

from fim.refactor import tracking_optim as topt  # noqa: E402


@pytest.mark.unit
class TestUnravelFlatIndices:
    def test_numpy_path_when_torch_api_absent(self, monkeypatch) -> None:
        monkeypatch.setattr(torch, "unravel_index", None, raising=False)
        idx = torch.tensor([0, 5], dtype=torch.long)
        x, y, z = topt._unravel_flat_indices(idx, (2, 3, 4))
        assert x.tolist() == [0, 0]
        assert y.tolist() == [0, 1]
        assert z.tolist() == [0, 1]

    def test_torch_callable_branch(self, monkeypatch) -> None:
        used: list[int] = []

        def fake_unravel(idx, shape):
            used.append(1)
            exp = np.unravel_index(idx.detach().cpu().numpy().astype(np.int64), shape)
            return tuple(torch.as_tensor(x, device=idx.device, dtype=torch.long) for x in exp)

        monkeypatch.setattr(torch, "unravel_index", fake_unravel, raising=False)
        idx = torch.tensor([5], dtype=torch.long)
        x, y, z = topt._unravel_flat_indices(idx, (2, 3, 4))
        assert used == [1]
        assert int(x[0]) == 0 and int(y[0]) == 1 and int(z[0]) == 1


@pytest.mark.unit
class TestAxisAngleRotmat:
    def test_zero_angle_is_identity(self) -> None:
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        angle = torch.tensor(0.0, dtype=torch.float32)
        r = topt.axis_angle_rotmat(axis, angle)
        assert r.shape == (3, 3)
        assert torch.allclose(r, torch.eye(3, dtype=r.dtype), atol=1e-6, rtol=1e-5)

    def test_z_axis_small_angle_orthogonal_rows(self) -> None:
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        angle = torch.tensor(0.3, dtype=torch.float64)
        r = topt.axis_angle_rotmat(axis, angle)
        assert torch.allclose(r @ r.T, torch.eye(3, dtype=r.dtype), atol=1e-5, rtol=1e-5)


@pytest.mark.unit
class TestTotalVariationLoss:
    def test_constant_field_zero(self) -> None:
        x = torch.ones(5, 5, 5, 3)
        assert topt.total_variation_loss(x).item() == pytest.approx(0.0, abs=1e-7)

    def test_random_field_positive(self) -> None:
        rng = torch.Generator().manual_seed(7)
        x = torch.randn(5, 5, 5, 3, generator=rng)
        v = topt.total_variation_loss(x).item()
        assert v > 0.0


@pytest.mark.unit
class TestInterpolatedPrediction:
    def test_uniform_volume_interior(self) -> None:
        vol = torch.ones(6, 6, 6, dtype=torch.float32)
        x = torch.tensor([2.5, 3.0])
        y = torch.tensor([2.5, 3.0])
        z = torch.tensor([2.5, 3.0])
        out = topt.interpolated_prediction(x, y, z, vol, trilinear_interp=True)
        assert out.shape == (2,)
        assert torch.allclose(out, torch.ones(2))

    def test_gaussian_weights_normalize(self) -> None:
        vol = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4)
        x = torch.tensor([1.25])
        y = torch.tensor([1.25])
        z = torch.tensor([1.25])
        out = topt.interpolated_prediction(x, y, z, vol, trilinear_interp=False, sig_proj=0.5)
        assert out.shape == (1,)
        assert torch.isfinite(out).all()


@pytest.mark.unit
class TestEstimateInitialShift:
    @patch.object(topt, "phase_cross_correlation")
    def test_combines_xy_and_yz(self, mock_pcc) -> None:
        mock_pcc.side_effect = [
            (np.array([1.5, -2.0]), None, None),
            (np.array([9.0, 3.25]), None, None),
        ]
        vol = np.ones((4, 4, 4), dtype=np.float32)
        xy, z_shift = topt.estimate_initial_shift(vol, vol)
        assert np.allclose(xy, np.array([1.5, -2.0]))
        assert z_shift == pytest.approx(3.25)
