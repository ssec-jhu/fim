"""Optimization math for deformation tracking (torch-heavy, GPU-friendly).

Functions here are pure (no file I/O) and operate on ``torch.Tensor`` /
``numpy.ndarray`` inputs. They are the building blocks of the forward model
and initial-alignment estimate used by ``main()``.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.registration import phase_cross_correlation


def _unravel_flat_indices(
    indices: torch.Tensor,
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map flat linear indices to per-axis voxel indices for batch sampling.

    Parameters
    ----------
    indices : torch.Tensor
        Flat indices ``(N,)``, typically ``torch.long`` on *device*.
    shape : tuple[int, ...]
        ``(nx, ny, nz)`` volume shape.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(ix, iy, iz)`` each ``(N,)`` ``torch.long`` on the same device as *indices*.

    Notes
    -----
    Uses ``torch.unravel_index`` when available (PyTorch 2.0+); otherwise NumPy on CPU.
    """
    fn = getattr(torch, "unravel_index", None)  # Newer PyTorch has this; older versions take the NumPy path below.
    if fn is not None:
        return fn(indices, shape)
    unr = np.unravel_index(
        indices.detach().cpu().numpy().astype(np.int64), shape
    )  # Same math on CPU: one array per axis (i, j, k).
    return tuple(
        torch.as_tensor(x, device=indices.device, dtype=torch.long) for x in unr
    )  # x runs over axes; move each back to the GPU/CPU tensor device.


def axis_angle_rotmat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrix from axis-angle via Rodrigues (axis normalized internally).

    Parameters
    ----------
    axis : torch.Tensor
        Shape ``(3,)``, rotation axis.
    angle : torch.Tensor
        Scalar angle in radians.

    Returns
    -------
    torch.Tensor
        ``(3, 3)`` rotation ``R``. With sample rows ``(1×3)``, ``forward_model`` uses ``v @ R``.
    """
    axis_unit = torch.nn.functional.normalize(axis, dim=0)  # Turn the learned axis into a unit vector (length 1).
    cos = torch.cos(angle)  # Cosine of the rotation angle.
    sin = torch.sin(angle)  # Sine of the rotation angle.
    ux, uy, uz = axis_unit[0], axis_unit[1], axis_unit[2]  # x,y,z parts of that unit axis.

    r00 = cos + ux**2 * (1 - cos)  # Rodrigues formula: entry that maps old X into new X.
    r01 = ux * uy * (1 - cos) - uz * sin  # Entry that maps old Y into new X.
    r02 = ux * uz * (1 - cos) + uy * sin  # Entry that maps old Z into new X.
    r10 = ux * uy * (1 - cos) + uz * sin  # Entry that maps old X into new Y.
    r11 = cos + uy**2 * (1 - cos)  # Entry that maps old Y into new Y.
    r12 = uy * uz * (1 - cos) - ux * sin  # Entry that maps old Z into new Y.
    r20 = ux * uz * (1 - cos) - uy * sin  # Entry that maps old X into new Z.
    r21 = uy * uz * (1 - cos) + ux * sin  # Entry that maps old Y into new Z.
    r22 = cos + uz**2 * (1 - cos)  # Entry that maps old Z into new Z.

    return torch.stack([torch.stack([r00, r01, r02]), torch.stack([r10, r11, r12]), torch.stack([r20, r21, r22])])


def interpolated_prediction(
    x_float: torch.Tensor,
    y_float: torch.Tensor,
    z_float: torch.Tensor,
    volume: torch.Tensor,
    trilinear_interp: bool = True,
    sig_proj: float = 0.42465,
) -> torch.Tensor:
    """Sample *volume* at fractional voxel indices (trilinear or Gaussian weights).

    Coordinates are clamped to ``[0, nx-2]`` etc. so each voxel always has a full
    2×2×2 neighborhood for interpolation.

    Parameters
    ----------
    x_float, y_float, z_float : torch.Tensor
        Fractional indices along ``volume``'s first three dims (broadcastable).
    volume : torch.Tensor
        Shape ``(nx, ny, nz, ...)``; higher dims interpolated per-voxel like the first three.
    trilinear_interp : bool
        If True, linear weights; if False, Gaussian weights + normalization.
    sig_proj : float
        Gaussian width (voxel units) when ``trilinear_interp`` is False.

    Returns
    -------
    torch.Tensor
        Interpolated values; leading dims follow broadcast rules.
    """
    x_float = torch.clamp(
        x_float, min=0, max=volume.shape[0] - 2
    )  # Stay inside so we always have a full 2×2×2 box of neighbors in X.
    y_float = torch.clamp(y_float, min=0, max=volume.shape[1] - 2)  # Same in Y.
    z_float = torch.clamp(z_float, min=0, max=volume.shape[2] - 2)  # Same in Z.

    x_floor = torch.floor(x_float)  # Integer voxel index below the sample point in X.
    y_floor = torch.floor(y_float)  # Integer index below in Y.
    z_floor = torch.floor(z_float)  # Integer index below in Z.
    x_ceil = x_floor + 1  # Next voxel index up in X.
    y_ceil = y_floor + 1  # Next index up in Y.
    z_ceil = z_floor + 1  # Next index up in Z.

    fx = x_float - x_floor  # How far we are from the lower X face (0 = on the face, 1 = at the upper face).
    fy = y_float - y_floor  # Same idea in Y.
    fz = z_float - z_floor  # Same idea in Z.
    cx = x_ceil - x_float  # Distance to the upper X face (pairs with fx for linear blending).
    cy = y_ceil - y_float  # Distance to upper Y face.
    cz = z_ceil - z_float  # Distance to upper Z face.

    x_floor = x_floor.to(torch.int32)  # Whole-number indices so we can index the volume array.
    y_floor = y_floor.to(torch.int32)
    z_floor = z_floor.to(torch.int32)
    x_ceil = x_ceil.to(torch.int32)
    y_ceil = y_ceil.to(torch.int32)
    z_ceil = z_ceil.to(torch.int32)

    if trilinear_interp:
        # Trick so the eight corner weights line up with standard trilinear interpolation (swap "near/far" weights).
        fx, fy, fz, cx, cy, cz = cx, cy, cz, fx, fy, fz
    else:
        fx = torch.exp(-(fx**2) / (2 * sig_proj**2))
        fy = torch.exp(-(fy**2) / (2 * sig_proj**2))
        fz = torch.exp(-(fz**2) / (2 * sig_proj**2))
        cx = torch.exp(-(cx**2) / (2 * sig_proj**2))
        cy = torch.exp(-(cy**2) / (2 * sig_proj**2))
        cz = torch.exp(-(cz**2) / (2 * sig_proj**2))

    f1 = fx * fy * fz  # Weight for the voxel at (floor X, floor Y, floor Z).
    f2 = fx * cy * fz  # Weight for (floor X, ceil Y, floor Z).
    f3 = cx * fy * fz  # Weight for (ceil X, floor Y, floor Z).
    f4 = cx * cy * fz  # Weight for (ceil X, ceil Y, floor Z).
    f5 = fx * fy * cz  # Weight for (floor X, floor Y, ceil Z).
    f6 = fx * cy * cz  # Weight for (floor X, ceil Y, ceil Z).
    f7 = cx * fy * cz  # Weight for (ceil X, floor Y, ceil Z).
    f8 = cx * cy * cz  # Weight for (ceil X, ceil Y, ceil Z).

    fff = volume[x_floor, y_floor, z_floor]  # Brightness at that first corner voxel.
    fcf = volume[x_floor, y_ceil, z_floor]  # Brightness at second corner.
    cff = volume[x_ceil, y_floor, z_floor]  # Third corner.
    ccf = volume[x_ceil, y_ceil, z_floor]  # Fourth corner.
    ffc = volume[x_floor, y_floor, z_ceil]  # Fifth corner.
    fcc = volume[x_floor, y_ceil, z_ceil]  # Sixth corner.
    cfc = volume[x_ceil, y_floor, z_ceil]  # Seventh corner.
    ccc = volume[x_ceil, y_ceil, z_ceil]  # Eighth corner.

    forward = (  # Add up: each corner value times its weight; extra dimensions of volume broadcast here.
        ccc * f8[(...,) + (None,) * (volume.ndim - 3)]
        + ccf * f4[(...,) + (None,) * (volume.ndim - 3)]
        + cff * f3[(...,) + (None,) * (volume.ndim - 3)]
        + cfc * f7[(...,) + (None,) * (volume.ndim - 3)]
        + fcc * f6[(...,) + (None,) * (volume.ndim - 3)]
        + fcf * f2[(...,) + (None,) * (volume.ndim - 3)]
        + fff * f1[(...,) + (None,) * (volume.ndim - 3)]
        + ffc * f5[(...,) + (None,) * (volume.ndim - 3)]
    )

    if not trilinear_interp:
        forward = forward / (f8 + f4 + f3 + f7 + f6 + f2 + f1 + f5)  # Gaussian mode: divide so weights sum to 1.

    return forward


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """Mean squared finite differences of a 3D vector field (TV² regularizer).

    Parameters
    ----------
    x : torch.Tensor
        ``(nx, ny, nz, 3)``, e.g. coarse ``r_deform_um``.

    Returns
    -------
    torch.Tensor
        Scalar: mean of squared backward differences along x, y, z.
    """
    dx = x[1:, :-1, :-1, :] - x[:-1, :-1, :-1, :]  # How much the field changes between neighbors along X.
    dy = x[:-1, 1:, :-1, :] - x[:-1, :-1, :-1, :]  # Same along Y.
    dz = x[:-1, :-1, 1:, :] - x[:-1, :-1, :-1, :]  # Same along Z.
    return (dx.pow(2) + dy.pow(2) + dz.pow(2)).mean()  # One number: average squared "roughness" of the field.


def estimate_initial_shift(
    stack_with_sphere_xyz: np.ndarray, stack_without_sphere_xyz: np.ndarray
) -> tuple[np.ndarray, float]:
    """Initial global shift in **pixels** from phase cross-correlation on 2D projections.

    XY shift from max over Z (XY image); Z shift from second component of shift on
    max-over-X projections (shape ``(Y,Z)``).

    Returns
    -------
    xy_shift : numpy.ndarray
        ``(2,)`` float64, ``(shift_x, shift_y)`` in pixels.
    z_shift : float
        Z shift in pixels.
    """
    max_with = stack_with_sphere_xyz.max(2)  # Squash Z: one 2D "top-down" picture of the deformed volume.
    max_without = stack_without_sphere_xyz.max(2)  # Same for the reference volume.
    xy_shift, _, _ = phase_cross_correlation(
        max_with, max_without, upsample_factor=32
    )  # How many pixels to shift in X and Y to line those 2D images up.

    yz_with = stack_with_sphere_xyz.max(0)  # Squash X instead: a (Y,Z) picture from the side.
    yz_without = stack_without_sphere_xyz.max(0)  # Side view of the reference.
    yz_shift, _, _ = phase_cross_correlation(
        yz_with, yz_without, upsample_factor=32
    )  # Shift in the side view (second number is along Z).
    z_shift = float(yz_shift[1])  # Use the Z component of that side-view shift as initial Z offset (pixels).
    return np.array(xy_shift, dtype=np.float64), z_shift
