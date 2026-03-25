"""
Deformation Tracking (Optimization-based)
========================================

This script estimates a 3D displacement field u(X) that maps *reference* positions to
*deformed* positions using gradient-based optimization on volumetric image data.

Convention:  u = x_deformed - X_reference   (standard continuum mechanics)
             F = I + grad(u)                (deformation gradient used by VFM)

Internally the optimizer solves a *backward warp* (deformed → reference) to find the
displacement that best aligns the deformed (with-sphere) volume to the reference
(no-sphere) volume.  The output is then negated so that Ux, Uy, Uz represent the
standard reference → deformed displacement.

Design goals:
- Minimal + clear: only core deformation prediction (no visualization / metrics / notebooks code)
- Fixed axis convention: assume TIFF stacks load as (Z, Y, X), convert to (X, Y, Z)
- CLI-first: paths can be passed via CLI; if omitted, defaults are provided

Outputs (written to --out_dir):
- Ux.npy, Uy.npy, Uz.npy           displacement components in meters, shape (X,Y,Z)
                                     convention: u = x_deformed - X_reference
- X.npy, Y.npy, Z.npy              3D coordinate grids in meters, shape (X,Y,Z)
- volume_matrix.npy                per-voxel volume weights in m^3, shape (X,Y,Z)

Notes:
- Reruns always replace outputs: existing U*.npy / grid .npy files in the output folder are
  removed before writing. With --skip_grids, old X/Y/Z/volume_matrix files are deleted so
  they cannot mismatch the new displacement fields.
- dxy_um / dz_um are voxel spacings in microns.
- Computation internally uses *centered* coordinates for stable rotation estimation,
  but outputs use coordinates starting at 0 (compatible with VFM code that centers later).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch
from numpy.lib.format import open_memmap
from skimage import io
from skimage.registration import phase_cross_correlation
from tqdm.auto import tqdm

# ----------------------------
# Defaults: small simulated stacks under fim/test_data/simulate/ (override via CLI).
# ----------------------------
_SIM_TIFF_DIR = Path(__file__).resolve().parent.parent / "test_data" / "simulate"
DEFAULT_WITHOUT_SPHERE = str(_SIM_TIFF_DIR / "ref_image.tif")
DEFAULT_WITH_SPHERE = str(_SIM_TIFF_DIR / "def_image.tif")


def _unravel_flat_indices(
    indices: torch.Tensor,
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map flat voxel indices to coordinate tensors (same batch shape as *indices*).

    ``torch.unravel_index`` exists from PyTorch 2.0+; older installs fall back to NumPy.
    """
    fn = getattr(torch, "unravel_index", None)
    if fn is not None:
        return fn(indices, shape)
    unr = np.unravel_index(indices.detach().cpu().numpy().astype(np.int64), shape)
    return tuple(torch.as_tensor(x, device=indices.device, dtype=torch.long) for x in unr)


def axis_angle_rotmat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues' rotation formula for axis-angle to rotation matrix."""
    axis_unit = torch.nn.functional.normalize(axis, dim=0)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    ux, uy, uz = axis_unit[0], axis_unit[1], axis_unit[2]

    r00 = cos + ux**2 * (1 - cos)
    r01 = ux * uy * (1 - cos) - uz * sin
    r02 = ux * uz * (1 - cos) + uy * sin
    r10 = ux * uy * (1 - cos) + uz * sin
    r11 = cos + uy**2 * (1 - cos)
    r12 = uy * uz * (1 - cos) - ux * sin
    r20 = ux * uz * (1 - cos) - uy * sin
    r21 = uy * uz * (1 - cos) + ux * sin
    r22 = cos + uz**2 * (1 - cos)

    return torch.stack([torch.stack([r00, r01, r02]), torch.stack([r10, r11, r12]), torch.stack([r20, r21, r22])])


def interpolated_prediction(
    x_float: torch.Tensor,
    y_float: torch.Tensor,
    z_float: torch.Tensor,
    volume: torch.Tensor,
    trilinear_interp: bool = True,
    sig_proj: float = 0.42465,
) -> torch.Tensor:
    """Trilinear (or Gaussian-weighted) interpolation for sub-pixel 3D sampling."""
    x_float = torch.clamp(x_float, min=0, max=volume.shape[0] - 2)
    y_float = torch.clamp(y_float, min=0, max=volume.shape[1] - 2)
    z_float = torch.clamp(z_float, min=0, max=volume.shape[2] - 2)

    x_floor = torch.floor(x_float)
    y_floor = torch.floor(y_float)
    z_floor = torch.floor(z_float)
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1
    z_ceil = z_floor + 1

    fx = x_float - x_floor
    fy = y_float - y_floor
    fz = z_float - z_floor
    cx = x_ceil - x_float
    cy = y_ceil - y_float
    cz = z_ceil - z_float

    x_floor = x_floor.to(torch.int32)
    y_floor = y_floor.to(torch.int32)
    z_floor = z_floor.to(torch.int32)
    x_ceil = x_ceil.to(torch.int32)
    y_ceil = y_ceil.to(torch.int32)
    z_ceil = z_ceil.to(torch.int32)

    if trilinear_interp:
        fx, fy, fz, cx, cy, cz = cx, cy, cz, fx, fy, fz
    else:
        fx = torch.exp(-(fx**2) / (2 * sig_proj**2))
        fy = torch.exp(-(fy**2) / (2 * sig_proj**2))
        fz = torch.exp(-(fz**2) / (2 * sig_proj**2))
        cx = torch.exp(-(cx**2) / (2 * sig_proj**2))
        cy = torch.exp(-(cy**2) / (2 * sig_proj**2))
        cz = torch.exp(-(cz**2) / (2 * sig_proj**2))

    f1 = fx * fy * fz
    f2 = fx * cy * fz
    f3 = cx * fy * fz
    f4 = cx * cy * fz
    f5 = fx * fy * cz
    f6 = fx * cy * cz
    f7 = cx * fy * cz
    f8 = cx * cy * cz

    fff = volume[x_floor, y_floor, z_floor]
    fcf = volume[x_floor, y_ceil, z_floor]
    cff = volume[x_ceil, y_floor, z_floor]
    ccf = volume[x_ceil, y_ceil, z_floor]
    ffc = volume[x_floor, y_floor, z_ceil]
    fcc = volume[x_floor, y_ceil, z_ceil]
    cfc = volume[x_ceil, y_floor, z_ceil]
    ccc = volume[x_ceil, y_ceil, z_ceil]

    forward = (
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
        forward = forward / (f8 + f4 + f3 + f7 + f6 + f2 + f1 + f5)

    return forward


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """TV2 loss for tensor of shape (nx, ny, nz, 3)."""
    dx = x[1:, :-1, :-1, :] - x[:-1, :-1, :-1, :]
    dy = x[:-1, 1:, :-1, :] - x[:-1, :-1, :-1, :]
    dz = x[:-1, :-1, 1:, :] - x[:-1, :-1, :-1, :]
    return (dx.pow(2) + dy.pow(2) + dz.pow(2)).mean()


def load_tiff_zyx_to_xyz(path: str, z_start: int, z_end: int | None, downsamp_xy: int, downsamp_z: int) -> np.ndarray:
    """
    Load TIFF assumed to be (Z, Y, X), apply crop/downsample, then convert to (X, Y, Z).
    """
    vol_zyx = io.imread(path)
    vol_zyx = vol_zyx[z_start:z_end:downsamp_z, ::downsamp_xy, ::downsamp_xy]
    # (Z,Y,X) -> (X,Y,Z)
    return vol_zyx.transpose(2, 1, 0)


def estimate_initial_shift(
    stack_with_sphere_xyz: np.ndarray, stack_without_sphere_xyz: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Estimate initial XY and Z shifts (in pixels) using phase cross correlation on projections.

    Returns:
      xy_shift: np.ndarray shape (2,) as (shift_x, shift_y) in pixels
      z_shift: float shift_z in pixels
    """
    max_with = stack_with_sphere_xyz.max(2)
    max_without = stack_without_sphere_xyz.max(2)
    xy_shift, _, _ = phase_cross_correlation(max_with, max_without, upsample_factor=32)

    yz_with = stack_with_sphere_xyz.max(0)  # (Y,Z)
    yz_without = stack_without_sphere_xyz.max(0)
    yz_shift, _, _ = phase_cross_correlation(yz_with, yz_without, upsample_factor=32)
    z_shift = float(yz_shift[1])
    return np.array(xy_shift, dtype=np.float64), z_shift


def _unlink_if_exists(path: Path) -> None:
    """Remove a file so the next write always replaces it (handles shape/dtype changes)."""
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def write_xyz_grids_m(
    out_dir: Path,
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    z_axis_m: np.ndarray,
    shape: tuple[int, int, int],
    dtype: np.dtype = np.float32,
    chunk_z: int = 8,
) -> None:
    """Write X/Y/Z 3D grids as disk-backed .npy files without allocating full meshgrid in RAM."""
    nx, ny, nz = shape
    if (len(x_axis_m), len(y_axis_m), len(z_axis_m)) != (nx, ny, nz):
        raise ValueError("Axis lengths do not match volume shape.")

    for name in ("X.npy", "Y.npy", "Z.npy"):
        _unlink_if_exists(out_dir / name)
    X_mm = open_memmap(out_dir / "X.npy", mode="w+", dtype=dtype, shape=shape)
    Y_mm = open_memmap(out_dir / "Y.npy", mode="w+", dtype=dtype, shape=shape)
    Z_mm = open_memmap(out_dir / "Z.npy", mode="w+", dtype=dtype, shape=shape)

    x_col = x_axis_m.astype(dtype, copy=False)[:, None, None]
    y_row = y_axis_m.astype(dtype, copy=False)[None, :, None]
    z_row = z_axis_m.astype(dtype, copy=False)[None, None, :]

    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        X_mm[:, :, z0:z1] = x_col
        Y_mm[:, :, z0:z1] = y_row
        Z_mm[:, :, z0:z1] = z_row[:, :, z0:z1]

    del X_mm, Y_mm, Z_mm


def write_volume_matrix_m3(
    out_dir: Path, shape: tuple[int, int, int], voxel_volume_m3: float, dtype=np.float32
) -> None:
    """Write volume_matrix.npy as disk-backed array filled with a constant voxel volume."""
    _unlink_if_exists(out_dir / "volume_matrix.npy")
    vol_mm = open_memmap(out_dir / "volume_matrix.npy", mode="w+", dtype=dtype, shape=shape)
    nz = shape[2]
    chunk_z = 8
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        vol_mm[:, :, z0:z1] = voxel_volume_m3
    del vol_mm


def smooth_displacement_field(
    U: np.ndarray,
    method: str,
    sigma: float,
) -> np.ndarray:
    """Apply post-processing smoothing to a 3D displacement component.

    Parameters
    ----------
    U : ndarray, shape (nx, ny, nz)
        One component of the displacement field (e.g. Ux).
    method : str
        ``"gaussian"``  — Gaussian low-pass filter (isotropic, ``sigma`` in pixels).
            Directly smooths the field. Good for general noise reduction.
        ``"laplacian"`` — Iterative Laplacian diffusion smoothing.
            Solves ``U_new = U + sigma * laplacian(U)`` for one step.
            ``sigma`` controls the diffusion strength (typical: 0.1–1.0).
            Smooths while respecting the local structure of the field;
            commonly used for mesh/displacement field regularization.
    sigma : float
        Kernel size / diffusion strength (see *method*).

    Returns
    -------
    ndarray, same shape as *U*.
    """
    if method == "gaussian":
        return scipy.ndimage.gaussian_filter(U, sigma=sigma, output=np.empty_like(U))
    if method == "laplacian":
        # Iterative Laplacian diffusion: U_new = U + alpha * laplacian(U)
        # Number of iterations scales with sigma; alpha kept small for stability.
        n_iter = max(1, int(round(sigma)))
        alpha = np.float32(min(sigma / max(n_iter, 1), 1.0 / 6.0))  # stability limit for 3D
        result = U.copy()
        lap = np.empty_like(result)  # reuse buffer
        for _ in range(n_iter):
            scipy.ndimage.laplace(result, output=lap)
            result += alpha * lap
        return result
    raise ValueError(f"Unknown smoothing method: {method}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Optimization-based deformation tracking (minimal).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--with_sphere", type=str, default=DEFAULT_WITH_SPHERE, help="Path to deformed TIFF (with sphere)")
    p.add_argument(
        "--without_sphere", type=str, default=DEFAULT_WITHOUT_SPHERE, help="Path to reference TIFF (no sphere)"
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "output" / "deformation-tracking"),
        help="Output directory",
    )
    p.add_argument("--dxy_um", type=float, default=0.492, help="Voxel size in XY (microns)")
    p.add_argument("--dz_um", type=float, default=3.0, help="Voxel size in Z (microns)")
    p.add_argument(
        "--sphere_diameter_mm",
        type=float,
        default=0.5,
        help="Sphere / indenter diameter in mm (metadata; kept in sync with distortion step in the UI).",
    )
    p.add_argument("--downsamp_xy", type=int, default=2, help="Downsample factor in XY")
    p.add_argument("--downsamp_z", type=int, default=1, help="Downsample factor in Z")
    p.add_argument("--z_start", type=int, default=0, help="Z crop start (in original Z index)")
    p.add_argument("--z_end", type=int, default=None, help="Z crop end (exclusive), or omit for full")

    p.add_argument("--num_iter", type=int, default=1001, help="Optimization iterations")
    p.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Emit a progress line every N iterations (for UI polling).",
    )
    p.add_argument("--batch_size", type=int, default=500000, help="Random samples per iteration")
    p.add_argument("--lr_shift", type=float, default=5e-2, help="Learning rate for global shift")
    p.add_argument("--lr_axis", type=float, default=1e-3, help="Learning rate for rotation axis")
    p.add_argument("--lr_angle", type=float, default=1e-5, help="Learning rate for rotation angle")
    p.add_argument("--lr_deform", type=float, default=0.1, help="Learning rate for deformation field")
    p.add_argument("--TV2_reg", type=float, default=30, help="TV2 regularization weight (0 disables)")
    p.add_argument("--lock_global_shift", action="store_true", help="Lock global shift at initial estimate")

    p.add_argument("--deform_downsample_factor_xy", type=int, default=10, help="Coarse deformation grid factor XY")
    p.add_argument("--deform_downsample_factor_z", type=int, default=8, help="Coarse deformation grid factor Z")
    p.add_argument(
        "--indentation_constraint",
        action="store_true",
        help="Indentation constraint: drive the saved output Uz to be mostly negative (downward displacement).",
    )
    p.add_argument(
        "--Uz_penalty_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for indentation constraint penalty. Larger values more strongly enforce Uz < 0 in the final output."
        ),
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    p.add_argument(
        "--output_downsample_xy",
        type=int,
        default=10,
        help="Downsample final outputs by this factor in XY (applied after optimization).",
    )
    p.add_argument(
        "--output_downsample_z",
        type=int,
        default=3,
        help="Downsample final outputs by this factor in Z (applied after optimization).",
    )
    p.add_argument(
        "--upsample_order",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Interpolation order for upsampling deformation field (1=linear, 2=quadratic, 3=cubic). Legacy uses 3.",
    )
    p.add_argument(
        "--skip_grids",
        action="store_true",
        help="Skip writing X/Y/Z grids and volume_matrix (saves disk I/O when running with inverse modeling next). "
        "A grid_params.json is always saved so main_VFM.py can recreate them.",
    )
    p.add_argument(
        "--smooth_method",
        type=str,
        default="none",
        choices=["none", "gaussian", "laplacian"],
        help="Post-processing smoothing on final displacement field. "
        "gaussian: Gaussian low-pass filter (best for general noise). "
        "laplacian: iterative Laplacian diffusion (structure-preserving, common in mechanics).",
    )
    p.add_argument(
        "--smooth_sigma",
        type=float,
        default=1.0,
        help="Smoothing strength: sigma (pixels) for gaussian; diffusion strength for laplacian (typical 0.1-5.0).",
    )

    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0_total = time.perf_counter()

    if args.device == "cuda":
        device = torch.device("cuda:0")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # ----------------------------
    # Load volumes (fixed ZYX -> XYZ)
    # ----------------------------
    stack_with_xyz = load_tiff_zyx_to_xyz(args.with_sphere, args.z_start, args.z_end, args.downsamp_xy, args.downsamp_z)
    stack_without_xyz = load_tiff_zyx_to_xyz(
        args.without_sphere, args.z_start, args.z_end, args.downsamp_xy, args.downsamp_z
    )
    if stack_with_xyz.shape != stack_without_xyz.shape:
        raise ValueError(f"Shape mismatch after preprocessing: {stack_with_xyz.shape} vs {stack_without_xyz.shape}")
    if stack_with_xyz.size == 0:
        raise ValueError("Empty volume after crop/downsample.")

    nx, ny, nz = stack_with_xyz.shape
    t_load_s = time.perf_counter() - t0_total

    # ----------------------------
    # Initial shift estimate (pixels)
    # ----------------------------
    xy_shift_px, z_shift_px = estimate_initial_shift(stack_with_xyz, stack_without_xyz)

    # ----------------------------
    # Coordinate axes for optimization (centered, microns)
    # ----------------------------
    dxy_eff_um = args.dxy_um * args.downsamp_xy
    dz_eff_um = args.dz_um * args.downsamp_z

    x_axis_um_centered = torch.arange(nx, device=device, dtype=torch.float32) * dxy_eff_um
    y_axis_um_centered = torch.arange(ny, device=device, dtype=torch.float32) * dxy_eff_um
    z_axis_um_centered = torch.arange(nz, device=device, dtype=torch.float32) * dz_eff_um
    x_axis_um_centered -= x_axis_um_centered.mean()
    y_axis_um_centered -= y_axis_um_centered.mean()
    z_axis_um_centered -= z_axis_um_centered.mean()

    # Output axes (start at 0, meters)
    x_axis_m = (np.arange(nx, dtype=np.float64) * dxy_eff_um) * 1e-6
    y_axis_m = (np.arange(ny, dtype=np.float64) * dxy_eff_um) * 1e-6
    z_axis_m = (np.arange(nz, dtype=np.float64) * dz_eff_um) * 1e-6

    # ----------------------------
    # Optimizable parameters
    # ----------------------------
    xyz_shift_global_um = torch.tensor(
        [-xy_shift_px[0] * dxy_eff_um, -xy_shift_px[1] * dxy_eff_um, -z_shift_px * dz_eff_um],
        dtype=torch.float32,
        requires_grad=(not args.lock_global_shift),
        device=device,
    )
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, requires_grad=True, device=device)
    angle = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)

    # Coarse deformation grid
    nx_c = max(2, nx // args.deform_downsample_factor_xy)
    ny_c = max(2, ny // args.deform_downsample_factor_xy)
    nz_c = max(2, nz // args.deform_downsample_factor_z)
    r_deform_um = torch.zeros((nx_c, ny_c, nz_c, 3), dtype=torch.float32, requires_grad=True, device=device)

    # Prepare images on device
    stack_with_t = torch.tensor(stack_with_xyz.astype(np.float32), device=device)
    stack_without_t = torch.tensor(stack_without_xyz.astype(np.float32), device=device)

    # Per-variable optimizers (keeps LR clear)
    optimizers = []
    if not args.lock_global_shift:
        optimizers.append(torch.optim.Adam([xyz_shift_global_um], lr=args.lr_shift))
    optimizers.append(torch.optim.Adam([axis], lr=args.lr_axis))
    optimizers.append(torch.optim.Adam([angle], lr=args.lr_angle))
    optimizers.append(torch.optim.Adam([r_deform_um], lr=args.lr_deform))

    def forward_model(rand_ind: torch.Tensor) -> torch.Tensor:
        # unravel indices (x,y,z)
        xyz = _unravel_flat_indices(rand_ind, stack_with_t.shape)

        # nominal physical coordinates (microns), centered
        delta_r = torch.stack(
            [x_axis_um_centered[xyz[0]], y_axis_um_centered[xyz[1]], z_axis_um_centered[xyz[2]]],
            dim=1,
        )

        # deformation field indices (float) in coarse grid coordinates
        x_def = torch.clamp(xyz[0] / args.deform_downsample_factor_xy, 0, r_deform_um.shape[0] - 1)
        y_def = torch.clamp(xyz[1] / args.deform_downsample_factor_xy, 0, r_deform_um.shape[1] - 1)
        z_def = torch.clamp(xyz[2] / args.deform_downsample_factor_z, 0, r_deform_um.shape[2] - 1)
        local_def = interpolated_prediction(x_def, y_def, z_def, r_deform_um, trilinear_interp=True)

        rot = axis_angle_rotmat(axis, angle)
        r_deformed = (delta_r + local_def) @ rot + xyz_shift_global_um[None, :]

        # microns -> pixel coordinates in reference volume
        x_float = r_deformed[:, 0] / dxy_eff_um + (nx / 2)
        y_float = r_deformed[:, 1] / dxy_eff_um + (ny / 2)
        z_float = r_deformed[:, 2] / dz_eff_um + (nz / 2)

        pred = interpolated_prediction(x_float, y_float, z_float, stack_without_t, trilinear_interp=True)
        tgt = stack_with_t[xyz[0], xyz[1], xyz[2]]
        return torch.mean((tgt - pred) ** 2)

    # ----------------------------
    # Optimization loop
    # ----------------------------
    t0_opt = time.perf_counter()
    total_vox = nx * ny * nz
    ui_no_tqdm = os.environ.get("FIM_UI_NO_TQDM", "0") == "1"
    # If user asks for progress updates less frequently than the total iterations (e.g. every 100 iters,
    # but num_iter=11), fall back to printing every iteration so the UI still shows live progress.
    progress_every = int(args.progress_every) if args.progress_every is not None else 0
    if progress_every > 0 and progress_every > args.num_iter:
        progress_every = 1
    for i in tqdm(range(args.num_iter), desc="optim", disable=ui_no_tqdm):
        rand_ind = torch.randint(total_vox, (args.batch_size,), device=device)
        mse = forward_model(rand_ind)
        loss = mse
        if args.TV2_reg and args.TV2_reg > 0:
            loss = loss + (args.TV2_reg * total_variation_loss(r_deform_um))
        if args.indentation_constraint and args.Uz_penalty_weight > 0:
            # Output convention: Uz_m = -(Uz_rot_um + shift_um[2]) * 1e-6.
            # Therefore, to push output Uz_m < 0 (downward), we penalize negative internal r_deform_um z
            # so the optimizer drives internal_z >= 0.
            negative_internal_uz = torch.relu(-r_deform_um[:, :, :, 2])
            loss = loss + (args.Uz_penalty_weight * torch.mean(negative_internal_uz**2))

        loss.backward()
        for opt in optimizers:
            opt.step()
            opt.zero_grad()

        # UI-friendly progress marker (stdout, parseable)
        if progress_every and progress_every > 0:
            if (i + 1) % progress_every == 0 or (i + 1) == args.num_iter:
                msg = f"FIM_PROGRESS iter={i + 1} total={args.num_iter}"
                if ui_no_tqdm:
                    # In UI mode we disable tqdm, so print cleanly without redraw artifacts.
                    print(msg, file=sys.stderr, flush=True)
                else:
                    # Use tqdm.write to avoid corrupting the live progress bar line.
                    tqdm.write(msg, file=sys.stderr)

    t_opt_s = time.perf_counter() - t0_opt

    # ----------------------------
    # Upsample deformation to full resolution and compose (rotation + shift)
    # ----------------------------
    r_np = r_deform_um.detach().cpu().numpy()  # microns, coarse
    zoom_factors = (nx / r_np.shape[0], ny / r_np.shape[1], nz / r_np.shape[2])

    # Upsample each component (order=1 linear, 2 quadratic, 3 cubic)
    upsample_order = args.upsample_order
    Ux_um = scipy.ndimage.zoom(r_np[:, :, :, 0], zoom_factors, order=upsample_order)
    Uy_um = scipy.ndimage.zoom(r_np[:, :, :, 1], zoom_factors, order=upsample_order)
    Uz_um = scipy.ndimage.zoom(r_np[:, :, :, 2], zoom_factors, order=upsample_order)

    rot_np = axis_angle_rotmat(axis, angle).detach().cpu().numpy().astype(np.float64)
    shift_um = xyz_shift_global_um.detach().cpu().numpy().astype(np.float64)  # microns

    # Rotate displacement vectors without stacking a full (nx,ny,nz,3) array
    Ux_rot_um = rot_np[0, 0] * Ux_um + rot_np[0, 1] * Uy_um + rot_np[0, 2] * Uz_um
    Uy_rot_um = rot_np[1, 0] * Ux_um + rot_np[1, 1] * Uy_um + rot_np[1, 2] * Uz_um
    Uz_rot_um = rot_np[2, 0] * Ux_um + rot_np[2, 1] * Uy_um + rot_np[2, 2] * Uz_um

    # Add global shift (microns), then convert to meters.
    # The internal displacement maps deformed → reference (backward warp).
    # Negate to output the standard mechanics convention: reference → deformed
    # (u = x_deformed - X_reference), consistent with F = I + grad(u) in VFM.
    Ux_m = -(Ux_rot_um + shift_um[0]) * 1e-6
    Uy_m = -(Uy_rot_um + shift_um[1]) * 1e-6
    Uz_m = -(Uz_rot_um + shift_um[2]) * 1e-6

    # ----------------------------
    # Optional output downsampling (after optimization)
    # ----------------------------
    ds_xy = int(args.output_downsample_xy) if args.output_downsample_xy is not None else 1
    ds_z = int(args.output_downsample_z) if args.output_downsample_z is not None else 1
    if ds_xy < 1:
        ds_xy = 1
    if ds_z < 1:
        ds_z = 1

    if ds_xy > 1 or ds_z > 1:
        Ux_m = Ux_m[::ds_xy, ::ds_xy, ::ds_z]
        Uy_m = Uy_m[::ds_xy, ::ds_xy, ::ds_z]
        Uz_m = Uz_m[::ds_xy, ::ds_xy, ::ds_z]
        x_axis_m = x_axis_m[::ds_xy]
        y_axis_m = y_axis_m[::ds_xy]
        z_axis_m = z_axis_m[::ds_z]
        nx, ny, nz = Ux_m.shape

    # ----------------------------
    # Optional post-processing smoothing (applied after downsampling to save memory)
    # ----------------------------
    if args.smooth_method != "none":
        print(
            f"Smoothing displacement field ({Ux_m.shape}): method={args.smooth_method}, sigma={args.smooth_sigma}",
            file=sys.stderr,
            flush=True,
        )
        Ux_m = smooth_displacement_field(Ux_m, args.smooth_method, args.smooth_sigma)
        Uy_m = smooth_displacement_field(Uy_m, args.smooth_method, args.smooth_sigma)
        Uz_m = smooth_displacement_field(Uz_m, args.smooth_method, args.smooth_sigma)

    # Save U fields (meters); remove old files first so reruns always replace (shape may change).
    for name in ("Ux.npy", "Uy.npy", "Uz.npy"):
        _unlink_if_exists(out_dir / name)
    np.save(out_dir / "Ux.npy", Ux_m.astype(np.float32, copy=False))
    np.save(out_dir / "Uy.npy", Uy_m.astype(np.float32, copy=False))
    np.save(out_dir / "Uz.npy", Uz_m.astype(np.float32, copy=False))

    dxy_final_um = dxy_eff_um * ds_xy
    dz_final_um = dz_eff_um * ds_z
    voxel_volume_m3 = float((dxy_final_um * 1e-6) * (dxy_final_um * 1e-6) * (dz_final_um * 1e-6))

    # Always save grid metadata so main_VFM.py can recreate grids if needed
    grid_meta = {
        "shape": [nx, ny, nz],
        "dxy_m": dxy_final_um * 1e-6,
        "dz_m": dz_final_um * 1e-6,
        "voxel_volume_m3": voxel_volume_m3,
    }
    (out_dir / "grid_params.json").write_text(json.dumps(grid_meta, indent=2) + "\n", encoding="utf-8")

    if args.skip_grids:
        print("Skipping X/Y/Z grids and volume_matrix (--skip_grids)", file=sys.stderr, flush=True)
        # Do not leave stale grid files from a previous run (would confuse inverse / main_VFM).
        for name in ("X.npy", "Y.npy", "Z.npy", "volume_matrix.npy"):
            _unlink_if_exists(out_dir / name)
    else:
        write_xyz_grids_m(out_dir, x_axis_m, y_axis_m, z_axis_m, shape=(nx, ny, nz), dtype=np.float32, chunk_z=8)
        write_volume_matrix_m3(out_dir, shape=(nx, ny, nz), voxel_volume_m3=voxel_volume_m3, dtype=np.float32)

    # Minimal run metadata
    (out_dir / "run_info.txt").write_text(
        "\n".join(
            [
                "deformation_tracking.py",
                f"with_sphere={args.with_sphere}",
                f"without_sphere={args.without_sphere}",
                f"shape_xyz={(nx, ny, nz)}",
                f"dxy_um={args.dxy_um} dz_um={args.dz_um}",
                f"downsamp_xy={args.downsamp_xy} downsamp_z={args.downsamp_z}",
                f"xy_shift_px={xy_shift_px.tolist()} z_shift_px={z_shift_px}",
                f"lock_global_shift={args.lock_global_shift}",
                f"TV2_reg={args.TV2_reg}",
                f"indentation_constraint={args.indentation_constraint} Uz_penalty_weight={args.Uz_penalty_weight}",
                f"device={device}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    t_total_s = time.perf_counter() - t0_total
    t_save_s = max(0.0, t_total_s - t_load_s - t_opt_s)
    sec_per_iter = (t_opt_s / args.num_iter) if args.num_iter else float("nan")
    print(
        (
            "Timing (seconds): "
            f"load={t_load_s:.2f}, optimize={t_opt_s:.2f} ({sec_per_iter:.4f} s/iter), "
            f"save+post={t_save_s:.2f}, total={t_total_s:.2f}"
        ),
        file=sys.stderr,
        flush=True,
    )

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
