"""Optimization-based 3D deformation tracking for volumetric TIFF stacks.

The optimizer aligns the deformed volume (with sphere) to the reference volume
(without sphere) using a backward warp, then writes displacement fields in the
standard convention:

    u = x_deformed - X_reference

Outputs in ``--out_dir``:
- ``Ux.npy``, ``Uy.npy``, ``Uz.npy`` (meters, shape ``(X, Y, Z)``)
- ``X.npy``, ``Y.npy``, ``Z.npy`` coordinate grids (meters)
- ``volume_matrix.npy`` voxel volume weights (m^3)

Optional ``--remap_to_reference`` rewrites ``u`` onto undeformed reference
coordinates by solving ``x = X + u(x)`` with a fixed-point inverse map
(``scipy.ndimage.map_coordinates``), applied on the coarse deformation grid
before upsampling for lower memory and runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch
from numpy.lib.format import open_memmap
from skimage import io
from skimage.metrics import structural_similarity as structural_similarity_2d
from skimage.registration import phase_cross_correlation
from tqdm.auto import tqdm

# ----------------------------
# Defaults: if you do not pass paths on the command line, these sample stacks are used.
# ----------------------------
_SIM_TIFF_DIR = (
    Path(__file__).resolve().parent.parent / "test_data" / "simulate"
)  # Folder that ships with small demo TIFFs.
DEFAULT_WITHOUT_SPHERE = str(_SIM_TIFF_DIR / "ref_image.tif")  # Default “no sphere” (reference) image path.
DEFAULT_WITH_SPHERE = str(_SIM_TIFF_DIR / "def_image.tif")  # Default “with sphere” (deformed) image path.


def _display_input_name(path: str) -> str:
    """Short display name for stderr logs (strip UUID-like filename prefixes).

    Parameters
    ----------
    path : str
        Full filesystem path.

    Returns
    -------
    str
        Basename, or text after ``<32hex>_`` when the prefix looks like an upload id.
    """
    name = Path(path).name  # Just the filename (no folders), for cleaner log lines.
    if "_" not in name:
        return name
    prefix, rest = name.split("_", 1)  # Split “hex_rest.tif” so we can drop a long upload id in prefix.
    if len(prefix) == 32 and all(c in "0123456789abcdef" for c in prefix.lower()):
        return rest
    return name


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
        # Trick so the eight corner weights line up with standard trilinear interpolation (swap “near/far” weights).
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
    return (dx.pow(2) + dy.pow(2) + dz.pow(2)).mean()  # One number: average squared “roughness” of the field.


def load_tiff_zyx_to_xyz(path: str, z_start: int, z_end: int | None, downsamp_xy: int, downsamp_z: int) -> np.ndarray:
    """
    Load TIFF as (Z,Y,X), crop/downsample, transpose to ``(X,Y,Z)`` for the pipeline.

    Parameters
    ----------
    path : str
        TIFF path.
    z_start, z_end : int | None
        Z slice ``z_start:z_end`` on the original Z axis; ``z_end`` may be ``None``.
    downsamp_xy, downsamp_z : int
        Strides after crop (XY: ``::downsamp_xy``, Z: ``::downsamp_z`` in slice).

    Returns
    -------
    numpy.ndarray
        ``(nx, ny, nz)``; dtype from file.
    """
    vol_zyx = io.imread(path)  # Image comes in as depth × height × width (Z,Y,X).
    vol_zyx = vol_zyx[
        z_start:z_end:downsamp_z, ::downsamp_xy, ::downsamp_xy
    ]  # Optional crop and skip voxels to shrink the stack.
    # Reorder axes to X,Y,Z for the rest of the code.
    return vol_zyx.transpose(2, 1, 0)


def estimate_initial_shift(
    stack_with_sphere_xyz: np.ndarray, stack_without_sphere_xyz: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Initial global shift in **pixels** from phase cross-correlation on 2D projections.

    XY shift from max over Z (XY image); Z shift from second component of shift on
    max-over-X projections (shape ``(Y,Z)``).

    Returns
    -------
    xy_shift : numpy.ndarray
        ``(2,)`` float64, ``(shift_x, shift_y)`` in pixels.
    z_shift : float
        Z shift in pixels.
    """
    max_with = stack_with_sphere_xyz.max(2)  # Squash Z: one 2D “top-down” picture of the deformed volume.
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


def _corr_rmse_ssim_2d(pred2d: np.ndarray, data2d: np.ndarray) -> tuple[float, float, float]:
    """Image metrics for comparison figure row titles: Pearson r, RMSE, SSIM.

    Correlation is NaN if either image is (near) constant.
    """
    p = np.asarray(pred2d, dtype=np.float64)  # Prediction image as double-precision numbers.
    d = np.asarray(data2d, dtype=np.float64)  # Measured (data) image the same way.
    rmse = float(np.sqrt(np.mean((p - d) ** 2)))  # Typical pixel error size (root mean square).
    pr, dr = p.ravel(), d.ravel()  # Long 1D lists of all pixels for correlation.
    if np.std(pr) > 1e-12 and np.std(dr) > 1e-12:
        corr = float(np.corrcoef(pr, dr)[0, 1])  # How linearly related the two images are (−1 to 1).
    else:
        corr = float("nan")  # Not defined if an image is flat (no variation).
    dmin = float(min(p.min(), d.min()))  # Darkest value across both images (for SSIM scaling).
    dmax = float(max(p.max(), d.max()))  # Brightest value across both.
    data_range = max(dmax - dmin, 1e-12)  # Overall contrast; SSIM needs this so “similar” is meaningful.
    ssim_val = float(
        structural_similarity_2d(p, d, data_range=data_range)
    )  # SSIM: closer to 1 means more alike structurally.
    return corr, rmse, ssim_val


def _save_comparison_figure(
    out_dir: Path,
    stack_with_xyz: np.ndarray,
    stack_without_xyz: np.ndarray,
    Ux_um: np.ndarray,
    Uy_um: np.ndarray,
    Uz_um: np.ndarray,
    rot_np: np.ndarray,
    shift_um: np.ndarray,
    dxy_eff_um: float,
    dz_eff_um: float,
) -> Path | None:
    """Save prediction-vs-data comparison PNG (4×3 grid: XY/YZ projections and slices).

    Columns: prediction, data, difference (pred − data). Each row reports Corr, RMSE, SSIM.
    Returns the saved figure path, or ``None`` when plotting dependencies are unavailable.

    The ``Ux_um`` / ``Uy_um`` / ``Uz_um`` inputs match the *internal* sign before the final
    output negation in ``main``; ``map_coordinates`` pulls ``stack_without_xyz`` at rotated,
    shifted positions to form ``pred``.

    Memory: builds ``pred`` one Z-slab at a time (no full-volume ``meshgrid``), so peak RAM
    stays ~O(nx·ny) temporaries + one ``pred`` volume (same order as the input stacks).
    """
    try:
        import matplotlib  # Only load plotting if we are saving a figure (keeps base script lighter).

        matplotlib.use("Agg")  # Draw to a file, not an on-screen window (works on servers).
        import matplotlib.pyplot as plt  # The usual pyplot interface for building the figure.
    except Exception:
        print("Skipping comparison figure: matplotlib is not available.", file=sys.stderr, flush=True)
        return None

    nx, ny, nz = stack_with_xyz.shape  # Number of voxels along X, Y, and Z.
    x_axis_um_centered = (
        np.arange(nx, dtype=np.float64) * dxy_eff_um
    )  # X positions in microns, centered so the middle of the box is ~0.
    y_axis_um_centered = np.arange(ny, dtype=np.float64) * dxy_eff_um  # Same for Y.
    z_axis_um_centered = np.arange(nz, dtype=np.float64) * dz_eff_um  # Same for Z.
    x_axis_um_centered -= x_axis_um_centered.mean()
    y_axis_um_centered -= y_axis_um_centered.mean()
    z_axis_um_centered -= z_axis_um_centered.mean()

    vol = stack_without_xyz.astype(
        np.float32, copy=False
    )  # Reference stack we resample (float32 is enough for plotting).
    pred = np.empty(
        (nx, ny, nz), dtype=np.float32
    )  # We fill this: “what the reference would look like” after the warp.
    xi = x_axis_um_centered[:, None]  # X coordinate as a column (varies along rows only).
    yj = y_axis_um_centered[None, :]  # Y coordinate as a row (varies along columns only).
    r00, r01, r02 = rot_np[0, 0], rot_np[0, 1], rot_np[0, 2]  # First row of the rotation matrix.
    r10, r11, r12 = rot_np[1, 0], rot_np[1, 1], rot_np[1, 2]  # Second row.
    r20, r21, r22 = rot_np[2, 0], rot_np[2, 1], rot_np[2, 2]  # Third row.
    sx, sy, sz = shift_um[0], shift_um[1], shift_um[2]  # Extra translation in microns after rotation.
    nx2, ny2, nz2 = (
        nx / 2.0,
        ny / 2.0,
        nz / 2.0,
    )  # Half the grid size: used to put the origin in the middle like the training code.

    for k in range(nz):  # k picks one Z layer at a time (saves memory).
        # For this slab: add displacement, apply rotation and shift, then turn physical
        # position into voxel indices in the reference volume.
        rx = xi + Ux_um[:, :, k]  # Reference X (μm) plus local displacement Ux on this layer.
        ry = yj + Uy_um[:, :, k]  # Same for Y.
        rz = z_axis_um_centered[k] + Uz_um[:, :, k]  # Z position of this layer plus Uz.
        x_um = r00 * rx + r01 * ry + r02 * rz + sx  # After rigid rotation and global shift: X in microns.
        y_um = r10 * rx + r11 * ry + r12 * rz + sy  # Rotated+shifted Y.
        z_um = r20 * rx + r21 * ry + r22 * rz + sz  # Rotated+shifted Z.
        # Match forward_model: voxel index = position/spacing + half the grid length.
        x_pix = np.clip(
            x_um / dxy_eff_um + nx2, 0.0, nx - 1.0001
        )  # Where to read reference intensity in X (fractional index).
        y_pix = np.clip(y_um / dxy_eff_um + ny2, 0.0, ny - 1.0001)  # Same in Y.
        z_pix = np.clip(z_um / dz_eff_um + nz2, 0.0, nz - 1.0001)  # Same in Z.
        pred[:, :, k] = scipy.ndimage.map_coordinates(vol, [x_pix, y_pix, z_pix], order=1, mode="nearest")

    z_mid = nz // 2  # Middle slice index for a top-down style view.
    x_mid = nx // 2  # Middle slice for a side view.

    data_proj_xy = stack_with_xyz.max(axis=2)  # Brightest value along Z at each XY pixel (data).
    pred_proj_xy = pred.max(axis=2)  # Same projection for the predicted volume.
    data_xy = stack_with_xyz[:, :, z_mid]  # Single horizontal slice of the data at z_mid.
    pred_xy = pred[:, :, z_mid]  # Same slice of the prediction.

    data_proj_yz = stack_with_xyz.max(axis=0)  # Brightest along X: a front/back view (data).
    pred_proj_yz = pred.max(axis=0)  # Same for prediction.
    data_yz = stack_with_xyz[x_mid, :, :]  # Single vertical slice at x_mid (data).
    pred_yz = pred[x_mid, :, :]  # Same slice of prediction.
    # imshow expects row = up/down; transpose so Y is horizontal and Z vertical on screen.
    data_proj_yz = np.asarray(data_proj_yz).T
    pred_proj_yz = np.asarray(pred_proj_yz).T
    data_yz = np.asarray(data_yz).T
    pred_yz = np.asarray(pred_yz).T

    rows: list[tuple[str, np.ndarray, np.ndarray]] = [  # Each row: caption, prediction 2D, measured 2D.
        ("XY projection (max Z)", pred_proj_xy, data_proj_xy),
        (f"XY slice (Z={z_mid})", pred_xy, data_xy),
        ("YZ projection (max X)", pred_proj_yz, data_proj_yz),
        (f"YZ slice (X={x_mid})", pred_yz, data_yz),
    ]

    fig, axes = plt.subplots(
        4, 3, figsize=(16, 18), constrained_layout=True
    )  # Four rows (views) by three columns (pred, data, difference).
    cmap_data = "viridis"  # Purple–yellow scale for intensity images.
    cmap_diff = "RdBu_r"  # Red–blue scale centered at zero for differences.

    for i, (row_label, p2d, d2d) in enumerate(rows):  # i counts which of the four view rows we are on.
        diff2d = p2d.astype(np.float64) - d2d.astype(np.float64)  # Prediction minus data at each pixel.
        corr, rmse, ssim_val = _corr_rmse_ssim_2d(p2d, d2d)  # Numbers printed in the title for this row.
        vmax_data = float(max(np.max(p2d), np.max(d2d), 1e-9))  # Top of the color scale for pred/data panels.
        vmin_data = 0.0  # Bottom of the scale (assume non-negative intensity).

        abs_diff = np.abs(diff2d)  # Size of the error at each pixel.
        vmax_diff = float(
            max(np.percentile(abs_diff, 99.5), np.max(abs_diff) * 1e-6, 1e-9)
        )  # Symmetric range so ±errors use the same colors.
        vmin_diff = -vmax_diff  # Negative side of the diverging scale.

        row_title = f"{row_label}\nCorr: {corr:.4f} | RMSE: {rmse:.2f} | SSIM: {ssim_val:.4f}"
        # Text under the middle column for this row.

        for j, (img, vmin, vmax, cmap) in enumerate(  # j = 0 prediction, 1 data, 2 difference.
            [
                (p2d, vmin_data, vmax_data, cmap_data),
                (d2d, vmin_data, vmax_data, cmap_data),
                (diff2d, vmin_diff, vmax_diff, cmap_diff),
            ]
        ):
            ax = axes[i, j]  # One small plot in the grid.
            im = ax.imshow(
                img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest"
            )  # Draw the 2D array as a colored image.
            ax.set_xticks([])  # Hide axis numbers (figures are for visual inspection only).
            ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Legend strip showing what colors mean.
            cbar.ax.tick_params(labelsize=7)

        if i == 0:
            axes[0, 0].set_title("Prediction", fontsize=11)
            axes[0, 1].set_title("Data\n" + row_title, fontsize=10)
            axes[0, 2].set_title("Difference (Pred − Data)", fontsize=11)
        else:
            axes[i, 1].set_title(row_title, fontsize=10)

    fig.suptitle(f"Prediction vs data — {out_dir.name}", fontsize=12, y=1.01)

    fig_path = out_dir / "comparison_prediction_vs_data.png"  # Where the PNG is written on disk.
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    print(f"Saved comparison figure to: {fig_path}", file=sys.stderr, flush=True)
    return fig_path


def _save_mse_curve_png(out_dir: Path, mse_trace: np.ndarray) -> Path | None:
    """Save iteration vs minibatch MSE as a line plot (headless). Returns path or None if matplotlib missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("Skipping MSE curve figure: matplotlib is not available.", file=sys.stderr, flush=True)
        return None

    it = np.arange(1, mse_trace.shape[0] + 1, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(it, mse_trace, color="C0", linewidth=1.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Minibatch MSE (data term)")
    ax.set_title("Optimization MSE trace (stochastic; batch varies per step)")
    ax.grid(True, alpha=0.3)
    fig_path = out_dir / "mse_curve.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"Saved MSE curve to: {fig_path}", file=sys.stderr, flush=True)
    return fig_path


def _unlink_if_exists(path: Path) -> None:
    """Delete *path* if it is a file; ignore errors (safe no-op for missing files)."""
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
    """Write ``X/Y/Z.npy`` memmaps: broadcast 1D axes to a full grid in Z slabs (low RAM)."""
    nx, ny, nz = shape  # Target volume size we must write.
    if (len(x_axis_m), len(y_axis_m), len(z_axis_m)) != (nx, ny, nz):
        raise ValueError("Axis lengths do not match volume shape.")

    for name in ("X.npy", "Y.npy", "Z.npy"):  # name is each coordinate file we are about to (over)write.
        _unlink_if_exists(out_dir / name)
    X_mm = open_memmap(
        out_dir / "X.npy", mode="w+", dtype=dtype, shape=shape
    )  # On-disk array for X at every voxel (no full mesh in RAM).
    Y_mm = open_memmap(out_dir / "Y.npy", mode="w+", dtype=dtype, shape=shape)  # Same for Y.
    Z_mm = open_memmap(out_dir / "Z.npy", mode="w+", dtype=dtype, shape=shape)  # Same for Z.

    x_col = x_axis_m.astype(dtype, copy=False)[
        :, None, None
    ]  # Repeat the X 1D line across Y and Z without storing a huge array.
    y_row = y_axis_m.astype(dtype, copy=False)[None, :, None]  # Repeat Y across X and Z.
    z_row = z_axis_m.astype(dtype, copy=False)[None, None, :]  # Repeat Z across X and Y.

    for z0 in range(0, nz, chunk_z):  # Process Z in chunks so we never build the full 3D grid in memory.
        z1 = min(nz, z0 + chunk_z)  # End index of this Z chunk.
        X_mm[:, :, z0:z1] = x_col
        Y_mm[:, :, z0:z1] = y_row
        Z_mm[:, :, z0:z1] = z_row[:, :, z0:z1]

    del X_mm, Y_mm, Z_mm


def write_volume_matrix_m3(
    out_dir: Path, shape: tuple[int, int, int], voxel_volume_m3: float, dtype=np.float32
) -> None:
    """Write ``volume_matrix.npy``: constant physical voxel volume (m³) per grid cell."""
    _unlink_if_exists(out_dir / "volume_matrix.npy")
    vol_mm = open_memmap(
        out_dir / "volume_matrix.npy", mode="w+", dtype=dtype, shape=shape
    )  # One physical voxel volume per cell (for integration weights later).
    nz = shape[2]  # How many Z layers to loop over.
    chunk_z = 8  # Write a few Z slices at a time.
    for z0 in range(0, nz, chunk_z):  # Start index of this chunk in Z.
        z1 = min(nz, z0 + chunk_z)  # End index (exclusive).
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
        return scipy.ndimage.gaussian_filter(
            U, sigma=sigma, output=np.empty_like(U)
        )  # Blur each component with a Gaussian bell of width sigma (voxels).
    if method == "laplacian":
        # Smooth by repeatedly adding a small Laplacian step (like heat diffusion on the field).
        # More iterations when sigma is large; step size alpha stays small so it stays stable.
        n_iter = max(1, int(round(sigma)))  # How many diffusion steps to take.
        alpha = np.float32(min(sigma / max(n_iter, 1), 1.0 / 6.0))  # Step size per iteration (capped for 3D stability).
        result = U.copy()  # Working copy we update in place.
        lap = np.empty_like(result)  # Scratch array holding the Laplacian of result.
        for _ in range(n_iter):
            scipy.ndimage.laplace(result, output=lap)
            result += alpha * lap
        return result
    raise ValueError(f"Unknown smoothing method: {method}")


def coarse_reference_axes_m(
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    z_axis_m: np.ndarray,
    nx_c: int,
    ny_c: int,
    nz_c: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical coordinates (m) at coarse deformation nodes, aligned with ``ndimage.zoom`` layout.

    Coarse index ``ic`` maps to a fractional index along the full ``x_axis_m`` so that
    upsampling coarse data with ``zoom=(nx/nx_c, ...)`` targets the same geometry as the
    fine reference voxel centers.
    """
    nx_f, ny_f, nz_f = len(x_axis_m), len(y_axis_m), len(z_axis_m)

    def centers_1d(ax: np.ndarray, n_c: int, n_full: int) -> np.ndarray:
        if n_c < 1 or n_full < 1:
            raise ValueError("n_c and n_full must be positive")
        ax = np.asarray(ax, dtype=np.float64)
        if n_c == 1:
            idx = np.array([(n_full - 1) / 2.0], dtype=np.float64)
        else:
            idx = (np.arange(n_c, dtype=np.float64) + 0.5) * (n_full / n_c) - 0.5
        idx = np.clip(idx, 0.0, float(n_full - 1))
        i0 = np.floor(idx).astype(np.int64)
        i1 = np.minimum(i0 + 1, n_full - 1)
        w = idx - i0.astype(np.float64)
        return (1.0 - w) * ax[i0] + w * ax[i1]

    return (
        centers_1d(x_axis_m, nx_c, nx_f),
        centers_1d(y_axis_m, ny_c, ny_f),
        centers_1d(z_axis_m, nz_c, nz_f),
    )


def _sample_displacement_at_world_m(
    Ux: np.ndarray,
    Uy: np.ndarray,
    Uz: np.ndarray,
    xm: np.ndarray,
    ym: np.ndarray,
    zm: np.ndarray,
    x0: float,
    y0: float,
    z0: float,
    dx: float,
    dy: float,
    dz: float,
    *,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trilinear (order=1) or nearest (order=0) sample of (Ux,Uy,Uz) at world coords (m)."""
    fx = (xm - x0) / dx
    fy = (ym - y0) / dy
    fz = (zm - z0) / dz
    coords = np.stack([fx, fy, fz], axis=0)
    kw = {"order": order, "mode": "nearest", "prefilter": False}
    ux_s = scipy.ndimage.map_coordinates(Ux, coords, **kw)
    uy_s = scipy.ndimage.map_coordinates(Uy, coords, **kw)
    uz_s = scipy.ndimage.map_coordinates(Uz, coords, **kw)
    return ux_s, uy_s, uz_s


def remap_displacement_lagrangian_griddata(
    Ux_m: np.ndarray,
    Uy_m: np.ndarray,
    Uz_m: np.ndarray,
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    z_axis_m: np.ndarray,
    *,
    method: str = "linear",
    max_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate displacement onto a regular grid in reference (Lagrangian) coordinates.

    Eulerian storage: ``U*[i,j,k]`` is u at the imaging voxel indexed by ``(i,j,k)``.
    For each reference lattice node ``X`` (same ``meshgrid`` as ``X.npy``/``Y.npy``/``Z.npy``),
    the deformed position obeys ``x = X + u(x)`` with u sampled on the Eulerian grid.
    We solve that with fixed-point iteration and :func:`scipy.ndimage.map_coordinates`
    (O(N) per iteration, modest memory). Equivalent to inverting the scatter ``X = x - u``
    without building a 3D Delaunay triangulation over all voxels.

    Parameters
    ----------
    Ux_m, Uy_m, Uz_m
        Displacement components (m), shape ``(nx, ny, nz)``, convention
        ``u = x_deformed - X_reference``.
    x_axis_m, y_axis_m, z_axis_m
        Uniform 1D voxel-center coordinates (m), length ``nx, ny, nz``.
    method
        ``linear`` → ``order=1``; ``nearest`` → ``order=0`` for ``map_coordinates``.
    max_iter
        Maximum fixed-point steps (typically converges in few iterations for moderate strain).

    Returns
    -------
    tuple
        ``(Ux, Uy, Uz)`` same shape, float64, u expressed on the reference grid.
    """
    if Ux_m.shape != Uy_m.shape or Ux_m.shape != Uz_m.shape:
        raise ValueError("Ux_m, Uy_m, Uz_m must have the same shape")
    nx, ny, nz = Ux_m.shape
    if (len(x_axis_m), len(y_axis_m), len(z_axis_m)) != (nx, ny, nz):
        raise ValueError("Axis lengths must match displacement shape")

    order = 1 if method == "linear" else 0
    x_axis_m = np.asarray(x_axis_m, dtype=np.float64)
    y_axis_m = np.asarray(y_axis_m, dtype=np.float64)
    z_axis_m = np.asarray(z_axis_m, dtype=np.float64)
    dx = float((x_axis_m[-1] - x_axis_m[0]) / (nx - 1)) if nx > 1 else 1.0
    dy = float((y_axis_m[-1] - y_axis_m[0]) / (ny - 1)) if ny > 1 else 1.0
    dz = float((z_axis_m[-1] - z_axis_m[0]) / (nz - 1)) if nz > 1 else 1.0
    x0, y0, z0 = float(x_axis_m[0]), float(y_axis_m[0]), float(z_axis_m[0])

    Ux = np.asarray(Ux_m, dtype=np.float64, order="C")
    Uy = np.asarray(Uy_m, dtype=np.float64, order="C")
    Uz = np.asarray(Uz_m, dtype=np.float64, order="C")

    Xg, Yg, Zg = np.meshgrid(x_axis_m, y_axis_m, z_axis_m, indexing="ij")
    xm = np.asarray(Xg, dtype=np.float64)
    ym = np.asarray(Yg, dtype=np.float64)
    zm = np.asarray(Zg, dtype=np.float64)

    ux, uy, uz = _sample_displacement_at_world_m(Ux, Uy, Uz, xm, ym, zm, x0, y0, z0, dx, dy, dz, order=order)
    xm = Xg + ux
    ym = Yg + uy
    zm = Zg + uz

    tol = max(dx, dy, dz) * 1e-8
    if tol <= 0:
        tol = 1e-15

    last_delta = float("inf")
    for _ in range(max_iter):
        ux, uy, uz = _sample_displacement_at_world_m(Ux, Uy, Uz, xm, ym, zm, x0, y0, z0, dx, dy, dz, order=order)
        xm_new = Xg + ux
        ym_new = Yg + uy
        zm_new = Zg + uz
        last_delta = float(np.max(np.abs(xm_new - xm) + np.abs(ym_new - ym) + np.abs(zm_new - zm)))
        xm, ym, zm = xm_new, ym_new, zm_new
        if last_delta < tol:
            break

    if last_delta >= tol:
        print(
            f"Warning: remap_to_reference fixed-point did not reach tol={tol:g} "
            f"(last max node update L1 ≈ {last_delta:g} m); "
            "try increasing --remap_max_iter.",
            file=sys.stderr,
            flush=True,
        )

    ux_out, uy_out, uz_out = _sample_displacement_at_world_m(
        Ux, Uy, Uz, xm, ym, zm, x0, y0, z0, dx, dy, dz, order=order
    )
    return ux_out, uy_out, uz_out


def main() -> None:
    """CLI: load volumes, optimize MSE (backward warp), compose motion, write U*.npy and sidecar files."""
    p = argparse.ArgumentParser(  # Collects command-line options and defaults.
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
        "--Uz_penalty_weight",
        type=float,
        default=0.0,
        help=(
            "Penalty weight for upward internal Uz in the coarse deformation field (0 disables). "
            "Larger values more strongly discourage upward motion so saved Uz tends downward."
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
    p.add_argument(
        "--save_comparison_figure",
        action="store_true",
        help="Save comparison_prediction_vs_data.png (prediction vs data: XY max projection and middle XY slice).",
    )
    p.add_argument(
        "--remap_to_reference",
        action="store_true",
        help=(
            "After optimization, express u on the undeformed reference lattice (inverse map x=X+u(x)) "
            "on the coarse deformation grid, then upsample u (m) with the same zoom as the internal "
            "field. Skips full-resolution remap. Default: Eulerian storage after zoom+rotate."
        ),
    )
    p.add_argument(
        "--remap_interp",
        type=str,
        default="linear",
        choices=["linear", "nearest"],
        help="Interpolation for --remap_to_reference: linear (trilinear) or nearest.",
    )
    p.add_argument(
        "--remap_max_iter",
        type=int,
        default=25,
        help="Max fixed-point iterations for --remap_to_reference.",
    )

    p.add_argument(
        "--trace_mse",
        action="store_true",
        help=(
            "Record minibatch data MSE each iteration, save optimization_mse.npy and mse_curve.png in --out_dir. "
            "MSE is the forward_model term only (before TV2 / Uz penalty)."
        ),
    )

    args = p.parse_args()  # All flags the user passed (or defaults).

    out_dir = Path(args.out_dir).resolve()  # Full path to the folder where we write results.
    out_dir.mkdir(parents=True, exist_ok=True)  # Create it if missing.

    t0_total = time.perf_counter()  # Start the clock for total runtime.

    if args.device == "cuda":
        device = torch.device("cuda:0")  # Use the first NVIDIA GPU.
    elif args.device == "cpu":
        device = torch.device("cpu")  # Force everything on the CPU.
    else:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )  # Prefer GPU if present, else CPU.

    # Heavy torch tensors and optimizers sit on device; NumPy coordinate arrays stay on CPU until we save.

    # ----------------------------
    # 1. Load volumes (fixed ZYX -> XYZ)
    # ----------------------------
    print(f"Loading TIFF (with sphere): {_display_input_name(args.with_sphere)}", file=sys.stderr, flush=True)
    stack_with_xyz = load_tiff_zyx_to_xyz(
        args.with_sphere, args.z_start, args.z_end, args.downsamp_xy, args.downsamp_z
    )  # Deformed specimen (with sphere), as X,Y,Z array, possibly cropped/downsampled.
    print(
        f"Loading TIFF (without sphere): {_display_input_name(args.without_sphere)}",
        file=sys.stderr,
        flush=True,
    )
    stack_without_xyz = load_tiff_zyx_to_xyz(  # Reference (no sphere), same preprocessing.
        args.without_sphere, args.z_start, args.z_end, args.downsamp_xy, args.downsamp_z
    )
    if stack_with_xyz.shape != stack_without_xyz.shape:
        raise ValueError(f"Shape mismatch after preprocessing: {stack_with_xyz.shape} vs {stack_without_xyz.shape}")
    if stack_with_xyz.size == 0:
        raise ValueError("Empty volume after crop/downsample.")

    nx, ny, nz = stack_with_xyz.shape  # Grid size after crop and downsampling.
    print(f"Loaded preprocessed volumes with shape (X,Y,Z)=({nx}, {ny}, {nz})", file=sys.stderr, flush=True)
    t_load_s = time.perf_counter() - t0_total  # Time spent just loading and preprocessing.

    # ----------------------------
    # 2. Initial shift estimate (pixels)
    # ----------------------------
    xy_shift_px, z_shift_px = estimate_initial_shift(
        stack_with_xyz, stack_without_xyz
    )  # Rough alignment: (shift in X, shift in Y) and shift in Z, all in pixels.

    # ----------------------------
    # Coordinate axes for optimization (centered, microns)
    # ----------------------------
    dxy_eff_um = args.dxy_um * args.downsamp_xy  # True pixel size in XY after downsampling (microns per voxel step).
    dz_eff_um = args.dz_um * args.downsamp_z  # True step size along Z after downsampling.

    x_axis_um_centered = (
        torch.arange(nx, device=device, dtype=torch.float32) * dxy_eff_um
    )  # X coordinate of each column, in microns, mean-centered for stable rotation.
    y_axis_um_centered = torch.arange(ny, device=device, dtype=torch.float32) * dxy_eff_um  # Same for Y.
    z_axis_um_centered = torch.arange(nz, device=device, dtype=torch.float32) * dz_eff_um  # Same for Z.
    x_axis_um_centered -= x_axis_um_centered.mean()
    y_axis_um_centered -= y_axis_um_centered.mean()
    z_axis_um_centered -= z_axis_um_centered.mean()
    # Subtracting the mean puts the origin near the middle of the box so rotation is about the specimen, not a corner.
    # Files written for VFM use positions starting at 0 m along each edge (see x_axis_m below).

    # Output axes (start at 0, meters)
    x_axis_m = (
        np.arange(nx, dtype=np.float64) * dxy_eff_um
    ) * 1e-6  # X coordinate of each voxel for saved grids, in meters from 0.
    y_axis_m = (np.arange(ny, dtype=np.float64) * dxy_eff_um) * 1e-6  # Y from 0 (m).
    z_axis_m = (np.arange(nz, dtype=np.float64) * dz_eff_um) * 1e-6  # Z from 0 (m).

    # ----------------------------
    # Optimizable parameters
    # ----------------------------
    xyz_shift_global_um = torch.tensor(  # Whole-volume translation in microns (learnable unless you lock it).
        # Negative (estimated shift in pixels × microns per pixel): start close to where
        # phase correlation says overlap is best.
        [-xy_shift_px[0] * dxy_eff_um, -xy_shift_px[1] * dxy_eff_um, -z_shift_px * dz_eff_um],
        dtype=torch.float32,
        requires_grad=(not args.lock_global_shift),
        device=device,
    )
    # Rotation axis (learnable 3-vector; normalized in axis_angle_rotmat); init +Z.
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, requires_grad=True, device=device)
    angle = torch.tensor(
        0.0, dtype=torch.float32, requires_grad=True, device=device
    )  # How many radians to rotate about that axis; starts at zero (no rotation).

    # Coarse deformation grid
    nx_c = max(2, nx // args.deform_downsample_factor_xy)  # Number of coarse voxels along X (at least 2).
    ny_c = max(2, ny // args.deform_downsample_factor_xy)  # Same along Y.
    nz_c = max(2, nz // args.deform_downsample_factor_z)  # Same along Z.
    r_deform_um = torch.zeros((nx_c, ny_c, nz_c, 3), dtype=torch.float32, requires_grad=True, device=device)
    # Local displacement vectors (microns) on the coarse grid; we upsample to full resolution after training.

    # Prepare images on device
    stack_with_t = torch.tensor(
        stack_with_xyz.astype(np.float32), device=device
    )  # Deformed volume as a torch tensor on GPU/CPU.
    stack_without_t = torch.tensor(
        stack_without_xyz.astype(np.float32), device=device
    )  # Reference volume the same way.

    # Per-variable optimizers (keeps LR clear)
    optimizers = []  # We will attach one Adam optimizer per parameter group below.
    if not args.lock_global_shift:
        optimizers.append(torch.optim.Adam([xyz_shift_global_um], lr=args.lr_shift))
    optimizers.append(torch.optim.Adam([axis], lr=args.lr_axis))
    optimizers.append(torch.optim.Adam([angle], lr=args.lr_angle))
    optimizers.append(torch.optim.Adam([r_deform_um], lr=args.lr_deform))

    def forward_model(rand_ind: torch.Tensor) -> torch.Tensor:
        """Batch MSE: *with-sphere* intensity vs *reference* sampled at inferred backward-warp positions."""
        # Turn random flat indices into (i,j,k) voxel addresses in the deformed image.
        xyz = _unravel_flat_indices(
            rand_ind, stack_with_t.shape
        )  # Three integer index tensors: where we read the “data” voxel (with sphere).

        # Physical location (microns) of that voxel in the centered frame.
        delta_r = torch.stack(
            [x_axis_um_centered[xyz[0]], y_axis_um_centered[xyz[1]], z_axis_um_centered[xyz[2]]],
            dim=1,
        )  # N rows × 3 columns: x,y,z in microns for each sample.

        # Where that point falls inside the coarse displacement grid (fractional indices).
        x_def = torch.clamp(
            xyz[0] / args.deform_downsample_factor_xy, 0, r_deform_um.shape[0] - 1
        )  # Coarse-grid X coordinate for trilinear lookup.
        y_def = torch.clamp(xyz[1] / args.deform_downsample_factor_xy, 0, r_deform_um.shape[1] - 1)  # Coarse-grid Y.
        z_def = torch.clamp(xyz[2] / args.deform_downsample_factor_z, 0, r_deform_um.shape[2] - 1)  # Coarse-grid Z.
        local_def = interpolated_prediction(
            x_def, y_def, z_def, r_deform_um, trilinear_interp=True
        )  # Local displacement (Ux,Uy,Uz) in microns at each sample.

        rot = axis_angle_rotmat(axis, angle)  # 3×3 rotation from current axis and angle.
        r_deformed = (delta_r + local_def) @ rot + xyz_shift_global_um[
            None, :
        ]  # After adding local motion, rotating, and shifting: where we land in microns.

        # Convert microns to fractional voxel indices in the reference (without-sphere)
        # volume (origin at volume center).
        x_float = r_deformed[:, 0] / dxy_eff_um + (nx / 2)  # Reference X index (can be between voxels).
        y_float = r_deformed[:, 1] / dxy_eff_um + (ny / 2)  # Reference Y index.
        z_float = r_deformed[:, 2] / dz_eff_um + (nz / 2)  # Reference Z index.

        pred = interpolated_prediction(
            x_float, y_float, z_float, stack_without_t, trilinear_interp=True
        )  # Reference image brightness at those predicted locations.
        tgt = stack_with_t[xyz[0], xyz[1], xyz[2]]  # Measured brightness at the original deformed voxel (with sphere).
        return torch.mean((tgt - pred) ** 2)  # Average squared difference over this minibatch (one scalar).

    # ----------------------------
    # Optimization loop
    # ----------------------------
    t0_opt = time.perf_counter()  # When the optimization phase started.
    total_vox = nx * ny * nz  # Total voxels: upper bound for random index sampling.
    ui_no_tqdm = (
        os.environ.get("FIM_UI_NO_TQDM", "0") == "1"
    )  # If set, hide the tqdm bar (e.g. for a GUI that parses logs instead).
    # If “print every N steps” exceeds the total number of steps, print every step so short
    # runs still show progress.
    progress_every = (
        int(args.progress_every) if args.progress_every is not None else 0
    )  # How often to print FIM_PROGRESS lines (0 = never).
    if progress_every > 0 and progress_every > args.num_iter:
        progress_every = 1
    # Progress bar without ETA (timing is noisy with stochastic batches/device variance).
    bar_fmt = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
    mse_trace = np.empty(args.num_iter, dtype=np.float64) if args.trace_mse else None
    for i in tqdm(range(args.num_iter), desc="optim", disable=ui_no_tqdm, bar_format=bar_fmt):
        rand_ind = torch.randint(total_vox, (args.batch_size,), device=device)
        mse = forward_model(rand_ind)
        if mse_trace is not None:
            mse_trace[i] = float(mse.detach().cpu().item())
        loss = mse
        if args.TV2_reg and args.TV2_reg > 0:
            loss = loss + (args.TV2_reg * total_variation_loss(r_deform_um))
        if args.Uz_penalty_weight > 0:
            # Saved Uz uses a sign flip later; penalizing negative *internal* coarse Uz
            # encourages downward motion in the saved field.
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
                    print(msg, file=sys.stderr, flush=True)
                else:
                    tqdm.write(msg, file=sys.stderr)

    t_opt_s = time.perf_counter() - t0_opt  # Seconds spent in the optimization loop.

    if args.trace_mse:
        assert mse_trace is not None
        np.save(out_dir / "optimization_mse.npy", mse_trace)
        print(f"Saved MSE trace array to: {out_dir / 'optimization_mse.npy'}", file=sys.stderr, flush=True)
        _save_mse_curve_png(out_dir, mse_trace)

    # ----------------------------
    # Upsample internal field; compose rotation + shift for saved u (meters)
    # ----------------------------
    r_np = r_deform_um.detach().cpu().numpy()
    nx_c, ny_c, nz_c = r_np.shape[0], r_np.shape[1], r_np.shape[2]
    zoom_factors = (nx / nx_c, ny / ny_c, nz / nz_c)
    upsample_order = args.upsample_order
    # Keep the unrotated zoomed field for optional comparison plots.
    Ux_um = scipy.ndimage.zoom(r_np[:, :, :, 0], zoom_factors, order=upsample_order)
    Uy_um = scipy.ndimage.zoom(r_np[:, :, :, 1], zoom_factors, order=upsample_order)
    Uz_um = scipy.ndimage.zoom(r_np[:, :, :, 2], zoom_factors, order=upsample_order)

    rot_np = axis_angle_rotmat(axis, angle).detach().cpu().numpy().astype(np.float64)
    shift_um = xyz_shift_global_um.detach().cpu().numpy().astype(np.float64)

    if args.save_comparison_figure:
        _save_comparison_figure(
            out_dir=out_dir,
            stack_with_xyz=stack_with_xyz,
            stack_without_xyz=stack_without_xyz,
            Ux_um=Ux_um,
            Uy_um=Uy_um,
            Uz_um=Uz_um,
            rot_np=rot_np,
            shift_um=shift_um,
            dxy_eff_um=dxy_eff_um,
            dz_eff_um=dz_eff_um,
        )

    # Add global shift (microns), then convert to meters.
    # The internal displacement maps deformed → reference (backward warp).
    # Negate to output the standard mechanics convention: reference → deformed
    # (u = x_deformed - X_reference), consistent with F = I + grad(u) in VFM.
    if args.remap_to_reference:
        r64 = r_np.astype(np.float64, copy=False)
        Ux_rot_c = rot_np[0, 0] * r64[:, :, :, 0] + rot_np[0, 1] * r64[:, :, :, 1] + rot_np[0, 2] * r64[:, :, :, 2]
        Uy_rot_c = rot_np[1, 0] * r64[:, :, :, 0] + rot_np[1, 1] * r64[:, :, :, 1] + rot_np[1, 2] * r64[:, :, :, 2]
        Uz_rot_c = rot_np[2, 0] * r64[:, :, :, 0] + rot_np[2, 1] * r64[:, :, :, 1] + rot_np[2, 2] * r64[:, :, :, 2]
        Ux_m = -(Ux_rot_c + shift_um[0]) * 1e-6
        Uy_m = -(Uy_rot_c + shift_um[1]) * 1e-6
        Uz_m = -(Uz_rot_c + shift_um[2]) * 1e-6
        x_cm, y_cm, z_cm = coarse_reference_axes_m(x_axis_m, y_axis_m, z_axis_m, nx_c, ny_c, nz_c)
        print(
            "Remapping displacement to reference (Lagrangian) on coarse grid, then upsampling u "
            f"(inverse map, {args.remap_interp}, max_iter={args.remap_max_iter}) ...",
            file=sys.stderr,
            flush=True,
        )
        Ux_m, Uy_m, Uz_m = remap_displacement_lagrangian_griddata(
            Ux_m,
            Uy_m,
            Uz_m,
            x_cm,
            y_cm,
            z_cm,
            method=args.remap_interp,
            max_iter=max(1, int(args.remap_max_iter)),
        )
        Ux_m = scipy.ndimage.zoom(Ux_m, zoom_factors, order=upsample_order)
        Uy_m = scipy.ndimage.zoom(Uy_m, zoom_factors, order=upsample_order)
        Uz_m = scipy.ndimage.zoom(Uz_m, zoom_factors, order=upsample_order)
    else:
        # Default path: rotate full-resolution internal field, then convert to saved convention.
        Ux_rot_um = rot_np[0, 0] * Ux_um + rot_np[0, 1] * Uy_um + rot_np[0, 2] * Uz_um
        Uy_rot_um = rot_np[1, 0] * Ux_um + rot_np[1, 1] * Uy_um + rot_np[1, 2] * Uz_um
        Uz_rot_um = rot_np[2, 0] * Ux_um + rot_np[2, 1] * Uy_um + rot_np[2, 2] * Uz_um
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
    print(f"Saving displacement fields (Ux/Uy/Uz) to: {out_dir}", file=sys.stderr, flush=True)
    for name in ("Ux.npy", "Uy.npy", "Uz.npy"):  # name: displacement component filename
        _unlink_if_exists(out_dir / name)
    np.save(out_dir / "Ux.npy", Ux_m.astype(np.float32, copy=False))  # write Ux as float32
    np.save(out_dir / "Uy.npy", Uy_m.astype(np.float32, copy=False))  # write Uy
    np.save(out_dir / "Uz.npy", Uz_m.astype(np.float32, copy=False))  # write Uz

    dxy_final_um = dxy_eff_um * ds_xy  # spacing after optional output stride
    dz_final_um = dz_eff_um * ds_z  # final Z spacing (μm)
    voxel_volume_m3 = float((dxy_final_um * 1e-6) * (dxy_final_um * 1e-6) * (dz_final_um * 1e-6))  # orthotope cell m³

    # Always save grid metadata so main_VFM.py can recreate grids if needed
    grid_meta = {  # JSON for inverse / VFM: final shape & spacings (not CLI defaults)
        "shape": [nx, ny, nz],
        "dxy_m": dxy_final_um * 1e-6,
        "dz_m": dz_final_um * 1e-6,
        "voxel_volume_m3": voxel_volume_m3,
    }
    (out_dir / "grid_params.json").write_text(json.dumps(grid_meta, indent=2) + "\n", encoding="utf-8")

    if args.skip_grids:
        print("Skipping X/Y/Z grids and volume_matrix (--skip_grids)", file=sys.stderr, flush=True)
        # Do not leave stale grid files from a previous run (would confuse inverse / main_VFM).
        for name in (
            "X.npy",
            "Y.npy",
            "Z.npy",
            "volume_matrix.npy",
        ):  # Remove old grid files so nothing is left from a previous run.
            _unlink_if_exists(out_dir / name)
    else:
        print("Saving X/Y/Z grids and volume_matrix ...", file=sys.stderr, flush=True)
        write_xyz_grids_m(out_dir, x_axis_m, y_axis_m, z_axis_m, shape=(nx, ny, nz), dtype=np.float32, chunk_z=8)
        write_volume_matrix_m3(out_dir, shape=(nx, ny, nz), voxel_volume_m3=voxel_volume_m3, dtype=np.float32)

    t_total_s = time.perf_counter() - t0_total  # Entire script wall time from start to finish.
    t_save_s = max(0.0, t_total_s - t_load_s - t_opt_s)  # Whatever time is left after load and optimize (save + misc).
    sec_per_iter = (t_opt_s / args.num_iter) if args.num_iter else float("nan")  # Average optimization step duration.
    print(
        (
            "Tracking step runtime (seconds): "
            f"load={t_load_s:.2f}, optimize={t_opt_s:.2f} ({sec_per_iter:.4f} s/iter), "
            f"save+post={t_save_s:.2f}, total={t_total_s:.2f}"
        ),
        file=sys.stderr,
        flush=True,
    )

    print(f"Saved outputs to: {out_dir}")
    print(f"Total running time: {t_total_s:.2f} s", file=sys.stderr, flush=True)

    # Build a copy-paste CLI command for reproducing this run.
    repro_parts = [
        "python",
        "fim/refactor/deformation_tracking.py",
        "--out_dir",
        shlex.quote(str(out_dir)),
        "--with_sphere",
        shlex.quote(str(args.with_sphere)),
        "--without_sphere",
        shlex.quote(str(args.without_sphere)),
        "--dxy_um",
        str(args.dxy_um),
        "--dz_um",
        str(args.dz_um),
        "--sphere_diameter_mm",
        str(args.sphere_diameter_mm),
        "--downsamp_xy",
        str(args.downsamp_xy),
        "--downsamp_z",
        str(args.downsamp_z),
        "--z_start",
        str(args.z_start),
        "--num_iter",
        str(args.num_iter),
        "--progress_every",
        str(args.progress_every),
        "--batch_size",
        str(args.batch_size),
        "--lr_shift",
        str(args.lr_shift),
        "--lr_axis",
        str(args.lr_axis),
        "--lr_angle",
        str(args.lr_angle),
        "--lr_deform",
        str(args.lr_deform),
        "--TV2_reg",
        str(args.TV2_reg),
        "--deform_downsample_factor_xy",
        str(args.deform_downsample_factor_xy),
        "--deform_downsample_factor_z",
        str(args.deform_downsample_factor_z),
        "--Uz_penalty_weight",
        str(args.Uz_penalty_weight),
        "--device",
        str(args.device),
        "--output_downsample_xy",
        str(args.output_downsample_xy),
        "--output_downsample_z",
        str(args.output_downsample_z),
        "--upsample_order",
        str(args.upsample_order),
        "--smooth_method",
        str(args.smooth_method),
        "--smooth_sigma",
        str(args.smooth_sigma),
        "--remap_interp",
        str(args.remap_interp),
        "--remap_max_iter",
        str(args.remap_max_iter),
    ]
    if args.z_end is not None:
        repro_parts.extend(["--z_end", str(args.z_end)])
    if args.lock_global_shift:
        repro_parts.append("--lock_global_shift")
    if args.skip_grids:
        repro_parts.append("--skip_grids")
    if args.save_comparison_figure:
        repro_parts.append("--save_comparison_figure")
    if args.trace_mse:
        repro_parts.append("--trace_mse")
    if args.remap_to_reference:
        repro_parts.append("--remap_to_reference")
    repro_command = " ".join(repro_parts)

    run_info_lines = [
        "deformation_tracking.py",
        f"out_dir={out_dir}",
        f"with_sphere={args.with_sphere}",
        f"without_sphere={args.without_sphere}",
        f"dxy_um={args.dxy_um}",
        f"dz_um={args.dz_um}",
        f"sphere_diameter_mm={args.sphere_diameter_mm}",
        f"downsamp_xy={args.downsamp_xy}",
        f"downsamp_z={args.downsamp_z}",
        f"z_start={args.z_start}",
        f"z_end={args.z_end}",
        f"num_iter={args.num_iter}",
        f"progress_every={args.progress_every}",
        f"batch_size={args.batch_size}",
        f"lr_shift={args.lr_shift}",
        f"lr_axis={args.lr_axis}",
        f"lr_angle={args.lr_angle}",
        f"lr_deform={args.lr_deform}",
        f"TV2_reg={args.TV2_reg}",
        f"lock_global_shift={args.lock_global_shift}",
        f"deform_downsample_factor_xy={args.deform_downsample_factor_xy}",
        f"deform_downsample_factor_z={args.deform_downsample_factor_z}",
        f"Uz_penalty_weight={args.Uz_penalty_weight}",
        f"device_flag={args.device}",
        f"output_downsample_xy={args.output_downsample_xy}",
        f"output_downsample_z={args.output_downsample_z}",
        f"upsample_order={args.upsample_order}",
        f"skip_grids={args.skip_grids}",
        f"smooth_method={args.smooth_method}",
        f"smooth_sigma={args.smooth_sigma}",
        f"save_comparison_figure={args.save_comparison_figure}",
        f"trace_mse={args.trace_mse}",
        f"remap_to_reference={args.remap_to_reference}",
        f"remap_interp={args.remap_interp}",
        f"remap_max_iter={args.remap_max_iter}",
        "---",
        f"shape_xyz_final={(nx, ny, nz)}",
        f"dxy_final_um={dxy_final_um}",
        f"dz_final_um={dz_final_um}",
        f"voxel_volume_m3={voxel_volume_m3}",
        f"xy_shift_px={xy_shift_px.tolist()}",
        f"z_shift_px={z_shift_px}",
        f"shift_um={shift_um.tolist()}",
        f"torch_device={device}",
        f"load_s={t_load_s:.4f}",
        f"optimize_s={t_opt_s:.4f}",
        f"total_running_time_s={t_total_s:.4f}",
    ]
    if args.trace_mse:
        run_info_lines.extend(
            [
                f"optimization_mse_npy={out_dir / 'optimization_mse.npy'}",
                f"mse_curve_png={out_dir / 'mse_curve.png'}",
            ]
        )
    if args.remap_to_reference:
        run_info_lines.append("remap_stage=coarse_then_upsample")
    run_info_lines.extend(["---", "repro_cli_command=", repro_command])
    (out_dir / "run_info.txt").write_text("\n".join(run_info_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
