"""Disk and figure I/O helpers for deformation tracking.

This module is intentionally free of :mod:`torch`. Only functions that touch
files (TIFFs, ``.npy`` memmaps, PNG figures) or derive metrics from 2D arrays
live here so they can be reused by non-GPU scripts and analysis notebooks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.ndimage
from numpy.lib.format import open_memmap
from skimage import io
from skimage.metrics import structural_similarity as structural_similarity_2d


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
    prefix, rest = name.split("_", 1)  # Split "hex_rest.tif" so we can drop a long upload id in prefix.
    if len(prefix) == 32 and all(c in "0123456789abcdef" for c in prefix.lower()):
        return rest
    return name


def load_tiff_zyx_to_xyz(path: str, z_start: int, z_end: int | None, downsamp_xy: int, downsamp_z: int) -> np.ndarray:
    """Load TIFF as (Z,Y,X), crop/downsample, transpose to ``(X,Y,Z)`` for the pipeline.

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
    data_range = max(dmax - dmin, 1e-12)  # Overall contrast; SSIM needs this so "similar" is meaningful.
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
    )  # We fill this: "what the reference would look like" after the warp.
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
