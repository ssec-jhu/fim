"""Optimization-based 3D deformation tracking for volumetric TIFF stacks.

This module now contains only the CLI entry point and the orchestration in
:func:`main`. The reusable pieces have been split into three focused modules
for easier reuse and testing:

- :mod:`fim.refactor.tracking_io`     — TIFF / ``.npy`` I/O and figure output.
- :mod:`fim.refactor.tracking_optim`  — torch-based forward model and the
  initial-shift estimator used at startup.
- :mod:`fim.refactor.tracking_remap`  — post-processing: smoothing and the
  Lagrangian reference-grid remap.

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
from tqdm.auto import tqdm

from fim.refactor import tracking_io as tio
from fim.refactor import tracking_optim as topt
from fim.refactor import tracking_remap as trem

# ----------------------------
# Defaults: if you do not pass paths on the command line, these sample stacks are used.
# ----------------------------
_SIM_TIFF_DIR = (
    Path(__file__).resolve().parent.parent / "test_data" / "simulate"
)  # Folder that ships with small demo TIFFs.
DEFAULT_WITHOUT_SPHERE = str(_SIM_TIFF_DIR / "ref_image.tif")  # Default "no sphere" (reference) image path.
DEFAULT_WITH_SPHERE = str(_SIM_TIFF_DIR / "def_image.tif")  # Default "with sphere" (deformed) image path.


def build_parser() -> argparse.ArgumentParser:
    """Build the ``deformation_tracking`` CLI parser.

    Extracted from :func:`main` so the flag surface is testable in isolation
    (defaults, choices, presence) without launching the full pipeline. Mirrors
    the pattern used in :mod:`fim.refactor.main_VFM`.
    """
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
        default=1,
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
    p.add_argument("--batch_size", type=int, default=750000, help="Random samples per iteration")
    p.add_argument("--lr_shift", type=float, default=5e-2, help="Learning rate for global shift")
    p.add_argument("--lr_axis", type=float, default=1e-3, help="Learning rate for rotation axis")
    p.add_argument("--lr_angle", type=float, default=1e-5, help="Learning rate for rotation angle")
    p.add_argument("--lr_deform", type=float, default=0.3, help="Learning rate for deformation field")
    p.add_argument("--TV2_reg", type=float, default=30, help="TV2 regularization weight (0 disables)")
    p.add_argument("--lock_global_shift", action="store_true", help="Lock global shift at initial estimate")

    p.add_argument("--deform_downsample_factor_xy", type=int, default=10, help="Coarse deformation grid factor XY")
    p.add_argument("--deform_downsample_factor_z", type=int, default=2, help="Coarse deformation grid factor Z")
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
    return p


def main() -> None:
    """CLI: load volumes, optimize MSE (backward warp), compose motion, write U*.npy and sidecar files."""
    args = build_parser().parse_args()  # All flags the user passed (or defaults).

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
    print(f"Loading TIFF (with sphere): {tio._display_input_name(args.with_sphere)}", file=sys.stderr, flush=True)
    stack_with_xyz = tio.load_tiff_zyx_to_xyz(
        args.with_sphere, args.z_start, args.z_end, args.downsamp_xy, args.downsamp_z
    )  # Deformed specimen (with sphere), as X,Y,Z array, possibly cropped/downsampled.
    print(
        f"Loading TIFF (without sphere): {tio._display_input_name(args.without_sphere)}",
        file=sys.stderr,
        flush=True,
    )
    stack_without_xyz = tio.load_tiff_zyx_to_xyz(  # Reference (no sphere), same preprocessing.
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
    xy_shift_px, z_shift_px = topt.estimate_initial_shift(
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
        xyz = topt._unravel_flat_indices(
            rand_ind, stack_with_t.shape
        )  # Three integer index tensors: where we read the "data" voxel (with sphere).

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
        local_def = topt.interpolated_prediction(
            x_def, y_def, z_def, r_deform_um, trilinear_interp=True
        )  # Local displacement (Ux,Uy,Uz) in microns at each sample.

        rot = topt.axis_angle_rotmat(axis, angle)  # 3×3 rotation from current axis and angle.
        r_deformed = (delta_r + local_def) @ rot + xyz_shift_global_um[
            None, :
        ]  # After adding local motion, rotating, and shifting: where we land in microns.

        # Convert microns to fractional voxel indices in the reference (without-sphere)
        # volume (origin at volume center).
        x_float = r_deformed[:, 0] / dxy_eff_um + (nx / 2)  # Reference X index (can be between voxels).
        y_float = r_deformed[:, 1] / dxy_eff_um + (ny / 2)  # Reference Y index.
        z_float = r_deformed[:, 2] / dz_eff_um + (nz / 2)  # Reference Z index.

        pred = topt.interpolated_prediction(
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
    # If "print every N steps" exceeds the total number of steps, print every step so short
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
            loss = loss + (args.TV2_reg * topt.total_variation_loss(r_deform_um))
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
        tio._save_mse_curve_png(out_dir, mse_trace)

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

    rot_np = topt.axis_angle_rotmat(axis, angle).detach().cpu().numpy().astype(np.float64)
    shift_um = xyz_shift_global_um.detach().cpu().numpy().astype(np.float64)

    if args.save_comparison_figure:
        tio._save_comparison_figure(
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
        x_cm, y_cm, z_cm = trem.coarse_reference_axes_m(x_axis_m, y_axis_m, z_axis_m, nx_c, ny_c, nz_c)
        print(
            "Remapping displacement to reference (Lagrangian) on coarse grid, then upsampling u "
            f"(inverse map, {args.remap_interp}, max_iter={args.remap_max_iter}) ...",
            file=sys.stderr,
            flush=True,
        )
        Ux_m, Uy_m, Uz_m = trem.remap_displacement_lagrangian_griddata(
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
        Ux_m = trem.smooth_displacement_field(Ux_m, args.smooth_method, args.smooth_sigma)
        Uy_m = trem.smooth_displacement_field(Uy_m, args.smooth_method, args.smooth_sigma)
        Uz_m = trem.smooth_displacement_field(Uz_m, args.smooth_method, args.smooth_sigma)

    # Save U fields (meters); remove old files first so reruns always replace (shape may change).
    print(f"Saving displacement fields (Ux/Uy/Uz) to: {out_dir}", file=sys.stderr, flush=True)
    for name in ("Ux.npy", "Uy.npy", "Uz.npy"):  # name: displacement component filename
        tio._unlink_if_exists(out_dir / name)
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
            tio._unlink_if_exists(out_dir / name)
    else:
        print("Saving X/Y/Z grids and volume_matrix ...", file=sys.stderr, flush=True)
        tio.write_xyz_grids_m(out_dir, x_axis_m, y_axis_m, z_axis_m, shape=(nx, ny, nz), dtype=np.float32, chunk_z=8)
        tio.write_volume_matrix_m3(out_dir, shape=(nx, ny, nz), voxel_volume_m3=voxel_volume_m3, dtype=np.float32)

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
