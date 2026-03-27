"""Main script for running VFM inverse modeling with different material models"""

import argparse
import logging
import os
import time

import numpy as np
from scipy.optimize import least_squares

from fim.refactor.material_model import MaterialModel
from fim.refactor.vws_models import (
    central_differentiation,
    increase_matrix_size,
    map_elements_to_centraldiff,
    read_input_file,
    set_depth_indentation_from_Uz,
)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define root path to test data
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))
DEFAULT_PATHS = {
    "linear": os.path.join(DATA_ROOT, "80um"),
    "hgo": os.path.join(DATA_ROOT, "HGO"),
    "nh": os.path.join(DATA_ROOT, "NH"),
}

# CLI
parser = argparse.ArgumentParser(description="Run FIM Material Model Evaluation")
parser.add_argument("--model", type=str, default="linear", choices=["linear", "hgo", "nh"], help="Material model type")
parser.add_argument("--data_path", type=str, help="Path to input data folder")

# Shared across models
parser.add_argument("--force_n", type=float, default=4.18e-05, help="Applied indentation force (N)")

# Linear model parameters
parser.add_argument("--E1_init", type=float, default=15000, help="E1 initial guess (Pa)")
parser.add_argument("--E2_init", type=float, default=15000, help="E2 initial guess (Pa)")
parser.add_argument("--v12", type=float, default=0.49, help="Poisson ratio v12")
parser.add_argument("--v23", type=float, default=0.49, help="Poisson ratio v23")
parser.add_argument("--Gt", type=float, default=500, help="Shear modulus Gt (Pa)")

# HGO model parameters
parser.add_argument("--C10_init", type=float, default=500, help="C10 initial guess (Pa) — isotropic ground matrix")
parser.add_argument("--D1_init", type=float, default=1e-5, help="D1 initial guess — compressibility")
parser.add_argument("--k1", type=float, default=2000, help="Fiber stiffness k1 (Pa)")
parser.add_argument("--k2", type=float, default=5, help="Fiber nonlinearity k2")
parser.add_argument("--kappa_init", type=float, default=0.05, help="Fiber dispersion kappa (0=aligned, 1/3=isotropic)")

# NH model parameters (C10_init and D1_init are shared with HGO)

# Optional mesh file for L/W/H dimensions (falls back to coordinate grids)
parser.add_argument(
    "--mesh_file",
    type=str,
    default=None,
    help="Optional .inp mesh file for sample dimensions. If omitted, L/W/H are computed from coordinate grids.",
)

args = parser.parse_args()

# Resolve inputs
model_name = args.model
data_path = args.data_path if args.data_path else DEFAULT_PATHS[model_name]


def run_inverse_model(displacement_field, X, Y, Z, volume_matrix, initial_guess, bounds, material_model):
    """Optimize material parameters using least squares based on internal vs external virtual work."""
    name = material_model.name

    if name == "linear":

        def residual(x):
            E1, E2 = x
            v12 = material_model.get_parameter("v12")
            v23 = material_model.get_parameter("v23")
            Gt = material_model.get_parameter("Gt")
            L = material_model.get_parameter("L")
            H = material_model.get_parameter("H")
            Force = material_model.get_parameter("Force")
            return material_model.model_func(
                displacement_field, E1, E2, v12, v23, Gt, X, Y, Z, Force, volume_matrix, L, H
            )

    elif name == "hgo":

        def residual(x):
            C10, D1, kappa = x
            k1 = material_model.get_parameter("k1")
            k2 = material_model.get_parameter("k2")
            L = material_model.get_parameter("L")
            H = material_model.get_parameter("H")
            Force = material_model.get_parameter("Force")
            volume_matrix = material_model.get_parameter("volume_matrix")
            return material_model.model_func(
                displacement_field, X, Y, Z, C10, D1, k1, k2, kappa, volume_matrix, Force, L, H
            )

    elif name == "nh":

        def residual(x):
            C10, D1 = x
            L = material_model.get_parameter("L")
            H = material_model.get_parameter("H")
            Force = material_model.get_parameter("Force")
            volume_matrix = material_model.get_parameter("volume_matrix")
            return material_model.model_func(displacement_field, X, Y, Z, C10, D1, volume_matrix, Force, L, H)

    else:
        raise ValueError("Unknown material model type")

    logging.info("Running least squares for model: %s", name)
    result = least_squares(residual, initial_guess, bounds=bounds)
    return result.x


def _needs_xy_swap(X: np.ndarray) -> bool:
    """Detect if X varies along axis 0 (physics convention: X,Y,Z)
    instead of axis 1 (image convention: Y,X,Z that VFM code expects).

    Returns True when a swap of axes 0 and 1 is needed.
    """
    if X.ndim != 3 or X.shape[0] < 2 or X.shape[1] < 2:
        return False
    x_varies_ax0 = abs(float(X[1, 0, 0]) - float(X[0, 0, 0])) > 1e-30
    x_varies_ax1 = abs(float(X[0, 1, 0]) - float(X[0, 0, 0])) > 1e-30
    # X should vary along axis 1 for VFM.  If it varies along axis 0 instead → swap.
    return x_varies_ax0 and not x_varies_ax1


def _auto_crop_blank_z(Ux, Uy, Uz, *grids, threshold=1e-30):
    """Remove leading/trailing Z slices (axis 2) that are entirely zero.

    Blank slices create artificial gradient spikes at the zero/non-zero boundary
    which corrupt the strain calculation.

    Returns cropped copies of (Ux, Uy, Uz, *grids).
    """
    mag_per_z = np.sqrt(Ux**2 + Uy**2 + Uz**2).max(axis=(0, 1))
    nonzero = np.where(mag_per_z > threshold)[0]
    if nonzero.size == 0 or (nonzero[0] == 0 and nonzero[-1] == Ux.shape[2] - 1):
        return (Ux, Uy, Uz) + grids  # nothing to crop
    z0, z1 = int(nonzero[0]), int(nonzero[-1]) + 1
    logging.info(
        "Auto Z-crop: keeping Z slices [%d:%d] of %d (removed %d blank slices).",
        z0,
        z1,
        Ux.shape[2],
        Ux.shape[2] - (z1 - z0),
    )
    cropped = [a[:, :, z0:z1] for a in (Ux, Uy, Uz) + grids]
    return tuple(cropped)


def _create_grids_from_params(folder, shape):
    """Create X, Y, Z grids and volume_matrix from grid_params.json when .npy files are absent.

    The grids use the (X,Y,Z) physics convention (X along axis 0) — the caller handles
    any axis swap needed for VFM.
    """
    import json

    meta_path = os.path.join(folder, "grid_params.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    dxy_m = float(meta["dxy_m"])
    dz_m = float(meta["dz_m"])
    voxel_vol = float(meta["voxel_volume_m3"])

    nx, ny, nz = shape
    x_axis = np.arange(nx, dtype=np.float64) * dxy_m
    y_axis = np.arange(ny, dtype=np.float64) * dxy_m
    z_axis = np.arange(nz, dtype=np.float64) * dz_m

    X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    volume_matrix = np.full(shape, voxel_vol, dtype=np.float64)

    logging.info(
        "Created grids from grid_params.json: shape=%s, dxy=%.3e m, dz=%.3e m",
        shape,
        dxy_m,
        dz_m,
    )
    return X, Y, Z, volume_matrix


def load_common_fields(folder):
    """Loads nodal coordinates and displacement fields, computes deformation gradient tensors,
    and load volume_matrix.

    If X.npy is missing but grid_params.json exists (pipeline mode with --skip_grids),
    the coordinate grids and volume_matrix are created on-the-fly from the metadata.

    If X.npy exists but was left over from an older run (different shape than Ux/Uy/Uz),
    grids are rebuilt from grid_params.json when present. This avoids shape mismatches
    after tracking with --skip_grids when stale grid .npy files remain in the folder.

    Returns:
        X, Y, Z: 3D coordinate grids
        tensor_displacement_list: ndarray of shape (Nx, Ny, Nz, 3, 3)
        volume_matrix: per-voxel volume weights matrix
    """
    Ux = np.load(f"{folder}/Ux.npy")
    Uy = np.load(f"{folder}/Uy.npy")
    Uz = np.load(f"{folder}/Uz.npy")

    disp_shape = Ux.shape
    if Uy.shape != disp_shape or Uz.shape != disp_shape:
        raise ValueError(f"Ux/Uy/Uz shape mismatch: {Ux.shape}, {Uy.shape}, {Uz.shape}")

    x_path = os.path.join(folder, "X.npy")
    y_path = os.path.join(folder, "Y.npy")
    z_path = os.path.join(folder, "Z.npy")
    vol_path = os.path.join(folder, "volume_matrix.npy")
    grid_params_path = os.path.join(folder, "grid_params.json")

    def _load_arrays_from_disk():
        return (
            np.load(x_path),
            np.load(y_path),
            np.load(z_path),
            np.load(vol_path),
        )

    have_disk_grids = all(os.path.exists(p) for p in (x_path, y_path, z_path, vol_path))

    if have_disk_grids:
        X, Y, Z, volume_matrix = _load_arrays_from_disk()
        same_shape = (
            X.shape == disp_shape
            and Y.shape == disp_shape
            and Z.shape == disp_shape
            and volume_matrix.shape == disp_shape
        )
        if not same_shape:
            logging.warning(
                "Grid files in %s have shapes X%s Y%s Z%s vol%s but displacement fields are %s. "
                "Rebuilding coordinate grids from grid_params.json (typical cause: --skip_grids run "
                "with stale X/Y/Z/volume_matrix from an older run).",
                folder,
                X.shape,
                Y.shape,
                Z.shape,
                volume_matrix.shape,
                disp_shape,
            )
            if not os.path.exists(grid_params_path):
                raise ValueError(
                    f"Displacement arrays have shape {disp_shape} but on-disk grids do not match and "
                    f"{grid_params_path} is missing. Delete X.npy, Y.npy, Z.npy, volume_matrix.npy in "
                    f"{folder} or re-run deformation tracking so grid_params.json matches Ux.npy."
                )
            X, Y, Z, volume_matrix = _create_grids_from_params(folder, disp_shape)
    elif os.path.exists(grid_params_path):
        X, Y, Z, volume_matrix = _create_grids_from_params(folder, disp_shape)
    else:
        raise FileNotFoundError(
            f"Neither X.npy nor grid_params.json found in {folder}. "
            "Run deformation_tracking.py first, or provide coordinate grid files."
        )

    # Auto-detect axis convention.
    # central_differentiation expects X along axis 1, Y along axis 0 (image convention).
    # deformation_tracking.py outputs X along axis 0, Y along axis 1 (physics convention).
    # If needed, swap axes 0 and 1 so downstream code always sees the image convention.
    swap = _needs_xy_swap(X)
    if swap:
        logging.info("Detected (X,Y,Z) axis order — swapping axes 0,1 to match VFM convention (Y,X,Z).")
        X = np.swapaxes(X, 0, 1)
        Y = np.swapaxes(Y, 0, 1)
        Z = np.swapaxes(Z, 0, 1)
        Ux = np.swapaxes(Ux, 0, 1)
        Uy = np.swapaxes(Uy, 0, 1)
        Uz = np.swapaxes(Uz, 0, 1)
        volume_matrix = np.swapaxes(volume_matrix, 0, 1)

    # Remove leading/trailing blank Z slices to avoid artificial gradient spikes.
    Ux, Uy, Uz, X, Y, Z, volume_matrix = _auto_crop_blank_z(Ux, Uy, Uz, X, Y, Z, volume_matrix)

    # Update indentation depth used by virtual fields:
    set_depth_indentation_from_Uz(Uz)

    X_e = increase_matrix_size(X)
    Y_e = increase_matrix_size(Y)
    Z_e = increase_matrix_size(Z)
    Ux_e = increase_matrix_size(Ux)
    Uy_e = increase_matrix_size(Uy)
    Uz_e = increase_matrix_size(Uz)

    grads = central_differentiation(Ux_e, Uy_e, Uz_e, X_e, Y_e, Z_e)
    tensor_displacement_list = map_elements_to_centraldiff(*grads)

    return X, Y, Z, tensor_displacement_list, volume_matrix


def _get_dimensions(X, Y, Z, mesh_file=None):
    """Get sample dimensions L, W, H.

    If mesh_file is provided, reads an .inp file for precise node-based extents.
    Otherwise computes from coordinate grids.
    """
    if mesh_file:
        logging.info("Reading dimensions from mesh file: %s", mesh_file)
        nodes, _ = read_input_file(mesh_file)
        L = abs(np.max(nodes[:, 1]) - np.min(nodes[:, 1]))
        W = abs(np.max(nodes[:, 2]) - np.min(nodes[:, 2]))
        H = abs(np.max(nodes[:, 3]) - np.min(nodes[:, 3]))
    else:
        L = np.ceil((np.max(X) - np.min(X)) * 1e4) / 1e4
        W = np.ceil((np.max(Y) - np.min(Y)) * 1e4) / 1e4
        H = np.ceil((np.max(Z) - np.min(Z)) * 1e4) / 1e4
    return L, W, H


if __name__ == "__main__":
    start_time = time.time()

    logging.info(f"Using model: {model_name}, data_path: {data_path}")

    if model_name == "linear":
        # === Linear Model ===
        X, Y, Z, disp_tensor, volume_matrix = load_common_fields(data_path)
        L, W, H = _get_dimensions(X, Y, Z, args.mesh_file)

        linear_params = {
            "E1": args.E1_init,
            "E2": args.E2_init,
            "v12": args.v12,
            "v23": args.v23,
            "Gt": args.Gt,
            "L": L,
            "W": W,
            "H": H,
            "Force": args.force_n,
        }
        linear_model = MaterialModel("linear", linear_params)

        initial_guess = [args.E1_init, args.E2_init]
        bounds = ((1000, 1000), (25000, 25000))

        # Print initial values and bounds
        logging.info("Linear model initial values: E1 = %.2f, E2 = %.2f", *initial_guess)
        logging.info(
            "Linear model bounds: E1 = [%.2f, %.2f], E2 = [%.2f, %.2f]",
            bounds[0][0],
            bounds[1][0],
            bounds[0][1],
            bounds[1][1],
        )

        # Run optimization
        result = run_inverse_model(disp_tensor, X, Y, Z, volume_matrix, initial_guess, bounds, linear_model)
        logging.info("Linear model optimized: E1 = %.2f, E2 = %.2f", *result)

        # Run sensitivity analysis
        deviation = 0.05
        linear_model.sensitivity_analysis_linear(disp_tensor, X, Y, Z, volume_matrix, L, H, deviation)

    elif model_name == "hgo":
        # === HGO Model ===
        X, Y, Z, disp_tensor, volume_matrix = load_common_fields(data_path)
        L, W, H = _get_dimensions(X, Y, Z, args.mesh_file)

        hgo_params = {
            "C10": args.C10_init,
            "D1": args.D1_init,
            "k1": args.k1,
            "k2": args.k2,
            "kappa": args.kappa_init,
            "L": L,
            "W": W,
            "H": H,
            "volume_matrix": volume_matrix,
            "Force": args.force_n,
        }
        hgo_model = MaterialModel("hgo", hgo_params)

        initial_guess = [args.C10_init, args.D1_init, args.kappa_init]
        bounds = ((0, 1e-5, 0), (1000, 1e-3, 0.33))

        # Print initial values and bounds
        logging.info(
            "HGO model initial values: C10 = %.2f, D1 = %.2e, k1 = %.2f, k2 = %.2f, kappa = %.3f",
            hgo_params["C10"],
            hgo_params["D1"],
            hgo_params["k1"],
            hgo_params["k2"],
            hgo_params["kappa"],
        )
        logging.info(
            "HGO model bounds: C10 = [%.2f, %.2f], D1 = [%.2e, %.2e], kappa = [%.3f, %.3f]",
            bounds[0][0],
            bounds[1][0],
            bounds[0][1],
            bounds[1][1],
            bounds[0][2],
            bounds[1][2],
        )
        logging.info("HGO model fixed: k1 = %.2f, k2 = %.2f", hgo_params["k1"], hgo_params["k2"])

        # Run optimization
        result_hgo = run_inverse_model(disp_tensor, X, Y, Z, volume_matrix, initial_guess, bounds, hgo_model)
        logging.info("HGO model optimized: C10 = %.2f, D1 = %.2e, kappa = %.3f", *result_hgo)

        # Update model parameters with optimized values for sensitivity analysis
        hgo_model.params["C10"] = result_hgo[0]
        hgo_model.params["D1"] = result_hgo[1]
        hgo_model.params["kappa"] = result_hgo[2]

        # Run sensitivity analysis with optimized parameters
        deviation = 0.05
        hgo_model.sensitivity_analysis_hgo(disp_tensor, X, Y, Z, volume_matrix, L, H, deviation)

    elif model_name == "nh":
        # === NH Model ===
        X, Y, Z, disp_tensor, volume_matrix = load_common_fields(data_path)
        L, W, H = _get_dimensions(X, Y, Z, args.mesh_file)
        nh_params = {
            "C10": args.C10_init,
            "D1": args.D1_init,
            "L": L,
            "W": W,
            "H": H,
            "volume_matrix": volume_matrix,
            "Force": args.force_n,
        }
        nh_model = MaterialModel("nh", nh_params)

        initial_guess = [args.C10_init, args.D1_init]
        bounds = ((100, 1e-6), (1000, 1e-3))

        # Print initial values and bounds
        logging.info("NH model initial values: C10 = %.2f, D1 = %.2e", *initial_guess)
        logging.info(
            "NH model bounds: C10 = [%.2f, %.2f], D1 = [%.2e, %.2e]",
            bounds[0][0],
            bounds[1][0],
            bounds[0][1],
            bounds[1][1],
        )

        # Run optimization
        result_nh = run_inverse_model(disp_tensor, X, Y, Z, volume_matrix, initial_guess, bounds, nh_model)
        logging.info("NH model optimized: C10 = %.2f, D1 = %.2e", *result_nh)

        # Update model parameters with optimized values for sensitivity analysis
        nh_model.params["C10"] = result_nh[0]
        nh_model.params["D1"] = result_nh[1]

        # Run sensitivity analysis with optimized parameters
        deviation = 0.05
        nh_model.sensitivity_analysis_nh(disp_tensor, X, Y, Z, volume_matrix, L, H, deviation)

    # Flush stdout so sensitivity matrix prints complete before the final timing line
    import sys

    sys.stdout.flush()
    logging.info(f"Inverse step runtime: {time.time() - start_time:.1f} seconds")
