"""Main script for running VFM inverse modeling with different material models"""

import logging
import os
import time

import numpy as np
from material_model import MaterialModel
from scipy.optimize import least_squares
from vws_models import central_differentiation, increase_matrix_size, map_elements_to_centraldiff, read_input_file

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define root path to test data
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))


def run_inverse_model(displacement_field, X, Y, Z, cube_size, initial_guess, bounds, material_model):
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
            return material_model.model_func(displacement_field, E1, E2, v12, v23, Gt, X, Y, Z, Force, cube_size, L, H)

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

    else:
        raise ValueError("Unknown material model type")

    logging.info("Running least squares for model: %s", name)
    result = least_squares(residual, initial_guess, bounds=bounds)
    return result.x


def load_common_fields(folder):
    """Loads nodal coordinates and displacement fields, computes deformation gradient tensors,
    and estimates cube element volume.

    Returns:
        X, Y, Z: 3D coordinate grids
        tensor_displacement_list: ndarray of shape (Nx, Ny, Nz, 3, 3)
        cube_size: scalar volume per voxel
    """
    X = np.load(f"{folder}/X.npy")
    Y = np.load(f"{folder}/Y.npy")
    Z = np.load(f"{folder}/Z.npy")
    Ux = np.load(f"{folder}/Ux.npy")
    Uy = np.load(f"{folder}/Uy.npy")
    Uz = np.load(f"{folder}/Uz.npy")

    X_e = increase_matrix_size(X)
    Y_e = increase_matrix_size(Y)
    Z_e = increase_matrix_size(Z)
    Ux_e = increase_matrix_size(Ux)
    Uy_e = increase_matrix_size(Uy)
    Uz_e = increase_matrix_size(Uz)

    grads = central_differentiation(Ux_e, Uy_e, Uz_e, X_e, Y_e, Z_e)
    tensor_displacement_list = map_elements_to_centraldiff(*grads)

    L_dim = np.mean(np.diff(X[0, :, 0]))
    W_dim = np.mean(np.diff(Y[:, 0, 0]))
    H_dim = np.mean(np.diff(Z[0, 0, :]))
    cube_size = L_dim * W_dim * H_dim

    return X, Y, Z, tensor_displacement_list, cube_size


def load_hgo_fields(folder):
    """Loads HGO-specific displacement, volume, and mesh dimensions."""
    X, Y, Z, tensor_displacement_list, cube_size = load_common_fields(folder)
    volume_matrix = np.load(f"{folder}/volume_matrix.npy")

    undeformed_nodes, connectivity = read_input_file(f"{folder}/350k.inp")

    L = abs(np.max(undeformed_nodes[:, 1]) - np.min(undeformed_nodes[:, 1]))
    W = abs(np.max(undeformed_nodes[:, 2]) - np.min(undeformed_nodes[:, 2]))
    H = abs(np.max(undeformed_nodes[:, 3]) - np.min(undeformed_nodes[:, 3]))

    return X, Y, Z, tensor_displacement_list, L, W, H, volume_matrix


if __name__ == "__main__":
    start_time = time.time()

    Model = "hgo"  # or 'hgo'

    if Model.lower() == "linear":
        # === Linear Model ===
        data_path = os.path.join(DATA_ROOT, "80um")
        X, Y, Z, disp_tensor, cube_size = load_common_fields(data_path)
        L = np.ceil((np.max(X) - np.min(X)) * 1e4) / 1e4
        W = np.ceil((np.max(Y) - np.min(Y)) * 1e4) / 1e4
        H = np.ceil((np.max(Z) - np.min(Z)) * 1e4) / 1e4

        linear_params = {
            "E1": 7000,
            "E2": 500,
            "v12": 0.49,
            "v23": 0.49,
            "Gt": 0.5e3,
            "L": L,
            "W": W,
            "H": H,
            "Force": 9.49803e-06,
        }
        linear_model = MaterialModel("linear", linear_params)

        initial_guess = [linear_params["E1"], linear_params["E2"]]
        bounds = ((2000, 500), (9000, 2500))

        # Run optimization
        result = run_inverse_model(disp_tensor, X, Y, Z, cube_size, initial_guess, bounds, linear_model)
        # logging.info(f"Linear model result: {result_linear}")
        logging.info("Linear model result: E1 = %.2f, E2 = %.2f", *result)

        # Run sensitivity analysis
        deviation = 0.05
        sens = linear_model.sensitivity_analysis(disp_tensor, X, Y, Z, cube_size, L, H, deviation)

    if Model.lower() == "hgo":
        # === HGO Model ===
        data_path = os.path.join(DATA_ROOT, "HGO")
        X, Y, Z, disp_tensor, L, W, H, volume_matrix = load_hgo_fields(data_path)

        hgo_params = {
            "C10": 500,
            "D1": 1e-5,
            "k1": 2000,
            "k2": 5,
            "kappa": 0.05,
            "L": L,
            "W": W,
            "H": H,
            "volume_matrix": volume_matrix,
            "Force": 1.20202e-05,
        }
        hgo_model = MaterialModel("hgo", hgo_params)

        initial_guess = [hgo_params["C10"], hgo_params["D1"], hgo_params["kappa"]]
        bounds = ((0, 1e-5, 0), (1000, 1e-3, 0.33))

        # Run optimization
        result_hgo = run_inverse_model(disp_tensor, X, Y, Z, volume_matrix, initial_guess, bounds, hgo_model)
        # logging.info(f"HGO model result: {result_hgo}")
        logging.info("HGO model result: C10 = %.2f, D1 = %.2e, kappa = %.3f", *result_hgo)

    logging.info(f"Total runtime: {time.time() - start_time}")
