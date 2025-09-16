"""
VWS Model Functions (XYZ convention)
------------------------------------
This module implements full-field virtual fields method (VWS) computations for supported
material models. Refactored for strict **(x, y, z)** ordering across all functions and arrays.

Original function purposes:
- `U_star_*` families: define virtual displacement fields used in the Virtual Fields Method.
- `*_dev*` versions: spatial derivatives of the virtual displacement fields.
- `increase_matrix_size`: utility to pad 3D grids by replication for differentiation.
- `read_input_file`: parse Abaqus-style mesh input (nodes, connectivity).
- `central_differentiation`: compute central finite differences of displacement fields.
- `map_elements_to_centraldiff`: assemble gradient tensors from scalar derivatives.
- `calculate_VWS_linear/hgo/nh`: material models (linear orthotropic, Holzapfel-Gasser-Ogden,
Neo-Hookean).
- `calculate_VWS_virtual_work`: compute internal/external virtual work with predefined fields.
- `sensitivity_full`: finite-difference sensitivity analysis for the linear model.

Axis convention in this refactored version:
axis=0 → x, axis=1 → y, axis=2 → z.
"""

from __future__ import annotations

import numpy as np

# --- Indentation/sample constants ------------------------------------------------
depth_indentation = 3.2e-05
sphere_radius = 5e-4
contact_radius = np.sqrt(depth_indentation * sphere_radius)


# --- Virtual displacement primitives (all in xyz order) --------------------------


def U_star_z_cos(x, y, z, L, H):
    t = z / H
    a_0 = contact_radius * 2
    a = t * a_0
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    return np.where(
        d <= a, c * t, np.where((d > a) & (d < L / 2), c * t * np.cos(np.pi / 2 * (d - a) / (L / 2 - a)), 0)
    )


def U_star_z_pw(x, y, z, L, H):
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    a_0 = contact_radius * 1.25
    t = z / H
    a = a_0 * t
    return np.where(d <= a, c * t, np.where((d > a) & (d < L / 2), c * t * (L / 2 - d) / (L / 2 - a), 0))


def U_star_x_para(x, y, z, L, H):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H
    return np.where(d <= L / 2, c * x * (L / 2 - d) * t, 0)


def U_star_y_para(x, y, z, L, H):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H
    return np.where(d <= L / 2, c * y * (L / 2 - d) * t, 0)


def U_star_x_sin(x, y, z, L, H):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H
    return np.where((d > 0) & (d < L / 2), c * x / d * t * np.sin(2 * np.pi * d / (L / 2)), 0)


def U_star_y_sin(x, y, z, L, H):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H
    return np.where((d > 0) & (d < L / 2), c * y / d * t * np.sin(2 * np.pi * d / (L / 2)), 0)


# --- Derivatives of the virtual fields (xyz order) -------------------------------


def U_star_z_cos_devX(x, y, z, L, H):
    t = z / H
    a_0 = contact_radius * 2
    a = t * a_0
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    safe_d = np.where(d == 0, 1e-10, d)
    expr = -c * t * np.sin(np.pi / 2 * (d - a) / (L / 2 - a)) * (np.pi / (2 * (L / 2 - a))) * x / safe_d
    return np.where((d > a) & (d < L / 2), expr, 0)


def U_star_z_cos_devY(x, y, z, L, H):
    t = z / H
    a_0 = contact_radius * 2
    a = t * a_0
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    safe_d = np.where(d == 0, 1e-10, d)
    expr = -c * t * np.sin(np.pi / 2 * (d - a) / (L / 2 - a)) * (np.pi / (2 * (L / 2 - a))) * y / safe_d
    return np.where((d > a) & (d < L / 2), expr, 0)


def U_star_z_cos_devZ(x, y, z, L, H):
    t = z / H
    a_0 = contact_radius * 2
    a = t * a_0
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    expr1 = c / H
    expr2 = c * np.cos(np.pi / 2 * (d - a) / (L / 2 - a)) / H + c * t * (
        -np.sin(np.pi / 2 * (d - a) / (L / 2 - a))
    ) * np.pi / 2 * (d - L / 2) * a_0 / (H**2 * (L / 2 - a) ** 2)
    return np.where(d <= a, expr1, np.where((d > a) & (d < L / 2), expr2, 0))


def U_star_z_pw_devX(x, y, z, L, H):
    c = 5e-5
    a_0 = contact_radius * 1.25
    t = z / H
    a = a_0 * t
    d = np.sqrt(x**2 + y**2)
    safe_d = np.where(d == 0, 1e-10, d)
    expr = -2 * c * x * t / ((L - 2 * a) * safe_d)
    return np.where((d > a) & (d < L / 2), expr, 0)


def U_star_z_pw_devY(x, y, z, L, H):
    c = 5e-5
    a_0 = contact_radius * 1.25
    t = z / H
    a = a_0 * t
    d = np.sqrt(x**2 + y**2)
    safe_d = np.where(d == 0, 1e-10, d)
    expr = -2 * c * y * t / ((L - 2 * a) * safe_d)
    return np.where((d > a) & (d < L / 2), expr, 0)


def U_star_z_pw_devZ(x, y, z, L, H):
    c = 5e-5
    a_0 = contact_radius * 1.25
    d = np.sqrt(x**2 + y**2)
    t = z / H
    a = a_0 * t
    expr1 = c / H
    expr2 = c * (L / 2 - d) / (H * (L / 2 - a)) + z * c * (L / 2 - d) * a_0 / (H**2 * (L / 2 - a) ** 2)
    return np.where(d <= a, expr1, np.where((d > a) & (d < L / 2), expr2, 0))


def U_star_x_para_devX(x, y, z, L, H):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H
    safe_d = np.where(d == 0, 1e-10, d)
    expr = c * ((L / 2 - d) - x**2 / safe_d) * t
    return np.where(d <= L / 2, expr, 0)


def U_star_x_para_devY(x, y, z, L, H):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H
    safe_d = np.where(d == 0, 1e-10, d)
    expr = -c * x * y * t / safe_d
    return np.where(d <= L / 2, expr, 0)


def U_star_x_para_devZ(x, y, z, L, H):
    c = 10
    d = np.sqrt(x**2 + y**2)
    expr = c * x * (L / 2 - d) / H
    return np.where(d <= L / 2, expr, 0)


def U_star_y_para_devX(x, y, z, L, H):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H
    safe_d = np.where(d == 0, 1e-10, d)
    expr = -c * y * x * t / safe_d
    return np.where(d <= L / 2, expr, 0)


def U_star_y_para_devY(x, y, z, L, H):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H
    safe_d = np.where(d == 0, 1e-10, d)
    expr = c * ((L / 2 - d) - y**2 / safe_d) * t
    return np.where(d <= L / 2, expr, 0)


def U_star_y_para_devZ(x, y, z, L, H):
    c = 10
    d = np.sqrt(x**2 + y**2)
    expr = c * y * (L / 2 - d) / H
    return np.where(d <= L / 2, expr, 0)


def U_star_x_sin_devX(x, y, z, L, H):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H
    expr1 = c * t * np.pi / (L / 2)
    sin_term = np.sin(2 * np.pi * d / (L / 2))
    cos_term = np.cos(2 * np.pi * d / (L / 2))
    expr2 = c * t * ((sin_term / d) - (x**2 * sin_term / d**3) + (x**2 * cos_term * 2 * np.pi / (d**2 * (L / 2))))
    return np.where(d == 0, expr1, np.where(d < L / 2, expr2, 0))


def U_star_x_sin_devY(x, y, z, L, H):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H
    sin_term = np.sin(2 * np.pi * d / (L / 2))
    cos_term = np.cos(2 * np.pi * d / (L / 2))
    expr = c * x * t * ((-y / d**3) * sin_term + (cos_term * 2 * np.pi * y / (d**2 * (L / 2))))
    return np.where((d > 0) & (d < L / 2), expr, 0)


def U_star_x_sin_devZ(x, y, z, L, H):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    expr = c * x / d * (1 / H) * np.sin(2 * np.pi * d / (L / 2))
    return np.where((d > 0) & (d < L / 2), expr, 0)


def U_star_y_sin_devX(x, y, z, L, H):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H
    sin_term = np.sin(2 * np.pi * d / (L / 2))
    cos_term = np.cos(2 * np.pi * d / (L / 2))
    expr = c * y * t * ((-x / d**3) * sin_term + (cos_term * 2 * np.pi * x / (d**2 * (L / 2))))
    return np.where((d > 0) & (d < L / 2), expr, 0)


def U_star_y_sin_devY(x, y, z, L, H):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H
    expr1 = c * t * np.pi / (L / 2)
    sin_term = np.sin(2 * np.pi * d / (L / 2))
    cos_term = np.cos(2 * np.pi * d / (L / 2))
    expr2 = c * t * ((sin_term / d) - (y**2 * sin_term / d**3) + (y**2 * cos_term * 2 * np.pi / (d**2 * (L / 2))))
    return np.where(d == 0, expr1, np.where(d < L / 2, expr2, 0))


def U_star_y_sin_devZ(x, y, z, L, H):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    expr = c * y / d * (1 / H) * np.sin(2 * np.pi * d / (L / 2))
    return np.where((d > 0) & (d < L / 2), expr, 0)


# --- Volumetric variant used by HGO/NH modes ------------------------------------


def U_star_z_pw_vol(x, y, z, L, H):
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    a_0 = contact_radius * 2
    t = z / H
    a = a_0
    k = (c / 2 * (L / 4 - a) - c * L / 4) / ((L / 4 - a) / 2 + L / 8)

    # Scalar path (e.g., EVW at (0,0,H))
    if np.ndim(d) == 0:
        if d <= a:
            return c * t
        elif d <= (L / 4):
            return t * (c + (k - c) * (d - a) / (L / 4 - a))
        elif d < (L / 2):
            return t * (k - 4 * k * (d - L / 4) / L)
        else:
            return 0.0

    # Array path (vectorized)
    result = np.zeros_like(d)
    mask1 = d <= a
    mask2 = (d > a) & (d <= (L / 4))
    mask3 = (d > (L / 4)) & (d < L / 2)

    result[mask1] = c * t[mask1]
    result[mask2] = t[mask2] * (c + (k - c) * (d[mask2] - a) / (L / 4 - a))
    result[mask3] = t[mask3] * (k - 4 * k * (d[mask3] - L / 4) / L)
    return result


def U_star_z_pw_vol_devX(x, y, z, L, H):
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    a = contact_radius * 2
    t = z / H
    k = (c / 2 * (L / 4 - a) - c * L / 4) / ((L / 4 - a) / 2 + L / 8)

    # scalar path
    if np.ndim(d) == 0:
        if d <= a:
            return 0.0
        if d <= (L / 4):
            return t * ((k - c) * (x / d) / (L / 4 - a))
        if d < (L / 2):
            return t * (-4 * k * (x / d) / L)
        return 0.0

    # vectorized path
    out = np.zeros_like(d)
    m2 = (d > a) & (d <= (L / 4))
    m3 = (d > (L / 4)) & (d < L / 2)
    out[m2] = t[m2] * ((k - c) * (x[m2] / d[m2]) / (L / 4 - a))
    out[m3] = t[m3] * (-4 * k * (x[m3] / d[m3]) / L)
    return out


def U_star_z_pw_vol_devY(x, y, z, L, H):
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    a = contact_radius * 2
    t = z / H
    k = (c / 2 * (L / 4 - a) - c * L / 4) / ((L / 4 - a) / 2 + L / 8)

    if np.ndim(d) == 0:
        if d <= a:
            return 0.0
        if d <= (L / 4):
            return t * ((k - c) * (y / d) / (L / 4 - a))
        if d < (L / 2):
            return t * (-4 * k * (y / d) / L)
        return 0.0

    out = np.zeros_like(d)
    m2 = (d > a) & (d <= (L / 4))
    m3 = (d > (L / 4)) & (d < L / 2)
    out[m2] = t[m2] * ((k - c) * (y[m2] / d[m2]) / (L / 4 - a))
    out[m3] = t[m3] * (-4 * k * (y[m3] / d[m3]) / L)
    return out


def U_star_z_pw_vol_devZ(x, y, z, L, H):
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    a = contact_radius * 2
    k = (c / 2 * (L / 4 - a) - c * L / 4) / ((L / 4 - a) / 2 + L / 8)

    if np.ndim(d) == 0:
        if d <= a:
            return c / H
        if d <= (L / 4):
            return (1 / H) * (c + (k - c) * (d - a) / (L / 4 - a))
        if d < (L / 2):
            return (1 / H) * (k - 4 * k * (d - L / 4) / L)
        return 0.0

    out = np.zeros_like(d)
    m1 = d <= a
    m2 = (d > a) & (d <= (L / 4))
    m3 = (d > (L / 4)) & (d < L / 2)
    out[m1] = c / H
    out[m2] = (1 / H) * (c + (k - c) * (d[m2] - a) / (L / 4 - a))
    out[m3] = (1 / H) * (k - 4 * k * (d[m3] - L / 4) / L)
    return out


# --- Utilities -------------------------------------------------------------------


def increase_matrix_size(matrix: np.ndarray) -> np.ndarray:
    """Pad a 3D (x,y,z) matrix by one voxel on each face by replication."""
    nx, ny, nz = matrix.shape
    out = np.zeros((nx + 2, ny + 2, nz + 2), dtype=matrix.dtype)
    out[1:-1, 1:-1, 1:-1] = matrix

    # Replicate borders
    out[0, :, :] = out[1, :, :]
    out[-1, :, :] = out[-2, :, :]
    out[:, 0, :] = out[:, 1, :]
    out[:, -1, :] = out[:, -2, :]
    out[:, :, 0] = out[:, :, 1]
    out[:, :, -1] = out[:, :, -2]
    return out


def read_input_file(file_path):
    nodes, connectivity = [], []
    in_tissue = in_node = in_elem = False

    with open(file_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("*"):
                low = line.lower().replace(" ", "")
                if low.startswith("*part") and "name=tissue" in low:
                    in_tissue, in_node, in_elem = True, False, False
                    continue
                if low.startswith("*endpart"):
                    in_tissue = in_node = in_elem = False
                    continue
                if in_tissue and low.startswith("*node"):
                    in_node, in_elem = True, False
                    continue
                if in_tissue and low.startswith("*element"):
                    in_node, in_elem = False, True
                    continue
                # other star lines end sub-section but keep in_tissue as-is
                in_node = in_elem = False
                continue

            if in_tissue and in_node:
                vals = [v.strip() for v in line.split(",")]
                if len(vals) >= 4:
                    nid = int(vals[0])
                    x, y, z = map(float, vals[1:4])
                    nodes.append([nid, x, y, z])
                continue

            if in_tissue and in_elem:
                vals = [v.strip() for v in line.split(",")]
                # Support C3D4 (5 ints) and C3D8 (9 ints)
                if len(vals) in (5, 9):
                    try:
                        connectivity.append([int(v) for v in vals])
                    except ValueError:
                        pass
                continue

    nodes = np.asarray(nodes, dtype=float)
    connectivity = np.asarray(connectivity, dtype=int)

    if nodes.size:
        nodes[:, 1] -= nodes[:, 1].min()
        nodes[:, 2] -= nodes[:, 2].min()
        nodes[:, 3] -= nodes[:, 3].min()

    return nodes, connectivity


# --- Core: central differences in strict (x,y,z) order ---------------------------


def central_differentiation(Ux, Uy, Uz, X, Y, Z):
    """Central finite differences for displacement fields on an (x,y, z) grid.

    Inputs are 3D arrays of identical shape (Nx, Ny, Nz). Returns nine arrays
    (all shape (Nx-2, Ny-2, Nz-2)) corresponding to \n
        [dUx/dx, dUy/dx, dUz/dx,
         dUx/dy, dUy/dy, dUz/dy,
         dUx/dz, dUy/dz, dUz/dz]

    Axis mapping: axis0=x, axis1=y, axis2=z.
    """
    if not (Ux.shape == Uy.shape == Uz.shape == X.shape == Y.shape == Z.shape):
        raise ValueError("All input arrays must have the same shape.")

    # Spacings along each axis (x,y,z)
    dx = X[2:, 1:-1, 1:-1] - X[:-2, 1:-1, 1:-1]
    dy = Y[1:-1, 2:, 1:-1] - Y[1:-1, :-2, 1:-1]
    dz = Z[1:-1, 1:-1, 2:] - Z[1:-1, 1:-1, :-2]

    eps = 1e-12
    dx = np.where(np.abs(dx) > eps, dx, np.sign(dx) * eps + eps)
    dy = np.where(np.abs(dy) > eps, dy, np.sign(dy) * eps + eps)
    dz = np.where(np.abs(dz) > eps, dz, np.sign(dz) * eps + eps)

    # Derivatives wrt x (axis 0)
    dUx_dx = (Ux[2:, 1:-1, 1:-1] - Ux[:-2, 1:-1, 1:-1]) / dx
    dUy_dx = (Uy[2:, 1:-1, 1:-1] - Uy[:-2, 1:-1, 1:-1]) / dx
    dUz_dx = (Uz[2:, 1:-1, 1:-1] - Uz[:-2, 1:-1, 1:-1]) / dx

    # Derivatives wrt y (axis 1)
    dUx_dy = (Ux[1:-1, 2:, 1:-1] - Ux[1:-1, :-2, 1:-1]) / dy
    dUy_dy = (Uy[1:-1, 2:, 1:-1] - Uy[1:-1, :-2, 1:-1]) / dy
    dUz_dy = (Uz[1:-1, 2:, 1:-1] - Uz[1:-1, :-2, 1:-1]) / dy

    # Derivatives wrt z (axis 2)
    dUx_dz = (Ux[1:-1, 1:-1, 2:] - Ux[1:-1, 1:-1, :-2]) / dz
    dUy_dz = (Uy[1:-1, 1:-1, 2:] - Uy[1:-1, 1:-1, :-2]) / dz
    dUz_dz = (Uz[1:-1, 1:-1, 2:] - Uz[1:-1, 1:-1, :-2]) / dz

    return dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz


def map_elements_to_centraldiff(dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz):
    """Pack scalar gradients into a (Nx,Ny,Nz,3,3) displacement-gradient tensor.
    All inputs must share the same (Nx,Ny,Nz) shape.
    """
    shape = dUx_dx.shape
    for arr in [dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz]:
        assert arr.shape == shape, "All central difference fields must have the same shape"

    nx, ny, nz = shape
    tensor = np.zeros((nx, ny, nz, 3, 3), dtype=np.float64)

    tensor[..., 0, 0] = dUx_dx
    tensor[..., 0, 1] = dUx_dy
    tensor[..., 0, 2] = dUx_dz
    tensor[..., 1, 0] = dUy_dx
    tensor[..., 1, 1] = dUy_dy
    tensor[..., 1, 2] = dUy_dz
    tensor[..., 2, 0] = dUz_dx
    tensor[..., 2, 1] = dUz_dy
    tensor[..., 2, 2] = dUz_dz
    return tensor


# --- Linear / HGO / NH evaluators (unchanged math; xyz everywhere) ---------------


def calculate_VWS_linear(
    tensor_displacement_list,
    E1,
    E2,
    v12,
    v23,
    Gt,
    X,
    Y,
    Z,
    Force,
    volume_matrix,
    L,
    H,
    output=1,
):
    E3 = E2
    v21 = (E2 / E1) * v12
    v13 = v12
    v31 = (E3 / E1) * v13
    v32 = v23
    Gp = E2 / (2 * (1 + v23))

    G12 = Gt
    G13 = Gt
    G23 = Gp

    S = np.array(
        [
            [1 / E1, -v21 / E2, -v31 / E3, 0, 0, 0],
            [-v12 / E1, 1 / E2, -v32 / E3, 0, 0, 0],
            [-v13 / E1, -v23 / E2, 1 / E3, 0, 0, 0],
            [0, 0, 0, 1 / G12, 0, 0],
            [0, 0, 0, 0, 1 / G13, 0],
            [0, 0, 0, 0, 0, 1 / G23],
        ]
    )

    C_stiffness = np.linalg.inv(S)
    I3 = np.eye(3)

    F = np.array(tensor_displacement_list) + I3
    J = np.linalg.det(F)
    F_inv = np.linalg.inv(F)

    C = np.einsum("...ji,...jk->...ik", F, F)
    E = 0.5 * (C - I3)

    e_vec = np.stack(
        [E[..., 0, 0], E[..., 1, 1], E[..., 2, 2], 2 * E[..., 0, 1], 2 * E[..., 0, 2], 2 * E[..., 1, 2]], axis=-1
    )

    sigma_vec = np.einsum("ij,...j->...i", C_stiffness, e_vec)

    sigma = np.zeros_like(F)
    sigma[..., 0, 0] = sigma_vec[..., 0]
    sigma[..., 1, 1] = sigma_vec[..., 1]
    sigma[..., 2, 2] = sigma_vec[..., 2]
    sigma[..., 0, 1] = sigma[..., 1, 0] = sigma_vec[..., 3]
    sigma[..., 0, 2] = sigma[..., 2, 0] = sigma_vec[..., 4]
    sigma[..., 1, 2] = sigma[..., 2, 1] = sigma_vec[..., 5]

    pk1 = J[..., None, None] * np.einsum("...ij,...kj->...ik", sigma, F_inv)

    phi = calculate_VWS_virtual_work(pk1, X, Y, Z, volume_matrix, Force, L, H, mode="linear")
    return phi * 1e10 if output == 1 else np.sqrt(phi[0] ** 2)


def calculate_VWS_hgo(tensor_displacement_list, X, Y, Z, C10, D1, k1, k2, kappa, volume_matrix, Force, L, H):
    a04 = np.array([1, 0, 0]).T
    I3 = np.eye(3)
    f = np.array(tensor_displacement_list) + I3[None, None, None, :, :]
    J = np.linalg.det(f)
    f_inv = np.linalg.inv(f)
    f_iso = J[..., None, None] ** (-1 / 3) * f

    c = np.einsum("...ji,...jk->...ik", f, f)
    b = np.einsum("...ik,...jk->...ij", f, f)
    I1 = np.trace(c, axis1=-2, axis2=-1)
    I1_iso = J ** (-2 / 3) * I1

    a4 = np.einsum("...ij,j->...i", f, a04)
    a4_iso = np.einsum("...ij,j->...i", f_iso, a04)
    I4 = np.einsum("...i,...i->...", a4, a4)
    I4_iso = J ** (-2 / 3) * I4
    A4_iso = np.einsum("...i,...j->...ij", a4_iso, a4_iso)

    E_iso = kappa * I1_iso + (1 - 3 * kappa) * I4_iso - 1

    sigma_iso = (2 * C10 / J ** (5 / 3))[..., None, None] * (b - I3 * I1[..., None, None] / 3)
    sigma_aniso = (
        (2 * k1 * E_iso / J ** (5 / 3))[..., None, None]
        * np.exp(k2 * E_iso**2)[..., None, None]
        * (
            kappa * b
            + (1 - 3 * kappa) * A4_iso
            - (1 / 3) * I3 * (kappa * I1[..., None, None] + (1 - 3 * kappa) * I4[..., None, None])
        )
    )
    sigma_vol = (1 / D1) * (J - 1 / J)[..., None, None] * I3
    sigma = sigma_iso + sigma_aniso + sigma_vol

    pk1 = J[..., None, None] * np.einsum("...ij,...kj->...ik", sigma, f_inv)

    phi = calculate_VWS_virtual_work(pk1, X, Y, Z, volume_matrix, Force, L, H, mode="hgo")
    return phi * 1e10


def calculate_VWS_nh(tensor_displacement_list, X, Y, Z, C10, D1, volume_matrix, Force, L, H):
    I3 = np.eye(3)
    F = tensor_displacement_list + I3
    J = np.linalg.det(F)
    Finv = np.linalg.inv(F)

    b = np.einsum("...ik,...jk->...ij", F, F)
    c = np.einsum("...ji,...jk->...ik", F, F)
    I1 = np.trace(c, axis1=-2, axis2=-1)[..., None, None]

    sigma_iso = (2 * C10 * J ** (-5 / 3))[..., None, None] * (b - (I3 * I1 / 3))
    sigma_vol = ((1 / D1) * (J - 1 / J))[..., None, None] * I3
    sigma = sigma_iso + sigma_vol

    pk1 = J[..., None, None] * np.einsum("...ij,...kj->...ik", sigma, Finv)

    phi = calculate_VWS_virtual_work(pk1, X, Y, Z, volume_matrix, Force, L, H, mode="nh")
    return phi * 1e10


# --- Virtual work assembly (xyz everywhere) --------------------------------------


def calculate_VWS_virtual_work(pk1, X, Y, Z, volume_element, Force, L, H, mode):
    """Compute IVW/EVW terms for the predefined virtual fields.
    Inputs X,Y,Z are the (x,y,z) coordinate grids.
    """
    X1 = X - L / 2
    X2 = Y - L / 2
    X3 = Z

    du_star_1 = np.zeros_like(pk1)
    du_star_1[..., 2, 0] = U_star_z_cos_devX(X1, X2, X3, L, H)
    du_star_1[..., 2, 1] = U_star_z_cos_devY(X1, X2, X3, L, H)
    du_star_1[..., 2, 2] = U_star_z_cos_devZ(X1, X2, X3, L, H)

    du_star_2 = np.zeros_like(pk1)
    du_star_2[..., 2, 0] = U_star_z_pw_devX(X1, X2, X3, L, H)
    du_star_2[..., 2, 1] = U_star_z_pw_devY(X1, X2, X3, L, H)
    du_star_2[..., 2, 2] = U_star_z_pw_devZ(X1, X2, X3, L, H)

    du_star_3 = np.zeros_like(pk1)
    du_star_3[..., 0, 0] = U_star_x_para_devX(X1, X2, X3, L, H)
    du_star_3[..., 0, 1] = U_star_x_para_devY(X1, X2, X3, L, H)
    du_star_3[..., 0, 2] = U_star_x_para_devZ(X1, X2, X3, L, H)
    du_star_3[..., 1, 0] = U_star_y_para_devX(X1, X2, X3, L, H)
    du_star_3[..., 1, 1] = U_star_y_para_devY(X1, X2, X3, L, H)
    du_star_3[..., 1, 2] = U_star_y_para_devZ(X1, X2, X3, L, H)

    du_star_4 = np.zeros_like(pk1)
    du_star_4[..., 0, 0] = U_star_x_sin_devX(X1, X2, X3, L, H)
    du_star_4[..., 0, 1] = U_star_x_sin_devY(X1, X2, X3, L, H)
    du_star_4[..., 0, 2] = U_star_x_sin_devZ(X1, X2, X3, L, H)
    du_star_4[..., 1, 0] = U_star_y_sin_devX(X1, X2, X3, L, H)
    du_star_4[..., 1, 1] = U_star_y_sin_devY(X1, X2, X3, L, H)
    du_star_4[..., 1, 2] = U_star_y_sin_devZ(X1, X2, X3, L, H)

    # IVW for fields we use
    ivw_2 = np.sum(pk1 * du_star_2, axis=(-2, -1)) * volume_element
    ivw_3 = np.sum(pk1 * du_star_3, axis=(-2, -1)) * volume_element
    total_IVW_2 = np.sum(ivw_2)
    total_IVW_3 = np.sum(ivw_3)

    evw_2 = -Force * U_star_z_pw(0, 0, H, L, H)

    if mode == "linear":
        phi = np.array([total_IVW_2 - evw_2, total_IVW_3])

    elif mode == "hgo":
        du_star_5 = np.zeros_like(pk1)
        du_star_5[..., 2, 0] = U_star_z_pw_vol_devX(X1, X2, X3, L, H)
        du_star_5[..., 2, 1] = U_star_z_pw_vol_devY(X1, X2, X3, L, H)
        du_star_5[..., 2, 2] = U_star_z_pw_vol_devZ(X1, X2, X3, L, H)
        ivw_5 = np.sum(pk1 * du_star_5, axis=(-2, -1)) * volume_element
        total_IVW_5 = np.sum(ivw_5)
        evw_5 = -Force * U_star_z_pw_vol(0, 0, H, L, H)
        phi = np.array([total_IVW_5 - evw_5, total_IVW_3])

    elif mode == "nh":
        du_star_5 = np.zeros_like(pk1)
        du_star_5[..., 2, 0] = U_star_z_pw_vol_devX(X1, X2, X3, L, H)
        du_star_5[..., 2, 1] = U_star_z_pw_vol_devY(X1, X2, X3, L, H)
        du_star_5[..., 2, 2] = U_star_z_pw_vol_devZ(X1, X2, X3, L, H)
        ivw_5 = np.sum(pk1 * du_star_5, axis=(-2, -1)) * volume_element
        total_IVW_5 = np.sum(ivw_5)
        evw_5 = -Force * U_star_z_pw_vol(0, 0, H, L, H)
        phi = np.array([total_IVW_3, total_IVW_5 - evw_5])
        phi = np.sqrt(phi[0] ** 2 + phi[1] ** 2)

    else:
        raise ValueError(f"Unsupported mode '{mode}', choose 'linear' or 'hgo'.")

    return phi


# --- Sensitivity analysis (unchanged numerics; xyz everywhere) -------------------


def sensitivity_full(
    tensor_displacement_list,
    E1,
    E2,
    v12,
    v23,
    Gt,
    X,
    Y,
    Z,
    Force,
    volume_matrix,
    L,
    H,
    deviation,
):
    sens_matrix = np.zeros((5, 5))

    E1_1 = E1 * (1 + deviation)
    E2_1 = E2 * (1 + deviation)
    v12_1 = v12 * (1 + deviation)
    v23_1 = v23 * (1 + deviation)
    Gt_1 = Gt * (1 + deviation)

    phi_base = calculate_VWS_linear(
        tensor_displacement_list, E1, E2, v12, v23, Gt, X, Y, Z, Force, volume_matrix, L, H, output=2
    )

    phi_E1_1 = calculate_VWS_linear(
        tensor_displacement_list, E1_1, E2, v12, v23, Gt, X, Y, Z, Force, volume_matrix, L, H, output=2
    )
    phi_E2_1 = calculate_VWS_linear(
        tensor_displacement_list, E1, E2_1, v12, v23, Gt, X, Y, Z, Force, volume_matrix, L, H, output=2
    )
    phi_v12_1 = calculate_VWS_linear(
        tensor_displacement_list, E1, E2, v12_1, v23, Gt, X, Y, Z, Force, volume_matrix, L, H, output=2
    )
    phi_v23_1 = calculate_VWS_linear(
        tensor_displacement_list, E1, E2, v12, v23_1, Gt, X, Y, Z, Force, volume_matrix, L, H, output=2
    )
    phi_Gt_1 = calculate_VWS_linear(
        tensor_displacement_list, E1, E2, v12, v23, Gt_1, X, Y, Z, Force, volume_matrix, L, H, output=2
    )

    sens_matrix[0, 0] = ((phi_E1_1 - phi_base) / (E1 * deviation)) ** 2
    sens_matrix[1, 1] = ((phi_E2_1 - phi_base) / (E2 * deviation)) ** 2
    sens_matrix[2, 2] = ((phi_v12_1 - phi_base) / (v12 * deviation)) ** 2
    sens_matrix[3, 3] = ((phi_v23_1 - phi_base) / (v23 * deviation)) ** 2
    sens_matrix[4, 4] = ((phi_Gt_1 - phi_base) / (Gt_1)) ** 2

    sens_matrix[0, 1] = ((phi_E1_1 - phi_base) / (E1 * deviation)) * ((phi_E2_1 - phi_base) / (E2 * deviation))
    sens_matrix[0, 2] = ((phi_E1_1 - phi_base) / (E1 * deviation)) * ((phi_v12_1 - phi_base) / (v12 * deviation))
    sens_matrix[0, 3] = ((phi_E1_1 - phi_base) / (E1 * deviation)) * ((phi_v23_1 - phi_base) / (v23 * deviation))
    sens_matrix[0, 4] = ((phi_E1_1 - phi_base) / (E1 * deviation)) * ((phi_Gt_1 - phi_base) / (Gt * deviation))
    sens_matrix[1, 0] = sens_matrix[0, 1]
    sens_matrix[1, 2] = ((phi_E2_1 - phi_base) / (E2 * deviation)) * ((phi_v12_1 - phi_base) / (v12 * deviation))
    sens_matrix[1, 3] = ((phi_E2_1 - phi_base) / (E2 * deviation)) * ((phi_v23_1 - phi_base) / (v23 * deviation))
    sens_matrix[1, 4] = ((phi_E2_1 - phi_base) / (E2 * deviation)) * ((phi_Gt_1 - phi_base) / (Gt * deviation))
    sens_matrix[2, 0] = sens_matrix[0, 2]
    sens_matrix[2, 1] = sens_matrix[1, 2]
    sens_matrix[2, 3] = ((phi_v12_1 - phi_base) / (v12 * deviation)) * ((phi_v23_1 - phi_base) / (v23 * deviation))
    sens_matrix[2, 4] = ((phi_v12_1 - phi_base) / (v12 * deviation)) * ((phi_Gt_1 - phi_base) / (Gt * deviation))
    sens_matrix[3, 0] = sens_matrix[0, 3]
    sens_matrix[3, 1] = sens_matrix[1, 3]
    sens_matrix[3, 2] = sens_matrix[2, 3]
    sens_matrix[3, 4] = ((phi_v23_1 - phi_base) / (v23 * deviation)) * ((phi_Gt_1 - phi_base) / (Gt * deviation))
    sens_matrix[4, 0] = sens_matrix[0, 4]
    sens_matrix[4, 1] = sens_matrix[1, 4]
    sens_matrix[4, 2] = sens_matrix[2, 4]
    sens_matrix[4, 3] = sens_matrix[3, 4]

    sens_matrix = np.abs(sens_matrix)
    m = np.min(sens_matrix)
    eps = 1e-12
    if not np.isfinite(m) or m <= eps:
        return sens_matrix

    print("Sensitivity Matrix (5x5):")
    for row in sens_matrix:
        print(" ".join(f"{value:10.4f}" for value in row))
    return sens_matrix / m
