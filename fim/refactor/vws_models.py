"""VWS Model Functions
Description: Implements full-field virtual fields method computations for supported material models.
"""

import numpy as np

depth_indentation = 3.2e-05
sphere_radius = 5e-4
contact_radius = np.sqrt(depth_indentation * sphere_radius)


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


def U_star_z_pw_vol(x, y, z, L, H):
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    a_0 = contact_radius * 2
    t = z / H
    a = a_0
    k = (c / 2 * (L / 4 - a) - c * L / 4) / ((L / 4 - a) / 2 + L / 8)

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
    a_0 = contact_radius * 2
    t = z / H
    a = a_0
    k = (c / 2 * (L / 4 - a) - c * L / 4) / ((L / 4 - a) / 2 + L / 8)

    result = np.zeros_like(d)
    # mask1 = d <= a
    mask2 = (d > a) & (d <= (L / 4))
    mask3 = (d > (L / 4)) & (d < L / 2)

    result[mask2] = t[mask2] * ((k - c) * (x[mask2] / d[mask2]) / (L / 4 - a))
    result[mask3] = t[mask3] * (-4 * k * (x[mask3] / d[mask3]) / L)
    return result


def U_star_z_pw_vol_devY(x, y, z, L, H):
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    a_0 = contact_radius * 2
    t = z / H
    a = a_0
    k = (c / 2 * (L / 4 - a) - c * L / 4) / ((L / 4 - a) / 2 + L / 8)

    result = np.zeros_like(d)
    # mask1 = d <= a
    mask2 = (d > a) & (d <= (L / 4))
    mask3 = (d > (L / 4)) & (d < L / 2)

    result[mask2] = t[mask2] * ((k - c) * (y[mask2] / d[mask2]) / (L / 4 - a))
    result[mask3] = t[mask3] * (-4 * k * (y[mask3] / d[mask3]) / L)
    return result


def U_star_z_pw_vol_devZ(x, y, z, L, H):
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    a_0 = contact_radius * 2
    a = a_0
    k = (c / 2 * (L / 4 - a) - c * L / 4) / ((L / 4 - a) / 2 + L / 8)

    result = np.zeros_like(d)
    mask1 = d <= a
    mask2 = (d > a) & (d <= (L / 4))
    mask3 = (d > (L / 4)) & (d < L / 2)

    result[mask1] = c / H
    result[mask2] = (1 / H) * (c + (k - c) * (d[mask2] - a) / (L / 4 - a))
    result[mask3] = (1 / H) * (k - 4 * k * (d[mask3] - L / 4) / L)
    return result


def increase_matrix_size(matrix):
    # Get the original dimensions of the matrix
    rows, cols, deps = matrix.shape

    # Create a new matrix with increased size
    new_deps = deps + 2
    new_rows = rows + 2
    new_cols = cols + 2
    new_matrix = np.zeros((new_rows, new_cols, new_deps), dtype=matrix.dtype)

    # Copy the original data from the input matrix to the inner region of the new matrix
    new_matrix[1:-1, 1:-1, 1:-1] = matrix

    # Extend the border of the new matrix by duplicating the values from the first inner layer
    new_matrix[0, :, :] = new_matrix[1, :, :]
    new_matrix[-1, :, :] = new_matrix[-2, :, :]
    new_matrix[:, 0, :] = new_matrix[:, 1, :]
    new_matrix[:, -1, :] = new_matrix[:, -2, :]
    new_matrix[:, :, 0] = new_matrix[:, :, 1]
    new_matrix[:, :, -1] = new_matrix[:, :, -2]

    return new_matrix


def read_input_file(file_path):
    nodes = []  # List to store nodal coordinates
    connectivity = []  # List to store element connectivity
    in_node_section = False
    in_element_section = False
    file = open(file_path)
    Lines = file.readlines()
    flag = 0

    for line in Lines:
        line = line.strip()
        # change according to the input file
        if line.startswith("*") and line.startswith("*Part, name=tissue"):
            flag = 1
            continue
        if line.startswith("*Node") and flag == 1:
            in_node_section = True
            in_element_section = False
            continue
        if line.startswith("*Element") and flag == 1:
            in_node_section = False
            in_element_section = True
            continue
        if line.startswith("*"):
            flag = 0
            continue
        if in_node_section and not in_element_section and flag == 1:
            values_n = line.split(",")
            node_info = [int(values_n[0])] + [np.double(values_n[i].strip()) for i in range(1, 4)]
            nodes.append(node_info)
            continue

        if not in_node_section and in_element_section and flag == 1:
            values_e = line.split(",")
            if len(values_e) == 9:
                element_info = [int(value) for value in values_e]
                connectivity.append(element_info)
            continue

        continue

    nodes = np.array(nodes)
    connectivity = np.array(connectivity)

    x_min = np.min(nodes[:, 1])
    y_min = np.min(nodes[:, 2])
    z_min = np.min(nodes[:, 3])

    nodes[:, 1] = np.around((nodes[:, 1] - x_min), 15)
    nodes[:, 2] = np.double(nodes[:, 2] - y_min)
    nodes[:, 3] = np.double(nodes[:, 3] - z_min)

    return nodes, connectivity


def central_differentiation(Ux, Uy, Uz, X, Y, Z):
    """Compute central finite differences of displacement fields (Ux, Uy, Uz) over a 3D grid (X, Y, Z).

    Parameters:
        Ux, Uy, Uz : ndarray
            Displacement components with shape (rows, cols, deps).
        X, Y, Z : ndarray
            Coordinate grids matching the shape of displacement fields.
        eps : float, optional
            Minimum allowed spacing to avoid division by zero.

    Returns:
        Tuple of 9 ndarrays:
            dUx_dx, dUy_dx, dUz_dx,
            dUx_dy, dUy_dy, dUz_dy,
            dUx_dz, dUy_dz, dUz_dz
    """
    if not (Ux.shape == Uy.shape == Uz.shape == X.shape == Y.shape == Z.shape):
        raise ValueError("All input arrays must have the same shape.")

    dx = X[1:-1, 2:, 1:-1] - X[1:-1, :-2, 1:-1]
    dy = Y[2:, 1:-1, 1:-1] - Y[:-2, 1:-1, 1:-1]
    dz = Z[1:-1, 1:-1, 2:] - Z[1:-1, 1:-1, :-2]

    eps = 1e-12
    dx = np.where(np.abs(dx) > eps, dx, eps)
    dy = np.where(np.abs(dy) > eps, dy, eps)
    dz = np.where(np.abs(dz) > eps, dz, eps)

    dUx_dx = (Ux[1:-1, 2:, 1:-1] - Ux[1:-1, :-2, 1:-1]) / dx
    dUy_dx = (Uy[1:-1, 2:, 1:-1] - Uy[1:-1, :-2, 1:-1]) / dx
    dUz_dx = (Uz[1:-1, 2:, 1:-1] - Uz[1:-1, :-2, 1:-1]) / dx

    dUx_dy = (Ux[2:, 1:-1, 1:-1] - Ux[:-2, 1:-1, 1:-1]) / dy
    dUy_dy = (Uy[2:, 1:-1, 1:-1] - Uy[:-2, 1:-1, 1:-1]) / dy
    dUz_dy = (Uz[2:, 1:-1, 1:-1] - Uz[:-2, 1:-1, 1:-1]) / dy

    dUx_dz = (Ux[1:-1, 1:-1, 2:] - Ux[1:-1, 1:-1, :-2]) / dz
    dUy_dz = (Uy[1:-1, 1:-1, 2:] - Uy[1:-1, 1:-1, :-2]) / dz
    dUz_dz = (Uz[1:-1, 1:-1, 2:] - Uz[1:-1, 1:-1, :-2]) / dz

    return dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz


def map_elements_to_centraldiff(dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz):
    """Map scalar gradient components to full 3x3 displacement gradient tensor at each voxel.

    Returns:
        tensor_array: ndarray of shape (Nx, Ny, Nz, 3, 3)
    """
    shape = dUx_dx.shape
    for arr in [dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz]:
        assert arr.shape == shape, "All central difference fields must have the same shape"

    rows, cols, deps = shape
    tensor_array = np.zeros((rows, cols, deps, 3, 3), dtype=np.float64)

    tensor_array[..., 0, 0] = dUx_dx
    tensor_array[..., 0, 1] = dUx_dy
    tensor_array[..., 0, 2] = dUx_dz
    tensor_array[..., 1, 0] = dUy_dx
    tensor_array[..., 1, 1] = dUy_dy
    tensor_array[..., 1, 2] = dUy_dz
    tensor_array[..., 2, 0] = dUz_dx
    tensor_array[..., 2, 1] = dUz_dy
    tensor_array[..., 2, 2] = dUz_dz

    return tensor_array


def calculate_VWS_linear(tensor_displacement_list, E1, E2, v12, v23, Gt, X, Y, Z, Force, cube_size, L, H, output=1):
    """Calculate the virtual work residuals for a linear orthotropic material model.

    Args:
        tensor_displacement_list: ndarray (Nx, Ny, Nz, 3, 3)
        E1, E2, v12, v23, Gt: material properties
        X, Y, Z: coordinate grids
        Force: applied load
        cube_size: voxel volume
        L, H: sample dimensions
        mode: 'phi' or 'residual'

    Returns:
        Virtual work difference phi or residual magnitude.
    """
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

    deformation_gradients = np.array(tensor_displacement_list)

    # Precompute deformation gradient F
    F = deformation_gradients + I3

    # Determinant and Inverse
    J = np.linalg.det(F)
    F_inv = np.linalg.inv(F)

    # Cauchy-Green Tensor and Green-Lagrange Strain
    C = np.einsum("...ji,...jk->...ik", F, F)
    E = 0.5 * (C - I3)

    # Strain in Voigt notation
    e_vec = np.stack(
        [E[..., 0, 0], E[..., 1, 1], E[..., 2, 2], 2 * E[..., 0, 1], 2 * E[..., 0, 2], 2 * E[..., 1, 2]], axis=-1
    )

    # Stress in Voigt notation and tensor form
    sigma_vec = np.einsum("ij,...j->...i", C_stiffness, e_vec)

    sigma = np.zeros_like(F)
    sigma[..., 0, 0] = sigma_vec[..., 0]
    sigma[..., 1, 1] = sigma_vec[..., 1]
    sigma[..., 2, 2] = sigma_vec[..., 2]
    sigma[..., 0, 1] = sigma[..., 1, 0] = sigma_vec[..., 3]
    sigma[..., 0, 2] = sigma[..., 2, 0] = sigma_vec[..., 4]
    sigma[..., 1, 2] = sigma[..., 2, 1] = sigma_vec[..., 5]

    # Piola-Kirchhoff stress
    pk1 = J[..., None, None] * np.einsum("...ij,...kj->...ik", sigma, F_inv)

    # Compute the virtual work residuals using predefined virtual displacement fields (Linear mode)
    phi = calculate_VWS_virtual_work(pk1, X, Y, Z, cube_size, Force, L, H, mode="linear")

    return phi * 1e10 if output == 1 else np.sqrt(phi[0] ** 2)


def calculate_VWS_hgo(tensor_displacement_list, X, Y, Z, C10, D1, k1, k2, kappa, volume_matrix, Force, L, H):
    """Computes the internal and external virtual work terms for HGO hyperelastic material model.

    Args:
        tensor_displacement_list: ndarray of deformation gradients (Nx, Ny, Nz, 3, 3)
        X, Y, Z: 3D mesh coordinates
        C10, D1, k1, k2, kappa: material parameters
        volume_matrix: per-voxel volume weights
        Force: scalar load
        L, H: sample size in X/Y and Z

    Returns:
        phi: array of residuals [IVW5 - EVW5, IVW3] scaled by 1e10
    """
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
    # A4 = np.einsum("...i,...j->...ij", a4, a4)
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

    # pk1 = J[..., None, None] * np.einsum("...ij,...jk->...ik", sigma, f_inv.transpose(0, 1, 2, 4, 3))
    pk1 = J[..., None, None] * np.einsum("...ij,...kj->...ik", sigma, f_inv)

    # Compute the virtual work residuals using predefined virtual displacement fields (HGO mode)
    phi = calculate_VWS_virtual_work(pk1, X, Y, Z, volume_matrix, Force, L, H, mode="hgo")
    return phi * 1e10


def calculate_VWS_virtual_work(pk1, X, Y, Z, volume_element, Force, L, H, mode):
    """Compute internal (IVW) and external (EVW) virtual work contributions for five
    predefined virtual fields, and return the residual vector phi for the given mode.
    """
    # Shift coordinates so that X1, X2 are centered (origin at mid-plane) and X3 is vertical
    X1 = X - L / 2
    X2 = Y - L / 2
    X3 = Z

    # Virtual field derivatives (du_star) for five virtual displacement fields:
    # du_star_1: z_cos in u3 only (vertical cosine-shaped field)
    du_star_1 = np.zeros_like(pk1)
    du_star_1[..., 2, 0] = U_star_z_cos_devX(X1, X2, X3, L, H)
    du_star_1[..., 2, 1] = U_star_z_cos_devY(X1, X2, X3, L, H)
    du_star_1[..., 2, 2] = U_star_z_cos_devZ(X1, X2, X3, L, H)

    # du_star_2: z_pw in u3 only (vertical piecewise field)
    du_star_2 = np.zeros_like(pk1)
    du_star_2[..., 2, 0] = U_star_z_pw_devX(X1, X2, X3, L, H)
    du_star_2[..., 2, 1] = U_star_z_pw_devY(X1, X2, X3, L, H)
    du_star_2[..., 2, 2] = U_star_z_pw_devZ(X1, X2, X3, L, H)

    # du_star_3: x_para in u1, y_para in u2 (in-plane parabolic field)
    du_star_3 = np.zeros_like(pk1)
    du_star_3[..., 0, 0] = U_star_x_para_devX(X1, X2, X3, L, H)
    du_star_3[..., 0, 1] = U_star_x_para_devY(X1, X2, X3, L, H)
    du_star_3[..., 0, 2] = U_star_x_para_devZ(X1, X2, X3, L, H)
    du_star_3[..., 1, 0] = U_star_y_para_devX(X1, X2, X3, L, H)
    du_star_3[..., 1, 1] = U_star_y_para_devY(X1, X2, X3, L, H)
    du_star_3[..., 1, 2] = U_star_y_para_devZ(X1, X2, X3, L, H)

    # du_star_4: x_sin in u1, y_sin in u2 (in-plane sinusoidal field)
    du_star_4 = np.zeros_like(pk1)
    du_star_4[..., 0, 0] = U_star_x_sin_devX(X1, X2, X3, L, H)
    du_star_4[..., 0, 1] = U_star_x_sin_devY(X1, X2, X3, L, H)
    du_star_4[..., 0, 2] = U_star_x_sin_devZ(X1, X2, X3, L, H)
    du_star_4[..., 1, 0] = U_star_y_sin_devX(X1, X2, X3, L, H)
    du_star_4[..., 1, 1] = U_star_y_sin_devY(X1, X2, X3, L, H)
    du_star_4[..., 1, 2] = U_star_y_sin_devZ(X1, X2, X3, L, H)

    # Internal virtual work (IVW) for each field: integrate pk1 : du_star over volume
    # ivw_1 = np.sum(pk1 * du_star_1, axis=(-2, -1)) * volume_element
    ivw_2 = np.sum(pk1 * du_star_2, axis=(-2, -1)) * volume_element
    ivw_3 = np.sum(pk1 * du_star_3, axis=(-2, -1)) * volume_element
    # ivw_4 = np.sum(pk1 * du_star_4, axis=(-2, -1)) * volume_element

    # Total internal virtual work for each field (sum over all elements)
    # total_IVW_1 = np.sum(ivw_1)
    total_IVW_2 = np.sum(ivw_2)
    total_IVW_3 = np.sum(ivw_3)
    # total_IVW_4 = np.sum(ivw_4)

    # External virtual work (EVW) for each field using applied force and geometry
    # evw_1 = -Force * U_star_z_cos(0, 0, H, L, H)
    evw_2 = -Force * U_star_z_pw(0, 0, H, L, H)
    # evw_3 = 0.0  # no external work for purely in-plane virtual field
    # evw_4 = -Force * U_star_z_pw(0, 0, H, L, H)  # (same as field 2 shape at top)

    # Assemble residual vector phi based on mode
    if mode == "linear":
        # Linear mode: use virtual fields 2 and 3
        phi = np.array([total_IVW_2 - evw_2, total_IVW_3])

    elif mode == "hgo":
        # du_star_5: z_pw_vol in u3 only (vertical piecewise volumetric field)
        du_star_5 = np.zeros_like(pk1)
        du_star_5[..., 2, 0] = U_star_z_pw_vol_devX(X1, X2, X3, L, H)
        du_star_5[..., 2, 1] = U_star_z_pw_vol_devY(X1, X2, X3, L, H)
        du_star_5[..., 2, 2] = U_star_z_pw_vol_devZ(X1, X2, X3, L, H)

        ivw_5 = np.sum(pk1 * du_star_5, axis=(-2, -1)) * volume_element
        total_IVW_5 = np.sum(ivw_5)
        evw_5 = -Force * U_star_z_pw_vol(0, 0, H, L, H)

        # HGO mode: use virtual fields 5 and 3
        phi = np.array([total_IVW_5 - evw_5, total_IVW_3])

    else:
        raise ValueError(f"Unsupported mode '{mode}', choose 'linear' or 'hgo'.")

    # Apply scaling for numerical stability
    return phi


def sensitivity_full(tensor_displacement_list, E1, E2, v12, v23, Gt, X, Y, Z, Force, cube_size, L, H, deviation):
    """Perform sensitivity analysis for the linear material model using finite difference.

    Parameters:
        tensor_displacement_list: ndarray of shape (Nx, Ny, Nz, 3, 3)
        E1, E2, v12, v23, Gt: material parameters
        X, Y, Z: coordinate grids
        Force: scalar external load
        cube_size: voxel volume
        L, H: sample dimensions
        deviation: relative perturbation ratio (e.g., 0.05 for 5%)

    Returns:
        sens_matrix: 5x5 normalized sensitivity matrix
    """
    # Perturb each parameter by (1 + deviation)
    sens_matrix = np.zeros((5, 5))
    E1_1 = E1 * (1 + deviation)
    E2_1 = E2 * (1 + deviation)
    v12_1 = v12 * (1 + deviation)
    v23_1 = v23 * (1 + deviation)
    Gt_1 = Gt * (1 + deviation)

    # Compute the baseline residual
    phi_base = calculate_VWS_linear(
        tensor_displacement_list, E1, E2, v12, v23, Gt, X, Y, Z, Force, cube_size, L, H, output=2
    )

    # Compute residuals for each perturbed parameter
    phi_E1_1 = calculate_VWS_linear(
        tensor_displacement_list, E1_1, E2, v12, v23, Gt, X, Y, Z, Force, cube_size, L, H, output=2
    )
    phi_E2_1 = calculate_VWS_linear(
        tensor_displacement_list, E1, E2_1, v12, v23, Gt, X, Y, Z, Force, cube_size, L, H, output=2
    )
    phi_v12_1 = calculate_VWS_linear(
        tensor_displacement_list, E1, E2, v12_1, v23, Gt, X, Y, Z, Force, cube_size, L, H, output=2
    )
    phi_v23_1 = calculate_VWS_linear(
        tensor_displacement_list, E1, E2, v12, v23_1, Gt, X, Y, Z, Force, cube_size, L, H, output=2
    )
    phi_Gt_1 = calculate_VWS_linear(
        tensor_displacement_list, E1, E2, v12, v23, Gt_1, X, Y, Z, Force, cube_size, L, H, output=2
    )

    # Fill diagonal entries (squared normalized differences)
    sens_matrix[0, 0] = ((phi_E1_1 - phi_base) / (E1 * deviation)) ** 2
    sens_matrix[1, 1] = ((phi_E2_1 - phi_base) / (E2 * deviation)) ** 2
    sens_matrix[2, 2] = ((phi_v12_1 - phi_base) / (v12 * deviation)) ** 2
    sens_matrix[3, 3] = ((phi_v23_1 - phi_base) / (v23 * deviation)) ** 2
    sens_matrix[4, 4] = ((phi_Gt_1 - phi_base) / (Gt_1)) ** 2

    # Fill off-diagonal entries (cross sensitivity)
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

    # Normalize matrix by the minimum value (ensures relative scaling)
    sens_matrix = np.abs(sens_matrix)
    sens_matrix = sens_matrix / np.min(sens_matrix)

    # Print for inspection
    print("Sensitivity Matrix (5x5):")
    for row in sens_matrix:
        print(" ".join(f"{value:10.4f}" for value in row))

    return sens_matrix
