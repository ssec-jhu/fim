from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares, minimize

depth_indentation = 3.2e-05
sphere_radius = 5e-4
Force = 9.49803e-06
contact_radius = np.sqrt(depth_indentation * sphere_radius)


def U_star_x_sin(x, y, z):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H

    if d == 0:
        return 0
    elif d < L / 2 and d > 0:
        return c * x / d * t * np.sin(2 * np.pi * d / (L / 2))
    else:
        return 0


def U_star_x_sin_devX(x, y, z):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H

    if d == 0:
        return c * t * np.pi / (L / 2)
    elif d < L / 2 and d > 0:
        return (
            c
            * t
            * (
                (np.sin(2 * np.pi * d / (L / 2))) / d
                - x**2 * (np.sin(2 * np.pi * d / (L / 2))) / d**3
                + x**2 * (np.cos(2 * np.pi * d / (L / 2))) * 2 * np.pi / (d**2 * L / 2)
            )
        )
    else:
        return 0


def U_star_x_sin_devY(x, y, z):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H

    if d == 0:
        return 0
    elif d < L / 2 and d > 0:
        return (
            c
            * x
            * t
            * (
                (-y / d**3) * (np.sin(2 * np.pi * d / (L / 2)))
                + (np.cos(2 * np.pi * d / (L / 2))) * 2 * np.pi * y / (d**2 * (L / 2))
            )
        )
    else:
        return 0


def U_star_x_sin_devZ(x, y, z):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H

    if d == 0:
        return 0
    elif d < L / 2 and d > 0:
        return c * x / d * (1 / H) * np.sin(2 * np.pi * d / (L / 2))
    else:
        return 0


def U_star_y_sin(x, y, z):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H

    if d == 0:
        return 0
    elif d < L / 2 and d > 0:
        return c * y / d * t * np.sin(2 * np.pi * d / (L / 2))
    else:
        return 0


def U_star_y_sin_devX(x, y, z):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H

    if d == 0:
        return 0
    elif d < L / 2 and d > 0:
        return (
            c
            * y
            * t
            * (
                (-x / d**3) * (np.sin(2 * np.pi * d / (L / 2)))
                + (np.cos(2 * np.pi * d / (L / 2))) * 2 * np.pi * x / (d**2 * (L / 2))
            )
        )
    else:
        return 0


def U_star_y_sin_devY(x, y, z):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H

    if d == 0:
        return c * t * np.pi / (L / 2)
    elif d < L / 2 and d > 0:
        return (
            c
            * t
            * (
                (np.sin(2 * np.pi * d / (L / 2))) / d
                - y**2 * (np.sin(2 * np.pi * d / (L / 2))) / d**3
                + y**2 * (np.cos(2 * np.pi * d / (L / 2))) * 2 * np.pi / (d**2 * L / 2)
            )
        )
    else:
        return 0


def U_star_y_sin_devZ(x, y, z):
    c = 2e-5
    d = np.sqrt(x**2 + y**2)
    t = z / H

    if d == 0:
        return 0
    elif d < L / 2 and d > 0:
        return c * y / d * (1 / H) * np.sin(2 * np.pi * d / (L / 2))
    else:
        return 0


def U_star_z_pw(x, y, z):
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    a_0 = contact_radius * 1.25
    t = z / H
    a = a_0 * t
    if d <= a:
        return c * t
    elif d > a and d < L / 2:
        # return (-2*c/(L-2*a))*d*t + (c*L/(L-2*a))*t
        return c * t * (L / 2 - d) / (L / 2 - a)
    else:
        return 0


def U_star_z_pw_devX(x, y, z):
    c = 5e-5
    a_0 = contact_radius * 1.25
    d = np.sqrt(x**2 + y**2)
    t = z / H
    a = a_0 * t
    if d <= a:
        return 0
    elif d > a and d < L / 2:
        return -2 * c * x * t / ((L - 2 * a) * d)
    else:
        return 0


def U_star_z_pw_devY(x, y, z):
    c = 5e-5
    a_0 = contact_radius * 1.25
    d = np.sqrt(x**2 + y**2)
    t = z / H
    a = a_0 * t
    if d <= a:
        return 0
    elif d > a and d < L / 2:
        return -2 * c * y * t / ((L - 2 * a) * d)
    else:
        return 0


def U_star_z_pw_devZ(x, y, z):
    c = 5e-5
    a_0 = contact_radius * 1.25
    d = np.sqrt(x**2 + y**2)
    t = z / H
    a = a_0 * t
    if d <= a:
        return c / H
    elif d > a and d < L / 2:
        # return (-2*c*H*L*d+c*H*L**2)/(L*H-2*z*a)**2
        return c * (L / 2 - d) / (H * (L / 2 - a)) + z * c * (L / 2 - d) * (a_0) / (H**2 * (L / 2 - a) ** 2)
    else:
        return 0


def U_star_x_para(x, y, z):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H

    if d <= L / 2:
        return c * x * (L / 2 - d) * t
    else:
        return 0


def U_star_x_para_devX(x, y, z):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H
    if d == 0:
        return c * (L / 2 - d) * t
    elif d <= L / 2:
        return c * ((L / 2 - d) - x**2 / d) * t
    else:
        return 0


def U_star_x_para_devY(x, y, z):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H
    if d == 0:
        return 0
    elif d <= L / 2:
        return c * x * (-y / d) * t
    else:
        return 0


def U_star_x_para_devZ(x, y, z):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H
    if d <= L / 2:
        return c * x * (L / 2 - d) / H
    else:
        return 0


def U_star_y_para(x, y, z):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H
    if d <= L / 2:
        return c * y * (L / 2 - d) * t
    else:
        return 0


def U_star_y_para_devX(x, y, z):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H
    if d <= L / 2:
        return c * y * (-x / d) * t
    else:
        return 0


def U_star_y_para_devY(x, y, z):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H
    if d == 0:
        return c * (L / 2 - d) * t
    elif d <= L / 2:
        return c * ((L / 2 - d) - y**2 / d) * t
    else:
        return 0


def U_star_y_para_devZ(x, y, z):
    c = 10
    d = np.sqrt(x**2 + y**2)
    t = z / H

    if d <= L / 2:
        return c * y * (L / 2 - d) / H
    else:
        return 0


def U_star_z_cos(x, y, z):
    t = z / H
    a_0 = contact_radius * 2
    a = t * a_0
    c = 5e-5

    d = np.sqrt(x**2 + y**2)

    if d <= a:
        return c * t
    elif d > a and d < L / 2:
        return c * t * (np.cos(np.pi / 2 * (d - a) / (L / 2 - a)))
    else:
        return 0


def U_star_z_cos_devX(x, y, z):
    t = z / H
    a_0 = contact_radius * 2
    a = t * a_0
    c = 5e-5

    d = np.sqrt(x**2 + y**2)

    if d <= a:
        return 0
    elif d > a and d < L / 2:
        return -c * t * (np.sin(np.pi / 2 * (d - a) / (L / 2 - a)) * np.pi / (2 * (L / 2 - a)) * x / d)
    else:
        return 0


def U_star_z_cos_devY(x, y, z):
    t = z / H
    a_0 = contact_radius * 2
    a = t * a_0
    c = 5e-5

    d = np.sqrt(x**2 + y**2)

    if d <= a:
        return 0
    elif d > a and d < L / 2:
        return -c * t * (np.sin(np.pi / 2 * (d - a) / (L / 2 - a)) * np.pi / (2 * (L / 2 - a)) * y / d)
    else:
        return 0


def U_star_z_cos_devZ(x, y, z):
    t = z / H
    a_0 = contact_radius * 2
    a = t * a_0
    c = 5e-5

    d = np.sqrt(x**2 + y**2)

    if d <= a:
        return c / H
    elif d > a and d < L / 2:
        return (
            c * (np.cos(np.pi / 2 * (d - a) / (L / 2 - a))) / H
            + c
            * t
            * (-(np.sin(np.pi / 2 * (d - a) / (L / 2 - a))))
            * np.pi
            / 2
            * (d - L / 2)
            / (L / 2 - a) ** 2
            * a_0
            / H
        )
    else:
        return 0


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


def central_differentiation(Ux, Uy, Uz, X, Y, Z):
    rows, cols, deps = len(Ux), len(Ux[0]), len(Ux[0][0])
    dUx_dx_enalarged = np.zeros((rows, cols, deps), dtype=np.double)
    dUy_dx_enalarged = np.zeros((rows, cols, deps), dtype=np.double)
    dUz_dx_enalarged = np.zeros((rows, cols, deps), dtype=np.double)
    dUx_dy_enalarged = np.zeros((rows, cols, deps), dtype=np.double)
    dUy_dy_enalarged = np.zeros((rows, cols, deps), dtype=np.double)
    dUz_dy_enalarged = np.zeros((rows, cols, deps), dtype=np.double)
    dUx_dz_enalarged = np.zeros((rows, cols, deps), dtype=np.double)
    dUy_dz_enalarged = np.zeros((rows, cols, deps), dtype=np.double)
    dUz_dz_enalarged = np.zeros((rows, cols, deps), dtype=np.double)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            for k in range(1, deps - 1):
                dUx_dx_enalarged[i, j, k] = (Ux[i, j + 1, k] - Ux[i, j - 1, k]) / (X[i, j + 1, k] - X[i, j - 1, k])
                dUy_dx_enalarged[i, j, k] = (Uy[i, j + 1, k] - Uy[i, j - 1, k]) / (X[i, j + 1, k] - X[i, j - 1, k])
                dUz_dx_enalarged[i, j, k] = (Uz[i, j + 1, k] - Uz[i, j - 1, k]) / (X[i, j + 1, k] - X[i, j - 1, k])
                dUx_dy_enalarged[i, j, k] = (Ux[i + 1, j, k] - Ux[i - 1, j, k]) / (Y[i + 1, j, k] - Y[i - 1, j, k])
                dUy_dy_enalarged[i, j, k] = (Uy[i + 1, j, k] - Uy[i - 1, j, k]) / (Y[i + 1, j, k] - Y[i - 1, j, k])
                dUz_dy_enalarged[i, j, k] = (Uz[i + 1, j, k] - Uz[i - 1, j, k]) / (Y[i + 1, j, k] - Y[i - 1, j, k])
                dUx_dz_enalarged[i, j, k] = (Ux[i, j, k + 1] - Ux[i, j, k - 1]) / (Z[i, j, k + 1] - Z[i, j, k - 1])
                dUy_dz_enalarged[i, j, k] = (Uy[i, j, k + 1] - Uy[i, j, k - 1]) / (Z[i, j, k + 1] - Z[i, j, k - 1])
                dUz_dz_enalarged[i, j, k] = (Uz[i, j, k + 1] - Uz[i, j, k - 1]) / (Z[i, j, k + 1] - Z[i, j, k - 1])

    dUx_dx = np.zeros((rows - 2, cols - 2, deps - 2), dtype=float)
    dUy_dx = np.zeros((rows - 2, cols - 2, deps - 2), dtype=float)
    dUz_dx = np.zeros((rows - 2, cols - 2, deps - 2), dtype=float)
    dUx_dy = np.zeros((rows - 2, cols - 2, deps - 2), dtype=float)
    dUy_dy = np.zeros((rows - 2, cols - 2, deps - 2), dtype=float)
    dUz_dy = np.zeros((rows - 2, cols - 2, deps - 2), dtype=float)
    dUx_dz = np.zeros((rows - 2, cols - 2, deps - 2), dtype=float)
    dUy_dz = np.zeros((rows - 2, cols - 2, deps - 2), dtype=float)
    dUz_dz = np.zeros((rows - 2, cols - 2, deps - 2), dtype=float)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            for k in range(1, deps - 1):
                dUx_dx[i - 1, j - 1, k - 1] = dUx_dx_enalarged[i, j, k]
                dUy_dx[i - 1, j - 1, k - 1] = dUy_dx_enalarged[i, j, k]
                dUz_dx[i - 1, j - 1, k - 1] = dUz_dx_enalarged[i, j, k]
                dUx_dy[i - 1, j - 1, k - 1] = dUx_dy_enalarged[i, j, k]
                dUy_dy[i - 1, j - 1, k - 1] = dUy_dy_enalarged[i, j, k]
                dUz_dy[i - 1, j - 1, k - 1] = dUz_dy_enalarged[i, j, k]
                dUx_dz[i - 1, j - 1, k - 1] = dUx_dz_enalarged[i, j, k]
                dUy_dz[i - 1, j - 1, k - 1] = dUy_dz_enalarged[i, j, k]
                dUz_dz[i - 1, j - 1, k - 1] = dUz_dz_enalarged[i, j, k]

    return dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz


def map_elements_to_centraldiff(dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz):
    rows = len(dUx_dx)
    cols = len(dUx_dx[0])
    deps = len(dUx_dx[0][0])
    # Create a list to store tensor displacement for each element
    tensor_displacement_list = np.zeros((rows, cols, deps), dtype=object)

    for i in range(rows):
        for j in range(cols):
            for k in range(deps):
                arr = np.zeros((3, 3), dtype=float)
                arr[0, 0] = dUx_dx[i, j, k]
                arr[0, 1] = dUx_dy[i, j, k]
                arr[0, 2] = dUx_dz[i, j, k]
                arr[1, 0] = dUy_dx[i, j, k]
                arr[1, 1] = dUy_dy[i, j, k]
                arr[1, 2] = dUy_dz[i, j, k]
                arr[2, 0] = dUz_dx[i, j, k]
                arr[2, 1] = dUz_dy[i, j, k]
                arr[2, 2] = dUz_dz[i, j, k]
                tensor_displacement_list[i, j, k] = arr

    return tensor_displacement_list


def calculate_VWS(tensor_displacement_list, E1, E2, v12, v23, Gt, X, Y, Z, Force, cube_size):
    # calculaitng stiffness matrix from E and v prescribed
    E3 = E2
    v21 = E2 / E1 * v12

    v13 = v12  # in this case E1 is the stiffer direction
    v31 = E3 / E1 * v13

    v32 = v23

    Gp = E2 / (2 * (1 + v23))

    G12 = Gt
    G13 = Gt
    G23 = Gp

    S11 = 1 / E1
    S22 = 1 / E2
    S33 = 1 / E3
    S12 = -v21 / E2
    S21 = -v12 / E1
    S13 = -v31 / E3
    S31 = -v13 / E1
    S23 = -v32 / E3
    S32 = -v23 / E2
    S44 = 1 / G12
    S55 = 1 / G13
    S66 = 1 / G23

    S = [
        [S11, S12, S13, 0, 0, 0],
        [S21, S22, S23, 0, 0, 0],
        [S31, S32, S33, 0, 0, 0],
        [0, 0, 0, S44, 0, 0],
        [0, 0, 0, 0, S55, 0],
        [0, 0, 0, 0, 0, S66],
    ]

    C_stiffness = np.linalg.inv(S)
    deformation_gradients = np.array(tensor_displacement_list)
    rows = len(tensor_displacement_list)
    cols = len(tensor_displacement_list[0])
    deps = len(tensor_displacement_list[0][0])
    I = np.eye(3)

    IVW_1 = []
    IVW_2 = []
    IVW_3 = []
    IVW_4 = []

    f = np.zeros((3, 3))
    e = np.zeros((3, 3))
    sigma = np.zeros((3, 3))

    pk1 = np.zeros((3, 3))

    for i in range(rows):
        for j in range(cols):
            for k in range(deps):
                # initially deformation_gradients stored spatial derivatives of displacement

                f = deformation_gradients[i][j][k] + I
                J = np.linalg.det(f)
                f_inv = np.linalg.inv(f)
                C = np.dot(f.T, f)
                e = 1 / 2 * (C - np.eye(3))

                e_vec = [e[0, 0], e[1, 1], e[2, 2], 2 * e[0, 1], 2 * e[0, 2], 2 * e[1, 2]]

                sigma_vec = np.dot(C_stiffness, e_vec)

                sigma = [
                    [sigma_vec[0], sigma_vec[3], sigma_vec[4]],
                    [sigma_vec[3], sigma_vec[1], sigma_vec[5]],
                    [sigma_vec[4], sigma_vec[5], sigma_vec[2]],
                ]
                sigma = np.array(sigma)
                pk1 = J * np.dot(sigma, f_inv.T)

                # Calculate X1 and X2 for each centroid
                X1 = X[i, j, k] - L / 2
                X2 = Y[i, j, k] - L / 2
                X3 = Z[i, j, k]

                du1_dX1_1 = 0
                du1_dX2_1 = 0
                du1_dX3_1 = 0
                du2_dX1_1 = 0
                du2_dX2_1 = 0
                du2_dX3_1 = 0
                du3_dX1_1 = U_star_z_cos_devX(X1, X2, X3)
                du3_dX2_1 = U_star_z_cos_devY(X1, X2, X3)
                du3_dX3_1 = U_star_z_cos_devZ(X1, X2, X3)

                du1_dX1_2 = 0
                du1_dX2_2 = 0
                du1_dX3_2 = 0
                du2_dX1_2 = 0
                du2_dX2_2 = 0
                du2_dX3_2 = 0
                du3_dX1_2 = U_star_z_pw_devX(X1, X2, X3)
                du3_dX2_2 = U_star_z_pw_devY(X1, X2, X3)
                du3_dX3_2 = U_star_z_pw_devZ(X1, X2, X3)

                du1_dX1_3 = U_star_x_para_devX(X1, X2, X3)
                du1_dX2_3 = U_star_x_para_devY(X1, X2, X3)
                du1_dX3_3 = U_star_x_para_devZ(X1, X2, X3)
                du2_dX1_3 = U_star_y_para_devX(X1, X2, X3)
                du2_dX2_3 = U_star_y_para_devY(X1, X2, X3)
                du2_dX3_3 = U_star_y_para_devZ(X1, X2, X3)
                du3_dX1_3 = 0
                du3_dX2_3 = 0
                du3_dX3_3 = 0

                du1_dX1_4 = U_star_x_sin_devX(X1, X2, X3)
                du1_dX2_4 = U_star_x_sin_devY(X1, X2, X3)
                du1_dX3_4 = U_star_x_sin_devZ(X1, X2, X3)
                du2_dX1_4 = U_star_y_sin_devX(X1, X2, X3)
                du2_dX2_4 = U_star_y_sin_devY(X1, X2, X3)
                du2_dX3_4 = U_star_y_sin_devZ(X1, X2, X3)
                du3_dX1_4 = 0
                du3_dX2_4 = 0
                du3_dX3_4 = 0

                du_star_1_dX = np.array(
                    [
                        [du1_dX1_1, du1_dX2_1, du1_dX3_1],
                        [du2_dX1_1, du2_dX2_1, du2_dX3_1],
                        [du3_dX1_1, du3_dX2_1, du3_dX3_1],
                    ]
                )
                du_star_2_dX = np.array(
                    [
                        [du1_dX1_2, du1_dX2_2, du1_dX3_2],
                        [du2_dX1_2, du2_dX2_2, du2_dX3_2],
                        [du3_dX1_2, du3_dX2_2, du3_dX3_2],
                    ]
                )
                du_star_3_dX = np.array(
                    [
                        [du1_dX1_3, du1_dX2_3, du1_dX3_3],
                        [du2_dX1_3, du2_dX2_3, du2_dX3_3],
                        [du3_dX1_3, du3_dX2_3, du3_dX3_3],
                    ]
                )
                du_star_4_dX = np.array(
                    [
                        [du1_dX1_4, du1_dX2_4, du1_dX3_4],
                        [du2_dX1_4, du2_dX2_4, du2_dX3_4],
                        [du3_dX1_4, du3_dX2_4, du3_dX3_4],
                    ]
                )

                ivw_1 = np.tensordot(pk1, du_star_1_dX) * cube_size
                ivw_2 = np.tensordot(pk1, du_star_2_dX) * cube_size
                ivw_3 = np.tensordot(pk1, du_star_3_dX) * cube_size
                ivw_4 = np.tensordot(pk1, du_star_4_dX) * cube_size

                IVW_1.append(ivw_1)
                IVW_2.append(ivw_2)
                IVW_3.append(ivw_3)
                IVW_4.append(ivw_4)

    total_IVW_1 = np.sum(IVW_1)
    total_IVW_2 = np.sum(IVW_2)
    total_IVW_3 = np.sum(IVW_3)
    total_IVW_4 = np.sum(IVW_4)

    # Calculate external virtual work for second set of virtual fields

    evw_1 = -Force * U_star_z_cos(0, 0, H)
    evw_2 = -Force * U_star_z_pw(0, 0, H)
    evw_3 = 0
    evw_4 = 0

    phi = np.array([total_IVW_2 - evw_2, total_IVW_3])

    phi_res = np.sqrt(phi[0] ** 2)

    # return phi*1e10 #return for minimization
    return phi_res  # return for sensitivity matrix


def residual(x, tensor_displacement_list, v12, v23, Gt, X, Y, Z, Force, cube_size):
    E1 = x[0]
    E2 = x[1]

    return calculate_VWS(tensor_displacement_list, E1, E2, v12, v23, Gt, X, Y, Z, Force, cube_size)


def senstivity_full(tensor_displacement_list, E1, E2, v12, v23, Gt, X, Y, Z, Force, cube_size, deviation):
    sens_matrix = np.zeros((5, 5))
    E1_1 = E1 * (1 + deviation)
    E2_1 = E2 * (1 + deviation)
    v12_1 = v12 * (1 + deviation)
    v23_1 = v23 * (1 + deviation)
    Gt_1 = Gt * (1 + deviation)

    phi_base = calculate_VWS(tensor_displacement_list, E1, E2, v12, v23, Gt, X, Y, Z, Force, cube_size)

    phi_E1_1 = calculate_VWS(tensor_displacement_list, E1_1, E2, v12, v23, Gt, X, Y, Z, Force, cube_size)
    phi_E2_1 = calculate_VWS(tensor_displacement_list, E1, E2_1, v12, v23, Gt, X, Y, Z, Force, cube_size)
    phi_v12_1 = calculate_VWS(tensor_displacement_list, E1, E2, v12_1, v23, Gt, X, Y, Z, Force, cube_size)
    phi_v23_1 = calculate_VWS(tensor_displacement_list, E1, E2, v12, v23_1, Gt, X, Y, Z, Force, cube_size)
    phi_Gt_1 = calculate_VWS(tensor_displacement_list, E1, E2, v12, v23, Gt_1, X, Y, Z, Force, cube_size)

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
    sens_matrix = sens_matrix / np.min(sens_matrix)

    # Print the formatted 5x5 matrix
    print("Sensitivity Matrix (5x5):")
    for row in sens_matrix:
        print(" ".join(f"{value:10.4f}" for value in row))

    return sens_matrix


def main():
    X = np.load(r"80um\X.npy")
    Y = np.load(r"80um\Y.npy")
    Z = np.load(r"80um\Z.npy")
    Ux = np.load(r"80um\Ux.npy")
    Uy = np.load(r"80um\Uy.npy")
    Uz = np.load(r"80um\Uz.npy")

    X_enlarged = increase_matrix_size(X)
    Y_enlarged = increase_matrix_size(Y)
    Z_enlarged = increase_matrix_size(Z)
    Ux_enlarged = increase_matrix_size(Ux)
    Uy_enlarged = increase_matrix_size(Uy)
    Uz_enlarged = increase_matrix_size(Uz)

    dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz = central_differentiation(
        Ux_enlarged, Uy_enlarged, Uz_enlarged, X_enlarged, Y_enlarged, Z_enlarged
    )
    tensor_displacement_list = map_elements_to_centraldiff(
        dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz
    )

    global L, W, H
    L = np.ceil((np.max(X) - np.min(X)) * 1e4) / 1e4
    W = np.ceil((np.max(Y) - np.min(Y)) * 1e4) / 1e4
    H = np.ceil((np.max(Z) - np.min(Z)) * 1e4) / 1e4

    L_dim = np.mean(np.diff(X[0, :, 0]))
    W_dim = np.mean(np.diff(Y[:, 0, 0]))
    H_dim = np.mean(np.diff(Z[0, 0, :]))
    cube_size = L_dim * W_dim * H_dim

    E1 = 6000
    E2 = 1500
    v12 = 0.49
    v23 = 0.49
    Gt = 0.5e3

    initial_guess = np.array([7000, 500])
    bnds = ((2000, 500), (9000, 2500))
    res_1 = least_squares(
        residual, initial_guess, bounds=bnds, args=(tensor_displacement_list, v12, v23, Gt, X, Y, Z, Force, cube_size)
    )
    print(res_1.x)

    sens = senstivity_full(tensor_displacement_list, E1, E2, v12, v23, Gt, X, Y, Z, Force, cube_size, 0.05)


if __name__ == "__main__":
    main()
