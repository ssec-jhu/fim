import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares, minimize
from multiprocessing import Pool


decimal_places = 10
index = 0

depth_indentation = 3.2e-5
sphere_radius = 5e-4

Force = 1.20202e-05


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
    a_0 = contact_radius * 10
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
    a_0 = contact_radius * 10
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
    a_0 = contact_radius * 10
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
    a_0 = contact_radius * 10
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


def U_star_z_pw_vol(x, y, z):
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    a_0 = contact_radius * 2
    t = z / H
    a = a_0
    k = (c / 2 * (L / 4 - a) - c * L / 4) / ((L / 4 - a) / 2 + L / 8)
    if d <= a:
        return c * t
    elif d > a and d <= (L / 4):
        return t * (c + (k - c) * (d - a) / (L / 4 - a))
    elif d > (L / 4) and d < L / 2:
        return t * (k - 4 * k * (d - L / 4) / L)
    else:
        return 0


def U_star_z_pw_vol_devX(x, y, z):
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    a_0 = contact_radius * 2
    t = z / H
    a = a_0
    k = (c / 2 * (L / 4 - a) - c * L / 4) / ((L / 4 - a) / 2 + L / 8)
    if d <= a:
        return 0
    elif d > a and d <= (L / 4):
        return t * ((k - c) * (x / d) / (L / 4 - a))
    elif d > (L / 4) and d < L / 2:
        return t * (-4 * k * (x / d) / L)
    else:
        return 0


def U_star_z_pw_vol_devY(x, y, z):
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    a_0 = contact_radius * 2
    t = z / H
    a = a_0
    k = (c / 2 * (L / 4 - a) - c * L / 4) / ((L / 4 - a) / 2 + L / 8)
    if d <= a:
        return 0
    elif d > a and d <= (L / 4):
        return t * ((k - c) * (y / d) / (L / 4 - a))
    elif d > (L / 4) and d < L / 2:
        return t * (-4 * k * (y / d) / L)
    else:
        return 0


def U_star_z_pw_vol_devZ(x, y, z):
    c = 5e-5
    d = np.sqrt(x**2 + y**2)
    a_0 = contact_radius * 2
    t = z / H
    a = a_0
    k = (c / 2 * (L / 4 - a) - c * L / 4) / ((L / 4 - a) / 2 + L / 8)
    if d <= a:
        return c / H
    elif d > a and d <= (L / 4):
        return (1 / H) * (c + (k - c) * (d - a) / (L / 4 - a))
    elif d > (L / 4) and d < L / 2:
        return (1 / H) * (k - 4 * k * (d - L / 4) / L)
    else:
        return 0


def read_output_file(file_path):
    # drop columns other than the displacements/field output and select the part name to select displacements of ROI
    df = pd.read_csv(file_path)
    df = df.drop(
        columns=["ODB Name", "Step", "Frame", "Section Name", "Material Name", "Section Point", "X", "Y", "Z"], axis=1
    )

    nodes_displacement = df.loc[df["Part Instance Name"] == "TISSUE-1"]
    nodes_displacement = nodes_displacement.drop("Part Instance Name", axis=1)

    nodes_displacement = np.array(nodes_displacement)
    return nodes_displacement


def read_input_file(file_path):
    nodes = []  # List to store nodal coordinates
    connectivity = []  # List to store element connectivity
    in_node_section = False
    in_element_section = False
    file = open(file_path, "r")
    Lines = file.readlines()
    flag = 0

    for line in Lines:
        line = line.strip()
        # change according to the input file
        if line.startswith("*") and line.startswith("*Part, name=tissue"):
            flag = 1
            continue
        elif line.startswith("*Node") and flag == 1:
            in_node_section = True
            in_element_section = False
            continue
        elif line.startswith("*Element") and flag == 1:
            in_node_section = False
            in_element_section = True
            continue
        elif line.startswith("*"):
            flag = 0
            continue
        else:
            if in_node_section and not in_element_section and flag == 1:
                values_n = line.split(",")
                node_info = [int(values_n[0])] + [np.double(values_n[i].strip()) for i in range(1, 4)]
                nodes.append(node_info)
                continue

            elif not in_node_section and in_element_section and flag == 1:
                values_e = line.split(",")
                if len(values_e) == 9:
                    element_info = [int(value) for value in values_e]
                    connectivity.append(element_info)
                continue

            else:
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


def calculate_nodes_deformed(nodes, nodes_displacement):
    number_nodes = min(nodes.shape[0], nodes_displacement.shape[0])
    nodes_deformed = np.zeros((number_nodes, 4))
    nodes_deformed[:, 0] = nodes[:number_nodes, 0]
    nodes_deformed[:number_nodes, 1:] = np.double(nodes[:number_nodes, 1:4] + nodes_displacement[:number_nodes, 1:4])

    return nodes_deformed


def create_centroids(nodes, connectivity):
    centroids = []
    for element in connectivity:
        x, y, z = 0, 0, 0
        for i in element[1:]:
            x = x + nodes[i - 1, 1]
            y = y + nodes[i - 1, 2]
            z = z + nodes[i - 1, 3]
        x = x / 8
        y = y / 8
        z = z / 8
        centroids.append([int(element[0]), x, y, z])
    centroids = np.array(centroids)

    return centroids


def calculate_volumes(nodes, connectivity):
    centroids = []
    for element in connectivity:
        x = []
        y = []
        z = []
        for i in element[1:]:
            x.append(nodes[i - 1, 1])
            y.append(nodes[i - 1, 2])
            z.append(nodes[i - 1, 3])
        x = np.unique(x)
        y = np.unique(y)
        z = np.unique(z)
        vol = (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0])
        centroids.append([int(element[0]), vol, 0, 0])
    centroids = np.array(centroids)

    return centroids


def calculate_cube_size(centroids, decimal_places):
    x_coordinates = np.round(np.sort(np.unique(centroids[:, 1])), 8)
    y_coordinates = np.round(np.sort(np.unique(centroids[:, 2])), 8)
    z_coordinates = np.round(np.sort(np.unique(centroids[:, 3])), 8)

    # calculating the cube size
    cube_size = [
        x_coordinates[-1] - x_coordinates[-2],
        y_coordinates[-1] - y_coordinates[-2],
        z_coordinates[-1] - z_coordinates[-2],
    ]

    return [cube_size[0], cube_size[1], cube_size[2]]


def map_element_numbers_to_matrix(centroids, cube_size):
    x_coordinates = np.sort(np.unique(centroids[:, 1]))
    y_coordinates = np.sort(np.unique(centroids[:, 2]))
    z_coordinates = np.sort(np.unique(centroids[:, 3]))

    cols = np.size(x_coordinates)
    rows = np.size(y_coordinates)
    deps = np.size(z_coordinates)

    print(rows)
    print(cols)
    print(deps)

    # exit()

    matrix = np.zeros((rows, cols, deps), dtype=int)
    for centroid in centroids:
        element_number = centroid[0]
        x = centroid[1]
        y = centroid[2]
        z = centroid[3]

        # Calculate indices on matrix
        matrix_x = np.where(x_coordinates == x)[0][0]
        matrix_y = np.where(y_coordinates == y)[0][0]
        matrix_z = np.where(z_coordinates == z)[0][0]

        # if element_number==63:
        # print(matrix_y, matrix_x, matrix_z)
        # Assign element number to matrix
        matrix[matrix_y, matrix_x, matrix_z] = element_number

    return matrix


def map_elements_to_displacement(elements_tensor, displacement_centroids):
    # Initialize three matrices for X, Y, and Z displacements
    Ux = np.zeros_like(elements_tensor, dtype=np.double)
    Uy = np.zeros_like(elements_tensor, dtype=np.double)
    Uz = np.zeros_like(elements_tensor, dtype=np.double)

    # Fill the displacement matrices with data from displacement_centroids
    for displacement_data in displacement_centroids:
        element_number = int(displacement_data[0])
        x_displacement = displacement_data[1]
        y_displacement = displacement_data[2]
        z_displacement = displacement_data[3]

        # Find the coordinates of the element in the elements_tensor
        coords = np.argwhere(elements_tensor == element_number)
        if coords.size == 0:
            print(element_number)
            exit()
        y, x, z = coords[0]

        Ux[y, x, z] = x_displacement
        Uy[y, x, z] = y_displacement
        Uz[y, x, z] = z_displacement

    return Ux, Uy, Uz


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


def calculate_VWS(tensor_displacement_list, X, Y, Z, C10, D1, k1, k2, kappa, volume_matrix, Force):
    deformation_gradients = np.array(tensor_displacement_list)
    rows = len(tensor_displacement_list)
    cols = len(tensor_displacement_list[0])
    deps = len(tensor_displacement_list[0][0])
    I = np.eye(3)

    IVW_1 = []
    IVW_2 = []
    IVW_3 = []
    IVW_4 = []
    IVW_5 = []

    f = np.zeros((3, 3))
    sigma = np.zeros((3, 3))
    a04 = np.array([1, 0, 0]).T
    pk1 = np.zeros((3, 3))

    for i in range(rows):
        for j in range(cols):
            for k in range(deps):
                # initially deformation_gradients stored spatial derivatives of displacement
                # element_number=matrix[i,j,k]
                volume_element = volume_matrix[i, j, k]
                f = deformation_gradients[i][j][k] + I
                J = np.linalg.det(f)
                f_iso = J ** (-1 / 3) * f
                c = np.dot(f.T, f)
                b = np.dot(f, f.T)
                b_iso = J ** (-2 / 3) * b
                f_inv = np.linalg.inv(f)
                I = np.eye(3)
                I1 = np.trace(c)
                I1_iso = J ** (-2 / 3) * I1

                I4 = np.dot(a04, np.dot(c, a04))  # also lambda4 ^2
                I4_iso = (J ** (-2 / 3)) * I4
                a4 = np.dot(f, a04)
                a4_iso = np.dot(f_iso, a04)
                A4 = np.outer(a4, a4)
                A4_iso = np.outer(a4_iso, a4_iso)

                E_iso = kappa * I1_iso + (1 - 3 * kappa) * I4_iso - 1

                sigma_iso = (2 * C10 / J ** (5 / 3)) * (b - I * I1 / 3)
                sigma_aniso = (
                    (2 * k1 * E_iso / J ** (5 / 3))
                    * (np.exp(k2 * E_iso**2))
                    * (kappa * b + (1 - 3 * kappa) * A4 + (-1 / 3) * np.eye(3) * (kappa * I1 + (1 - 3 * kappa) * I4))
                )
                sigma_vol = (1 / D1) * (J - 1 / J) * I
                sigma = sigma_aniso + sigma_iso + sigma_vol
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

                du1_dX1_5 = 0
                du1_dX2_5 = 0
                du1_dX3_5 = 0
                du2_dX1_5 = 0
                du2_dX2_5 = 0
                du2_dX3_5 = 0
                du3_dX1_5 = U_star_z_pw_vol_devX(X1, X2, X3)
                du3_dX2_5 = U_star_z_pw_vol_devY(X1, X2, X3)
                du3_dX3_5 = U_star_z_pw_vol_devZ(X1, X2, X3)

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
                du_star_5_dX = np.array(
                    [
                        [du1_dX1_5, du1_dX2_5, du1_dX3_5],
                        [du2_dX1_5, du2_dX2_5, du2_dX3_5],
                        [du3_dX1_5, du3_dX2_5, du3_dX3_5],
                    ]
                )

                ivw_1 = np.tensordot(pk1, du_star_1_dX) * volume_element
                ivw_2 = np.tensordot(pk1, du_star_2_dX) * volume_element
                ivw_3 = np.tensordot(pk1, du_star_3_dX) * volume_element
                ivw_4 = np.tensordot(pk1, du_star_4_dX) * volume_element
                ivw_5 = np.tensordot(pk1, du_star_5_dX) * volume_element

                IVW_1.append(ivw_1)
                IVW_2.append(ivw_2)
                IVW_3.append(ivw_3)
                IVW_4.append(ivw_4)
                IVW_5.append(ivw_5)

    total_IVW_1 = np.sum(IVW_1)
    total_IVW_2 = np.sum(IVW_2)
    total_IVW_3 = np.sum(IVW_3)
    total_IVW_4 = np.sum(IVW_4)
    total_IVW_5 = np.sum(IVW_5)

    evw_1 = -Force * U_star_z_cos(0, 0, H)
    evw_2 = -Force * U_star_z_pw(0, 0, H)
    evw_3 = 0
    evw_4 = -Force * U_star_z_pw(0, 0, H)
    evw_5 = -Force * U_star_z_pw_vol(0, 0, H)

    # Cost function

    phi = np.array([total_IVW_5 - evw_5, total_IVW_3])
    phi_res = np.sqrt(phi[0] ** 2 + phi[1] ** 2)
    print(C10, D1, kappa, phi_res)

    return phi * 1e10
    # return phi_res


def residual_3(x, tensor_displacement_list, X, Y, Z, k1, k2, volume_matrix, Force):
    C10 = x[0]
    D1 = x[1]
    kappa = x[2]

    return calculate_VWS(tensor_displacement_list, X, Y, Z, C10, D1, k1, k2, kappa, volume_matrix, Force)


def senstivity_full(tensor_displacement_list, X, Y, Z, C10, D1, k1, k2, kappa, volume_matrix, Force, deviation):
    sens_matrix = np.zeros((5, 5))
    C10_1 = C10 * (1 + deviation)
    D1_1 = D1 * (1 + deviation)
    k1_1 = k1 * (1 + deviation)
    k2_1 = k2 * (1 + deviation)
    kappa_1 = kappa * (1 + deviation)

    phi_base = calculate_VWS(tensor_displacement_list, X, Y, Z, C10, D1, k1, k2, kappa, volume_matrix, Force)

    phi_C10_1 = calculate_VWS(tensor_displacement_list, X, Y, Z, C10_1, D1, k1, k2, kappa, volume_matrix, Force)
    phi_D1_1 = calculate_VWS(tensor_displacement_list, X, Y, Z, C10, D1_1, k1, k2, kappa, volume_matrix, Force)
    phi_k1_1 = calculate_VWS(tensor_displacement_list, X, Y, Z, C10, D1, k1_1, k2, kappa, volume_matrix, Force)
    phi_k2_1 = calculate_VWS(tensor_displacement_list, X, Y, Z, C10, D1, k1, k2_1, kappa, volume_matrix, Force)
    phi_kappa_1 = calculate_VWS(tensor_displacement_list, X, Y, Z, C10, D1, k1, k2, kappa_1, volume_matrix, Force)

    sens_matrix[0, 0] = ((phi_C10_1 - phi_base) / (C10 * deviation)) ** 2
    sens_matrix[1, 1] = ((phi_D1_1 - phi_base) / (D1 * deviation)) ** 2
    sens_matrix[2, 2] = ((phi_k1_1 - phi_base) / (k1 * deviation)) ** 2
    sens_matrix[3, 3] = ((phi_k2_1 - phi_base) / (k2 * deviation)) ** 2
    sens_matrix[4, 4] = ((phi_kappa_1 - phi_base) / (kappa * deviation)) ** 2

    sens_matrix[0, 1] = ((phi_C10_1 - phi_base) / (C10 * deviation)) * ((phi_D1_1 - phi_base) / (D1 * deviation))
    sens_matrix[0, 2] = ((phi_C10_1 - phi_base) / (C10 * deviation)) * ((phi_k1_1 - phi_base) / (k1 * deviation))
    sens_matrix[0, 3] = ((phi_C10_1 - phi_base) / (C10 * deviation)) * ((phi_k2_1 - phi_base) / (k2 * deviation))
    sens_matrix[0, 4] = ((phi_C10_1 - phi_base) / (C10 * deviation)) * ((phi_kappa_1 - phi_base) / (kappa * deviation))
    sens_matrix[1, 0] = sens_matrix[0, 1]
    sens_matrix[1, 2] = ((phi_D1_1 - phi_base) / (D1 * deviation)) * ((phi_k1_1 - phi_base) / (k1 * deviation))
    sens_matrix[1, 3] = ((phi_D1_1 - phi_base) / (D1 * deviation)) * ((phi_k2_1 - phi_base) / (k2 * deviation))
    sens_matrix[1, 4] = ((phi_D1_1 - phi_base) / (D1 * deviation)) * ((phi_kappa_1 - phi_base) / (kappa * deviation))
    sens_matrix[2, 0] = sens_matrix[0, 2]
    sens_matrix[2, 1] = sens_matrix[1, 2]
    sens_matrix[2, 3] = ((phi_k1_1 - phi_base) / (k1 * deviation)) * ((phi_k2_1 - phi_base) / (k2 * deviation))
    sens_matrix[2, 4] = ((phi_k1_1 - phi_base) / (k1 * deviation)) * ((phi_kappa_1 - phi_base) / (kappa * deviation))
    sens_matrix[3, 0] = sens_matrix[0, 3]
    sens_matrix[3, 1] = sens_matrix[1, 3]
    sens_matrix[3, 2] = sens_matrix[2, 3]
    sens_matrix[3, 4] = ((phi_k2_1 - phi_base) / (k2 * deviation)) * ((phi_kappa_1 - phi_base) / (kappa * deviation))
    sens_matrix[4, 0] = sens_matrix[0, 4]
    sens_matrix[4, 1] = sens_matrix[1, 4]
    sens_matrix[4, 2] = sens_matrix[2, 4]
    sens_matrix[4, 3] = sens_matrix[3, 4]

    sens_matrix = np.abs(sens_matrix)
    # sens_matrix = sens_matrix / np.min(sens_matrix)

    # Print the formatted 5x5 matrix
    print("Sensitivity Matrix (5x5):")
    for row in sens_matrix:
        print(" ".join(f"{value:10.4f}" for value in row))

    return sens_matrix


def main():
    file_path_undeformed = r"HGO/350k.inp"

    undeformed_nodes, connectivity = read_input_file(file_path_undeformed)

    global L, W, H
    L = abs(np.max(undeformed_nodes[:, 1]) - np.min(undeformed_nodes[:, 1]))
    W = abs(np.max(undeformed_nodes[:, 2]) - np.min(undeformed_nodes[:, 2]))
    H = abs(np.max(undeformed_nodes[:, 3]) - np.min(undeformed_nodes[:, 3]))

    X = np.load(r"HGO/X.npy")
    Y = np.load(r"HGO/Y.npy")
    Z = np.load(r"HGO/Z.npy")
    Ux = np.load(r"HGO/Ux.npy")
    Uy = np.load(r"HGO/Uy.npy")
    Uz = np.load(r"HGO/Uz.npy")
    volume_matrix = np.load(r"HGO/volume_matrix.npy")

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

    C10 = 267
    D1 = 8e-4
    k1 = 2000
    k2 = 5
    kappa = 0

    # needs to be fixed
    # sens=senstivity_full(tensor_displacement_list, X,Y,Z,C10,D1,k1,k2,kappa,volume_matrix,Force,0.05)

    initial_guess = np.array([500, 1e-5, 0.05])
    bnds = ((0, 1e-5, 0), (1000, 1e-3, 0.33))

    res_1 = least_squares(
        residual_3, initial_guess, bounds=bnds, args=(tensor_displacement_list, X, Y, Z, k1, k2, volume_matrix, Force)
    )
    print(res_1.x)


if __name__ == "__main__":
    main()
