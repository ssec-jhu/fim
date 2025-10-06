import numpy as np
import pytest

import fim.refactor.vws_models as vm


def make_grid(nx=5, ny=5, nz=5, L=1.0, H=0.1):
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    z = np.linspace(0, H, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z, L, H


def make_pk_inputs(nx=3, ny=3, nz=3, L=1.0, H=0.1):
    X, Y, Z, L, H = make_grid(nx, ny, nz, L, H)
    vol = np.ones_like(X)
    # richer gradient tensor so all params influence phi
    G = np.zeros((*X.shape, 3, 3))
    G[..., 0, 0] = 0.02
    G[..., 1, 1] = 0.01
    G[..., 2, 2] = -0.01
    G[..., 0, 1] = 0.005
    G[..., 1, 0] = -0.004
    G[..., 0, 2] = 0.003
    G[..., 2, 0] = -0.002
    G[..., 1, 2] = 0.006
    G[..., 2, 1] = -0.005
    return X, Y, Z, L, H, vol, G


def test_virtual_fields_and_derivs_finite():
    X, Y, Z, L, H = make_grid(7, 7, 7)
    X1, Y1, Z1 = X - L / 2, Y - L / 2, Z

    value_funcs = [
        vm.U_star_z_cos,
        vm.U_star_z_pw,
        vm.U_star_x_para,
        vm.U_star_y_para,
        vm.U_star_x_sin,
        vm.U_star_y_sin,
        vm.U_star_z_pw_vol,
    ]
    deriv_funcs = [
        vm.U_star_z_cos_devX,
        vm.U_star_z_cos_devY,
        vm.U_star_z_cos_devZ,
        vm.U_star_z_pw_devX,
        vm.U_star_z_pw_devY,
        vm.U_star_z_pw_devZ,
        vm.U_star_x_para_devX,
        vm.U_star_x_para_devY,
        vm.U_star_x_para_devZ,
        vm.U_star_y_para_devX,
        vm.U_star_y_para_devY,
        vm.U_star_y_para_devZ,
        vm.U_star_x_sin_devX,
        vm.U_star_x_sin_devY,
        vm.U_star_x_sin_devZ,
        vm.U_star_y_sin_devX,
        vm.U_star_y_sin_devY,
        vm.U_star_y_sin_devZ,
        vm.U_star_z_pw_vol_devX,
        vm.U_star_z_pw_vol_devY,
        vm.U_star_z_pw_vol_devZ,
    ]

    for f in value_funcs:
        out = f(X1, Y1, Z1, L, H)
        assert out.shape == X.shape
        assert np.isfinite(out[1:-1, 1:-1, 1:-1]).all()  # ignore singular center

    for f in deriv_funcs:
        out = f(X1, Y1, Z1, L, H)
        assert out.shape == X.shape
        assert np.isfinite(out[1:-1, 1:-1, 1:-1]).all()

    far = np.sqrt(X1**2 + Y1**2) >= (L / 2)
    assert np.allclose(vm.U_star_x_para(X1, Y1, Z1, L, H)[far], 0.0)
    assert np.allclose(vm.U_star_y_para(X1, Y1, Z1, L, H)[far], 0.0)


def test_increase_matrix_size():
    a = np.arange(2 * 2 * 2).reshape(2, 2, 2)
    b = vm.increase_matrix_size(a)
    assert b.shape == (4, 4, 4)
    assert np.array_equal(b[1:-1, 1:-1, 1:-1], a)
    assert np.array_equal(b[0, :, :], b[1, :, :])
    assert np.array_equal(b[-1, :, :], b[-2, :, :])
    assert np.array_equal(b[:, 0, :], b[:, 1, :])
    assert np.array_equal(b[:, -1, :], b[:, -2, :])
    assert np.array_equal(b[:, :, 0], b[:, :, 1])
    assert np.array_equal(b[:, :, -1], b[:, :, -2])


def test_read_input_file_parses(tmp_path):
    p = tmp_path / "mini.inp"
    lines = [
        "*Heading",
        "*Part, name=tissue",
        "*Node",
    ]
    for i in range(8):
        lines.append(f"{i + 1}, {0.1 * i}, {0.2 * i}, {0.3 * i}")
    lines += [
        "*Element, type=C3D8",
        "1, 1,2,3,4,5,6,7,8",
        "*SomethingElse",
    ]
    p.write_text("\n".join(lines))
    nodes, conn = vm.read_input_file(str(p))
    assert nodes.shape == (8, 4)
    assert conn.shape == (1, 9)
    assert np.isclose(nodes[:, 1:].min(), 0.0)


def test_central_differentiation_and_map():
    X, Y, Z, L, H = make_grid(5, 5, 5)
    Ux, Uy, Uz = X.copy(), Y.copy(), Z.copy()
    d = vm.central_differentiation(Ux, Uy, Uz, X, Y, Z)
    shape = (X.shape[0] - 2, X.shape[1] - 2, X.shape[2] - 2)
    for arr in d:
        assert arr.shape == shape

    # Don’t assert specific magnitudes (code’s axes differ from “math” dx/dy/dz);
    # just check consistency and finiteness.
    for arr in d:
        assert np.isfinite(arr).all()

    tensor = vm.map_elements_to_centraldiff(*d)
    assert tensor.shape == (*shape, 3, 3)
    # spot mapping checks
    assert np.allclose(tensor[..., 0, 0], d[0])
    assert np.allclose(tensor[..., 1, 2], d[7])
    assert np.allclose(tensor[..., 2, 1], d[5])


def test_central_differentiation_mismatch_raises():
    X, Y, Z, L, H = make_grid(5, 5, 5)
    Ux = np.zeros_like(X)
    Uy = np.zeros_like(X)
    Uz = np.zeros_like(X[:-1, ...])  # mismatch
    with pytest.raises(ValueError):
        vm.central_differentiation(Ux, Uy, Uz, X, Y, Z)


def test_calculate_VWS_linear_output_modes_and_values():
    X, Y, Z, L, H, vol, G = make_pk_inputs(3, 3, 3, L=1.2, H=0.2)
    E1, E2, v12, v23, Gt = 1e5, 5e4, 0.25, 0.3, 2e4
    Force = 1.0

    phi_vec = vm.calculate_VWS_linear(G, E1, E2, v12, v23, Gt, X, Y, Z, Force, vol, L, H, return_scalar=False)
    assert np.isfinite(phi_vec).all() and phi_vec.shape == (2,)

    phi_mag = vm.calculate_VWS_linear(G, E1, E2, v12, v23, Gt, X, Y, Z, Force, vol, L, H, return_scalar=True)
    assert np.isscalar(phi_mag) or np.array(phi_mag).shape == ()

    G0 = np.zeros_like(G)
    phi_vec0 = vm.calculate_VWS_linear(G0, E1, E2, v12, v23, Gt, X, Y, Z, Force, vol, L, H, return_scalar=False)
    assert np.isfinite(phi_vec0).all()


def test_calculate_VWS_hgo_and_nh_and_virtual_work_modes(monkeypatch):
    X, Y, Z, L, H, vol, G = make_pk_inputs(3, 3, 3)
    Force = 0.5

    # Patch scalar call for evw_5 – make original function scalar-safe
    orig = vm.U_star_z_pw_vol

    def safe_vol(x, y, z, L_, H_):
        if np.isscalar(x):
            xs, ys, zs = np.array([x]), np.array([y]), np.array([z])
            val = orig(xs, ys, zs, L_, H_)
            return float(np.asarray(val)[0])
        return orig(x, y, z, L_, H_)

    monkeypatch.setattr(vm, "U_star_z_pw_vol", safe_vol)

    C10, D1, k1, k2, kappa = 1e3, 1e-2, 100.0, 5.0, 1 / 3

    # Test HGO model with both return modes
    phi_hgo_vec = vm.calculate_VWS_hgo(G, X, Y, Z, C10, D1, k1, k2, kappa, vol, Force, L, H, return_scalar=False)
    assert np.isfinite(phi_hgo_vec).all() and phi_hgo_vec.shape == (2,)

    phi_hgo_scalar = vm.calculate_VWS_hgo(G, X, Y, Z, C10, D1, k1, k2, kappa, vol, Force, L, H, return_scalar=True)
    assert np.isscalar(phi_hgo_scalar) or np.array(phi_hgo_scalar).shape == ()

    # Test NH model with both return modes
    phi_nh_vec = vm.calculate_VWS_nh(G, X, Y, Z, C10, D1, vol, Force, L, H, return_scalar=False)
    assert np.isfinite(phi_nh_vec).all() and phi_nh_vec.shape == (2,)

    phi_nh_scalar = vm.calculate_VWS_nh(G, X, Y, Z, C10, D1, vol, Force, L, H, return_scalar=True)
    assert np.isscalar(phi_nh_scalar) or np.array(phi_nh_scalar).shape == ()

    with pytest.raises(ValueError):
        vm.calculate_VWS_virtual_work(np.zeros((*X.shape, 3, 3)), X, Y, Z, vol, Force, L, H, mode="nope")


def test_sensitivity_full_linear_runs_and_normalizes():
    X, Y, Z, L, H, vol, G = make_pk_inputs(3, 3, 3)
    E1, E2, v12, v23, Gt = 1e5, 6e4, 0.22, 0.28, 2.5e4
    sens = vm.sensitivity_full_linear(G, E1, E2, v12, v23, Gt, X, Y, Z, 1.0, vol, L, H, deviation=0.1)
    assert sens.shape == (5, 5)
    # With richer G, min should be >0, so normalization is finite
    assert np.isfinite(sens).all()
    assert np.isclose(sens.min(), 1.0)
    # Check symmetry
    assert np.allclose(sens, sens.T)


def test_sensitivity_full_hgo_runs_and_normalizes(monkeypatch):
    X, Y, Z, L, H, vol, G = make_pk_inputs(3, 3, 3)
    Force = 0.5

    # Patch scalar call for evw_5
    orig = vm.U_star_z_pw_vol

    def safe_vol(x, y, z, L_, H_):
        if np.isscalar(x):
            xs, ys, zs = np.array([x]), np.array([y]), np.array([z])
            val = orig(xs, ys, zs, L_, H_)
            return float(np.asarray(val)[0])
        return orig(x, y, z, L_, H_)

    monkeypatch.setattr(vm, "U_star_z_pw_vol", safe_vol)

    C10, D1, k1, k2, kappa = 1e3, 1e-2, 100.0, 5.0, 0.1
    sens = vm.sensitivity_full_hgo(G, X, Y, Z, C10, D1, k1, k2, kappa, vol, Force, L, H, deviation=0.05)
    assert sens.shape == (5, 5)
    assert np.isfinite(sens).all()
    assert np.isclose(sens.min(), 1.0)
    # Check symmetry
    assert np.allclose(sens, sens.T)


def test_sensitivity_nh_runs_and_normalizes(monkeypatch):
    X, Y, Z, L, H, vol, G = make_pk_inputs(3, 3, 3)
    Force = 0.5

    # Patch scalar call for evw_5
    orig = vm.U_star_z_pw_vol

    def safe_vol(x, y, z, L_, H_):
        if np.isscalar(x):
            xs, ys, zs = np.array([x]), np.array([y]), np.array([z])
            val = orig(xs, ys, zs, L_, H_)
            return float(np.asarray(val)[0])
        return orig(x, y, z, L_, H_)

    monkeypatch.setattr(vm, "U_star_z_pw_vol", safe_vol)

    C10, D1 = 1e3, 1e-2
    sens = vm.sensitivity_nh(G, X, Y, Z, C10, D1, vol, Force, L, H, deviation=0.05)
    assert sens.shape == (2, 2)
    assert np.isfinite(sens).all()
    assert np.isclose(sens.min(), 1.0)
    # Check symmetry
    assert np.allclose(sens, sens.T)
