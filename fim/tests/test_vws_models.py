import numpy as np
import pytest

import fim.refactor.vws_models as vws  # use the refactored module only


# ---------- small helpers / fixtures ----------
@pytest.fixture
def small_grid():
    """5x5x5 grid in strict (x,y,z) order with unit spacing."""
    n = 5
    xs = ys = zs = np.linspace(0.0, 4.0, n)
    # indexing='ij' ensures axis0=x, axis1=y, axis2=z
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    vol = np.ones_like(X)
    L, H = 4.0, 4.0
    return X, Y, Z, vol, L, H


@pytest.fixture
def linear_fields(small_grid):
    X, Y, Z, *_ = small_grid
    a, b, c = 0.01, -0.02, 0.03
    Ux = a * X
    Uy = b * Y
    Uz = c * Z
    return Ux, Uy, Uz, a, b, c


@pytest.fixture
def tensor_zero(small_grid):
    X, Y, Z, *_ = small_grid
    return np.zeros(X.shape + (3, 3), dtype=float)


# ---------- unit tests ----------
def test_increase_matrix_size_basic():
    A = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    B = vws.increase_matrix_size(A)
    assert B.shape == (4, 5, 6)
    np.testing.assert_array_equal(B[1:-1, 1:-1, 1:-1], A)
    assert np.all(B[0, :, :] == B[1, :, :])
    assert np.all(B[:, 0, :] == B[:, 1, :])
    assert np.all(B[:, :, 0] == B[:, :, 1])


def test_central_differentiation_linear_field(linear_fields, small_grid):
    Ux, Uy, Uz, a, b, c = linear_fields
    X, Y, Z, *_ = small_grid
    outs = vws.central_differentiation(Ux, Uy, Uz, X, Y, Z)
    dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz = outs
    assert dUx_dx.shape == (3, 3, 3)
    np.testing.assert_allclose(dUx_dx, a, atol=1e-10)
    np.testing.assert_allclose(dUy_dy, b, atol=1e-10)
    np.testing.assert_allclose(dUz_dz, c, atol=1e-10)
    for arr in [dUy_dx, dUz_dx, dUx_dy, dUz_dy, dUx_dz, dUy_dz]:
        np.testing.assert_allclose(arr, 0.0, atol=1e-10)


def test_map_elements_to_centraldiff_shape(linear_fields, small_grid):
    Ux, Uy, Uz, *_ = linear_fields
    X, Y, Z, *_ = small_grid
    grads = vws.central_differentiation(Ux, Uy, Uz, X, Y, Z)
    T = vws.map_elements_to_centraldiff(*grads)
    assert T.shape == (3, 3, 3, 3, 3)
    np.testing.assert_allclose(T[..., 0, 0], grads[0])
    np.testing.assert_allclose(T[..., 1, 1], grads[4])
    np.testing.assert_allclose(T[..., 2, 2], grads[8])


def test_calculate_VWS_linear_zero_grad_returns_finite_vector(tensor_zero, small_grid):
    X, Y, Z, vol, L, H = small_grid
    Force = 2.0
    E1, E2, v12, v23, Gt = 7000.0, 500.0, 0.3, 0.3, 400.0

    phi = vws.calculate_VWS_linear(tensor_zero, E1, E2, v12, v23, Gt, X, Y, Z, Force, vol, L, H, output=1)

    # From the code: returns phi * 1e10 where phi is from calculate_VWS_virtual_work
    # calculate_VWS_virtual_work with mode="linear" returns np.array([total_IVW_2 - evw_2, total_IVW_3])
    # So this should be a 2-element array
    assert isinstance(phi, np.ndarray), f"Expected array, got {type(phi)}"
    assert phi.shape == (2,), f"Expected shape (2,), got {phi.shape}"
    assert np.isfinite(phi).all()

    c = 5e-5
    np.testing.assert_allclose(phi[0], Force * c * 1e10, atol=1e-6)
    np.testing.assert_allclose(phi[1], 0.0, atol=1e-10)


def test_calculate_VWS_hgo_zero_grad_matches_expected_scale(tensor_zero, small_grid):
    X, Y, Z, vol, L, H = small_grid
    Force = 3.0
    C10, D1, k1, k2, kappa = 1.0, 1e-3, 2.0, 3.0, 0.2

    phi = vws.calculate_VWS_hgo(tensor_zero, X, Y, Z, C10, D1, k1, k2, kappa, vol, Force, L, H)

    # From the code: returns phi * 1e10 where phi is from calculate_VWS_virtual_work
    # calculate_VWS_virtual_work with mode="hgo" returns np.array([total_IVW_5 - evw_5, total_IVW_3])
    # So this should be a 2-element array
    assert isinstance(phi, np.ndarray), f"Expected array, got {type(phi)}"
    assert phi.shape == (2,), f"Expected shape (2,), got {phi.shape}"
    assert np.isfinite(phi).all()

    c = 5e-5
    np.testing.assert_allclose(phi[0], Force * c * 1e10, atol=1e-6)
    np.testing.assert_allclose(phi[1], 0.0, atol=1e-10)


def test_calculate_VWS_nh_zero_grad_scalar_matches_expected_scale(tensor_zero, small_grid):
    X, Y, Z, vol, L, H = small_grid
    Force = 1.5
    C10, D1 = 1.0, 1e-3

    phi = vws.calculate_VWS_nh(tensor_zero, X, Y, Z, C10, D1, vol, Force, L, H)

    # From the code: returns phi * 1e10 where phi is np.sqrt(phi[0] ** 2 + phi[1] ** 2)
    # Since phi[0] and phi[1] are scalars, np.sqrt returns a scalar
    # So this should be a scalar
    assert np.isscalar(phi), f"Expected scalar, got {type(phi)} with shape {getattr(phi, 'shape', 'no shape')}"
    assert np.isfinite(phi)

    c = 5e-5
    np.testing.assert_allclose(phi, Force * c * 1e10, atol=1e-6)


def test_virtual_work_modes_consistency_with_zero_pk1(small_grid):
    X, Y, Z, vol, L, H = small_grid
    pk1 = np.zeros(X.shape + (3, 3))
    Force = 2.0
    c = 5e-5

    # Test linear mode - should return 2-element array
    phi_lin = vws.calculate_VWS_virtual_work(pk1, X, Y, Z, vol, Force, L, H, mode="linear")
    assert isinstance(phi_lin, np.ndarray), f"Linear mode should return array, got {type(phi_lin)}"
    assert phi_lin.shape == (2,), f"Linear mode should return shape (2,), got {phi_lin.shape}"
    np.testing.assert_allclose(phi_lin[0], Force * c, atol=1e-12)
    np.testing.assert_allclose(phi_lin[1], 0.0, atol=1e-12)

    # Test HGO mode - should return 2-element array
    phi_hgo = vws.calculate_VWS_virtual_work(pk1, X, Y, Z, vol, Force, L, H, mode="hgo")
    assert isinstance(phi_hgo, np.ndarray), f"HGO mode should return array, got {type(phi_hgo)}"
    assert phi_hgo.shape == (2,), f"HGO mode should return shape (2,), got {phi_hgo.shape}"
    np.testing.assert_allclose(phi_hgo[0], Force * c, atol=1e-12)
    np.testing.assert_allclose(phi_hgo[1], 0.0, atol=1e-12)

    # Test NH mode - should return SCALAR (this is the key difference!)
    phi_nh = vws.calculate_VWS_virtual_work(pk1, X, Y, Z, vol, Force, L, H, mode="nh")
    # From the code: phi = np.sqrt(phi[0] ** 2 + phi[1] ** 2)
    # When phi[0] and phi[1] are scalars, np.sqrt returns a scalar
    assert np.isscalar(phi_nh), f"NH mode should return scalar, got {type(phi_nh)}"
    np.testing.assert_allclose(phi_nh, Force * c, atol=1e-12)


def test_sensitivity_full_shape_and_finite(tensor_zero, small_grid):
    X, Y, Z, vol, L, H = small_grid
    Force = 1.0
    E1, E2, v12, v23, Gt = 7000.0, 500.0, 0.3, 0.3, 400.0
    deviation = 0.05
    S = vws.sensitivity_full(tensor_zero, E1, E2, v12, v23, Gt, X, Y, Z, Force, vol, L, H, deviation)
    assert S.shape == (5, 5)
    assert np.isfinite(S).all()
    assert np.min(S) >= 0.0


def test_read_input_file_minimal_inp_parsing(tmp_path):
    # Let's try to figure out what format the parser actually expects
    # Based on the code, it looks for elements with specific counts
    content = """*Heading
Test file
*Part, name=tissue
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
5, 1.0, 1.0, 0.0
6, 1.0, 0.0, 1.0
7, 0.0, 1.0, 1.0
8, 1.0, 1.0, 1.0
*Element, type=C3D8
1, 1, 2, 5, 3, 4, 6, 8, 7
*End Part
"""
    p = tmp_path / "tiny.inp"
    p.write_text(content)

    nodes, conn = vws.read_input_file(str(p))

    # The parser expects nodes array with [id, x, y, z] and shifts coordinates to start at 0
    assert nodes.shape[0] == 8  # 8 nodes
    assert nodes.shape[1] == 4  # id + 3 coordinates

    # The parser looks for elements with 9 values (len(values_e) == 9)
    assert conn.shape[0] == 1  # 1 element
    assert conn.shape[1] == 9  # element_id + 8 node_ids

    # Coordinates should be shifted to start at 0
    assert np.all(nodes[:, 1:] >= 0.0)


# Additional tests that work with the actual implementation
def test_virtual_displacement_fields_basic():
    """Test basic functionality of virtual displacement fields."""
    L, H = 4.0, 4.0

    # Test at center
    u_center = vws.U_star_z_cos(0.0, 0.0, H, L, H)
    assert np.isfinite(u_center)

    # Test at origin
    u_origin = vws.U_star_z_cos(0.0, 0.0, 0.0, L, H)
    assert u_origin == 0.0  # Should be zero at z=0


def test_material_models_with_nonzero_deformation():
    """Test material models with small but nonzero deformation."""
    n = 3
    X, Y, Z = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    vol = np.ones_like(X) * (1.0 / (n - 1)) ** 3
    L, H = 1.0, 1.0
    Force = 1.0

    # Small uniform extension
    tensor_disp = np.zeros(X.shape + (3, 3))
    tensor_disp[..., 0, 0] = 0.001  # 0.1% strain
    tensor_disp[..., 1, 1] = 0.001
    tensor_disp[..., 2, 2] = 0.001

    # Test that functions don't crash and return finite values
    E1, E2, v12, v23, Gt = 1000.0, 1000.0, 0.3, 0.3, 400.0
    phi_lin = vws.calculate_VWS_linear(tensor_disp, E1, E2, v12, v23, Gt, X, Y, Z, Force, vol, L, H, output=1)
    assert isinstance(phi_lin, np.ndarray)
    assert np.isfinite(phi_lin).all()

    C10, D1, k1, k2, kappa = 100.0, 1e-3, 10.0, 1.0, 0.33
    phi_hgo = vws.calculate_VWS_hgo(tensor_disp, X, Y, Z, C10, D1, k1, k2, kappa, vol, Force, L, H)
    assert isinstance(phi_hgo, np.ndarray)
    assert np.isfinite(phi_hgo).all()

    C10, D1 = 100.0, 1e-3
    phi_nh = vws.calculate_VWS_nh(tensor_disp, X, Y, Z, C10, D1, vol, Force, L, H)
    assert np.isscalar(phi_nh)
    assert np.isfinite(phi_nh)


def test_error_conditions():
    """Test that functions raise appropriate errors for invalid inputs."""
    # Test central_differentiation with mismatched shapes
    Ux = np.ones((3, 3, 3))
    Uy = np.ones((3, 3, 3))
    Uz = np.ones((2, 2, 2))  # Wrong shape
    X = Y = Z = np.ones((3, 3, 3))

    with pytest.raises(ValueError):
        vws.central_differentiation(Ux, Uy, Uz, X, Y, Z)

    # Test virtual work with invalid mode
    pk1 = np.zeros((3, 3, 3, 3, 3))
    vol = np.ones((3, 3, 3))

    with pytest.raises(ValueError):
        vws.calculate_VWS_virtual_work(pk1, X, Y, Z, vol, 1.0, 1.0, 1.0, mode="invalid")


# --- Extra tests to bump coverage to ~100% for vws_models.py ---


def test_volumetric_virtual_fields_scalar_and_array():
    L, H = 4.0, 4.0

    # scalar path at center (d=0)
    u0 = vws.U_star_z_pw_vol(0.0, 0.0, H, L, H)
    dx0 = vws.U_star_z_pw_vol_devX(0.0, 0.0, H, L, H)
    dy0 = vws.U_star_z_pw_vol_devY(0.0, 0.0, H, L, H)
    dz0 = vws.U_star_z_pw_vol_devZ(0.0, 0.0, H, L, H)
    assert np.isfinite([u0, dx0, dy0, dz0]).all()
    # At d=0, only the Z-derivative should be nonzero and equal c/H
    np.testing.assert_allclose(dx0, 0.0)
    np.testing.assert_allclose(dy0, 0.0)
    np.testing.assert_allclose(dz0, 5e-5 / H, rtol=0, atol=1e-12)

    # array path exercising all three piecewise regions (and outside)
    xs = np.array([0.0, L / 8, 3 * L / 8, 3.0])
    ys = np.zeros_like(xs)
    zs = np.full_like(xs, H)
    u = vws.U_star_z_pw_vol(xs, ys, zs, L, H)
    dx = vws.U_star_z_pw_vol_devX(xs, ys, zs, L, H)
    dy = vws.U_star_z_pw_vol_devY(xs, ys, zs, L, H)
    dz = vws.U_star_z_pw_vol_devZ(xs, ys, zs, L, H)
    # Finite and last region outside (<L/2 false) gives zeros
    assert np.isfinite([u, dx, dy, dz]).all()
    assert u[-1] == 0.0 and dx[-1] == 0.0 and dy[-1] == 0.0 and dz[-1] == 0.0


def test_calculate_VWS_linear_output2_returns_scalar(tensor_zero, small_grid):
    X, Y, Z, vol, L, H = small_grid
    Force = 2.0
    E1, E2, v12, v23, Gt = 7000.0, 500.0, 0.3, 0.3, 400.0
    # output=2 path returns sqrt((IVW2 - EVW2)^2) WITHOUT 1e10 scaling
    phi = vws.calculate_VWS_linear(tensor_zero, E1, E2, v12, v23, Gt, X, Y, Z, Force, vol, L, H, output=2)
    assert np.isscalar(phi)
    c = 5e-5
    np.testing.assert_allclose(phi, Force * c, atol=1e-12)


def test_sensitivity_full_degenerate_normalization(monkeypatch, tensor_zero, small_grid):
    # Force the finite-difference deltas to be zero so min(m)==0 -> skip normalization branch
    X, Y, Z, vol, L, H = small_grid
    E1, E2, v12, v23, Gt = 1000.0, 800.0, 0.3, 0.3, 400.0

    # sensitivity_full calls calculate_VWS_linear(..., output=2). Return a constant to zero all diffs.
    monkeypatch.setattr(vws, "calculate_VWS_linear", lambda *a, **k: 1.2345)

    S = vws.sensitivity_full(
        tensor_zero, E1, E2, v12, v23, Gt, X, Y, Z, Force=1.0, volume_matrix=vol, L=L, H=H, deviation=0.05
    )
    # We should get the raw (all-zero) matrix back (no normalization)
    assert S.shape == (5, 5)
    assert np.all(S == 0.0)


def test_sine_virtual_field_derivative_safe_at_origin():
    # Hit d==0 branches in the sine-based derivative helpers
    L, H = 4.0, 4.0
    x = y = 0.0
    z = H / 2
    # These must not NaN and should match the special-case formulas used at d==0
    val_xx = vws.U_star_x_sin_devX(x, y, z, L, H)
    val_yy = vws.U_star_y_sin_devY(x, y, z, L, H)
    val_xz = vws.U_star_x_sin_devZ(x, y, z, L, H)
    val_yz = vws.U_star_y_sin_devZ(x, y, z, L, H)
    assert np.isfinite([val_xx, val_yy, val_xz, val_yz]).all()
    # At d==0, the Z-derivatives are defined via limits -> 0 for sin(0)
    np.testing.assert_allclose(val_xz, 0.0, atol=1e-12)
    np.testing.assert_allclose(val_yz, 0.0, atol=1e-12)


def test_virtual_work_with_nonzero_pk1_produces_finite_values(small_grid):
    # Exercise nonzero IVW paths across modes
    X, Y, Z, vol, L, H = small_grid
    pk1 = np.ones(X.shape + (3, 3)) * 1e-4  # tiny but nonzero
    Force = 0.7

    phi_lin = vws.calculate_VWS_virtual_work(pk1, X, Y, Z, vol, Force, L, H, mode="linear")
    phi_hgo = vws.calculate_VWS_virtual_work(pk1, X, Y, Z, vol, Force, L, H, mode="hgo")
    phi_nh = vws.calculate_VWS_virtual_work(pk1, X, Y, Z, vol, Force, L, H, mode="nh")

    assert isinstance(phi_lin, np.ndarray) and phi_lin.shape == (2,)
    assert isinstance(phi_hgo, np.ndarray) and phi_hgo.shape == (2,)
    assert np.isscalar(phi_nh)
    assert np.isfinite(phi_lin).all() and np.isfinite(phi_hgo).all() and np.isfinite(phi_nh)
