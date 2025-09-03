import numpy as np
import pytest

from fim.refactor import vws_models as VM
from fim.refactor.material_model import MaterialModel


def _tiny_fields(n=2):
    # coords (Nx,Ny,Nz)
    x = y = z = np.linspace(0.0, 1.0, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    vol = np.ones_like(X, dtype=float)
    # small, nonzero deformation gradients (Nx,Ny,Nz,3,3)
    rng = np.random.default_rng(0)
    F = rng.normal(scale=1e-6, size=(n, n, n, 3, 3))
    return X, Y, Z, F, vol


def test_material_model_selects_funcs_and_get_parameter():
    m_lin = MaterialModel("linear", {"E1": 1, "foo": 2})
    assert m_lin.model_func is VM.calculate_VWS_linear
    assert m_lin.get_parameter("E1") == 1
    assert m_lin.get_parameter("missing", default=7) == 7

    m_hgo = MaterialModel("hgo", {})
    assert m_hgo.model_func is VM.calculate_VWS_hgo

    m_nh = MaterialModel("nh", {})
    assert m_nh.model_func is VM.calculate_VWS_nh

    with pytest.raises(ValueError):
        MaterialModel("nope", {})


def test_sensitivity_analysis_linear_returns_matrix_no_nan():
    X, Y, Z, F, vol = _tiny_fields(2)
    model = MaterialModel(
        "linear",
        {
            "E1": 7000.0,
            "E2": 500.0,
            "v12": 0.49,
            "v23": 0.49,
            "Gt": 500.0,
            "L": 1.0,
            "H": 1.0,
            "Force": 1.0,
        },
    )
    out = model.sensitivity_analysis(F, X, Y, Z, vol, L=1.0, H=1.0)
    assert isinstance(out, np.ndarray) and out.shape == (5, 5)
    assert np.isfinite(out).all()


def test_sensitivity_analysis_others_not_implemented():
    X, Y, Z, F, vol = _tiny_fields(2)
    for name in ("hgo", "nh"):
        m = MaterialModel(name, {})
        with pytest.raises(NotImplementedError):
            m.sensitivity_analysis(F, X, Y, Z, vol, 1.0, 1.0)
