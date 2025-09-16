# tests/test_material_model.py
import numpy as np
import pytest

import fim.refactor.material_model as mm


# ---------- helpers ----------
@pytest.fixture
def dummy_fields():
    # minimal shapes to pass through evaluate_virtual_fields
    n = 3
    X, Y, Z = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    vol = np.ones_like(X)
    disp_grad = np.zeros(X.shape + (3, 3))
    return disp_grad, X, Y, Z, vol


@pytest.fixture
def linear_params():
    return dict(E1=1000.0, E2=800.0, v12=0.3, v23=0.3, Gt=400.0, Force=2.0, L=1.0, H=1.0)


# ---------- tests ----------
def test_model_selection_and_get_parameter(monkeypatch, linear_params):
    # patch the compute fns so we don't rely on _bak
    monkeypatch.setattr(mm, "calculate_VWS_linear", lambda *a, **k: "LIN", raising=True)
    monkeypatch.setattr(mm, "calculate_VWS_hgo", lambda *a, **k: "HGO", raising=True)
    monkeypatch.setattr(mm, "calculate_VWS_nh", lambda *a, **k: "NH", raising=True)

    m_lin = mm.MaterialModel("linear", linear_params)
    m_hgo = mm.MaterialModel("hgo", linear_params)
    m_nh = mm.MaterialModel("nh", linear_params)

    assert m_lin.model_func is mm.calculate_VWS_linear
    assert m_hgo.model_func is mm.calculate_VWS_hgo
    assert m_nh.model_func is mm.calculate_VWS_nh

    # get_parameter + default behavior
    assert m_lin.get_parameter("E1") == 1000.0
    assert m_lin.get_parameter("nope", default=42) == 42

    # bad name
    with pytest.raises(ValueError):
        mm.MaterialModel("unsupported", {})


def test_sensitivity_analysis_only_for_linear(monkeypatch, dummy_fields, linear_params):
    disp_grad, X, Y, Z, vol = dummy_fields

    # fake sensitivity_full returns a known matrix
    fake_S = np.arange(25, dtype=float).reshape(5, 5)
    monkeypatch.setattr(mm, "sensitivity_full", lambda *a, **k: fake_S, raising=True)

    m_lin = mm.MaterialModel("linear", linear_params)
    S = m_lin.sensitivity_analysis(disp_grad, X, Y, Z, vol, L=1.0, H=1.0, deviation=0.07)
    assert S.shape == (5, 5)
    np.testing.assert_array_equal(S, fake_S)

    # hgo/nh should raise
    for name in ("hgo", "nh"):
        m = mm.MaterialModel(name, linear_params)
        with pytest.raises(NotImplementedError):
            m.sensitivity_analysis(disp_grad, X, Y, Z, vol, L=1.0, H=1.0)


def test_evaluate_virtual_fields_delegates(monkeypatch, dummy_fields, linear_params):
    disp_grad, X, Y, Z, vol = dummy_fields
    Force = 2.5

    # Make the selected model function echo its inputs so we can assert the call shape
    def echo_model(displacement_field, X_, Y_, Z_, Force_, volume_matrix, params_):
        return dict(
            hit=True,
            shapes=(displacement_field.shape, X_.shape, volume_matrix.shape),
            force=Force_,
            passthrough=params_.get("E1"),
        )

    monkeypatch.setattr(mm, "calculate_VWS_linear", echo_model, raising=True)
    m_lin = mm.MaterialModel("linear", {**linear_params, "Force": Force})

    out = m_lin.evaluate_virtual_fields(disp_grad, X, Y, Z, Force, vol)
    assert out["hit"] is True
    assert out["shapes"][0][-2:] == (3, 3)  # tensor at the tail
    assert out["shapes"][1] == X.shape
    assert out["shapes"][2] == vol.shape
    assert out["force"] == Force
    assert out["passthrough"] == linear_params["E1"]


def test_info_and_to_dict_capture(capsys, linear_params):
    m = mm.MaterialModel("linear", linear_params)
    m.info()
    captured = capsys.readouterr().out
    assert "Model: linear" in captured
    assert "E1" in captured and "1000.0" in captured

    d = m.to_dict()
    assert d["model_name"] == "linear"
    assert d["parameters"]["E2"] == 800.0
