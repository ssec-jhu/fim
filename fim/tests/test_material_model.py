import numpy as np
import pytest

import fim.refactor.material_model as mm


@pytest.mark.unit
def test_get_material_model_dispatch_and_unsupported(monkeypatch):
    # Replace the imported functions with sentinels to test mapping
    monkeypatch.setattr(mm, "calculate_VWS_linear", object())
    monkeypatch.setattr(mm, "calculate_VWS_hgo", object())
    monkeypatch.setattr(mm, "calculate_VWS_nh", object())

    m_lin = mm.MaterialModel("linear", {})
    assert m_lin.model_func is mm.calculate_VWS_linear

    m_hgo = mm.MaterialModel("hgo", {})
    assert m_hgo.model_func is mm.calculate_VWS_hgo

    m_nh = mm.MaterialModel("nh", {})
    assert m_nh.model_func is mm.calculate_VWS_nh

    with pytest.raises(ValueError):
        mm.MaterialModel("banana", {})  # unsupported


def test_get_parameter_and_to_dict():
    params = {"E1": 1000, "Force": 1.2}
    m = mm.MaterialModel("nh", params)
    assert m.get_parameter("E1") == 1000
    assert m.get_parameter("missing", default=42) == 42
    assert m.to_dict() == {"model_name": "nh", "parameters": params}


def test_info_prints(capsys):
    m = mm.MaterialModel("hgo", {"k1": 10, "k2": 5})
    m.info()
    out = capsys.readouterr().out
    assert "Model: hgo" in out and "k1: 10" in out and "k2: 5" in out


def test_sensitivity_analysis_linear_calls_underlying(monkeypatch):
    # Stub out sensitivity_full_linear to capture args and return a known array
    captured = {}

    def fake_sens(tensor_displacement_list, E1, E2, v12, v23, Gt, X, Y, Z, Force, volume_matrix, L, H, deviation):
        captured.update(
            dict(
                E1=E1,
                E2=E2,
                v12=v12,
                v23=v23,
                Gt=Gt,
                Force=Force,
                dev=deviation,
                shapes=(np.shape(tensor_displacement_list), np.shape(X)),
            )
        )
        return np.ones((5, 5)) * 7.0

    monkeypatch.setattr(mm, "sensitivity_full_linear", fake_sens)

    params = dict(E1=1e5, E2=6e4, v12=0.2, v23=0.3, Gt=2e4, Force=1.5)
    m = mm.MaterialModel("linear", params)

    # Minimal fake data
    G = np.zeros((3, 3, 3, 3, 3))
    X = Y = Z = np.zeros((3, 3, 3))
    vol = np.ones_like(X)
    L, H = 1.0, 0.1

    out = m.sensitivity_analysis_linear(G, X, Y, Z, vol, L, H, deviation=0.08)
    assert np.all(out == 7.0)
    # Verify parameters plumbed correctly
    assert captured["E1"] == 1e5 and captured["E2"] == 6e4
    assert captured["v12"] == 0.2 and captured["v23"] == 0.3
    assert captured["Gt"] == 2e4 and captured["Force"] == 1.5
    assert captured["dev"] == 0.08
    assert captured["shapes"][0] == (3, 3, 3, 3, 3) and captured["shapes"][1] == (3, 3, 3)


def test_sensitivity_analysis_linear_wrong_model_raises():
    m = mm.MaterialModel("nh", {})
    with pytest.raises(NotImplementedError):
        m.sensitivity_analysis_linear(None, None, None, None, None, None, None)


def test_evaluate_virtual_fields_uses_bound_model_func(monkeypatch):
    # We explicitly override the bound model_func to avoid relying on real VWS signatures.
    m = mm.MaterialModel("linear", {"alpha": 123})

    called = {}

    def fake_model(displacement_field, X, Y, Z, Force, volume_matrix, params):
        called["args"] = (displacement_field, X, Y, Z, Force, volume_matrix)
        called["params"] = params
        return "OK"

    m.model_func = fake_model

    # Dummy inputs
    D = "disp"
    X = Y = Z = "grid"
    Force = "F"
    V = "vol"

    ret = m.evaluate_virtual_fields(D, X, Y, Z, Force, V)
    assert ret == "OK"
    assert called["args"] == (D, X, Y, Z, Force, V)
    # ensure the same dict object is passed through (not a copy)
    assert called["params"] is m.params


def test_sensitivity_analysis_hgo_calls_underlying(monkeypatch):
    """Test that HGO sensitivity analysis calls the underlying function with correct parameters."""
    # Stub out sensitivity_full_hgo to capture args and return a known array
    captured = {}

    def fake_sens(tensor_displacement_list, X, Y, Z, C10, D1, k1, k2, kappa, volume_matrix, Force, L, H, deviation):
        captured.update(
            dict(
                C10=C10,
                D1=D1,
                k1=k1,
                k2=k2,
                kappa=kappa,
                Force=Force,
                dev=deviation,
                shapes=(np.shape(tensor_displacement_list), np.shape(X)),
            )
        )
        return np.ones((5, 5)) * 8.0

    monkeypatch.setattr(mm, "sensitivity_full_hgo", fake_sens)

    params = dict(C10=1e3, D1=1e-2, k1=100.0, k2=5.0, kappa=1 / 3, Force=2.0)
    m = mm.MaterialModel("hgo", params)

    # Minimal fake data
    G = np.zeros((3, 3, 3, 3, 3))
    X = Y = Z = np.zeros((3, 3, 3))
    vol = np.ones_like(X)
    L, H = 1.0, 0.1

    out = m.sensitivity_analysis_hgo(G, X, Y, Z, vol, L, H, deviation=0.1)
    assert np.all(out == 8.0)
    # Verify parameters plumbed correctly
    assert captured["C10"] == 1e3 and captured["D1"] == 1e-2
    assert captured["k1"] == 100.0 and captured["k2"] == 5.0
    assert captured["kappa"] == 1 / 3 and captured["Force"] == 2.0
    assert captured["dev"] == 0.1
    assert captured["shapes"][0] == (3, 3, 3, 3, 3) and captured["shapes"][1] == (3, 3, 3)


def test_sensitivity_analysis_hgo_wrong_model_raises():
    """Test that HGO sensitivity analysis raises error for non-HGO models."""
    m = mm.MaterialModel("linear", {})
    with pytest.raises(NotImplementedError):
        m.sensitivity_analysis_hgo(None, None, None, None, None, None, None)


def test_sensitivity_analysis_nh_calls_underlying(monkeypatch):
    """Test that NH sensitivity analysis calls the underlying function with correct parameters."""
    # Stub out sensitivity_nh to capture args and return a known array
    captured = {}

    def fake_sens(tensor_displacement_list, X, Y, Z, C10, D1, volume_matrix, Force, L, H, deviation):
        captured.update(
            dict(
                C10=C10,
                D1=D1,
                Force=Force,
                dev=deviation,
                shapes=(np.shape(tensor_displacement_list), np.shape(X)),
            )
        )
        return np.ones((2, 2)) * 9.0

    monkeypatch.setattr(mm, "sensitivity_nh", fake_sens)

    params = dict(C10=500.0, D1=5e-3, Force=1.5)
    m = mm.MaterialModel("nh", params)

    # Minimal fake data
    G = np.zeros((3, 3, 3, 3, 3))
    X = Y = Z = np.zeros((3, 3, 3))
    vol = np.ones_like(X)
    L, H = 1.0, 0.1

    out = m.sensitivity_analysis_nh(G, X, Y, Z, vol, L, H, deviation=0.08)
    assert np.all(out == 9.0)
    # Verify parameters plumbed correctly
    assert captured["C10"] == 500.0 and captured["D1"] == 5e-3
    assert captured["Force"] == 1.5 and captured["dev"] == 0.08
    assert captured["shapes"][0] == (3, 3, 3, 3, 3) and captured["shapes"][1] == (3, 3, 3)


def test_sensitivity_analysis_nh_wrong_model_raises():
    """Test that NH sensitivity analysis raises error for non-NH models."""
    m = mm.MaterialModel("linear", {})
    with pytest.raises(NotImplementedError):
        m.sensitivity_analysis_nh(None, None, None, None, None, None, None)
