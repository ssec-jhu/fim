import numpy as np
from refactor.material_model import MaterialModel


def test_linear_model_parameters():
    model = MaterialModel("linear", {"v12": 0.3, "v23": 0.3, "Gt": 100, "L": 1, "H": 1, "Force": 1})
    assert model.name == "linear"
    assert np.isclose(model.get_parameter("v12"), 0.3)


def test_hgo_model_parameters():
    model = MaterialModel("hgo", {"C10": 500, "D1": 1e-5, "k1": 2000, "L": 1, "H": 1, "Force": 1})
    assert model.name == "hgo"
    assert np.isclose(model.get_parameter("k1"), 2000)


def test_sensitivity_analysis_runs():
    model = MaterialModel("linear", {"E1": 7000, "E2": 500, "L": 1.0, "H": 1.0, "Force": 1.0})
    dummy_disp = np.zeros((2, 2, 2, 3))
    dummy_X = dummy_Y = dummy_Z = np.zeros((2, 2, 2))
    test_params = [1000, 500]
    cube_size = 1.0
    L = 1.0
    H = 1.0

    result = model.sensitivity_analysis(dummy_disp, dummy_X, dummy_Y, dummy_Z, test_params, cube_size, L, H)

    assert isinstance(result, np.ndarray)
    assert result.size > 0
    assert not np.any(np.isnan(result))
