import numpy as np

from fim.refactor.material_model import MaterialModel


def test_linear_model_parameters():
    model = MaterialModel("linear", {"v12": 0.3, "v23": 0.3, "Gt": 100, "L": 1, "H": 1, "Force": 1})
    assert model.name == "linear"
    assert np.isclose(model.get_parameter("v12"), 0.3)


def test_hgo_model_parameters():
    model = MaterialModel("hgo", {"C10": 500, "D1": 1e-5, "k1": 2000, "L": 1, "H": 1, "Force": 1})
    assert model.name == "hgo"
    assert np.isclose(model.get_parameter("k1"), 2000)


def test_sensitivity_analysis_runs():
    model = MaterialModel(
        "linear",
        {
            "E1": 7000,
            "E2": 500,
            "v12": 0.49,
            "v23": 0.49,
            "Gt": 500.0,
            "L": 1.0,
            "H": 1.0,
            "Force": 1.0,
        },
    )

    # deformation gradient tensors: shape (Nx, Ny, Nz, 3, 3)
    rng = np.random.default_rng(0)
    dummy_disp = rng.normal(scale=1e-6, size=(2, 2, 2, 3, 3))

    # coordinate grids: shape (Nx, Ny, Nz)
    dummy_X, dummy_Y, dummy_Z = np.meshgrid(
        np.linspace(0, 1, 2), np.linspace(0, 1, 2), np.linspace(0, 1, 2), indexing="ij"
    )

    # per-voxel volume weights: shape (Nx, Ny, Nz)
    volume_matrix = np.ones((2, 2, 2))
    L, H = 1.0, 1.0

    result = model.sensitivity_analysis(dummy_disp, dummy_X, dummy_Y, dummy_Z, volume_matrix, L, H)

    assert isinstance(result, np.ndarray)
    assert result.size > 0
