import importlib
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

# =========================
# Utilities
# =========================


def make_grids(n=3, L=1.2, H=0.3):
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    z = np.linspace(0, H, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z, L, L, H


def ensure_fake_scipy(monkeypatch):
    """Provide a minimal fake scipy.optimize.least_squares so main_VFM imports cleanly."""
    if "scipy" not in sys.modules:
        scipy_mod = ModuleType("scipy")
        optimize_mod = ModuleType("optimize")

        def _ls(fun, x0, bounds):
            # default no-op; tests stub least_squares anyway
            return SimpleNamespace(x=np.array(x0))

        optimize_mod.least_squares = _ls
        scipy_mod.optimize = optimize_mod
        monkeypatch.setitem(sys.modules, "scipy", scipy_mod)
        monkeypatch.setitem(sys.modules, "scipy.optimize", optimize_mod)


def import_mvf(monkeypatch):
    """Import fim.refactor.main_VFM safely with clean argv and fake scipy."""
    ensure_fake_scipy(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["prog"])  # stop argparse from reading pytest args
    if "fim.refactor.main_VFM" in sys.modules:
        del sys.modules["fim.refactor.main_VFM"]
    return importlib.import_module("fim.refactor.main_VFM")


# =========================
# run_inverse_model — cover all branches
# =========================


def test_run_inverse_model_linear_hgo_nh(monkeypatch):
    mvf = import_mvf(monkeypatch)

    def fake_ls_factory(key, calls):
        def fake_ls(fun, x0, bounds):
            val = np.asarray(fun(np.array(x0)))
            calls[key] = val
            return SimpleNamespace(x=np.array(x0))

        return fake_ls

    X, Y, Z, L, W, H = make_grids(n=3)
    vol = np.ones_like(X)
    disp = np.zeros((*X.shape, 3, 3))
    calls = {"linear": None, "hgo": None, "nh": None}

    # ----- linear -----
    class MLin:
        name = "linear"

        def __init__(self):
            self._p = {"v12": 0.2, "v23": 0.3, "Gt": 100.0, "L": L, "H": H, "Force": 1.0}

        def get_parameter(self, k):
            return self._p[k]

        def model_func(self, *a, **k):
            return np.array([0.1, -0.2])

    monkeypatch.setattr(mvf, "least_squares", fake_ls_factory("linear", calls))
    out = mvf.run_inverse_model(disp, X, Y, Z, vol, [1000, 500], ((0, 0), (1e6, 1e6)), MLin())
    assert np.allclose(out, [1000, 500])
    assert calls["linear"].shape == (2,)

    # ----- hgo -----
    class MHGO:
        name = "hgo"

        def __init__(self):
            self._p = {"k1": 10, "k2": 5, "L": L, "H": H, "Force": 1.0, "volume_matrix": vol}

        def get_parameter(self, k):
            return self._p[k]

        def model_func(self, *a, **k):
            return np.array([0.3, 0.4])

    monkeypatch.setattr(mvf, "least_squares", fake_ls_factory("hgo", calls))
    out = mvf.run_inverse_model(disp, X, Y, Z, vol, [200, 1e-5, 0.1], ((0, 0, 0), (1e6, 1e-2, 0.33)), MHGO())
    assert np.allclose(out, [200, 1e-5, 0.1])
    assert calls["hgo"].shape == (2,)

    # ----- nh -----
    class MNH:
        name = "nh"

        def __init__(self):
            self._p = {"L": L, "H": H, "Force": 1.0, "volume_matrix": vol}

        def get_parameter(self, k):
            return self._p[k]

        def model_func(self, *a, **k):
            return np.array([0.5, 0.6])

    monkeypatch.setattr(mvf, "least_squares", fake_ls_factory("nh", calls))
    out = mvf.run_inverse_model(disp, X, Y, Z, vol, [50, 1e-3], ((0, 0), (1e6, 10)), MNH())
    assert np.allclose(out, [50, 1e-3])
    assert calls["nh"].shape == (2,)


# =========================
# load_common_fields / load_hgo_fields / load_nh_fields
# =========================


def _write_inp(path: Path):
    # Minimal .inp that read_input_file() can parse
    lines = [
        "*Heading",
        "*Part, name=tissue",
        "*Node",
        "1, 0.0, 0.0, 0.0",
        "2, 1.0, 0.0, 0.0",
        "3, 0.0, 1.0, 0.0",
        "4, 0.0, 0.0, 0.5",
        "*Element, type=C3D8",
        "1, 1,2,3,4,1,2,3,4",
    ]
    path.write_text("\n".join(lines))


def test_load_common_and_specific_fields(tmp_path, monkeypatch):
    mvf = import_mvf(monkeypatch)

    X, Y, Z, L, W, H = make_grids(n=3)
    Ux, Uy, Uz = X.copy(), Y.copy(), Z.copy()
    vol = np.ones_like(X)

    folder = tmp_path / "data"
    folder.mkdir()
    np.save(folder / "X.npy", X)
    np.save(folder / "Y.npy", Y)
    np.save(folder / "Z.npy", Z)
    np.save(folder / "Ux.npy", Ux)
    np.save(folder / "Uy.npy", Uy)
    np.save(folder / "Uz.npy", Uz)
    np.save(folder / "volume_matrix.npy", vol)

    # Common
    Xo, Yo, Zo, tensor, vol_out = mvf.load_common_fields(str(folder))
    assert Xo.shape == X.shape and vol_out.shape == X.shape
    assert tensor.shape[-2:] == (3, 3)

    # HGO
    _write_inp(folder / "350k.inp")
    Xh, Yh, Zh, Th, Lh, Wh, Hh, Vh = mvf.load_hgo_fields(str(folder))
    assert np.isclose([Lh, Wh, Hh], [1.0, 1.0, 0.5]).all()
    assert Th.shape[-2:] == (3, 3) and Vh.shape == vol.shape

    # NH
    _write_inp(folder / "335k_32um.inp")
    Xn, Yn, Zn, Tn, Ln, Wn, Hn, Vn = mvf.load_nh_fields(str(folder))
    assert np.isclose([Ln, Wn, Hn], [1.0, 1.0, 0.5]).all()
    assert Tn.shape[-2:] == (3, 3) and Vn.shape == vol.shape


# =========================
# __main__ block for each mode with stubs
# =========================


def _stub_numpy_load_factory():
    X, Y, Z, L, W, H = make_grids(n=3)
    vol = np.ones_like(X)
    U = np.zeros_like(X)

    def _fake_load(path, *a, **k):
        s = str(path)
        if s.endswith("X.npy"):
            return X
        if s.endswith("Y.npy"):
            return Y
        if s.endswith("Z.npy"):
            return Z
        if s.endswith("Ux.npy"):
            return U
        if s.endswith("Uy.npy"):
            return U
        if s.endswith("Uz.npy"):
            return U
        if s.endswith("volume_matrix.npy"):
            return vol
        raise AssertionError(f"Unexpected np.load: {s}")

    return _fake_load


class _FakeMaterialModel:
    def __init__(self, name, params):
        self.name, self.params = name, params

    def sensitivity_analysis_linear(self, *a, **k):
        return np.ones((5, 5))

    def sensitivity_analysis_hgo(self, *a, **k):
        return np.ones((5, 5))

    def sensitivity_analysis_nh(self, *a, **k):
        return np.ones((2, 2))

    def get_parameter(self, k, default=None):
        return self.params.get(k, default)

    def model_func(self, *a, **k):
        # return 2-vector to satisfy least_squares even if called
        return np.array([0.0, 0.0])


def _stub_least_squares(fun, x0, bounds):
    # keep fast; do not call residual
    return SimpleNamespace(x=np.array(x0))


def _stub_read_input_file(*a, **k):
    nodes = np.array([[1, 0.0, 0.0, 0.0], [2, 1.0, 0.0, 0.0], [3, 0.0, 1.0, 0.0], [4, 0.0, 0.0, 0.5]])
    conn = np.array([[1, 1, 2, 3, 4, 1, 2, 3, 4]])
    return nodes, conn


@pytest.mark.parametrize("mode", ["linear", "hgo", "nh"])
def test_main_block_executes_all_modes(monkeypatch, tmp_path, mode):
    ensure_fake_scipy(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["prog", "--model", mode, "--data_path", str(tmp_path)])

    # Patch heavy bits used inside __main__
    import fim.refactor.main_VFM as mvf_mod
    import fim.refactor.material_model as mm
    import fim.refactor.vws_models as vwm

    monkeypatch.setattr(np, "load", _stub_numpy_load_factory())
    monkeypatch.setattr(mvf_mod, "least_squares", _stub_least_squares)
    monkeypatch.setattr(mm, "MaterialModel", _FakeMaterialModel, raising=True)
    monkeypatch.setattr(vwm, "read_input_file", _stub_read_input_file, raising=True)

    # Execute the script block
    runpy.run_module("fim.refactor.main_VFM", run_name="__main__")
