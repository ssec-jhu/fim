import importlib
import logging
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

    # _get_dimensions from coordinate grids (no mesh file)
    Lg, Wg, Hg = mvf._get_dimensions(Xo, Yo, Zo)
    assert Lg > 0 and Wg > 0 and Hg > 0

    # _get_dimensions from .inp mesh file
    _write_inp(folder / "sample.inp")
    Lm, Wm, Hm = mvf._get_dimensions(Xo, Yo, Zo, mesh_file=str(folder / "sample.inp"))
    assert np.isclose([Lm, Wm, Hm], [1.0, 1.0, 0.5]).all()


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


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["linear", "hgo", "nh"])
def test_main_block_executes_all_modes(monkeypatch, tmp_path, mode):
    ensure_fake_scipy(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["prog", "--model", mode, "--data_path", str(tmp_path)])

    # Create stub .npy files so os.path.exists() checks pass inside load_common_fields
    dummy = np.zeros((3, 3, 3))
    for name in ("X", "Y", "Z", "Ux", "Uy", "Uz", "volume_matrix"):
        np.save(tmp_path / f"{name}.npy", dummy)

    # Patch heavy bits used inside __main__
    import fim.refactor.main_VFM as mvf_mod
    import fim.refactor.material_model as mm
    import fim.refactor.vws_models as vwm

    monkeypatch.setattr(np, "load", _stub_numpy_load_factory())
    monkeypatch.setattr(mvf_mod, "least_squares", _stub_least_squares)
    monkeypatch.setattr(mm, "MaterialModel", _FakeMaterialModel, raising=True)
    monkeypatch.setattr(vwm, "read_input_file", _stub_read_input_file, raising=True)

    # Execute the script block. main() now exits cleanly via raise SystemExit(0).
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("fim.refactor.main_VFM", run_name="__main__")
    assert excinfo.value.code == 0


# =========================
# Edge-case coverage
# =========================


def test_run_inverse_model_unknown_raises(monkeypatch):
    mvf = import_mvf(monkeypatch)

    class MUnknown:
        name = "bogus"

    with pytest.raises(ValueError, match="Unknown material model"):
        mvf.run_inverse_model(None, None, None, None, None, [1], ((0,), (1,)), MUnknown())


def test_needs_xy_swap_small_array(monkeypatch):
    mvf = import_mvf(monkeypatch)
    assert mvf._needs_xy_swap(np.zeros((1, 3, 3))) is False
    assert mvf._needs_xy_swap(np.zeros((3,))) is False


def test_auto_crop_blank_z_trims_edges(monkeypatch):
    mvf = import_mvf(monkeypatch)
    nz = 7
    Ux = np.zeros((3, 3, nz))
    Uy = np.zeros_like(Ux)
    Uz = np.zeros_like(Ux)
    Ux[:, :, 2:5] = 1.0  # only middle slices non-zero
    result = mvf._auto_crop_blank_z(Ux, Uy, Uz)
    assert result[0].shape[2] == 3  # slices 2,3,4 kept


def test_create_grids_from_params(monkeypatch, tmp_path):
    import json

    mvf = import_mvf(monkeypatch)
    meta = {"dxy_m": 1e-5, "dz_m": 2e-5, "voxel_volume_m3": 2e-15}
    (tmp_path / "grid_params.json").write_text(json.dumps(meta))
    shape = (4, 5, 6)
    X, Y, Z, vol = mvf._create_grids_from_params(str(tmp_path), shape)
    assert X.shape == shape
    assert vol.shape == shape
    assert np.isclose(vol[0, 0, 0], 2e-15)


def test_load_common_fields_missing_files_raises(monkeypatch, tmp_path):
    mvf = import_mvf(monkeypatch)
    folder = tmp_path / "empty"
    folder.mkdir()
    np.save(folder / "Ux.npy", np.zeros((3, 3, 3)))
    np.save(folder / "Uy.npy", np.zeros((3, 3, 3)))
    np.save(folder / "Uz.npy", np.zeros((3, 3, 3)))
    with pytest.raises(FileNotFoundError, match="Neither X.npy nor grid_params.json"):
        mvf.load_common_fields(str(folder))


def test_load_common_fields_ux_uy_uz_shape_mismatch_raises(monkeypatch, tmp_path):
    mvf = import_mvf(monkeypatch)
    folder = tmp_path / "mix"
    folder.mkdir()
    np.save(folder / "Ux.npy", np.zeros((2, 2, 2)))
    np.save(folder / "Uy.npy", np.zeros((2, 2, 2)))
    np.save(folder / "Uz.npy", np.zeros((3, 2, 2)))
    np.save(folder / "X.npy", np.zeros((2, 2, 2)))
    np.save(folder / "Y.npy", np.zeros((2, 2, 2)))
    np.save(folder / "Z.npy", np.zeros((2, 2, 2)))
    np.save(folder / "volume_matrix.npy", np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="Ux/Uy/Uz shape mismatch"):
        mvf.load_common_fields(str(folder))


def test_load_common_fields_stale_grids_without_grid_params_raises(monkeypatch, tmp_path):
    """Mismatching on-disk grids and no grid_params.json should error with actionable message."""
    mvf = import_mvf(monkeypatch)
    folder = tmp_path / "stale_no_meta"
    folder.mkdir()
    nu = 4
    U = np.zeros((nu, nu, nu), dtype=np.float32)
    np.save(folder / "Ux.npy", U)
    np.save(folder / "Uy.npy", U)
    np.save(folder / "Uz.npy", U)
    old = 8
    G = np.zeros((old, old, old))
    np.save(folder / "X.npy", G)
    np.save(folder / "Y.npy", G)
    np.save(folder / "Z.npy", G)
    np.save(folder / "volume_matrix.npy", np.ones_like(G))
    with pytest.raises(ValueError, match="grid_params.json is missing"):
        mvf.load_common_fields(str(folder))


def test_load_common_fields_partial_grids_load_from_grid_params_only(monkeypatch, tmp_path):
    """Only some X/Y/Z files present: rebuild coordinates from grid_params.json."""
    import json

    mvf = import_mvf(monkeypatch)
    folder = tmp_path / "partial"
    folder.mkdir()
    nu = 3
    U = np.zeros((nu, nu, nu), dtype=np.float32)
    np.save(folder / "Ux.npy", U)
    np.save(folder / "Uy.npy", U)
    np.save(folder / "Uz.npy", U)
    np.save(folder / "X.npy", np.zeros((2, 2, 2)))
    # Missing Y.npy, Z.npy, volume_matrix -> have_disk_grids False
    meta = {"shape": [nu, nu, nu], "dxy_m": 1e-5, "dz_m": 2e-5, "voxel_volume_m3": 2e-15}
    (folder / "grid_params.json").write_text(json.dumps(meta))
    Xo, Yo, Zo, tensor, vol_out = mvf.load_common_fields(str(folder))
    assert Xo.shape == (nu, nu, nu)
    assert vol_out.shape == (nu, nu, nu)


def test_import_has_no_side_effects(monkeypatch):
    """Importing ``fim.refactor.main_VFM`` must not parse argv, reconfigure
    logging, or trigger dataset downloads. Regression guard for the old
    module-level ``parser.parse_args()`` / ``logging.basicConfig()`` pattern.
    """
    ensure_fake_scipy(monkeypatch)
    # A hostile argv that would make argparse call sys.exit() if parsed at import.
    monkeypatch.setattr(sys, "argv", ["prog", "--this-flag-does-not-exist"])

    # Force a fresh import.
    if "fim.refactor.main_VFM" in sys.modules:
        del sys.modules["fim.refactor.main_VFM"]

    before = list(logging.getLogger().handlers)
    mvf = importlib.import_module("fim.refactor.main_VFM")
    after = list(logging.getLogger().handlers)

    assert before == after, "import must not reconfigure the root logger"
    assert not hasattr(mvf, "args"), "args must not be defined at module scope"
    assert not hasattr(mvf, "data_path"), "data_path must not be defined at module scope"
    assert not hasattr(mvf, "model_name"), "model_name must not be defined at module scope"
    assert callable(mvf.main)
    assert callable(mvf.build_parser)


def test_build_parser_defaults(monkeypatch):
    mvf = import_mvf(monkeypatch)
    args = mvf.build_parser().parse_args([])
    assert args.model == "linear"
    assert args.data_path is None
    assert args.mesh_file is None
    # Indentation flags default to the current vws_models state (auto depth).
    assert args.indent_depth is None
    assert args.sphere_radius == mvf._DEFAULT_INDENT.sphere_radius


def test_apply_indentation_overrides_defaults_preserve_auto_depth(monkeypatch):
    """Without --indent_depth, the depth already set from max|Uz| is kept."""
    mvf = import_mvf(monkeypatch)
    import fim.refactor.vws_models as vm

    original = vm.get_indentation()
    try:
        # Simulate what load_common_fields() does when processing a Uz field.
        vm.set_depth_indentation_from_Uz(np.array([[1e-4, -3e-5]]))
        prior_depth = vm.get_indentation().depth

        args = mvf.build_parser().parse_args(["--sphere_radius", "2e-3"])
        params = mvf._apply_indentation_overrides(args)

        assert params.depth == prior_depth  # auto-from-Uz depth survives
        assert params.sphere_radius == 2e-3
        assert vm.get_indentation() == params
    finally:
        vm.set_indentation(original)


def test_apply_indentation_overrides_pin_depth(monkeypatch):
    """--indent_depth replaces whatever was set from max|Uz|."""
    mvf = import_mvf(monkeypatch)
    import fim.refactor.vws_models as vm

    original = vm.get_indentation()
    try:
        vm.set_depth_indentation_from_Uz(np.array([[9e-4]]))
        args = mvf.build_parser().parse_args(["--indent_depth", "5e-5", "--sphere_radius", "1e-3"])
        params = mvf._apply_indentation_overrides(args)
        assert params.depth == 5e-5
        assert params.sphere_radius == 1e-3
        # contact_radius is derived; it must follow the pinned depth.
        assert np.isclose(params.contact_radius, np.sqrt(5e-5 * 1e-3))
    finally:
        vm.set_indentation(original)


def test_resolve_default_data_path_invokes_fim_util(monkeypatch, tmp_path):
    mvf = import_mvf(monkeypatch)

    recorded = {}

    def fake_resolve(name, *, auto_fetch):
        recorded["name"] = name
        recorded["auto_fetch"] = auto_fetch
        return tmp_path / name

    monkeypatch.setattr(mvf.fim_util, "resolve_dataset", fake_resolve)

    path = mvf._resolve_default_data_path("linear")
    assert recorded == {"name": "80um", "auto_fetch": True}
    assert path == str(tmp_path / "80um")


def test_load_common_fields_stale_grids_rebuilt_from_grid_params(monkeypatch, tmp_path):
    """--skip_grids leaves no new X.npy; stale full-res grids + new downsampled U must still work."""
    import json

    mvf = import_mvf(monkeypatch)
    folder = tmp_path / "out"
    folder.mkdir()

    # New tracking output (e.g. output_downsample_xy=10 → smaller grid)
    nu = 4
    U = np.zeros((nu, nu, nu), dtype=np.float32)
    np.save(folder / "Ux.npy", U)
    np.save(folder / "Uy.npy", U)
    np.save(folder / "Uz.npy", U)

    # Stale full grids from an older run (wrong shape)
    old = 20
    Xold = np.zeros((old, old, old))
    np.save(folder / "X.npy", Xold)
    np.save(folder / "Y.npy", Xold)
    np.save(folder / "Z.npy", Xold)
    np.save(folder / "volume_matrix.npy", np.ones_like(Xold))

    meta = {"shape": [nu, nu, nu], "dxy_m": 1e-5, "dz_m": 2e-5, "voxel_volume_m3": 2e-15}
    (folder / "grid_params.json").write_text(json.dumps(meta))

    Xo, Yo, Zo, tensor, vol_out = mvf.load_common_fields(str(folder))
    assert Xo.shape == (nu, nu, nu)
    assert vol_out.shape == (nu, nu, nu)
    assert tensor.shape[:3] == (nu, nu, nu)
