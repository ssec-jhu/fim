# FIM — Full-Field Indentation Microscopy

[![CI](https://github.com/ssec-jhu/fim/actions/workflows/ci.yml/badge.svg)](https://github.com/ssec-jhu/fim/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/fim/badge/?version=latest)](https://fim.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/ssec-jhu/fim/branch/main/graph/badge.svg?token=0KPNKHRC2V)](https://codecov.io/gh/ssec-jhu/fim)
[![Security](https://github.com/ssec-jhu/fim/actions/workflows/security.yml/badge.svg)](https://github.com/ssec-jhu/fim/actions/workflows/security.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15287306.svg)](https://doi.org/10.5281/zenodo.15287306)

![SSEC-JHU Logo](docs/_static/SSEC_logo_horiz_blue_1152x263.png)

3D deformation tracking and inverse material characterization from volumetric
(OCT / confocal) image stacks of indentation experiments. The pipeline has
three steps:

1. **Distortion Correction** — Remove optical distortions from raw image
   volumes to obtain undistorted reference and deformed stacks. *(Can be
   skipped if undistorted images are acquired directly, e.g. by scanning the
   material from the bottom.)*
2. **Deformation Tracking** — Estimate the 3D displacement field between a
   reference and deformed volume using optimization-based image registration.
3. **Virtual Fields Method (VFM)** — Compute material properties (elastic
   moduli, fiber parameters) from the displacement field using the virtual
   work principle.

A lightweight web UI and a schema-driven CLI are provided to run each step
with dynamically rendered parameters.

---

## Installation

### pip (editable install)

**Requirements:** Git, **Python 3.11+**, and `pip`. All dependencies (including **PyTorch**) are declared in **`pyproject.toml`** and installed by `pip install -e .`.

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#installing) or [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) if you do not have `conda` yet.

```bash
# Download and install either miniconda or anaconda first
# See: https://docs.conda.io/en/latest/miniconda.html#installing
# Or: https://docs.anaconda.com/free/anaconda/install/index.html

# 1. Create conda env with Python
conda create -n fim_env python=3.11
conda activate fim_env

# 2. Clone the repo
git clone https://github.com/ssec-jhu/fim.git
cd fim

# 3. Install FIM and all dependencies
pip install -e .
```

If you already have Python 3.11+ (venv or system), skip step 1 and run steps 2–3 in that environment.

Run **`fim-ui`** and open **http://127.0.0.1:8000/** (`fim-ui --port 8001` if 8000 is busy).

For **CUDA** on Linux/NVIDIA, install a matching PyTorch build from [pytorch.org](https://pytorch.org) first, then run `pip install -e .` again.

### Docker

```bash
docker build -t fim .
docker run -d -p 8000:8000 fim
```

Or use a pre-built image:

```bash
docker pull ghcr.io/ssec-jhu/fim:latest
docker run -d -p 8000:8000 ghcr.io/ssec-jhu/fim:latest
```

The container serves the web UI on port **8000**.

---

## Running the Pipeline

### Web UI

```bash
fim-ui
```

Same as `uvicorn fim.app.main:app --reload`. Use `--host 0.0.0.0` to listen on all interfaces and `--no-reload` in production.

Parameters are rendered
dynamically from `fim/app/schemas/fim_params.schema.json` — update the schema
to add or remove parameters without editing UI code.

### CLI (schema-driven)

```bash
python -m fim.app.cli list-steps
python -m fim.app.cli show-step tracking
python -m fim.app.cli run tracking --set out_dir=/tmp/fim-out --set num_iter=200
```

### Direct script execution

```bash
# Deformation tracking
python -m fim.refactor.deformation_tracking \
    --with_sphere path/to/deformed_tiffs \
    --without_sphere path/to/reference_tiffs \
    --out_dir path/to/output

# Virtual Fields Method
python -m fim.refactor.main_VFM \
    --data_path path/to/output \
    --model linear
```

---

## Testing

Install dev dependencies:

```bash
pip install -e ".[dev]"
```

### Using tox (recommended)

```bash
tox              # run all checks: lint, security, tests, docs, build
tox -e test      # tests only
tox -e check-style  # lint only
```

### Outside of tox

```bash
pytest .                                        # all tests
pytest tests/test_util.py::test_base_dummy      # single test
ruff check . --select E --select F --select I   # lint
bandit --severity-level=medium -r fim            # security
```

### Build docs

```bash
pip install -e ".[docs]"
cd docs && make clean html
open _build/html/index.html
```

---

## Project Layout

```
fim/
├── app/               # FastAPI web UI + CLI + pipeline runner
│   ├── main.py        # FastAPI application
│   ├── cli.py         # Schema-driven CLI
│   └── schemas/       # JSON parameter schemas
├── refactor/          # Core algorithms
│   ├── deformation_tracking.py   # Step 1: 3D displacement estimation
│   ├── main_VFM.py              # Step 2: Inverse material characterization
│   └── vws_models.py            # VFM model implementations
├── tests/             # Unit tests
└── legacy/            # Previous implementations (not actively maintained)
```

---

## License

See [LICENSE](LICENSE).
