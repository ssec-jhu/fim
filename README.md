# FIM — Full-field Indentation Measurement

[![CI](https://github.com/ssec-jhu/fim/actions/workflows/ci.yml/badge.svg)](https://github.com/ssec-jhu/fim/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/fim/badge/?version=latest)](https://fim.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/ssec-jhu/fim/branch/main/graph/badge.svg?token=0KPNKHRC2V)](https://codecov.io/gh/ssec-jhu/fim)
[![Security](https://github.com/ssec-jhu/fim/actions/workflows/security.yml/badge.svg)](https://github.com/ssec-jhu/fim/actions/workflows/security.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15287306.svg)](https://doi.org/10.5281/zenodo.15287306)

![SSEC-JHU Logo](docs/_static/SSEC_logo_horiz_blue_1152x263.png)

3D deformation tracking and inverse material characterization from volumetric
(OCT / confocal) image stacks of indentation experiments. The pipeline has
three steps:

1. **Deformation Tracking** — Estimate the 3D displacement field between a
   reference and deformed volume using optimization-based image registration.
2. **Virtual Fields Method (VFM)** — Compute material properties (elastic
   moduli, fiber parameters) from the displacement field using the virtual
   work principle.
3. **Reconstruction** — (Optional) Reconstruct the deformed geometry from the
   displacement field for visualization.

A lightweight web UI and a schema-driven CLI are provided to run each step
with dynamically rendered parameters.

---

## Quick Start

### Option A: Conda (recommended)

This is the easiest path, especially for deformation tracking which requires
PyTorch.

```bash
git clone https://github.com/ssec-jhu/fim.git
cd fim
conda env create -f fim_env.yml
conda activate fim_env
pip install -e .
```

> To enable GPU/CUDA support for PyTorch, edit `fim_env.yml` and ensure the
> `cpuonly` line stays commented out before creating the environment.

### Option B: pip only

```bash
git clone https://github.com/ssec-jhu/fim.git
cd fim
pip install -e ".[tracking]"   # runtime deps + PyTorch
```

Or, if you only need the UI / VFM (no PyTorch):

```bash
pip install -e .
```

### Verify the installation

```bash
python -c "import fim; print('fim installed')"
```

---

## Running the Pipeline

### Web UI

```bash
uvicorn fim.app.main:app --reload
```

Open `http://127.0.0.1:8000/ui` in your browser. Parameters are rendered
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
python fim/refactor/deformation_tracking.py \
    --ref_dir path/to/reference_tiffs \
    --def_dir path/to/deformed_tiffs \
    --out_dir path/to/output

# Virtual Fields Method
python fim/refactor/main_VFM.py \
    --data_path path/to/output \
    --material_model linear
```

---

## Build with Docker

```bash
docker build -t fim .
docker run -d -p 8000:8000 fim
```

Or pull a pre-built image:

```bash
docker pull ghcr.io/ssec-jhu/fim:latest
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
pip install -r requirements/docs.txt
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
