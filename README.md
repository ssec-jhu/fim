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

## Setup and run (choose one path)

FIM offers two installation paths depending on your role and technical needs:

| | **Path A — Conda / pip install** | **Path B — Docker container** |
|---|-----------------------------------|--------------------------------|
| **Audience** | Developers who edit code, add models, or tune parameters | Users who run FIM in the web browser; no local Python installation is required |
| **Prerequisites** | Git, Conda | Docker Desktop |
| **Install** | `git clone … && pip install -e .` | `docker run --rm -p 8000:8000 -v ~/fim-output:/data ghcr.io/ssec-jhu/fim` |
| **Run options** | Web UI, CLI, or direct script execution | Web UI only |

Choose one path — they are independent. Full steps are under **Path A** and **Path B** below.

### Path A — Conda / pip install

---

**Installation — Conda environment, clone, and editable install**

**Prerequisites**
- Git
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html#installing) or [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)

```bash
# Create and activate environment (Conda example)
conda create -n fim_env python=3.11
conda activate fim_env

# Clone and enter repo (`pip` uses the active env)
git clone https://github.com/ssec-jhu/fim.git
cd fim

# Install FIM and all dependencies:
pip install -e .
```

If you use another Python 3.11+ environment instead of Conda, activate it and run the same `pip install -e .` after cloning.

For **CUDA** on Linux/NVIDIA, install a matching PyTorch build from [pytorch.org](https://pytorch.org) first, then run `pip install -e .` again.

---

**Run — Web UI, CLI, or direct scripts**

After installation you can run the pipeline in **three** ways:

1. **Web UI** — run `fim-ui`:

```bash
fim-ui
```

Open **http://127.0.0.1:8000/** (use `fim-ui --port 8001` if 8000 is busy).  
Equivalent to `uvicorn fim.app.main:app --reload`. In production use `--no-reload` and tune `--host` as needed.

Parameters come from `fim/app/schemas/fim_params.schema.json`.

2. **Direct script execution**

```bash
python -m fim.refactor.deformation_tracking \
    --with_sphere path/to/deformed_tiffs \
    --without_sphere path/to/reference_tiffs \
    --out_dir path/to/output

python -m fim.refactor.main_VFM \
    --data_path path/to/output \
    --model linear
```

3. **CLI (schema-driven)**

```bash
python -m fim.app.cli list-steps
python -m fim.app.cli show-step tracking
python -m fim.app.cli run tracking --set out_dir=/tmp/fim-out --set num_iter=200
```

### Path B — Docker container

The image serves the UI with **uvicorn** on port **8000** (CPU PyTorch).  
**Prerequisite:** install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (macOS/Windows) or Docker Engine + Compose (Linux).

---

**For users — run the published image**

Pull the FIM image from GHCR, then run in UI mode:

```bash
# Latest UI image from GitHub Container Registry
docker pull ghcr.io/ssec-jhu/fim:latest

# Host folder for pipeline output; appears as /data inside the container
mkdir -p ~/fim-output
# Publish UI on port 8000; bind ~/fim-output → /data; --rm drops container when stopped
docker run --rm -p 8000:8000 -v ~/fim-output:/data ghcr.io/ssec-jhu/fim:latest
```

Open **http://localhost:8000** (not `http://0.0.0.0:8000` — that address only means “listen on all interfaces” inside the container).

To refresh the image later (if the Docker image was updated):

```bash
docker pull ghcr.io/ssec-jhu/fim:latest
```

In the UI, outputs should live under **`/data`** (mapped to your host folder above). Override the host path with `FIM_HOST_DATA_DIR` if you use Compose (see below).

---

**For maintainers — build / update images locally**

**Run from a local build** (clone the repo first, then):

```bash
docker compose up --build
```

Defaults: `./fim-docker-data` → `/data`, container name `fim-app`. Rebuild/restart:

```bash
docker compose down
docker compose up -d --build
docker compose logs -f fim
```

**Build and publish** (multi-arch push to GHCR):

```bash
docker login ghcr.io
make build   # local tag: fim:local (single architecture)
make push    # ghcr.io/ssec-jhu/fim:latest (linux/amd64 + linux/arm64)
```

The default published image is **CPU-only** (no CUDA GPU).

---

## Testing (Conda setup)

Install developer dependencies:

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
pytest fim/tests/test_util.py::test_base_dummy  # single test
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
