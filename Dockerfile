# FIM web UI — production-oriented image (CPU PyTorch wheel from PyPI).
# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# setuptools_scm needs a version when .git is not copied into the image
ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+docker
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}

COPY pyproject.toml README.md LICENSE ./
COPY fim ./fim

RUN pip install --upgrade pip \
    && pip install .

RUN useradd --create-home --shell /bin/bash --uid 1000 fim \
    && chown -R fim:fim /app \
    && mkdir -p /data \
    && chown fim:fim /data

# Folder picker and job outputs: map a host directory to /data (see README / docker-compose).
ENV FIM_FS_LIST_ROOT=/data

USER fim

EXPOSE 8000

# Matches FastAPI route /healthz in fim.app.main
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

CMD ["uvicorn", "fim.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
