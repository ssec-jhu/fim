# FIM Docker images
#
# Note: Docker cannot load a multi-architecture manifest into a single local tag
# (`docker buildx build --platform linux/amd64,linux/arm64 --load` is not supported).
# `make build` builds for the host platform only and loads `fim:local`.
# `make push` publishes a multi-arch manifest to GHCR (requires `docker login ghcr.io`).

IMAGE ?= ghcr.io/ssec-jhu/fim
TAG ?= latest
PLATFORMS ?= linux/amd64,linux/arm64

SETUPTOOLS_SCM_PRETEND_VERSION ?= 0.0.0+docker

.PHONY: help build push

help:
	@echo "make build  - build and load fim:local for this machine (single architecture)"
	@echo "make push   - build $(IMAGE):$(TAG) for $(PLATFORMS) and push to GHCR"

build:
	docker buildx build \
		--load \
		--build-arg SETUPTOOLS_SCM_PRETEND_VERSION=$(SETUPTOOLS_SCM_PRETEND_VERSION) \
		-t fim:local \
		.

push:
	docker buildx build \
		--platform $(PLATFORMS) \
		--build-arg SETUPTOOLS_SCM_PRETEND_VERSION=$(SETUPTOOLS_SCM_PRETEND_VERSION) \
		-t $(IMAGE):$(TAG) \
		--push \
		.
