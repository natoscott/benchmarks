# LMCache-enabled llm-d Image

This directory contains a Dockerfile that extends the official llm-d CUDA image with LMCache support.

## Image Details

- **Base Image**: `ghcr.io/llm-d/llm-d-cuda:v0.4.0`
- **Added**: LMCache Python package installed into the vllm venv
- **Repository**: `quay.io/nathans/llm-d-cuda-lmcache:v0.4.0`

## Building

The image is built automatically by Quay.io from this repository.

Alternatively, to build locally:

```bash
podman build -f Dockerfile.lmcache -t quay.io/nathans/llm-d-cuda-lmcache:v0.4.0 .
podman push quay.io/nathans/llm-d-cuda-lmcache:v0.4.0
```

## Usage

This image is used by the lmcache benchmark configurations in `scripts/run-all.sh`:
- `lmcache-local`
- `lmcache-redis`
- `lmcache-valkey`
