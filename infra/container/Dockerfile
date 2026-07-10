# syntax=docker/dockerfile:1.7
ARG TARGETPLATFORM=linux/amd64
FROM --platform=${TARGETPLATFORM} pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime@sha256:0279f7aa29974bf64e61d0ff6e979b41a249b3662a46e30778dbf80b8c99c361

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_CLI_VERSION=2.35.21
ARG AWS_CLI_SHA256=1fe665267a6149dfb8551cec52b419fa6e82533fab6dd7678939209246e792ee
ARG DINOV2_COMMIT=7764ea0f912e53c92e82eb78a2a1631e92725fc8
ARG MUJOCO210_SHA256=a436ca2f4144c38b837205635bbd60ffe1162d5b44c87df22232795978d7d012
ARG PREWARM_DINO=1
ARG SOURCE_REVISION=unknown

LABEL org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.source="https://github.com/danielhacobian/temporal-straightening"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN python -c "import platform; assert platform.python_version().startswith('3.10.'), platform.python_version()" \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libegl1 \
        libglew-dev \
        libglib2.0-0 \
        libgl1 \
        libglfw3 \
        libglx-mesa0 \
        libosmesa6-dev \
        libsm6 \
        libxext6 \
        libxrender1 \
        patchelf \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/temporal-straightening

COPY infra/container/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir \
        pip==24.0 setuptools==70.0.0 wheel==0.43.0 \
    && python -m pip install --no-cache-dir --no-deps \
        --index-url https://download.pytorch.org/whl/cu121 \
        torchvision==0.18.0 \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python - <<'PY'
import torch
import torchvision

assert torch.__version__.split('+', 1)[0] == '2.3.0', torch.__version__
assert torchvision.__version__.split('+', 1)[0] == '0.18.0', torchvision.__version__
PY

# The legacy PointMaze environment still imports mujoco-py/MuJoCo 2.1 even
# though the modern mujoco package is also installed.
RUN mkdir -p /opt/mujoco \
    && curl --fail --location --retry 5 \
        https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
        --output /tmp/mujoco210.tar.gz \
    && echo "${MUJOCO210_SHA256}  /tmp/mujoco210.tar.gz" | sha256sum --check --strict \
    && tar -xzf /tmp/mujoco210.tar.gz -C /opt/mujoco \
    && rm /tmp/mujoco210.tar.gz

# AWS CLI v2 is used only for bounded S3 input/output syncs in the entrypoint.
RUN curl --fail --location --retry 5 \
        "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-${AWS_CLI_VERSION}.zip" \
        --output /tmp/awscliv2.zip \
    && echo "${AWS_CLI_SHA256}  /tmp/awscliv2.zip" | sha256sum --check --strict \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip

COPY . /opt/temporal-straightening

ENV D4RL_SUPPRESS_IMPORT_ERROR=1 \
    DATASET_DIR=/workspace/data \
    DINOV2_HUB_REPO=facebookresearch/dinov2:${DINOV2_COMMIT} \
    LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
    MUJOCO_GL=osmesa \
    MUJOCO_PY_FORCE_CPU=1 \
    MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
    EXPECTED_RUN_UID=10001 \
    PYTHONUNBUFFERED=1 \
    RUN_ROOT=/workspace/run \
    SOURCE_REVISION=${SOURCE_REVISION} \
    TORCH_HOME=/opt/torch-cache \
    WANDB_MODE=disabled

# mujoco-py compiles its Cython extension on first import. Build it into the
# immutable image while site-packages is still writable; GPU workers are also
# forced to reuse this OSMesa/CPU extension rather than compiling as UID 10001.
RUN python - <<'PY'
import os
from pathlib import Path

import mujoco_py

expected_runtime = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.environ.get("LD_PRELOAD") != expected_runtime:
    raise RuntimeError("system libstdc++ must be preloaded before importing mujoco-py")
extension = Path(mujoco_py.cymj.__file__).resolve()
if not extension.is_file():
    raise RuntimeError(f"mujoco-py extension was not built: {extension}")
print(f"Prebuilt mujoco-py extension: {extension}")
PY

# Cache both the DINOv2 source and ViT-S/14 weights in the immutable image so
# Batch jobs do not pay a download penalty and can survive a transient GitHub
# or model-host outage. Set --build-arg PREWARM_DINO=0 only when debugging an
# upstream availability problem.
RUN mkdir -p "${TORCH_HOME}" \
    && if [[ "${PREWARM_DINO}" == "1" ]]; then \
         python infra/container/prewarm_dino.py; \
       fi \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin tsrunner \
    && touch /opt/conda/lib/python3.10/site-packages/mujoco_py/generated/mujocopy-buildlock \
    && chown tsrunner:tsrunner \
         /opt/conda/lib/python3.10/site-packages/mujoco_py/generated/mujocopy-buildlock \
    && mkdir -p /scratch /workspace/data /workspace/run \
    && chmod 0755 /opt/temporal-straightening/infra/container/batch_entrypoint.sh \
    && chown -R tsrunner:tsrunner /scratch /workspace "${TORCH_HOME}"

USER 10001:10001

ENTRYPOINT ["/opt/temporal-straightening/infra/container/batch_entrypoint.sh"]
CMD ["python", "-m", "infra.container.smoke_test"]
