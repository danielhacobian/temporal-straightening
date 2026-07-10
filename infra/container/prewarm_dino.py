#!/usr/bin/env python3
"""Download the exact DINOv2 source ref and ViT-S/14 weights at image build."""

from __future__ import annotations

import os

import torch


torch.hub.set_dir("/opt/torch-cache/hub")
model = torch.hub.load(
    os.environ["DINOV2_HUB_REPO"],
    "dinov2_vits14",
    trust_repo=True,
    verbose=False,
)
if model.num_features != 384:
    raise RuntimeError(f"Unexpected DINOv2 feature width: {model.num_features}")
