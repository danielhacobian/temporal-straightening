#!/usr/bin/env python3
"""Fast runtime validation used by the default CPU and GPU job definitions."""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

import gym
import hydra
import numpy as np
import mujoco_py
import torch
import torchvision


def exercise_point_maze() -> dict:
    """Reset, render, and step the repository's legacy PointMaze as non-root."""
    expected_uid = os.environ.get("EXPECTED_RUN_UID")
    if expected_uid is not None and os.geteuid() != int(expected_uid):
        raise RuntimeError(f"Expected UID {expected_uid}, got {os.geteuid()}")
    if os.environ.get("MUJOCO_PY_FORCE_CPU") != "1":
        raise RuntimeError("MUJOCO_PY_FORCE_CPU=1 is required")
    expected_runtime = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    if os.environ.get("LD_PRELOAD") != expected_runtime:
        raise RuntimeError(f"Expected LD_PRELOAD={expected_runtime}")

    extension = Path(mujoco_py.cymj.__file__).resolve()
    if not extension.is_file():
        raise RuntimeError(f"mujoco-py extension is missing: {extension}")
    build_lock = extension.parent / "mujocopy-buildlock"
    if not build_lock.is_file() or not os.access(build_lock, os.W_OK):
        raise RuntimeError(f"mujoco-py build lock is not writable: {build_lock}")

    from env.pointmaze.maze_model import U_MAZE
    from env.pointmaze.point_maze_wrapper import PointMazeWrapper

    environment = PointMazeWrapper(
        maze_spec=U_MAZE,
        reward_type="sparse",
        reset_target=False,
    )
    initial_state = np.array([1.0856, 1.9746, 0.0, 0.0], dtype=np.float64)
    try:
        observation, state = environment.prepare(seed=0, init_state=initial_state)
        next_observation, reward, done, info = environment.step(
            np.array([0.0, 0.25], dtype=np.float32)
        )

        for name, current in (("reset", observation), ("step", next_observation)):
            visual = np.asarray(current["visual"])
            proprio = np.asarray(current["proprio"])
            if visual.shape != (224, 224, 3) or visual.dtype != np.uint8:
                raise RuntimeError(
                    f"{name} visual has shape={visual.shape}, dtype={visual.dtype}"
                )
            if np.ptp(visual) == 0:
                raise RuntimeError(f"{name} rendered a blank frame")
            if proprio.shape != (4,) or not np.isfinite(proprio).all():
                raise RuntimeError(f"{name} proprio is invalid: {proprio}")

        stepped_state = np.asarray(info["state"])
        if stepped_state.shape != (4,) or not np.isfinite(stepped_state).all():
            raise RuntimeError(f"step state is invalid: {stepped_state}")
        if np.allclose(stepped_state, state):
            raise RuntimeError("non-zero PointMaze action did not change state")
        if done is not False:
            raise RuntimeError(f"unexpected PointMaze termination: {done}")

        return {
            "uid": os.geteuid(),
            "mujoco_py": package_version("mujoco-py"),
            "mujoco_py_extension": str(extension),
            "mujoco_py_build_lock": str(build_lock),
            "libstdcxx_preload": os.environ["LD_PRELOAD"],
            "visual_shape": list(next_observation["visual"].shape),
            "state_shape": list(stepped_state.shape),
            "reward": float(reward),
        }
    finally:
        environment.close()


def main() -> None:
    expect_gpu = os.environ.get("EXPECT_GPU", "0") == "1"
    if platform.python_version_tuple()[:2] != ("3", "10"):
        raise RuntimeError(f"Expected Python 3.10, got {platform.python_version()}")
    if torch.__version__.split("+", 1)[0] != "2.3.0":
        raise RuntimeError(f"Expected torch 2.3.0, got {torch.__version__}")
    if torchvision.__version__.split("+", 1)[0] != "0.18.0":
        raise RuntimeError(f"Expected torchvision 0.18.0, got {torchvision.__version__}")
    if expect_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU Batch job started without a visible CUDA device")
    # The CPU-only infrastructure smoke also runs under amd64 emulation on
    # Apple Silicon during local validation. oneDNN can select instructions
    # that the emulator cannot execute after mujoco-py initializes; GPU jobs
    # never take this path, so their production kernels remain unchanged.
    if not expect_gpu:
        torch.backends.mkldnn.enabled = False

    from models.dino import DinoV2Encoder

    encoder = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    device = torch.device("cuda" if expect_gpu else "cpu")
    encoder = encoder.eval().to(device)
    with torch.inference_mode():
        latent = encoder(torch.zeros(1, 3, 224, 224, device=device))
    point_maze = exercise_point_maze()

    payload = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "gym": gym.__version__,
        "hydra": hydra.__version__,
        "mujoco": package_version("mujoco"),
        "point_maze": point_maze,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "latent_shape": list(latent.shape),
    }
    output = Path(os.environ.get("RUN_ROOT", "/workspace/run")) / "smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
