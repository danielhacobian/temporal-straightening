"""Build deterministic UMaze planner goals and a disjoint proxy set.

The default archive mode reads only selected tensors from the released ZIP,
uses the seed-42 validation split, and requires neither MuJoCo nor a model.
The legacy directory mode remains available when action replay is desired.
Proxy frames are selected in time before BFS shortest-path labels are computed;
remaining trajectory time is never used as a distance target.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import re
import stat
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .proxy_signal import MAZES, shortest_grid_steps


SPLIT_SEED = 42
SELECTION_SEED = 42
PROXY_SELECTION_SEED = 43
PROXY_SEED_OFFSET = 1_000_000
TRAIN_FRACTION = 0.9
_ARCHIVE_EPISODE_RE = re.compile(r"^episode_(\d+)\.pth$")


@dataclass(frozen=True)
class PlannerSegment:
    """A half-open action interval and its inclusive terminal state index."""

    episode_id: int
    start: int
    goal_index: int

    @property
    def horizon(self) -> int:
        return self.goal_index - self.start


@dataclass(frozen=True)
class ArchiveLayout:
    """Validated members for one released PointMaze leaf inside a ZIP."""

    source_leaf: PurePosixPath
    core_members: Mapping[str, zipfile.ZipInfo]
    episode_members: Mapping[int, zipfile.ZipInfo]


def validation_episode_indices(
    episode_count: int,
    *,
    train_fraction: float = TRAIN_FRACTION,
    split_seed: int = SPLIT_SEED,
) -> list[int]:
    """Match ``datasets.traj_dset.split_traj_datasets`` exactly."""
    if episode_count < 2:
        raise ValueError("the released dataset must contain at least two episodes")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between zero and one")
    train_count = int(train_fraction * episode_count)
    if train_count == 0 or train_count == episode_count:
        raise ValueError("train_fraction produced an empty train or validation split")
    permutation = torch.randperm(
        episode_count,
        generator=torch.Generator().manual_seed(split_seed),
    ).tolist()
    return [int(value) for value in permutation[train_count:]]


def select_planner_segments(
    seq_lengths: Sequence[int],
    validation_indices: Sequence[int],
    *,
    count: int,
    horizon: int,
    selection_seed: int = SELECTION_SEED,
) -> list[PlannerSegment]:
    """Choose one deterministic, valid planner segment per held-out episode."""
    if count <= 0:
        raise ValueError("planner goal count must be positive")
    if horizon <= 0:
        raise ValueError("planner horizon must be positive")
    eligible = [
        int(index)
        for index in validation_indices
        if int(seq_lengths[int(index)]) >= horizon + 1
    ]
    if len(eligible) < count:
        raise ValueError(
            f"need {count} distinct held-out episodes with >= {horizon + 1} "
            f"states; found {len(eligible)}"
        )

    rng = np.random.default_rng(selection_seed)
    order = rng.permutation(np.asarray(eligible, dtype=np.int64))[:count]
    segments: list[PlannerSegment] = []
    for episode_id in order.tolist():
        length = int(seq_lengths[episode_id])
        # A horizon-H replay consumes actions [start, start + H) and compares
        # against states [start, start + H], hence the exclusive upper bound.
        start = int(rng.integers(0, length - horizon))
        segments.append(
            PlannerSegment(
                episode_id=int(episode_id),
                start=start,
                goal_index=start + horizon,
            )
        )
    return segments


def select_disjoint_proxy_segments(
    seq_lengths: Sequence[int],
    validation_indices: Sequence[int],
    planner_segments: Sequence[PlannerSegment],
    *,
    count: int,
    episode_length: int,
    selection_seed: int = PROXY_SELECTION_SEED,
    states: np.ndarray | None = None,
    max_samples: int = 8,
    maze: Sequence[str] = MAZES["umaze"],
) -> list[PlannerSegment]:
    """Choose fixed proxy segments from validation episodes unused by planning.

    When states are supplied, the candidate episode order and temporal segment
    are fixed before any BFS labels are computed. Candidates whose sampled
    target is constant (fewer than three distinct distance ranks) are excluded
    because a within-trajectory Spearman correlation is undefined or too
    degenerate there. This eligibility check never observes model latents and
    never changes the already-selected frames to improve a correlation.
    """
    if episode_length < 3:
        raise ValueError("proxy episode_length must be at least three")
    planner_episode_ids = {segment.episode_id for segment in planner_segments}
    remaining = [
        int(index)
        for index in validation_indices
        if int(index) not in planner_episode_ids
    ]
    if states is None:
        proxy_segments = select_planner_segments(
            seq_lengths,
            remaining,
            count=count,
            horizon=episode_length - 1,
            selection_seed=selection_seed,
        )
    else:
        states = np.asarray(states)
        if (
            states.ndim != 3
            or states.shape[0] != len(seq_lengths)
            or states.shape[2] < 2
        ):
            raise ValueError(
                "states must have shape [episode, time, state_dim>=2] for "
                "proxy eligibility"
            )
        horizon = episode_length - 1
        eligible = [
            episode_id
            for episode_id in remaining
            if int(seq_lengths[episode_id]) >= episode_length
        ]
        rng = np.random.default_rng(selection_seed)
        order = rng.permutation(np.asarray(eligible, dtype=np.int64)).tolist()
        # Materialize every start before inspecting any label. Rejected
        # candidates therefore cannot influence later temporal choices.
        candidates = []
        for episode_id in order:
            episode_id = int(episode_id)
            start = int(
                rng.integers(0, int(seq_lengths[episode_id]) - horizon)
            )
            candidates.append(
                PlannerSegment(
                    episode_id=episode_id,
                    start=start,
                    goal_index=start + horizon,
                )
            )
        proxy_segments = []
        for candidate in candidates:
            segment_states = states[
                candidate.episode_id,
                candidate.start : candidate.goal_index + 1,
            ]
            try:
                spatial_proxy_sample_indices(
                    segment_states,
                    max_samples=max_samples,
                    maze=maze,
                )
            except ValueError as exc:
                if "fewer than three sampled grid distances" not in str(exc):
                    raise
                continue
            proxy_segments.append(candidate)
            if len(proxy_segments) == count:
                break
        if len(proxy_segments) < count:
            raise ValueError(
                f"need {count} disjoint proxy trajectories with at least three "
                f"sampled BFS distance levels; found {len(proxy_segments)}"
            )
    proxy_episode_ids = {segment.episode_id for segment in proxy_segments}
    if planner_episode_ids & proxy_episode_ids:
        raise AssertionError("planner and proxy episode sets must be disjoint")
    return proxy_segments


def action_statistics(
    actions: np.ndarray, seq_lengths: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce PointMazeDataset's valid-step mean and sample std."""
    if actions.ndim != 3:
        raise ValueError("actions must have shape [episode, time, action_dim]")
    # PointMazeDataset calls .float() before computing torch.mean/torch.std.
    # Preserve that behavior exactly instead of normalizing the archived
    # float64 tensors in higher precision.
    action_tensor = torch.as_tensor(actions, dtype=torch.float32)
    valid = [
        action_tensor[index, : int(length)]
        for index, length in enumerate(seq_lengths)
    ]
    if not valid or sum(len(part) for part in valid) < 2:
        raise ValueError("at least two valid action samples are required")
    flattened = torch.vstack(valid)
    mean = torch.mean(flattened, dim=0).cpu().numpy()
    std = torch.std(flattened, dim=0).cpu().numpy()
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("action statistics contain non-finite values")
    if np.any(std <= 0):
        raise ValueError("every action dimension must have non-zero variance")
    return mean, std


def normalize_and_group_actions(
    raw_actions: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    *,
    frameskip: int,
) -> np.ndarray:
    """Normalize low-level actions and concatenate each frameskip-sized group."""
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    if raw_actions.ndim != 2:
        raise ValueError("raw_actions must have shape [time, action_dim]")
    if frameskip <= 0 or len(raw_actions) % frameskip:
        raise ValueError("action count must be divisible by a positive frameskip")
    normalized = (
        raw_actions
        - np.asarray(action_mean, dtype=np.float32)
    ) / np.asarray(action_std, dtype=np.float32)
    return normalized.reshape(len(raw_actions) // frameskip, -1)


def spatial_proxy_sample_indices(
    states: np.ndarray,
    *,
    max_samples: int,
    maze: Sequence[str] = MAZES["umaze"],
) -> tuple[list[int], list[int]]:
    """Pick evenly spaced frames, then label them with BFS distance to the goal.

    Frame selection is deliberately independent of the distance labels. This
    retains ties and backtracking that a proxy must explain instead of choosing
    one convenient frame per distance and inflating rank correlation.
    """
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] < 2 or len(states) < 3:
        raise ValueError("proxy states must have shape [time>=3, state_dim>=2]")
    if max_samples < 3:
        raise ValueError("max_samples must be at least three")
    goal_index = len(states) - 1
    sample_count = min(max_samples, len(states))
    chosen = sorted(
        {int(round(value)) for value in np.linspace(0, goal_index, sample_count)}
    )
    if len(chosen) < 3:
        raise ValueError("proxy sampling produced fewer than three frames")
    path_steps = [
        int(shortest_grid_steps(states[index], states[goal_index], maze))
        for index in chosen
    ]
    if len(set(path_steps)) < 3:
        raise ValueError("proxy rollout spans fewer than three sampled grid distances")
    return chosen, path_steps


def make_proxy_trajectory(
    visuals: np.ndarray,
    states: np.ndarray,
    *,
    seed: int | None = None,
    max_samples: int,
    identifier: str | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one proxy group without importing MuJoCo-dependent modules."""
    visuals = np.asarray(visuals)
    states = np.asarray(states)
    if visuals.ndim != 4 or visuals.shape[-1] != 3:
        raise ValueError("proxy visuals must be raw HWC frames [time, H, W, 3]")
    if visuals.dtype != np.uint8:
        raise ValueError("proxy visuals must retain raw uint8 environment pixels")
    if len(visuals) != len(states):
        raise ValueError("proxy visuals and states must have the same length")
    indices, path_steps = spatial_proxy_sample_indices(
        states, max_samples=max_samples
    )
    goal_index = len(states) - 1
    if identifier is None:
        if seed is None:
            raise ValueError("seed or identifier is required for proxy provenance")
        identifier = f"umaze-proxy-seed-{seed}"
    source_payload: dict[str, Any]
    if source is None:
        if seed is None:
            raise ValueError("seed is required when proxy source is not provided")
        source_payload = {
            "kind": "independent_waypoint_rollout",
            "seed": int(seed),
        }
    else:
        source_payload = dict(source)
    segment_start = int(source_payload.get("segment_start", 0))
    source_payload.update(
        {
            "selected_frame_indices": [segment_start + index for index in indices],
            "goal_frame_index": segment_start + goal_index,
        }
    )
    return {
        "trajectory_id": identifier,
        "goal_id": f"{identifier}-final",
        "source": source_payload,
        "observations": {
            "visual": np.expand_dims(visuals[indices], axis=1),
            # Environment observations expose float32 proprio even though the
            # archived state tensors are float64 for replay verification.
            "proprio": np.expand_dims(states[indices].astype(np.float32), axis=1),
        },
        "goal_observation": {
            "visual": np.expand_dims(visuals[goal_index : goal_index + 1], axis=1),
            "proprio": np.expand_dims(
                states[goal_index : goal_index + 1].astype(np.float32), axis=1
            ),
        },
        "states": states[indices],
        "goal_state": states[goal_index],
        "shortest_path_steps": path_steps,
        "path_distance_source": "precomputed_grid_bfs:umaze",
    }


def assemble_planner_targets(
    records: Sequence[Mapping[str, Any]],
    *,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    frameskip: int,
    goal_horizon: int,
) -> dict[str, Any]:
    """Convert replay records into the saved-target contract used by plan.py."""
    if not records:
        raise ValueError("at least one planner replay record is required")
    if goal_horizon % frameskip:
        raise ValueError("goal_horizon must be divisible by frameskip")
    for record in records:
        if len(record["actions"]) != goal_horizon:
            raise ValueError("every planner record must contain goal_horizon actions")

    observation_keys = ("visual", "proprio")
    obs_0 = {
        key: np.expand_dims(
            np.stack([np.asarray(record["observations"][key][0]) for record in records]),
            axis=1,
        )
        for key in observation_keys
    }
    obs_g = {
        key: np.expand_dims(
            np.stack([np.asarray(record["observations"][key][-1]) for record in records]),
            axis=1,
        )
        for key in observation_keys
    }
    return {
        "obs_0": obs_0,
        "obs_g": obs_g,
        "state_0": np.stack(
            [np.asarray(record["states"][0], dtype=np.float32) for record in records]
        ),
        "state_g": np.stack(
            [np.asarray(record["states"][-1], dtype=np.float32) for record in records]
        ),
        "gt_actions": np.stack(
            [
                normalize_and_group_actions(
                    np.asarray(record["actions"]),
                    action_mean,
                    action_std,
                    frameskip=frameskip,
                )
                for record in records
            ]
        ),
        "goal_H": int(goal_horizon),
        "goal_ids": [str(record["goal_id"]) for record in records],
        "goal_sources": [dict(record["source"]) for record in records],
    }


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path.name} must contain a torch.Tensor")
    return value.detach().cpu()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_archive_path(name: str) -> PurePosixPath:
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"dataset archive contains an unsafe path: {name!r}")
    raw = name[:-1] if name.endswith("/") else name
    parts = raw.split("/")
    if not raw or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"dataset archive contains an unsafe path: {name!r}")
    path = PurePosixPath(*parts)
    if path.is_absolute():
        raise ValueError(f"dataset archive contains an unsafe path: {name!r}")
    return path


def inspect_point_maze_archive(
    handle: zipfile.ZipFile, dataset_leaf: str = "point_maze"
) -> ArchiveLayout:
    """Validate the complete ZIP directory before reading any tensor payload."""
    members: dict[PurePosixPath, tuple[zipfile.ZipInfo, bool]] = {}
    for member in handle.infolist():
        path = _validated_archive_path(member.filename)
        if path in members:
            raise ValueError(
                f"dataset archive contains a duplicate path: {member.filename!r}"
            )
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(
                f"dataset archive contains a symbolic link: {member.filename!r}"
            )
        is_directory = member.is_dir()
        file_type = stat.S_IFMT(mode)
        expected_type = stat.S_IFDIR if is_directory else stat.S_IFREG
        if file_type not in (0, expected_type):
            raise ValueError(
                f"dataset archive contains a non-file entry: {member.filename!r}"
            )
        if member.flag_bits & 0x1:
            raise ValueError(
                f"dataset archive contains an encrypted entry: {member.filename!r}"
            )
        members[path] = (member, is_directory)

    file_paths = {
        path for path, (_member, is_directory) in members.items() if not is_directory
    }
    required = ("states.pth", "actions.pth", "seq_lengths.pth")
    candidate_sets = [
        {
            path.parent
            for path in file_paths
            if path.name == name and path.parent.name == dataset_leaf
        }
        for name in required
    ]
    candidates = set.intersection(*candidate_sets) if candidate_sets else set()
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"dataset archive must contain exactly one {dataset_leaf!r} directory "
            f"with {', '.join(required)}"
        )
    source_leaf = candidates.pop()
    core_members = {
        name: members[source_leaf / name][0]
        for name in required
    }

    episode_members: dict[int, zipfile.ZipInfo] = {}
    for path, (member, is_directory) in members.items():
        try:
            relative = path.relative_to(source_leaf)
        except ValueError:
            continue
        if not relative.parts or relative.parts[0] != "obses":
            continue
        if is_directory:
            if len(relative.parts) != 1:
                raise ValueError(
                    f"dataset archive contains a malformed observation path: "
                    f"{member.filename!r}"
                )
            continue
        if len(relative.parts) != 2 or relative.parent != PurePosixPath("obses"):
            raise ValueError(
                f"dataset archive contains a malformed observation path: "
                f"{member.filename!r}"
            )
        match = _ARCHIVE_EPISODE_RE.fullmatch(relative.name)
        if not match:
            raise ValueError(
                f"dataset archive contains a malformed observation file: "
                f"{member.filename!r}"
            )
        episode_id = int(match.group(1))
        if relative.name != f"episode_{episode_id:03d}.pth":
            raise ValueError(
                f"dataset archive contains a non-canonical observation file: "
                f"{member.filename!r}"
            )
        if episode_id in episode_members:
            raise ValueError(f"dataset archive contains duplicate episode {episode_id}")
        episode_members[episode_id] = member

    episode_ids = sorted(episode_members)
    if not episode_ids:
        raise FileNotFoundError("dataset archive contains no observation episodes")
    if episode_ids != list(range(len(episode_ids))):
        raise ValueError(
            "dataset archive observation episodes must be contiguous from zero"
        )
    return ArchiveLayout(
        source_leaf=source_leaf,
        core_members=core_members,
        episode_members=episode_members,
    )


def _load_archive_tensor(
    handle: zipfile.ZipFile, member: zipfile.ZipInfo, label: str
) -> tuple[torch.Tensor, str]:
    payload = handle.read(member)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        value = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(f"archive member {label} is not a readable tensor") from exc
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"archive member {label} must contain a torch.Tensor")
    return value.detach().cpu(), digest


def _validated_dataset_arrays(
    states_tensor: torch.Tensor,
    actions_tensor: torch.Tensor,
    lengths_tensor: torch.Tensor,
    *,
    expected_episodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if lengths_tensor.ndim != 1 or lengths_tensor.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("seq_lengths.pth must be a one-dimensional integer tensor")
    states = states_tensor.numpy()
    actions = actions_tensor.numpy()
    seq_lengths = [int(value) for value in lengths_tensor.reshape(-1).tolist()]
    if states.ndim != 3 or states.shape[2] < 4:
        raise ValueError("states.pth must have shape [episode, time, state_dim>=4]")
    if actions.ndim != 3 or actions.shape[2] != 2:
        raise ValueError("actions.pth must have shape [episode, time, 2]")
    if states.shape[:2] != actions.shape[:2] or len(seq_lengths) != len(states):
        raise ValueError("states, actions, and seq_lengths episode dimensions differ")
    if expected_episodes is not None and len(states) != expected_episodes:
        raise ValueError(
            "core tensor and observation episode counts differ: "
            f"{len(states)} != {expected_episodes}"
        )
    if any(length <= 0 or length > states.shape[1] for length in seq_lengths):
        raise ValueError("seq_lengths contains an invalid episode length")
    return states, actions, seq_lengths


def _state_fingerprints(states: np.ndarray, seq_lengths: Sequence[int]) -> set[bytes]:
    fingerprints: set[bytes] = set()
    for episode_id, length in enumerate(seq_lengths):
        for state in states[episode_id, : int(length)]:
            fingerprints.add(np.asarray(state[:4], dtype="<f8").tobytes())
    return fingerprints


def _replay_segment(
    env: Any,
    initial_state: np.ndarray,
    actions: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    observation, state = env.prepare(seed, initial_state)
    observations: dict[str, list[np.ndarray]] = {
        "visual": [np.asarray(observation["visual"])],
        "proprio": [np.asarray(observation["proprio"])],
    }
    states = [np.asarray(state)]
    for action in actions:
        observation, _reward, _done, info = env.step(action)
        observations["visual"].append(np.asarray(observation["visual"]))
        observations["proprio"].append(np.asarray(observation["proprio"]))
        states.append(np.asarray(info["state"]))
    stacked = {key: np.stack(values) for key, values in observations.items()}
    if stacked["visual"].ndim != 4 or stacked["visual"].shape[-1] != 3:
        raise ValueError("UMaze replay did not produce raw HWC visual frames")
    if stacked["visual"].dtype != np.uint8:
        raise ValueError("UMaze replay did not preserve raw uint8 visual pixels")
    return stacked, np.stack(states)


def _generate_proxy_trajectories(
    env: Any,
    *,
    maze_spec: str,
    released_fingerprints: set[bytes],
    count: int,
    seed_offset: int,
    episode_length: int,
    max_samples: int,
    max_attempts: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    # Imported lazily so unit tests for deterministic selection/assembly never
    # need mujoco_py, d4rl, or an OpenGL context.
    from generate_point_maze_medium import generate_episode

    trajectories: list[dict[str, Any]] = []
    rejected_low_diversity: list[int] = []
    for attempt in range(max_attempts):
        seed = seed_offset + attempt
        try:
            visuals, states, _actions, _info = generate_episode(
                env,
                maze_spec,
                seed,
                episode_length,
                policy="waypoint",
            )
        except (AssertionError, IndexError):
            continue
        visual_array = visuals.detach().cpu().numpy()
        state_array = states.detach().cpu().numpy()
        overlap = [
            index
            for index, state in enumerate(state_array)
            if np.asarray(state[:4], dtype="<f8").tobytes() in released_fingerprints
        ]
        if overlap:
            raise RuntimeError(
                f"independent proxy seed {seed} exactly overlaps released states "
                f"at frames {overlap[:5]}"
            )
        try:
            trajectory = make_proxy_trajectory(
                visual_array,
                state_array,
                seed=seed,
                max_samples=max_samples,
            )
        except ValueError as exc:
            message = str(exc)
            if "fewer than three" not in message or "grid distances" not in message:
                raise
            rejected_low_diversity.append(seed)
            continue
        trajectories.append(trajectory)
        if len(trajectories) == count:
            return trajectories, rejected_low_diversity
    raise RuntimeError(
        f"generated only {len(trajectories)}/{count} spatially diverse proxy "
        f"trajectories in {max_attempts} deterministic attempts"
    )


def write_bundle_atomic(bundle: Mapping[str, Any], destination: Path) -> str:
    """Atomically publish the pickle and its standard sha256sum sidecar."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            pickle.dump(dict(bundle), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256_file(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    sidecar = Path(f"{destination}.sha256")
    sidecar_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            sidecar_temporary = Path(handle.name)
            handle.write(f"{digest}  {destination.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(sidecar_temporary, sidecar)
    finally:
        if sidecar_temporary is not None:
            sidecar_temporary.unlink(missing_ok=True)
    try:
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass
    return digest


def _validated_episode_visuals(
    tensor: torch.Tensor, episode_id: int, seq_length: int
) -> np.ndarray:
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise ValueError(
            f"episode {episode_id} observations must have shape [time, H, W, 3]"
        )
    if tensor.dtype != torch.uint8:
        raise ValueError(f"episode {episode_id} observations must be raw uint8")
    if len(tensor) < seq_length:
        raise ValueError(
            f"episode {episode_id} has {len(tensor)} observations for "
            f"seq_length={seq_length}"
        )
    return tensor[:seq_length].numpy()


def build_bundle_from_archive(
    dataset_zip: Path,
    destination: Path,
    *,
    expected_dataset_sha256: str | None = None,
    dataset_version_id: str | None = None,
    dataset_leaf: str = "point_maze",
    planner_count: int = 50,
    proxy_count: int = 50,
    goal_horizon: int = 25,
    frameskip: int = 5,
    split_seed: int = SPLIT_SEED,
    selection_seed: int = SELECTION_SEED,
    proxy_selection_seed: int = PROXY_SELECTION_SEED,
    proxy_episode_length: int = 100,
    proxy_max_samples: int = 8,
) -> dict[str, Any]:
    """Build fixed goals directly from selected members of the released ZIP."""
    dataset_zip = dataset_zip.resolve()
    if not dataset_zip.is_file():
        raise FileNotFoundError(dataset_zip)
    if goal_horizon % frameskip:
        raise ValueError("goal_horizon must be divisible by frameskip")
    if planner_count <= 0 or proxy_count <= 0:
        raise ValueError("planner_count and proxy_count must be positive")

    archive_sha256 = _sha256_file(dataset_zip)
    if (
        expected_dataset_sha256 is not None
        and archive_sha256 != expected_dataset_sha256.lower()
    ):
        raise ValueError(
            "released dataset SHA-256 mismatch: "
            f"expected {expected_dataset_sha256.lower()}, observed {archive_sha256}"
        )

    with zipfile.ZipFile(dataset_zip) as handle:
        layout = inspect_point_maze_archive(handle, dataset_leaf)
        core_tensors: dict[str, torch.Tensor] = {}
        core_sha256: dict[str, str] = {}
        for name, member in layout.core_members.items():
            tensor, digest = _load_archive_tensor(handle, member, name)
            core_tensors[name] = tensor
            core_sha256[name] = digest
        states, actions, seq_lengths = _validated_dataset_arrays(
            core_tensors["states.pth"],
            core_tensors["actions.pth"],
            core_tensors["seq_lengths.pth"],
            expected_episodes=len(layout.episode_members),
        )
        validation_indices = validation_episode_indices(
            len(seq_lengths), split_seed=split_seed
        )
        planner_segments = select_planner_segments(
            seq_lengths,
            validation_indices,
            count=planner_count,
            horizon=goal_horizon,
            selection_seed=selection_seed,
        )
        proxy_segments = select_disjoint_proxy_segments(
            seq_lengths,
            validation_indices,
            planner_segments,
            count=proxy_count,
            episode_length=proxy_episode_length,
            selection_seed=proxy_selection_seed,
            states=states,
            max_samples=proxy_max_samples,
        )
        planner_episode_ids = {segment.episode_id for segment in planner_segments}
        proxy_episode_ids = {segment.episode_id for segment in proxy_segments}
        if planner_episode_ids & proxy_episode_ids:
            raise AssertionError("planner and proxy archive episodes overlap")

        action_mean, action_std = action_statistics(actions, seq_lengths)
        planner_records: list[dict[str, Any]] = []
        selected_observation_sha256: dict[str, str] = {}
        for segment in planner_segments:
            member = layout.episode_members[segment.episode_id]
            visual_tensor, member_sha256 = _load_archive_tensor(
                handle, member, member.filename
            )
            visuals = _validated_episode_visuals(
                visual_tensor,
                segment.episode_id,
                seq_lengths[segment.episode_id],
            )
            selected_observation_sha256[member.filename] = member_sha256
            endpoint_indices = [segment.start, segment.goal_index]
            endpoint_states = np.asarray(
                states[segment.episode_id, endpoint_indices]
            )
            source_actions = np.asarray(
                actions[
                    segment.episode_id,
                    segment.start : segment.goal_index,
                ]
            )
            goal_id = (
                f"released-val-e{segment.episode_id:04d}-"
                f"s{segment.start:03d}-h{segment.horizon:03d}"
            )
            planner_records.append(
                {
                    "goal_id": goal_id,
                    "source": {
                        "kind": "released_validation_archive_segment",
                        **asdict(segment),
                        "split_seed": int(split_seed),
                        "archive_member": member.filename,
                        "archive_member_sha256": member_sha256,
                    },
                    "observations": {
                        "visual": visuals[endpoint_indices],
                        "proprio": endpoint_states.astype(np.float32),
                    },
                    "states": endpoint_states,
                    "actions": source_actions,
                }
            )

        bundle = assemble_planner_targets(
            planner_records,
            action_mean=action_mean,
            action_std=action_std,
            frameskip=frameskip,
            goal_horizon=goal_horizon,
        )
        proxy_trajectories: list[dict[str, Any]] = []
        for segment in proxy_segments:
            member = layout.episode_members[segment.episode_id]
            visual_tensor, member_sha256 = _load_archive_tensor(
                handle, member, member.filename
            )
            visuals = _validated_episode_visuals(
                visual_tensor,
                segment.episode_id,
                seq_lengths[segment.episode_id],
            )
            selected_observation_sha256[member.filename] = member_sha256
            stop = segment.goal_index + 1
            trajectory = make_proxy_trajectory(
                visuals[segment.start:stop],
                np.asarray(states[segment.episode_id, segment.start:stop]),
                max_samples=proxy_max_samples,
                identifier=(
                    f"released-val-proxy-e{segment.episode_id:04d}-"
                    f"s{segment.start:03d}-h{segment.horizon:03d}"
                ),
                source={
                    "kind": "released_validation_archive_trajectory",
                    **asdict(segment),
                    "segment_start": segment.start,
                    "split_seed": int(split_seed),
                    "selection_seed": int(proxy_selection_seed),
                    "archive_member": member.filename,
                    "archive_member_sha256": member_sha256,
                    "temporal_sampling": "evenly_spaced_before_bfs_labelling",
                    "eligibility_rule": (
                        "at_least_3_distinct_bfs_levels_after_fixed_temporal_sampling"
                    ),
                    "eligibility_uses_model_latents": False,
                },
            )
            proxy_trajectories.append(trajectory)

    planner_episode_ids_sorted = sorted(planner_episode_ids)
    proxy_episode_ids_sorted = sorted(proxy_episode_ids)
    if set(planner_episode_ids_sorted) & set(proxy_episode_ids_sorted):
        raise AssertionError("planner and proxy episode provenance must be disjoint")
    bundle["proxy_trajectories"] = proxy_trajectories
    bundle["metadata"] = {
        "schema": "temporal-straightening.umaze-fixed-goals.v2",
        "builder": "infra.experiments.build_goal_bundle",
        "mode": "archive_no_mujoco",
        "dataset": {
            "archive_filename": dataset_zip.name,
            "archive_sha256": archive_sha256,
            "expected_archive_sha256": (
                expected_dataset_sha256.lower()
                if expected_dataset_sha256 is not None
                else None
            ),
            "version_id": dataset_version_id,
            "archive_leaf": str(layout.source_leaf),
            "episode_count": len(seq_lengths),
            "core_member_sha256": core_sha256,
            "selected_observation_member_sha256": selected_observation_sha256,
        },
        "split": {
            "strategy": "trajectory_randperm",
            "seed": int(split_seed),
            "train_fraction": TRAIN_FRACTION,
            "validation_episode_count": len(validation_indices),
            "validation_episode_ids": validation_indices,
        },
        "planner": {
            "count": planner_count,
            "selection_seed": int(selection_seed),
            "episode_ids": planner_episode_ids_sorted,
            "goal_horizon_environment_steps": goal_horizon,
            "frameskip": frameskip,
            "goal_horizon_model_steps": goal_horizon // frameskip,
            "observation_source": "archived_raw_hwc_frames",
            "action_normalization": {
                "source": "all_valid_released_dataset_steps",
                "std_ddof": 1,
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
            },
        },
        "proxy": {
            "count": proxy_count,
            "selection_seed": int(proxy_selection_seed),
            "episode_ids": proxy_episode_ids_sorted,
            "episode_length": proxy_episode_length,
            "max_samples_per_trajectory": proxy_max_samples,
            "temporal_sampling": "evenly_spaced_before_bfs_labelling",
            "trajectory_eligibility": (
                "at_least_3_distinct_bfs_levels_after_fixed_temporal_sampling"
            ),
            "eligibility_uses_model_latents": False,
            "distance_label": "grid_bfs_to_final_state",
            "uses_remaining_time": False,
            "planner_episode_overlap": False,
        },
    }
    digest = write_bundle_atomic(bundle, destination)
    return {
        "destination": str(destination.resolve()),
        "sha256": digest,
        "sidecar": str(Path(f"{destination.resolve()}.sha256")),
        "planner_goals": planner_count,
        "proxy_trajectories": proxy_count,
        "mode": "archive_no_mujoco",
    }


def build_bundle(
    dataset_dir: Path,
    destination: Path,
    *,
    planner_count: int = 50,
    proxy_count: int = 50,
    goal_horizon: int = 25,
    frameskip: int = 5,
    split_seed: int = SPLIT_SEED,
    selection_seed: int = SELECTION_SEED,
    proxy_seed_offset: int = PROXY_SEED_OFFSET,
    proxy_episode_length: int = 26,
    proxy_max_samples: int = 8,
    proxy_max_attempts: int = 1000,
    replay_atol: float = 1e-5,
) -> dict[str, Any]:
    """Build and atomically save a complete fixed-goal bundle."""
    dataset_dir = dataset_dir.resolve()
    required_paths = {
        name: dataset_dir / name
        for name in ("states.pth", "actions.pth", "seq_lengths.pth")
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"released PointMaze dataset is missing: {missing}")
    if goal_horizon % frameskip:
        raise ValueError("goal_horizon must be divisible by frameskip")
    if proxy_count <= 0 or proxy_episode_length < 3:
        raise ValueError("proxy_count must be positive and episode length >= 3")

    states_tensor = _load_tensor(required_paths["states.pth"])
    actions_tensor = _load_tensor(required_paths["actions.pth"])
    lengths_tensor = _load_tensor(required_paths["seq_lengths.pth"])
    states, actions, seq_lengths = _validated_dataset_arrays(
        states_tensor, actions_tensor, lengths_tensor
    )

    validation_indices = validation_episode_indices(
        len(seq_lengths), split_seed=split_seed
    )
    segments = select_planner_segments(
        seq_lengths,
        validation_indices,
        count=planner_count,
        horizon=goal_horizon,
        selection_seed=selection_seed,
    )
    action_mean, action_std = action_statistics(actions, seq_lengths)

    # Heavy environment imports stay below all tensor and deterministic-input
    # validation so failures are cheap and tests can exercise helpers alone.
    from env.pointmaze.maze_model import U_MAZE
    from env.pointmaze.point_maze_wrapper import PointMazeWrapper

    env = PointMazeWrapper(
        maze_spec=U_MAZE,
        reward_type="sparse",
        reset_target=False,
    )
    try:
        planner_records: list[dict[str, Any]] = []
        for ordinal, segment in enumerate(segments):
            source_actions = np.asarray(
                actions[
                    segment.episode_id,
                    segment.start : segment.goal_index,
                ]
            )
            observations, replay_states = _replay_segment(
                env,
                np.asarray(states[segment.episode_id, segment.start]),
                source_actions,
                seed=selection_seed + ordinal,
            )
            expected_states = np.asarray(
                states[
                    segment.episode_id,
                    segment.start : segment.goal_index + 1,
                ]
            )
            max_state_error = float(np.max(np.abs(replay_states - expected_states)))
            if not np.allclose(
                replay_states,
                expected_states,
                rtol=0.0,
                atol=replay_atol,
            ):
                raise RuntimeError(
                    f"UMaze replay diverged for episode {segment.episode_id} "
                    f"at start {segment.start}; max state error={max_state_error:.3e}"
                )
            goal_id = (
                f"released-val-e{segment.episode_id:04d}-"
                f"s{segment.start:03d}-h{segment.horizon:03d}"
            )
            planner_records.append(
                {
                    "goal_id": goal_id,
                    "source": {
                        "kind": "released_validation_action_replay",
                        **asdict(segment),
                        "split_seed": int(split_seed),
                        "replay_seed": int(selection_seed + ordinal),
                        "max_state_replay_error": max_state_error,
                    },
                    "observations": observations,
                    "states": replay_states,
                    "actions": source_actions,
                }
            )

        bundle = assemble_planner_targets(
            planner_records,
            action_mean=action_mean,
            action_std=action_std,
            frameskip=frameskip,
            goal_horizon=goal_horizon,
        )
        proxy_trajectories, rejected_proxy_seeds = _generate_proxy_trajectories(
            env,
            maze_spec=U_MAZE,
            released_fingerprints=_state_fingerprints(states, seq_lengths),
            count=proxy_count,
            seed_offset=proxy_seed_offset,
            episode_length=proxy_episode_length,
            max_samples=proxy_max_samples,
            max_attempts=proxy_max_attempts,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    bundle["proxy_trajectories"] = proxy_trajectories
    bundle["metadata"] = {
        "schema": "temporal-straightening.umaze-fixed-goals.v1",
        "builder": "infra.experiments.build_goal_bundle",
        "dataset": {
            "episode_count": len(seq_lengths),
            "tensor_sha256": {
                name: _sha256_file(path) for name, path in required_paths.items()
            },
        },
        "split": {
            "strategy": "trajectory_randperm",
            "seed": int(split_seed),
            "train_fraction": TRAIN_FRACTION,
            "validation_episode_count": len(validation_indices),
            "validation_episode_ids": validation_indices,
        },
        "planner": {
            "count": planner_count,
            "selection_seed": int(selection_seed),
            "goal_horizon_environment_steps": goal_horizon,
            "frameskip": frameskip,
            "goal_horizon_model_steps": goal_horizon // frameskip,
            "action_normalization": {
                "source": "all_valid_released_dataset_steps",
                "std_ddof": 1,
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
            },
        },
        "proxy": {
            "count": proxy_count,
            "source": "independent_waypoint_rollouts",
            "seed_offset": int(proxy_seed_offset),
            "accepted_seeds": [
                int(item["source"]["seed"]) for item in proxy_trajectories
            ],
            "rejected_low_spatial_diversity_seeds": rejected_proxy_seeds,
            "episode_length": proxy_episode_length,
            "max_samples_per_trajectory": proxy_max_samples,
            "distance_label": "grid_bfs_to_final_state",
            "uses_remaining_time": False,
            "exact_released_state_overlap_checked": True,
        },
    }
    digest = write_bundle_atomic(bundle, destination)
    return {
        "destination": str(destination.resolve()),
        "sha256": digest,
        "sidecar": str(Path(f"{destination.resolve()}.sha256")),
        "planner_goals": planner_count,
        "proxy_trajectories": proxy_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset-zip",
        type=Path,
        help="released PointMaze ZIP; reads selected members without extraction",
    )
    source.add_argument(
        "--dataset-dir",
        type=Path,
        help="extracted dataset directory for legacy MuJoCo replay mode",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-sha256")
    parser.add_argument("--dataset-version-id")
    parser.add_argument("--dataset-leaf", default="point_maze")
    parser.add_argument("--planner-count", type=int, default=50)
    parser.add_argument("--proxy-count", type=int, default=50)
    parser.add_argument("--goal-horizon", type=int, default=25)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument(
        "--proxy-selection-seed", type=int, default=PROXY_SELECTION_SEED
    )
    parser.add_argument("--proxy-seed-offset", type=int, default=PROXY_SEED_OFFSET)
    parser.add_argument("--proxy-episode-length", type=int)
    parser.add_argument("--proxy-max-samples", type=int, default=8)
    parser.add_argument("--proxy-max-attempts", type=int, default=1000)
    parser.add_argument("--replay-atol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.dataset_zip is not None:
        result = build_bundle_from_archive(
            args.dataset_zip,
            args.output,
            expected_dataset_sha256=args.dataset_sha256,
            dataset_version_id=args.dataset_version_id,
            dataset_leaf=args.dataset_leaf,
            planner_count=args.planner_count,
            proxy_count=args.proxy_count,
            goal_horizon=args.goal_horizon,
            frameskip=args.frameskip,
            split_seed=args.split_seed,
            selection_seed=args.selection_seed,
            proxy_selection_seed=args.proxy_selection_seed,
            proxy_episode_length=args.proxy_episode_length or 100,
            proxy_max_samples=args.proxy_max_samples,
        )
    else:
        result = build_bundle(
            args.dataset_dir,
            args.output,
            planner_count=args.planner_count,
            proxy_count=args.proxy_count,
            goal_horizon=args.goal_horizon,
            frameskip=args.frameskip,
            split_seed=args.split_seed,
            selection_seed=args.selection_seed,
            proxy_seed_offset=args.proxy_seed_offset,
            proxy_episode_length=args.proxy_episode_length or 26,
            proxy_max_samples=args.proxy_max_samples,
            proxy_max_attempts=args.proxy_max_attempts,
            replay_atol=args.replay_atol,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
