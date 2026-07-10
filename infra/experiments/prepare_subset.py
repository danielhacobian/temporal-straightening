"""Materialize a deterministic PointMaze rollout subset without copying images."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path


_EPISODE_RE = re.compile(r"^episode_(\d+)\.pth$")
CANONICAL_SPLIT_SEED = 42
CANONICAL_TRAIN_FRACTION = 0.9


def canonical_training_indices(total: int) -> list[int]:
    """Return the released-dataset training pool used by the fixed goal split."""
    if total < 2:
        raise ValueError("the released dataset must contain at least two episodes")
    import torch

    train_count = int(CANONICAL_TRAIN_FRACTION * total)
    permutation = torch.randperm(
        total,
        generator=torch.Generator().manual_seed(CANONICAL_SPLIT_SEED),
    ).tolist()
    return [int(index) for index in permutation[:train_count]]


def selected_indices(total: int, rollouts: int, seed: int) -> list[int]:
    """Sample subsets only from the canonical training pool.

    The fixed planner/proxy goals come from the complementary seed-42
    validation pool.  Restricting every partial dataset here prevents a larger
    screening subset from quietly training on its own evaluation episodes.
    """
    if rollouts <= 0 or rollouts > total:
        raise ValueError(f"rollouts must be in [1, {total}], got {rollouts}")
    # The released-dataset anchor consumes the full archive. Keep its canonical
    # order instead of pretending a data-seed reshuffle is an independent sample.
    if rollouts == total:
        return list(range(total))
    indices = canonical_training_indices(total)
    if rollouts > len(indices):
        raise ValueError(
            f"partial rollouts must be in [1, {len(indices)}] to preserve the "
            "canonical held-out pool"
        )
    random.Random(seed).shuffle(indices)
    return indices[:rollouts]


def _replace_symlink(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    destination.symlink_to(source.resolve())


def materialize(source: Path, destination: Path, rollouts: int, seed: int) -> dict:
    """Index tensors and remap episode files into an isolated dataset leaf."""
    import torch

    required = ("states.pth", "actions.pth", "seq_lengths.pth")
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    destination.mkdir(parents=True, exist_ok=True)
    states = torch.load(source / "states.pth", map_location="cpu")
    actions = torch.load(source / "actions.pth", map_location="cpu")
    lengths = torch.load(source / "seq_lengths.pth", map_location="cpu")
    total = len(states)
    if len(actions) != total or len(lengths) != total:
        raise ValueError("dataset tensors have inconsistent rollout counts")
    indices = selected_indices(total, rollouts, seed)
    tensor_indices = torch.tensor(indices, dtype=torch.long)
    torch.save(states.index_select(0, tensor_indices), destination / "states.pth")
    torch.save(actions.index_select(0, tensor_indices), destination / "actions.pth")
    torch.save(lengths.index_select(0, tensor_indices), destination / "seq_lengths.pth")

    source_obs = source / "obses"
    destination_obs = destination / "obses"
    destination_obs.mkdir(exist_ok=True)
    if not source_obs.is_dir():
        raise FileNotFoundError(source_obs)
    for new_index, old_index in enumerate(indices):
        old_episode = source_obs / f"episode_{old_index:03d}.pth"
        if not old_episode.is_file():
            raise FileNotFoundError(old_episode)
        _replace_symlink(old_episode, destination_obs / f"episode_{new_index:03d}.pth")

    for entry in source.iterdir():
        if entry.name in {*required, "obses"}:
            continue
        target = destination / entry.name
        if not target.exists() and not target.is_symlink():
            _replace_symlink(entry, target)

    metadata = {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "source_rollouts": total,
        "selected_rollouts": rollouts,
        "data_seed": seed,
        "sampling_pool": (
            "full_released_dataset"
            if rollouts == total
            else "canonical_seed42_training_split"
        ),
        "canonical_split_seed": CANONICAL_SPLIT_SEED,
        "canonical_train_fraction": CANONICAL_TRAIN_FRACTION,
        "indices": indices,
    }
    temporary = destination / f"subset.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination / "subset.json")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(args.source, args.destination, args.rollouts, args.seed),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
