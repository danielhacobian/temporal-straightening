"""Trim a checksum-pinned fixed-goal bundle for an economical run profile."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any


def _slice(value: Any, count: int) -> Any:
    if isinstance(value, dict):
        return {key: _slice(item, count) for key, item in value.items()}
    try:
        return value[:count]
    except (TypeError, IndexError):
        return value


def _batch_len(value: Any, field: str) -> int:
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"goal bundle field {field} cannot be empty")
        lengths = {
            _batch_len(item, f"{field}.{key}") for key, item in value.items()
        }
        if len(lengths) != 1:
            raise ValueError(
                f"goal bundle field {field} has inconsistent observation lengths: "
                f"{sorted(lengths)}"
            )
        return lengths.pop()
    try:
        return len(value)
    except TypeError as exc:
        raise ValueError(f"goal bundle field {field} has no batch dimension") from exc


def prepare(source: Path, destination: Path, count: int) -> dict[str, int]:
    """Keep the first N paired planner goals and proxy trajectory groups."""
    if count <= 0:
        raise ValueError("count must be positive")
    with source.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise ValueError("goal bundle must be a mapping")
    required = ("obs_0", "obs_g", "state_0", "state_g", "gt_actions", "goal_H")
    missing = [key for key in required if key not in bundle]
    if missing:
        raise ValueError(f"goal bundle is missing planner fields: {missing}")
    paired_fields = ("obs_0", "obs_g", "state_0", "state_g", "gt_actions")
    available = _batch_len(bundle["state_0"], "state_0")
    for key in paired_fields:
        observed = _batch_len(bundle[key], key)
        if observed != available:
            raise ValueError(
                f"goal bundle field {key} has {observed} entries; "
                f"expected {available}"
            )
    if "goal_ids" in bundle and _batch_len(bundle["goal_ids"], "goal_ids") != available:
        raise ValueError(
            f"goal bundle field goal_ids has {len(bundle['goal_ids'])} entries; "
            f"expected {available}"
        )
    if count > available:
        raise ValueError(f"requested {count} planner goals but bundle has {available}")

    prepared = dict(bundle)
    for key in paired_fields:
        prepared[key] = _slice(bundle[key], count)
    if "goal_ids" in bundle:
        prepared["goal_ids"] = _slice(bundle["goal_ids"], count)
    trajectories = bundle.get("proxy_trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError(
            "goal bundle needs non-empty proxy_trajectories; start/goal pairs are "
            "not a goal-distance monotonicity trajectory"
        )
    if count > len(trajectories):
        raise ValueError(f"requested {count} proxy groups but bundle has {len(trajectories)}")
    prepared["proxy_trajectories"] = trajectories[:count]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(prepared, handle)
    return {"planner_goals": count, "proxy_trajectories": count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.destination, args.count)
    print(f"prepared {result['planner_goals']} goals and proxy trajectories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
