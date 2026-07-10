"""Paired hierarchical bootstrap for planner success by training seed and goal."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


PairKey = tuple[str, str]


def load_records(paths: Iterable[Path]) -> dict[PairKey, bool]:
    """Load per-goal planner outputs keyed by (training seed, goal id)."""
    records: dict[PairKey, bool] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            raise ValueError(f"{path} is not a per_goal_results.json payload")
        for index, record in enumerate(payload["records"]):
            if not isinstance(record, dict):
                raise ValueError(f"{path} record {index} is not a mapping")
            train_seed = record.get("train_seed")
            goal_id = record.get("goal_id")
            success = record.get("success")
            if train_seed is None or goal_id is None or not isinstance(success, bool):
                raise ValueError(
                    f"{path} record {index} needs train_seed, goal_id, and boolean success"
                )
            key = (str(train_seed), str(goal_id))
            if key in records:
                raise ValueError(f"duplicate training-seed/goal pair {key} in {path}")
            records[key] = success
    if not records:
        raise ValueError("at least one per-goal record is required")
    return records


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(
    baseline: dict[PairKey, bool],
    candidate: dict[PairKey, bool],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Bootstrap a complete crossed seed-by-goal paired design.

    Every training seed must be evaluated on the same goal IDs.  Each bootstrap
    replicate draws the seed axis and one shared goal axis, then evaluates their
    Cartesian product.  This preserves goal pairing across seeds instead of
    inventing a different goal sample inside each sampled seed.
    """
    if baseline.keys() != candidate.keys():
        missing_candidate = sorted(baseline.keys() - candidate.keys())
        missing_baseline = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            "paired inputs must have identical training-seed/goal keys; "
            f"missing candidate={missing_candidate[:5]}, "
            f"missing baseline={missing_baseline[:5]}"
        )
    if resamples < 100:
        raise ValueError("resamples must be at least 100")

    grouped: dict[str, list[str]] = defaultdict(list)
    for train_seed, goal_id in baseline:
        grouped[train_seed].append(goal_id)
    seeds = sorted(grouped)
    for train_seed in seeds:
        grouped[train_seed].sort()
    goals = grouped[seeds[0]]
    for train_seed in seeds[1:]:
        if grouped[train_seed] != goals:
            raise ValueError(
                "paired inputs must form a complete crossed seed-by-goal design; "
                f"seed {train_seed} has different goal IDs"
            )

    deltas = {
        key: float(candidate[key]) - float(baseline[key]) for key in baseline
    }
    observed_delta = sum(deltas.values()) / len(deltas)
    baseline_rate = sum(map(float, baseline.values())) / len(baseline)
    candidate_rate = sum(map(float, candidate.values())) / len(candidate)

    generator = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        sampled_seeds = generator.choices(seeds, k=len(seeds))
        sampled_goals = generator.choices(goals, k=len(goals))
        draw = [
            deltas[(sampled_seed, sampled_goal)]
            for sampled_seed in sampled_seeds
            for sampled_goal in sampled_goals
        ]
        samples.append(sum(draw) / len(draw))

    return {
        "metric": "paired_planner_success_delta",
        "pairing_unit": ["training_seed", "goal_id"],
        "bootstrap_axes": ["training_seed", "shared_goal_id"],
        "training_seeds": len(seeds),
        "paired_goals": len(goals),
        "paired_cells": len(baseline),
        "goals_per_seed": {key: len(value) for key, value in grouped.items()},
        "baseline_success_rate": baseline_rate,
        "candidate_success_rate": candidate_rate,
        "candidate_minus_baseline": observed_delta,
        "confidence_level": 0.95,
        "ci_low": _percentile(samples, 0.025),
        "ci_high": _percentile(samples, 0.975),
        "resamples": resamples,
        "bootstrap_seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = paired_bootstrap(
        load_records(args.baseline),
        load_records(args.candidate),
        resamples=args.resamples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
