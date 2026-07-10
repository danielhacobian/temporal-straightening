"""Compile one allowlisted built-in experiment unit under a $5 retry ceiling."""

from __future__ import annotations

import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .manifest import ManifestError, expand_runs, load_manifest, validate_manifest
from .runner import BATCH_MAX_ATTEMPTS, build_plan, main as runner_main


MAX_TOTAL_USD = 5.0
MAX_PER_ATTEMPT_USD = MAX_TOTAL_USD / BATCH_MAX_ATTEMPTS
MANIFEST_DIRECTORY = Path(__file__).resolve().parent / "manifests"
BUILTIN_FAMILIES = {
    "anchor": "umaze_exact_anchor.yaml",
    "screen": "screening_funnel.yaml",
    "finalists": "finalists.yaml",
    "scale": "scaling_trend.yaml",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def source_manifest_path(family: str) -> Path:
    _require(family in BUILTIN_FAMILIES, f"unknown built-in family {family!r}")
    return MANIFEST_DIRECTORY / BUILTIN_FAMILIES[family]


def compile_builtin_chunk(
    family: str,
    variant: str,
    seed_index: int,
    rollouts: int,
) -> dict[str, Any]:
    """Select exactly one static variant/seed/rollout and clamp its envelope."""
    _require(
        isinstance(seed_index, int) and not isinstance(seed_index, bool),
        "seed index must be an integer",
    )
    _require(
        isinstance(rollouts, int) and not isinstance(rollouts, bool) and rollouts > 0,
        "rollouts must be a positive integer",
    )
    manifest = load_manifest(source_manifest_path(family))
    enabled = {
        item["name"]: item
        for item in manifest["variants"]
        if item.get("enabled", True)
    }
    _require(variant in enabled, f"variant must be one of {sorted(enabled)}")
    seeds = manifest["seed_sets"]
    _require(0 <= seed_index < len(seeds), f"seed index must be in [0, {len(seeds)})")
    source_rollouts = manifest["dataset"]["rollouts"]
    allowed_rollouts = (
        source_rollouts if isinstance(source_rollouts, list) else [source_rollouts]
    )
    _require(
        rollouts in allowed_rollouts,
        f"rollouts must be one of {sorted(allowed_rollouts)} for {family}",
    )

    chunk = deepcopy(manifest)
    chunk["dataset"]["rollouts"] = rollouts
    chunk["seed_sets"] = [deepcopy(seeds[seed_index])]
    chunk["variants"] = [deepcopy(enabled[variant])]
    chunk.pop("profiles", None)
    chunk["cost_envelope"]["max_usd"] = min(
        float(chunk["cost_envelope"]["max_usd"]), MAX_PER_ATTEMPT_USD
    )
    validate_manifest(chunk, resolved=False)
    runs = expand_runs(chunk)
    _require(len(runs) == 1, "built-in chunk must expand to exactly one run")
    maximum = min(runs[0].max_usd, runs[0].max_hours * runs[0].max_hourly_usd)
    _require(
        maximum * BATCH_MAX_ATTEMPTS <= MAX_TOTAL_USD,
        "built-in chunk exceeds the retry-inclusive $5 ceiling",
    )
    return chunk


def materialize_builtin_chunk(
    output_path: str | Path,
    *,
    family: str,
    variant: str,
    seed_index: int,
    rollouts: int,
) -> Path:
    chunk = compile_builtin_chunk(family, variant, seed_index, rollouts)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(chunk, sort_keys=False), encoding="utf-8")
    return output


def _compiled_temporary(
    family: str, variant: str, seed_index: int, rollouts: int
):
    temporary = tempfile.TemporaryDirectory(prefix="ts-builtin-chunk-")
    output = Path(temporary.name) / "compiled.yaml"
    materialize_builtin_chunk(
        output,
        family=family,
        variant=variant,
        seed_index=seed_index,
        rollouts=rollouts,
    )
    return temporary, output


def _add_selector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--family", choices=sorted(BUILTIN_FAMILIES), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--rollouts", type=int, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)

    compile_action = actions.add_parser("compile")
    _add_selector_arguments(compile_action)
    compile_action.add_argument("--output", type=Path, required=True)

    plan = actions.add_parser("plan")
    _add_selector_arguments(plan)
    plan.add_argument("--budget-usd", type=float, default=MAX_TOTAL_USD)
    plan.add_argument("--output", type=Path)

    run = actions.add_parser("run")
    _add_selector_arguments(run)
    run.add_argument("--index", type=int)
    run.add_argument("--hourly-usd", type=float)
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selector = {
        "family": args.family,
        "variant": args.variant,
        "seed_index": args.seed_index,
        "rollouts": args.rollouts,
    }
    if args.action == "compile":
        materialize_builtin_chunk(args.output, **selector)
        return 0

    temporary, compiled = _compiled_temporary(**selector)
    try:
        if args.action == "plan":
            payload = build_plan(compiled)
            if payload["array_size"] != 1:
                raise ManifestError("built-in chunk plan must contain exactly one run")
            if payload["maximum_total_usd"] > min(args.budget_usd, MAX_TOTAL_USD):
                raise ManifestError(
                    f"chunk maximum ${payload['maximum_total_usd']:.2f} exceeds "
                    f"${min(args.budget_usd, MAX_TOTAL_USD):.2f} submission ceiling"
                )
            payload["manifest"] = (
                f"builtin:{args.family}:{args.variant}:"
                f"s{args.seed_index}:r{args.rollouts}"
            )
            payload["chunk"] = selector
            rendered = json.dumps(payload, indent=2, sort_keys=True)
            if args.output:
                args.output.write_text(rendered + "\n", encoding="utf-8")
            else:
                print(rendered)
            return 0
        runner_args = ["run", str(compiled)]
        if args.index is not None:
            runner_args.extend(("--index", str(args.index)))
        if args.hourly_usd is not None:
            runner_args.extend(("--hourly-usd", str(args.hourly_usd)))
        if args.dry_run:
            runner_args.append("--dry-run")
        return runner_main(runner_args)
    finally:
        temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
