"""Compile a small declarative request into the approved UMaze manifest schema.

Custom request files are data, not Hydra manifests or command lines.  Every
accepted value is range checked and mapped through a literal allowlist before
the normal experiment runner sees it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from yaml.constructor import ConstructorError

from .manifest import ManifestError, validate_manifest
from .runner import BATCH_MAX_ATTEMPTS, build_plan, main as runner_main


CUSTOM_SCHEMA_VERSION = 1
MAX_SPEC_BYTES = 16 * 1024
MAX_TOTAL_USD = 5.0
MAX_HOURLY_USD = 0.40
MAX_SEEDS = 3
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,31}$")

_ROOT_KEYS = {
    "schema_version",
    "name",
    "variant",
    "rollouts",
    "epochs",
    "evaluation",
    "goal_count",
    "seeds",
    "limits",
}
_SEED_KEYS = {"data_seed", "train_seed", "planner_seed"}
_LIMIT_KEYS = {"max_hours", "max_usd"}

# This is deliberately a closed vocabulary.  A contributor cannot provide a
# Hydra key/value, Python module, executable, or shell fragment in the spec.
APPROVED_VARIANTS: dict[str, dict[str, Any]] = {
    "curvature": {
        "encoder": "dino_channel",
        "training.straighten": "cos1e-1",
        "training.encoder_lr": 1.0e-5,
    },
    "normalized_acceleration": {
        "encoder": "dino_channel",
        "training.straighten": "normacc1e-1",
        "training.encoder_lr": 1.0e-5,
    },
    "paper_curvature": {
        "encoder": "dino_channel",
        "training.straighten": "aggcos1e-1",
        "training.encoder_lr": 1.0e-5,
    },
    "projector_only": {
        "encoder": "dino_channel",
        "training.straighten": False,
        "training.encoder_lr": 1.0e-6,
    },
    "ratio_speed": {
        "encoder": "dino_channel",
        "training.straighten": "ratiospeed1e-1",
        "training.encoder_lr": 1.0e-5,
    },
}

_EVALUATION_STAGES = {
    "proxy": ["train", "proxy"],
    "proxy_and_plan": ["train", "proxy", "plan"],
}


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _strict_mapping(
    loader: _StrictSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "custom request mapping keys must be scalar values",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _strict_mapping
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _exact_keys(
    value: Mapping[str, Any], allowed: set[str], field: str
) -> None:
    unknown = set(value) - allowed
    missing = allowed - set(value)
    _require(not unknown, f"{field} has unsupported fields: {sorted(unknown)}")
    _require(not missing, f"{field} is missing required fields: {sorted(missing)}")


def _bounded_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{field} must be an integer",
    )
    _require(minimum <= value <= maximum, f"{field} must be in [{minimum}, {maximum}]")
    return value


def _bounded_number(
    value: Any, field: str, minimum: float, maximum: float
) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be a number",
    )
    number = float(value)
    _require(minimum <= number <= maximum, f"{field} must be in [{minimum}, {maximum}]")
    return number


def load_custom_spec(path: str | Path, *, expected_name: str | None = None) -> dict[str, Any]:
    """Load and strictly validate one declarative custom-run request."""
    source = Path(path)
    _require(source.is_file(), f"custom spec does not exist: {source}")
    _require(not source.is_symlink(), "custom spec cannot be a symbolic link")
    _require(source.stat().st_size <= MAX_SPEC_BYTES, "custom spec exceeds 16 KiB")
    try:
        with source.open("r", encoding="utf-8") as handle:
            spec = yaml.load(handle, Loader=_StrictSafeLoader)
    except yaml.YAMLError as exc:
        raise ManifestError(f"invalid custom request YAML: {exc}") from exc
    _require(isinstance(spec, dict), "custom spec root must be a mapping")
    _exact_keys(spec, _ROOT_KEYS, "custom spec")
    _require(
        isinstance(spec["schema_version"], int)
        and not isinstance(spec["schema_version"], bool)
        and spec["schema_version"] == CUSTOM_SCHEMA_VERSION,
        "custom schema_version must be 1",
    )

    name = spec["name"]
    _require(isinstance(name, str) and bool(_NAME_RE.fullmatch(name)), "invalid custom name")
    _require(source.stem == name, "custom name must match the YAML filename")
    if expected_name is not None:
        _require(name == expected_name, "training tag slug does not match custom spec name")

    variant = spec["variant"]
    _require(
        isinstance(variant, str) and variant in APPROVED_VARIANTS,
        f"variant must be one of {sorted(APPROVED_VARIANTS)}",
    )
    _bounded_int(spec["rollouts"], "rollouts", 10, 800)
    _bounded_int(spec["epochs"], "epochs", 1, 20)
    _require(
        isinstance(spec["evaluation"], str)
        and spec["evaluation"] in _EVALUATION_STAGES,
        f"evaluation must be one of {sorted(_EVALUATION_STAGES)}",
    )
    _bounded_int(spec["goal_count"], "goal_count", 2, 50)

    seeds = spec["seeds"]
    _require(isinstance(seeds, list), "seeds must be a list")
    _require(1 <= len(seeds) <= MAX_SEEDS, f"seeds must contain 1 to {MAX_SEEDS} entries")
    seen: set[tuple[int, int, int]] = set()
    for index, seed in enumerate(seeds):
        _require(isinstance(seed, dict), f"seeds[{index}] must be a mapping")
        _exact_keys(seed, _SEED_KEYS, f"seeds[{index}]")
        triple = tuple(
            _bounded_int(seed[key], f"seeds[{index}].{key}", 0, 2_147_483_647)
            for key in ("data_seed", "train_seed", "planner_seed")
        )
        _require(triple not in seen, f"duplicate seed triple {triple}")
        seen.add(triple)

    limits = spec["limits"]
    _require(isinstance(limits, dict), "limits must be a mapping")
    _exact_keys(limits, _LIMIT_KEYS, "limits")
    max_hours = _bounded_number(limits["max_hours"], "limits.max_hours", 0.25, 12.0)
    max_usd = _bounded_number(limits["max_usd"], "limits.max_usd", 0.25, 2.5)
    per_attempt = min(max_usd, max_hours * MAX_HOURLY_USD)
    maximum_total = per_attempt * len(seeds) * BATCH_MAX_ATTEMPTS
    _require(
        maximum_total <= MAX_TOTAL_USD,
        f"two-attempt custom maximum ${maximum_total:.2f} exceeds ${MAX_TOTAL_USD:.2f}",
    )
    return deepcopy(spec)


def compile_custom_manifest(
    spec: Mapping[str, Any], *, expected_name: str | None = None
) -> dict[str, Any]:
    """Map a validated request to one normal manifest with no free-form overrides."""
    # Revalidate mappings supplied programmatically by round-tripping through a
    # temporary YAML-independent path below.
    name = spec.get("name")
    _require(isinstance(name, str) and bool(_NAME_RE.fullmatch(name)), "invalid custom name")
    if expected_name is not None:
        _require(name == expected_name, "training tag slug does not match custom spec name")
    variant = spec.get("variant")
    _require(variant in APPROVED_VARIANTS, "unapproved custom variant")

    manifest = {
        "schema_version": 1,
        "name": f"custom_{name}",
        "paper_version": "declarative-custom-v1 based on arXiv:2603.12231v1",
        "environment": {
            "name": "PointMaze UMaze",
            "hydra": "point_maze",
            "maze": "umaze",
            "goal_horizon": 25,
        },
        "dataset": {
            "uri": "${TS_UMAZE_DATASET_S3_URI}",
            "sha256": "${TS_UMAZE_DATASET_SHA256}",
            "version_id": "${TS_UMAZE_DATASET_VERSION_ID}",
            "archive": "zip",
            "leaf": "point_maze",
            "rollouts": spec["rollouts"],
        },
        "goal_set": {
            "uri": "${TS_UMAZE_GOALS_S3_URI}",
            "sha256": "${TS_UMAZE_GOALS_SHA256}",
            "version_id": "${TS_UMAZE_GOALS_VERSION_ID}",
            "count": spec["goal_count"],
        },
        "seed_sets": deepcopy(spec["seeds"]),
        "variants": [
            {
                "name": variant,
                "hydra_overrides": deepcopy(APPROVED_VARIANTS[variant]),
            }
        ],
        "epochs": spec["epochs"],
        "hydra_overrides": {
            "train": {
                "training.batch_size": 32,
                "training.save_every_x_epoch": 1,
                "training.save_every_x_iterations": 0,
                "training.decoder_start_epoch": 999,
                "training.reconstruct_every_x_batch": 999999,
                "env.num_workers": 4,
                "plan_settings.plan_cfg_path": None,
            },
            "plan": {
                "objective.alpha": 0,
                "objective.mode": "last",
                "planner.max_iter": 1,
                "planner.n_taken_actions": 25,
                "planner.sub_planner.horizon": 25,
                "planner.sub_planner.lr": 0.01,
                "planner.sub_planner.sample_type": "zero",
                "planner.sub_planner.action_noise": 0,
                "planner.sub_planner.opt_steps": 100,
                "planner.sub_planner.eval_every": -1,
                "decode_for_viz": False,
                "+wandb_logging": False,
            },
        },
        "stages": deepcopy(_EVALUATION_STAGES[spec["evaluation"]]),
        "cost_envelope": {
            "max_hours": float(spec["limits"]["max_hours"]),
            "max_hourly_usd": MAX_HOURLY_USD,
            "max_usd": float(spec["limits"]["max_usd"]),
        },
        "artifact_prefix": "${TS_ARTIFACT_PREFIX}/custom",
    }
    validate_manifest(manifest, resolved=False)
    return manifest


def materialize_custom_manifest(
    spec_path: str | Path, output_path: str | Path, *, expected_name: str | None = None
) -> Path:
    spec = load_custom_spec(spec_path, expected_name=expected_name)
    manifest = compile_custom_manifest(spec, expected_name=expected_name)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)

    validate = actions.add_parser("validate")
    validate.add_argument("specs", type=Path, nargs="+")

    compile_action = actions.add_parser("compile")
    compile_action.add_argument("spec", type=Path)
    compile_action.add_argument("--expected-name")
    compile_action.add_argument("--output", type=Path, required=True)

    plan = actions.add_parser("plan")
    plan.add_argument("spec", type=Path)
    plan.add_argument("--expected-name")
    plan.add_argument("--budget-usd", type=float, default=MAX_TOTAL_USD)
    plan.add_argument("--output", type=Path)

    run = actions.add_parser("run")
    run.add_argument("spec", type=Path)
    run.add_argument("--expected-name")
    run.add_argument("--index", type=int)
    run.add_argument("--hourly-usd", type=float)
    run.add_argument("--dry-run", action="store_true")
    return parser


def _compiled_temporary(spec: Path, expected_name: str | None):
    temporary = tempfile.TemporaryDirectory(prefix="ts-custom-manifest-")
    output = Path(temporary.name) / "compiled.yaml"
    materialize_custom_manifest(spec, output, expected_name=expected_name)
    return temporary, output


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "validate":
        for spec_path in args.specs:
            spec = load_custom_spec(spec_path)
            compile_custom_manifest(spec)
            print(f"valid custom request: {spec_path}")
        return 0
    if args.action == "compile":
        materialize_custom_manifest(
            args.spec, args.output, expected_name=args.expected_name
        )
        return 0

    temporary, compiled = _compiled_temporary(args.spec, args.expected_name)
    try:
        if args.action == "plan":
            payload = build_plan(compiled)
            if payload["maximum_total_usd"] > args.budget_usd:
                raise ManifestError(
                    f"two-attempt plan maximum ${payload['maximum_total_usd']:.2f} "
                    f"exceeds ${args.budget_usd:.2f} submission budget"
                )
            payload["manifest"] = str(args.spec)
            payload["custom_spec"] = str(args.spec)
            rendered = json.dumps(payload, indent=2, sort_keys=True)
            if args.output:
                args.output.write_text(rendered + "\n", encoding="utf-8")
            else:
                print(rendered)
            return 0
        if args.action == "run":
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
    raise AssertionError("unreachable custom action")


if __name__ == "__main__":
    raise SystemExit(main())
