"""Load, validate, and deterministically expand experiment manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = 1
ALLOWED_STAGES = ("train", "proxy", "plan")
_S3_RE = re.compile(r"^s3://[^/]+/.+")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_VERSION_ID_RE = re.compile(r"^[^\s]{1,1024}$")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ENV_REF_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


class ManifestError(ValueError):
    """Raised when a manifest is unsafe, incomplete, or internally inconsistent."""


@dataclass(frozen=True)
class RunSpec:
    """One immutable member of a variant/seed/rollout experiment matrix."""

    manifest_name: str
    paper_version: str
    environment_name: str
    environment_hydra: str
    maze: str
    goal_horizon: int
    dataset_uri: str
    dataset_sha256: str
    dataset_version_id: str
    dataset_archive: str
    dataset_leaf: str
    rollouts: int
    goal_set_uri: str
    goal_set_sha256: str
    goal_set_version_id: str
    goal_count: int
    data_seed: int
    train_seed: int
    planner_seed: int
    variant: str
    epochs: int
    stages: tuple[str, ...]
    train_overrides: tuple[tuple[str, Any], ...]
    plan_overrides: tuple[tuple[str, Any], ...]
    max_hours: float
    max_hourly_usd: float
    max_usd: float
    artifact_uri: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation with override maps restored."""
        result = asdict(self)
        result["stages"] = list(self.stages)
        result["train_overrides"] = dict(self.train_overrides)
        result["plan_overrides"] = dict(self.plan_overrides)
        result["max_runtime_seconds_at_cap"] = allowed_runtime_seconds(
            self, self.max_hourly_usd
        )
        result["max_cost_usd"] = min(
            self.max_usd, self.max_hours * self.max_hourly_usd
        )
        return result


def _is_deferred(value: Any) -> bool:
    return isinstance(value, str) and bool(_ENV_REF_RE.search(value))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _positive_number(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0,
        f"{field} must be a positive number",
    )
    return float(value)


def _validate_overrides(value: Any, field: str) -> None:
    _require(isinstance(value, dict), f"{field} must be a mapping")
    for key, item in value.items():
        _require(isinstance(key, str) and key, f"{field} keys must be non-empty strings")
        _require(
            item is None or isinstance(item, (str, int, float, bool, list)),
            f"{field}.{key} must be a scalar, list, or null",
        )


def validate_manifest(manifest: Mapping[str, Any], *, resolved: bool = False) -> None:
    """Validate the schema without requiring jsonschema at worker runtime."""
    _require(isinstance(manifest, dict), "manifest root must be a mapping")
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "schema_version must be 1")

    name = manifest.get("name")
    _require(isinstance(name, str) and bool(_NAME_RE.fullmatch(name)), "invalid name")
    _require(
        isinstance(manifest.get("paper_version"), str) and manifest["paper_version"],
        "paper_version is required",
    )

    environment = manifest.get("environment")
    _require(isinstance(environment, dict), "environment must be a mapping")
    for key in ("name", "hydra", "maze"):
        _require(
            isinstance(environment.get(key), str) and environment[key],
            f"environment.{key} is required",
        )
    _require(
        isinstance(environment.get("goal_horizon"), int)
        and environment["goal_horizon"] > 0,
        "environment.goal_horizon must be a positive integer",
    )

    dataset = manifest.get("dataset")
    _require(isinstance(dataset, dict), "dataset must be a mapping")
    uri = dataset.get("uri")
    checksum = dataset.get("sha256")
    version_id = dataset.get("version_id")
    _require(
        isinstance(uri, str) and (_S3_RE.fullmatch(uri) or (not resolved and _is_deferred(uri))),
        "dataset.uri must be an S3 URI",
    )
    _require(
        isinstance(checksum, str)
        and (_SHA256_RE.fullmatch(checksum) or (not resolved and _is_deferred(checksum))),
        "dataset.sha256 must be a 64-character hex digest",
    )
    _require(
        isinstance(version_id, str)
        and (
            _VERSION_ID_RE.fullmatch(version_id)
            or (not resolved and _is_deferred(version_id))
        ),
        "dataset.version_id must be a non-empty S3 version ID",
    )
    _require(dataset.get("archive") in {"zip", "tar", "none"}, "invalid dataset.archive")
    _require(
        isinstance(dataset.get("leaf"), str) and dataset["leaf"],
        "dataset.leaf is required",
    )
    rollouts = dataset.get("rollouts")
    if isinstance(rollouts, list):
        _require(bool(rollouts), "dataset.rollouts cannot be empty")
        _require(
            all(isinstance(item, int) and item > 0 for item in rollouts),
            "dataset.rollouts entries must be positive integers",
        )
        _require(len(set(rollouts)) == len(rollouts), "dataset.rollouts must be unique")
    else:
        _require(
            isinstance(rollouts, int) and rollouts > 0,
            "dataset.rollouts must be a positive integer or list",
        )

    goal_set = manifest.get("goal_set")
    _require(isinstance(goal_set, dict), "goal_set must be a mapping")
    goal_uri = goal_set.get("uri")
    goal_sha = goal_set.get("sha256")
    goal_version_id = goal_set.get("version_id")
    _require(
        isinstance(goal_uri, str)
        and (_S3_RE.fullmatch(goal_uri) or (not resolved and _is_deferred(goal_uri))),
        "goal_set.uri must be an S3 URI",
    )
    _require(
        isinstance(goal_sha, str)
        and (_SHA256_RE.fullmatch(goal_sha) or (not resolved and _is_deferred(goal_sha))),
        "goal_set.sha256 must be a 64-character hex digest",
    )
    _require(
        isinstance(goal_version_id, str)
        and (
            _VERSION_ID_RE.fullmatch(goal_version_id)
            or (not resolved and _is_deferred(goal_version_id))
        ),
        "goal_set.version_id must be a non-empty S3 version ID",
    )
    _require(
        isinstance(goal_set.get("count"), int) and goal_set["count"] > 0,
        "goal_set.count must be a positive integer",
    )

    _require(
        isinstance(manifest.get("epochs"), int) and manifest["epochs"] > 0,
        "epochs must be a positive integer",
    )
    stages = manifest.get("stages")
    _require(isinstance(stages, list) and bool(stages), "stages must be a non-empty list")
    _require(len(stages) == len(set(stages)), "stages cannot contain duplicates")
    _require(all(stage in ALLOWED_STAGES for stage in stages), "unsupported stage")
    _require(stages[0] == "train", "the first stage must be train")
    if "plan" in stages and "proxy" in stages:
        _require(stages.index("proxy") < stages.index("plan"), "proxy must precede plan")

    seeds = manifest.get("seed_sets")
    _require(isinstance(seeds, list) and bool(seeds), "seed_sets must be a non-empty list")
    seen_seeds: set[tuple[int, int, int]] = set()
    for index, seed_set in enumerate(seeds):
        _require(isinstance(seed_set, dict), f"seed_sets[{index}] must be a mapping")
        for key in ("data_seed", "train_seed", "planner_seed"):
            _require(
                isinstance(seed_set.get(key), int) and seed_set[key] >= 0,
                f"seed_sets[{index}].{key} must be a non-negative integer",
            )
        triple = tuple(seed_set[key] for key in ("data_seed", "train_seed", "planner_seed"))
        _require(triple not in seen_seeds, f"duplicate seed set {triple}")
        seen_seeds.add(triple)

    variants = manifest.get("variants")
    _require(isinstance(variants, list) and bool(variants), "variants must be a non-empty list")
    variant_names: set[str] = set()
    enabled_count = 0
    for index, variant in enumerate(variants):
        _require(isinstance(variant, dict), f"variants[{index}] must be a mapping")
        variant_name = variant.get("name")
        _require(
            isinstance(variant_name, str) and bool(_NAME_RE.fullmatch(variant_name)),
            f"variants[{index}].name is invalid",
        )
        _require(variant_name not in variant_names, f"duplicate variant {variant_name}")
        variant_names.add(variant_name)
        enabled = variant.get("enabled", True)
        _require(isinstance(enabled, bool), f"variants[{index}].enabled must be boolean")
        if enabled:
            enabled_count += 1
        else:
            _require(
                isinstance(variant.get("blocked_reason"), str) and variant["blocked_reason"],
                f"disabled variant {variant_name} needs blocked_reason",
            )
        _validate_overrides(variant.get("hydra_overrides", {}), f"variants[{index}].hydra_overrides")
    _require(enabled_count > 0, "at least one variant must be enabled")

    profiles = manifest.get("profiles", {})
    _require(isinstance(profiles, dict), "profiles must be a mapping")
    for profile_name, selected in profiles.items():
        _require(bool(_NAME_RE.fullmatch(profile_name)), f"invalid profile {profile_name}")
        _require(isinstance(selected, list) and bool(selected), f"profile {profile_name} is empty")
        _require(len(selected) == len(set(selected)), f"profile {profile_name} has duplicates")
        _require(set(selected) <= variant_names, f"profile {profile_name} has unknown variants")
        disabled = {
            variant["name"]
            for variant in variants
            if not variant.get("enabled", True) and variant["name"] in selected
        }
        _require(not disabled, f"profile {profile_name} selects disabled variants: {sorted(disabled)}")

    overrides = manifest.get("hydra_overrides", {})
    _require(isinstance(overrides, dict), "hydra_overrides must be a mapping")
    for stage in ("train", "plan"):
        _validate_overrides(overrides.get(stage, {}), f"hydra_overrides.{stage}")

    envelope = manifest.get("cost_envelope")
    _require(isinstance(envelope, dict), "cost_envelope must be a mapping")
    _positive_number(envelope.get("max_hours"), "cost_envelope.max_hours")
    _positive_number(envelope.get("max_hourly_usd"), "cost_envelope.max_hourly_usd")
    _positive_number(envelope.get("max_usd"), "cost_envelope.max_usd")

    artifact_prefix = manifest.get("artifact_prefix")
    _require(
        isinstance(artifact_prefix, str)
        and (
            _S3_RE.fullmatch(artifact_prefix)
            or (not resolved and _is_deferred(artifact_prefix))
        ),
        "artifact_prefix must be an S3 URI",
    )


def _resolve_value(value: Any, environment: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        try:
            return Template(value).substitute(environment)
        except KeyError as exc:
            raise ManifestError(f"missing environment variable {exc.args[0]}") from exc
    if isinstance(value, list):
        return [_resolve_value(item, environment) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_value(item, environment) for key, item in value.items()}
    return value


def resolve_environment(
    manifest: Mapping[str, Any], environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Resolve only ``${NAME}`` placeholders, then perform strict validation."""
    resolved = _resolve_value(deepcopy(dict(manifest)), environment or os.environ)
    validate_manifest(resolved, resolved=True)
    return resolved


def load_manifest(
    path: str | Path,
    *,
    resolve: bool = False,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read one YAML manifest and optionally resolve its environment references."""
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    validate_manifest(manifest, resolved=False)
    return resolve_environment(manifest, environment) if resolve else manifest


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expand_runs(manifest: Mapping[str, Any], profile: str = "default") -> list[RunSpec]:
    """Expand a validated manifest into a stable AWS Batch array ordering."""
    validate_manifest(manifest, resolved=not any(_is_deferred(v) for v in _walk(manifest)))
    profiles = manifest.get("profiles", {})
    if profile in profiles:
        selected = set(profiles[profile])
    elif profile == "default" and not profiles:
        selected = {variant["name"] for variant in manifest["variants"]}
    elif profile == "default" and "default" not in profiles:
        selected = {variant["name"] for variant in manifest["variants"]}
    else:
        raise ManifestError(f"unknown profile {profile}")

    variants = [
        variant
        for variant in manifest["variants"]
        if variant.get("enabled", True) and variant["name"] in selected
    ]
    _require(bool(variants), f"profile {profile} selects no enabled variants")
    variants.sort(key=lambda item: item["name"])
    seed_sets = sorted(
        manifest["seed_sets"],
        key=lambda item: (item["data_seed"], item["train_seed"], item["planner_seed"]),
    )
    rollouts = manifest["dataset"]["rollouts"]
    rollout_values = sorted(rollouts if isinstance(rollouts, list) else [rollouts])
    common_train = manifest.get("hydra_overrides", {}).get("train", {})
    common_plan = manifest.get("hydra_overrides", {}).get("plan", {})
    envelope = manifest["cost_envelope"]

    runs: list[RunSpec] = []
    for rollout_count in rollout_values:
        for seed_set in seed_sets:
            for variant in variants:
                train_overrides = {**common_train, **variant.get("hydra_overrides", {})}
                identity = {
                    "schema_version": manifest["schema_version"],
                    "manifest": manifest["name"],
                    "paper_version": manifest["paper_version"],
                    "environment": manifest["environment"],
                    "dataset_sha256": manifest["dataset"]["sha256"],
                    "dataset_version_id": manifest["dataset"]["version_id"],
                    "goal_set_sha256": manifest["goal_set"]["sha256"],
                    "goal_set_version_id": manifest["goal_set"]["version_id"],
                    "rollouts": rollout_count,
                    "variant": variant["name"],
                    "seeds": seed_set,
                    "epochs": manifest["epochs"],
                    "stages": manifest["stages"],
                    "train_overrides": train_overrides,
                    "plan_overrides": common_plan,
                }
                suffix = _canonical_digest(identity)[:10]
                run_id = (
                    f"{manifest['name']}-{variant['name']}-r{rollout_count}"
                    f"-d{seed_set['data_seed']}-t{seed_set['train_seed']}"
                    f"-p{seed_set['planner_seed']}-{suffix}"
                )
                artifact_uri = (
                    f"{manifest['artifact_prefix'].rstrip('/')}/{manifest['name']}/{run_id}"
                )
                runs.append(
                    RunSpec(
                        manifest_name=manifest["name"],
                        paper_version=manifest["paper_version"],
                        environment_name=manifest["environment"]["name"],
                        environment_hydra=manifest["environment"]["hydra"],
                        maze=manifest["environment"]["maze"],
                        goal_horizon=manifest["environment"]["goal_horizon"],
                        dataset_uri=manifest["dataset"]["uri"],
                        dataset_sha256=manifest["dataset"]["sha256"].lower(),
                        dataset_version_id=manifest["dataset"]["version_id"],
                        dataset_archive=manifest["dataset"]["archive"],
                        dataset_leaf=manifest["dataset"]["leaf"],
                        rollouts=rollout_count,
                        goal_set_uri=manifest["goal_set"]["uri"],
                        goal_set_sha256=manifest["goal_set"]["sha256"].lower(),
                        goal_set_version_id=manifest["goal_set"]["version_id"],
                        goal_count=manifest["goal_set"]["count"],
                        data_seed=seed_set["data_seed"],
                        train_seed=seed_set["train_seed"],
                        planner_seed=seed_set["planner_seed"],
                        variant=variant["name"],
                        epochs=manifest["epochs"],
                        stages=tuple(manifest["stages"]),
                        train_overrides=tuple(sorted(train_overrides.items())),
                        plan_overrides=tuple(sorted(common_plan.items())),
                        max_hours=float(envelope["max_hours"]),
                        max_hourly_usd=float(envelope["max_hourly_usd"]),
                        max_usd=float(envelope["max_usd"]),
                        artifact_uri=artifact_uri,
                        run_id=run_id,
                    )
                )
    return runs


def _walk(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def allowed_runtime_seconds(run: RunSpec, actual_hourly_usd: float) -> int:
    """Apply hourly-rate, wall-clock, and dollar limits to one worker."""
    _require(actual_hourly_usd > 0, "actual hourly rate must be positive")
    _require(
        actual_hourly_usd <= run.max_hourly_usd,
        f"hourly rate ${actual_hourly_usd:.4f} exceeds ${run.max_hourly_usd:.4f} cap",
    )
    by_hours = run.max_hours * 3600
    by_cost = run.max_usd / actual_hourly_usd * 3600
    return max(1, int(min(by_hours, by_cost)))


def plan_digest(runs: list[RunSpec]) -> str:
    """Return a digest teammates can compare before submitting a job array."""
    return _canonical_digest({"runs": [run.to_dict() for run in runs]})
