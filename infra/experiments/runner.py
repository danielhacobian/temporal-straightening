"""Plan or execute one deterministic manifest member on an AWS Batch worker."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from .manifest import (
    ManifestError,
    RunSpec,
    allowed_runtime_seconds,
    expand_runs,
    load_manifest,
    plan_digest,
)
from .prepare_subset import selected_indices


DEFAULT_SCRATCH = Path("/scratch")
PYTHON = "python"
BATCH_MAX_ATTEMPTS = 2
_ACTIVE_PROCESS: subprocess.Popen | None = None
_TERMINATING = False
_ARCHIVED_EPISODE_RE = re.compile(r"^episode_(\d+)\.pth$")


class WorkerTerminated(RuntimeError):
    """Raised when Batch asks the worker to stop."""


def _signal_handler(signum, _frame) -> None:
    global _ACTIVE_PROCESS, _TERMINATING
    process = _ACTIVE_PROCESS
    if _TERMINATING:
        return
    _TERMINATING = True
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            # Training owns checkpoint serialization.  Give it a bounded window
            # to finish an atomic checkpoint before the runner uploads artifacts.
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    raise WorkerTerminated(f"received signal {signum}")


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _override_args(overrides: Mapping[str, Any]) -> list[str]:
    return [f"{key}={_hydra_value(value)}" for key, value in sorted(overrides.items())]


def paths_for(run: RunSpec, scratch_root: Path = DEFAULT_SCRATCH) -> dict[str, Path]:
    """Return the only writable paths used by a worker."""
    workspace = scratch_root / "temporal-straightening" / run.run_id
    return {
        "workspace": workspace,
        "inputs": workspace / "inputs",
        "extracted": workspace / "inputs" / "dataset",
        "subset_parent": workspace / "inputs" / "subset",
        "subset_leaf": workspace / "inputs" / "subset" / run.dataset_leaf,
        "run": workspace / "run",
        "train": workspace / "run" / "checkpoints" / "train",
        "checkpoints": workspace / "run" / "checkpoints",
        "results": workspace / "run" / "results",
        "logs": workspace / "run" / "logs",
        "goal_set": workspace / "inputs" / "fixed_goals.pkl",
        "goal_set_source": workspace / "inputs" / "fixed_goals_source.pkl",
    }


def build_stage_commands(
    run: RunSpec, scratch_root: Path = DEFAULT_SCRATCH
) -> list[dict[str, Any]]:
    """Build argv arrays in manifest order; no shell interpolation is involved."""
    paths = paths_for(run, scratch_root)
    commands: list[dict[str, Any]] = []
    train_overrides = dict(run.train_overrides)
    train_overrides.update(
        {
            "env.dataset.n_rollout": None,
            "training.seed": run.train_seed,
            "training.epochs": run.epochs,
            "+experiment_data_seed": run.data_seed,
            "ckpt_base_path": str(paths["checkpoints"]),
            "hydra.run.dir": str(paths["train"]),
            "hydra.job.chdir": True,
        }
    )
    plan_overrides = dict(run.plan_overrides)
    plan_overrides.update(
        {
            "ckpt_base_path": str(paths["train"]),
            "model_name": run.variant,
            "model_epoch": run.epochs,
            "n_evals": run.goal_count,
            "goal_H": run.goal_horizon,
            "goal_source": "file",
            "goal_file_path": str(paths["goal_set"]),
            "seed": run.planner_seed,
            "hydra.run.dir": str(paths["results"] / "plan"),
            "hydra.sweep.dir": str(paths["results"] / "plan"),
            "hydra.job.chdir": True,
        }
    )
    for stage in run.stages:
        if stage == "train":
            command = [
                PYTHON,
                "train.py",
                "--config-name",
                "train.yaml",
                f"env={run.environment_hydra}",
                *_override_args(train_overrides),
            ]
        elif stage == "proxy":
            command = [
                PYTHON,
                "-m",
                "infra.experiments.proxy_signal",
                "--model-dir",
                str(paths["train"]),
                "--model-epoch",
                str(run.epochs),
                "--goal-set",
                str(paths["goal_set"]),
                "--maze",
                run.maze,
                "--output",
                str(paths["results"] / "proxy.json"),
            ]
        elif stage == "plan":
            command = [
                PYTHON,
                "plan.py",
                "--config-name",
                "plan_gd.yaml",
                *_override_args(plan_overrides),
            ]
        else:  # Manifest validation makes this unreachable.
            raise ManifestError(f"unsupported stage {stage}")
        commands.append({"stage": stage, "argv": command})
    return commands


def deterministic_metadata(
    run: RunSpec, scratch_root: Path = DEFAULT_SCRATCH
) -> dict[str, Any]:
    """Describe inputs and exact commands without timestamps or host-specific state."""
    paths = paths_for(run, scratch_root)
    subset_command = [
        PYTHON,
        "-m",
        "infra.experiments.prepare_subset",
        "--source",
        str(paths["extracted"] / run.dataset_leaf),
        "--destination",
        str(paths["subset_leaf"]),
        "--rollouts",
        str(run.rollouts),
        "--seed",
        str(run.data_seed),
    ]
    goals_command = [
        PYTHON,
        "-m",
        "infra.experiments.prepare_goals",
        "--source",
        str(paths["goal_set_source"]),
        "--destination",
        str(paths["goal_set"]),
        "--count",
        str(run.goal_count),
    ]
    return {
        "run": run.to_dict(),
        "inputs": {
            "dataset": {"uri": run.dataset_uri, "sha256": run.dataset_sha256},
            "goal_set": {"uri": run.goal_set_uri, "sha256": run.goal_set_sha256},
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "prepare_subset_argv": subset_command,
        "prepare_goals_argv": goals_command,
        "commands": build_stage_commands(run, scratch_root),
    }


def build_plan(
    manifest_path: Path,
    *,
    profile: str = "default",
    environment: Mapping[str, str] | None = None,
    scratch_root: Path = DEFAULT_SCRATCH,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, resolve=True, environment=environment)
    runs = expand_runs(manifest, profile)
    per_run_max = [min(run.max_usd, run.max_hours * run.max_hourly_usd) for run in runs]
    one_attempt_total = round(sum(per_run_max), 4)
    return {
        "manifest": str(manifest_path),
        "profile": profile,
        "array_size": len(runs),
        "plan_digest": plan_digest(runs),
        "batch_max_attempts": BATCH_MAX_ATTEMPTS,
        "maximum_one_attempt_usd": one_attempt_total,
        "maximum_total_usd": round(one_attempt_total * BATCH_MAX_ATTEMPTS, 4),
        "runs": [
            {
                "array_index": index,
                **deterministic_metadata(run, scratch_root),
            }
            for index, run in enumerate(runs)
        ],
    }


def verify_submission_identity(runs: list[RunSpec], environment: Mapping[str, str]) -> dict:
    """Fail before paid work if the submitted plan and image revision do not match."""
    actual_plan_digest = plan_digest(runs)
    expected_plan_digest = environment.get("EXPECTED_PLAN_DIGEST")
    actual_source_revision = environment.get("SOURCE_REVISION")
    expected_source_revision = environment.get("EXPECTED_SOURCE_REVISION")
    in_batch = bool(environment.get("AWS_BATCH_JOB_ID"))
    if in_batch and (not expected_plan_digest or not expected_source_revision):
        raise ManifestError(
            "Batch experiment jobs require EXPECTED_PLAN_DIGEST and "
            "EXPECTED_SOURCE_REVISION"
        )
    if expected_plan_digest and expected_plan_digest != actual_plan_digest:
        raise ManifestError(
            f"submitted plan digest {expected_plan_digest} does not match "
            f"container plan {actual_plan_digest}"
        )
    if expected_source_revision and expected_source_revision != actual_source_revision:
        raise ManifestError(
            f"submitted source revision {expected_source_revision} does not match "
            f"container revision {actual_source_revision or '<unset>'}"
        )
    return {
        "plan_digest": actual_plan_digest,
        "source_revision": actual_source_revision,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_logged(
    argv: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    timeout_seconds: float,
) -> None:
    global _ACTIVE_PROCESS
    if timeout_seconds <= 0:
        raise TimeoutError("experiment cost/runtime envelope is exhausted")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        log.write((f"$ {shlex.join(argv)}\n").encode("utf-8"))
        log.flush()
        _ACTIVE_PROCESS = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = _ACTIVE_PROCESS.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            os.killpg(_ACTIVE_PROCESS.pid, signal.SIGTERM)
            try:
                _ACTIVE_PROCESS.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(_ACTIVE_PROCESS.pid, signal.SIGKILL)
                _ACTIVE_PROCESS.wait()
            raise TimeoutError(f"command exceeded remaining envelope: {shlex.join(argv)}") from exc
        finally:
            _ACTIVE_PROCESS = None
    if return_code:
        raise subprocess.CalledProcessError(return_code, argv)


def _s3_download(
    uri: str,
    destination: Path,
    timeout_seconds: float,
    *,
    version_id: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid S3 URI: {uri}")
    subprocess.run(
        [
            "aws",
            "s3api",
            "get-object",
            "--bucket",
            parsed.netloc,
            "--key",
            parsed.path.lstrip("/"),
            "--version-id",
            version_id,
            str(destination),
        ],
        check=True,
        timeout=timeout_seconds,
    )


def _download_verified(
    uri: str,
    digest: str,
    version_id: str,
    destination: Path,
    timeout_seconds: float,
) -> None:
    if destination.is_file() and _sha256(destination) == digest:
        return
    destination.unlink(missing_ok=True)
    _s3_download(uri, destination, timeout_seconds, version_id=version_id)
    observed = _sha256(destination)
    if observed != digest:
        destination.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {uri}: expected {digest}, observed {observed}")


def _dataset_input_path(run: RunSpec, paths: Mapping[str, Path]) -> Path:
    parsed = urlparse(run.dataset_uri)
    suffix = Path(parsed.path).suffix
    if run.dataset_archive == "tar" and parsed.path.endswith(".tar.gz"):
        suffix = ".tar.gz"
    return paths["inputs"] / f"dataset{suffix or '.bin'}"


def _validated_zip_path(name: str) -> PurePosixPath:
    """Return one unambiguous relative ZIP path or fail before extraction."""
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"dataset archive contains an unsafe path: {name!r}")
    raw = name[:-1] if name.endswith("/") else name
    segments = raw.split("/")
    if not raw or any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError(f"dataset archive contains an unsafe path: {name!r}")
    path = PurePosixPath(*segments)
    if path.is_absolute():
        raise ValueError(f"dataset archive contains an unsafe path: {name!r}")
    return path


def _zip_extraction_plan(
    run: RunSpec, handle: zipfile.ZipFile
) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    """Validate every member, then select only files needed by this run."""
    members: dict[PurePosixPath, tuple[zipfile.ZipInfo, bool]] = {}
    for member in handle.infolist():
        path = _validated_zip_path(member.filename)
        if path in members:
            raise ValueError(
                f"dataset archive contains a duplicate path: {member.filename!r}"
            )
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(
                f"dataset archive contains a symbolic link: {member.filename!r}"
            )
        file_type = stat.S_IFMT(mode)
        is_directory = member.is_dir()
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
            if path.name == name and path.parent.name == run.dataset_leaf
        }
        for name in required
    ]
    candidates = set.intersection(*candidate_sets) if candidate_sets else set()
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"dataset archive must contain exactly one {run.dataset_leaf!r} "
            f"directory with {', '.join(required)}"
        )
    source_leaf = candidates.pop()
    source_observations = source_leaf / "obses"

    episode_members: dict[int, tuple[zipfile.ZipInfo, PurePosixPath]] = {}
    leaf_files: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    for path, (member, is_directory) in members.items():
        try:
            relative = path.relative_to(source_leaf)
        except ValueError:
            continue
        if not relative.parts:
            continue
        inside_observations = relative.parts[0] == "obses"
        if is_directory:
            if inside_observations and len(relative.parts) != 1:
                raise ValueError(
                    f"dataset archive contains a malformed observation path: "
                    f"{member.filename!r}"
                )
            continue
        if not inside_observations:
            leaf_files.append((member, relative))
            continue
        if len(relative.parts) != 2 or relative.parent != PurePosixPath("obses"):
            raise ValueError(
                f"dataset archive contains a malformed observation path: "
                f"{member.filename!r}"
            )
        match = _ARCHIVED_EPISODE_RE.fullmatch(relative.name)
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
            raise ValueError(
                f"dataset archive contains duplicate episode {episode_id}"
            )
        episode_members[episode_id] = (member, relative)

    episode_ids = sorted(episode_members)
    if not episode_ids:
        raise FileNotFoundError(
            f"dataset archive contains no observations under {source_observations}"
        )
    expected_ids = list(range(len(episode_ids)))
    if episode_ids != expected_ids:
        raise ValueError(
            "dataset archive observation episodes must be contiguous from zero; "
            f"found {episode_ids[:5]}...{episode_ids[-5:]}"
        )
    # The selected observation IDs must be derived from the same total that
    # prepare_subset will read from the core tensors.  Inspecting these small
    # files here prevents a malformed archive from silently selecting a
    # different episode set and still avoids inflating any observation file.
    import torch

    tensor_counts: dict[str, int] = {}
    for name in required:
        member = members[source_leaf / name][0]
        try:
            value = torch.load(
                io.BytesIO(handle.read(member)),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise ValueError(
                f"dataset archive contains an unreadable core tensor: {name}"
            ) from exc
        if not isinstance(value, torch.Tensor) or value.ndim < 1:
            raise ValueError(
                f"dataset archive core file {name} must contain a tensor with "
                "a rollout dimension"
            )
        tensor_counts[name] = len(value)
    if set(tensor_counts.values()) != {len(episode_ids)}:
        raise ValueError(
            "dataset archive core tensors and observation episode counts differ: "
            f"{tensor_counts}, observations={len(episode_ids)}"
        )
    selected = set(selected_indices(len(episode_ids), run.rollouts, run.data_seed))
    plan = list(leaf_files)
    plan.extend(episode_members[index] for index in sorted(selected))
    return plan


def _extract_zip_dataset(run: RunSpec, archive: Path, destination: Path) -> None:
    """Extract a validated dataset leaf, omitting unused observation episodes."""
    with zipfile.ZipFile(archive) as handle:
        # The plan scans every central-directory member before the temporary
        # extraction directory is created, so an unsafe late entry writes
        # nothing and cannot partially replace a previous materialization.
        plan = _zip_extraction_plan(run, handle)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.extract-", dir=destination.parent
            )
        )
        try:
            target_leaf = temporary / run.dataset_leaf
            for member, relative in plan:
                target = target_leaf.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            if destination.exists():
                shutil.rmtree(destination)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def _extract_dataset(run: RunSpec, archive: Path, destination: Path) -> None:
    if run.dataset_archive == "zip":
        _extract_zip_dataset(run, archive, destination)
        return
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    root = destination.resolve()

    def safe_path(name: str) -> None:
        candidate = (root / name).resolve()
        if os.path.commonpath((root, candidate)) != str(root):
            raise ValueError(f"dataset archive contains an unsafe path: {name!r}")

    if run.dataset_archive == "tar":
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                safe_path(member.name)
                if not (member.isdir() or member.isreg()):
                    raise ValueError(
                        f"dataset archive contains a non-file entry: {member.name!r}"
                    )
            handle.extractall(destination)
    else:
        leaf = destination / run.dataset_leaf
        leaf.mkdir(parents=True)
        shutil.copy2(archive, leaf / archive.name)
    expected_leaf = destination / run.dataset_leaf
    if not expected_leaf.is_dir():
        candidates = list(destination.rglob(run.dataset_leaf))
        if len(candidates) == 1 and candidates[0].is_dir():
            expected_leaf.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidates[0]), str(expected_leaf))
        else:
            raise FileNotFoundError(
                f"dataset archive did not contain one {run.dataset_leaf!r} directory"
            )


def _remaining(deadline: float, reserve_seconds: float = 0) -> float:
    return max(0.0, deadline - time.monotonic() - reserve_seconds)


def _upload(run: RunSpec, run_root: Path, timeout_seconds: float = 600) -> None:
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            str(run_root),
            run.artifact_uri,
            "--only-show-errors",
        ],
        check=True,
        timeout=max(1, timeout_seconds),
    )


def _restore(run: RunSpec, run_root: Path, timeout_seconds: float = 600) -> None:
    """Restore checkpoints/logs from an interrupted attempt before writing metadata."""
    run_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            run.artifact_uri,
            str(run_root),
            "--only-show-errors",
        ],
        check=True,
        timeout=max(1, timeout_seconds),
    )


def execute_run(
    run: RunSpec,
    *,
    actual_hourly_usd: float | None = None,
    scratch_root: Path = DEFAULT_SCRATCH,
    repo_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute one array member, always attempting to preserve logs in S3."""
    if not scratch_root.is_absolute() or (
        not dry_run and scratch_root != DEFAULT_SCRATCH
    ):
        raise ManifestError("AWS Batch execution must use /scratch")
    repo_root = (repo_root or Path.cwd()).resolve()
    if not dry_run and not (repo_root / "train.py").is_file():
        raise FileNotFoundError(f"train.py not found under {repo_root}")
    hourly = actual_hourly_usd if actual_hourly_usd is not None else run.max_hourly_usd
    runtime_seconds = allowed_runtime_seconds(run, hourly)
    metadata = deterministic_metadata(run, scratch_root)
    metadata["submission_identity"] = {
        "plan_digest": os.environ.get("VERIFIED_PLAN_DIGEST"),
        "source_revision": os.environ.get("VERIFIED_SOURCE_REVISION"),
    }
    metadata["runtime_enforcement"] = {
        "hourly_usd": hourly,
        "hourly_rate_source": "provided" if actual_hourly_usd is not None else "envelope_cap",
        "allowed_runtime_seconds": runtime_seconds,
    }
    if dry_run:
        return metadata

    started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    deadline = started_monotonic + runtime_seconds
    state = "failed"
    error: str | None = None
    restored_complete_status: dict[str, Any] | None = None
    paths = paths_for(run, scratch_root)
    for key in ("inputs", "checkpoints", "results", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)

    for handled_signal in (signal.SIGTERM, signal.SIGINT):
        signal.signal(handled_signal, _signal_handler)
    try:
        _restore(
            run,
            paths["run"],
            timeout_seconds=min(600, _remaining(deadline, 300)),
        )
        restored_metadata_path = paths["run"] / "run_metadata.json"
        restored_status_path = paths["run"] / "status.json"
        if restored_metadata_path.is_file():
            restored_metadata = json.loads(
                restored_metadata_path.read_text(encoding="utf-8")
            )
            restored_identity = restored_metadata.get("submission_identity", {})
            if restored_identity != metadata["submission_identity"]:
                raise ManifestError(
                    "restored artifacts were produced by a different plan or source "
                    f"revision: {restored_identity!r} != "
                    f"{metadata['submission_identity']!r}"
                )
        elif restored_status_path.is_file() or any(paths["checkpoints"].rglob("*.pth")):
            raise ManifestError(
                "restored artifacts have no run_metadata.json provenance; refusing resume"
            )
        if restored_status_path.is_file():
            restored_status = json.loads(restored_status_path.read_text(encoding="utf-8"))
            if restored_status.get("state") == "complete":
                state = "complete"
                restored_complete_status = restored_status
                return restored_status
        _write_json(paths["run"] / "run_metadata.json", metadata)
        dataset_file = _dataset_input_path(run, paths)
        _download_verified(
            run.dataset_uri,
            run.dataset_sha256,
            run.dataset_version_id,
            dataset_file,
            _remaining(deadline, 300),
        )
        _download_verified(
            run.goal_set_uri,
            run.goal_set_sha256,
            run.goal_set_version_id,
            paths["goal_set_source"],
            _remaining(deadline, 300),
        )
        _extract_dataset(run, dataset_file, paths["extracted"])

        environment = os.environ.copy()
        environment.update(
            {
                "DATASET_DIR": str(paths["subset_parent"]),
                "WANDB_MODE": "disabled",
                "EXPERIMENT_RUN_ID": run.run_id,
                "EXPERIMENT_DATA_SEED": str(run.data_seed),
                "EXPERIMENT_TRAIN_SEED": str(run.train_seed),
                "EXPERIMENT_PLANNER_SEED": str(run.planner_seed),
                "EXPERIMENT_ARTIFACT_URI": run.artifact_uri,
            }
        )
        subset_argv = metadata["prepare_subset_argv"]
        _run_logged(
            subset_argv,
            cwd=repo_root,
            environment=environment,
            log_path=paths["logs"] / "prepare_subset.log",
            timeout_seconds=_remaining(deadline, 300),
        )
        _run_logged(
            metadata["prepare_goals_argv"],
            cwd=repo_root,
            environment=environment,
            log_path=paths["logs"] / "prepare_goals.log",
            timeout_seconds=_remaining(deadline, 300),
        )
        for stage in metadata["commands"]:
            _run_logged(
                stage["argv"],
                cwd=repo_root,
                environment=environment,
                log_path=paths["logs"] / f"{stage['stage']}.log",
                timeout_seconds=_remaining(deadline, 300),
            )
        state = "complete"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if restored_complete_status is None:
            elapsed_seconds = time.monotonic() - started_monotonic
            status = {
                "run_id": run.run_id,
                "state": state,
                "error": error,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "estimated_compute_usd": round(elapsed_seconds / 3600 * hourly, 6),
                "aws_batch_job_id": os.environ.get("AWS_BATCH_JOB_ID"),
                "aws_batch_array_index": os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX"),
            }
            _write_json(paths["run"] / "status.json", status)
            try:
                upload_timeout = 75 if _TERMINATING else 600
                _upload(
                    run,
                    paths["run"],
                    timeout_seconds=min(
                        upload_timeout,
                        max(1, _remaining(deadline)),
                    ),
                )
            except Exception as upload_error:
                print(f"artifact upload failed: {upload_error}", file=sys.stderr)
                if state == "complete":
                    raise
    return status


def _array_index(cli_index: int | None) -> int:
    if cli_index is not None:
        return cli_index
    value = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX")
    if value is None:
        raise ManifestError("--index or AWS_BATCH_JOB_ARRAY_INDEX is required")
    try:
        return int(value)
    except ValueError as exc:
        raise ManifestError("AWS_BATCH_JOB_ARRAY_INDEX must be an integer") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    validate = subparsers.add_parser("validate", help="validate without resolving S3 variables")
    validate.add_argument("manifests", type=Path, nargs="+")

    plan = subparsers.add_parser("plan", help="emit the deterministic Batch array plan")
    plan.add_argument("manifest", type=Path)
    plan.add_argument("--profile", default="default")
    plan.add_argument("--budget-usd", type=float)
    plan.add_argument("--output", type=Path)

    run = subparsers.add_parser("run", help="execute one Batch array member")
    run.add_argument("manifest", type=Path)
    run.add_argument("--profile", default="default")
    run.add_argument("--index", type=int)
    run.add_argument("--hourly-usd", type=float)
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "validate":
        for path in args.manifests:
            load_manifest(path)
            print(f"valid: {path}")
        return 0
    if args.action == "plan":
        payload = build_plan(args.manifest, profile=args.profile)
        if args.budget_usd is not None and payload["maximum_total_usd"] > args.budget_usd:
            raise ManifestError(
                f"two-attempt plan maximum ${payload['maximum_total_usd']:.2f} exceeds "
                f"${args.budget_usd:.2f} submission budget"
            )
        output = json.dumps(payload, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        return 0
    if args.action == "run":
        manifest = load_manifest(args.manifest, resolve=True)
        runs = expand_runs(manifest, args.profile)
        submission_identity = verify_submission_identity(runs, os.environ)
        os.environ["VERIFIED_PLAN_DIGEST"] = submission_identity["plan_digest"]
        os.environ["VERIFIED_SOURCE_REVISION"] = (
            submission_identity["source_revision"] or "unknown"
        )
        index = _array_index(args.index)
        if not 0 <= index < len(runs):
            raise ManifestError(f"array index {index} is outside [0, {len(runs)})")
        hourly = args.hourly_usd
        if hourly is None and os.environ.get("EXPERIMENT_HOURLY_USD"):
            hourly = float(os.environ["EXPERIMENT_HOURLY_USD"])
        result = execute_run(runs[index], actual_hourly_usd=hourly, dry_run=args.dry_run)
        result["submission_identity"] = submission_identity
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
