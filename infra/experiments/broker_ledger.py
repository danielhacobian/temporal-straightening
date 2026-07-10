"""Durable cost admission and recovery for the trusted AWS Batch broker.

An S3 compare-and-swap lease serializes the aggregate read plus reservation,
even when many independent GitHub tag workflows arrive together. Ledger entries
remain outstanding while a job is reserved, active, or waiting for the AWS
Budgets actual-spend value to settle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
BUDGET_LIMIT_USD = Decimal("20.00")
SAFETY_RESERVE_USD = Decimal("5.00")
ADMISSION_LIMIT_USD = BUDGET_LIMIT_USD - SAFETY_RESERVE_USD
MAX_REQUEST_USD = Decimal("5.00")
LEDGER_PREFIX = "budget-ledger/"
ADMISSION_LOCK_KEY = "broker-locks/budget-admission.json"
RESERVATION_GRACE = timedelta(minutes=15)
SETTLEMENT_HOLD = timedelta(days=7)
BROKER_WORKFLOW_TIMEOUT = timedelta(minutes=15)
LOCK_LEASE = timedelta(minutes=20)
LOCK_WAIT_SECONDS = 600
LOCK_POLL_SECONDS = 2
LOCK_MIN_REMAINING = timedelta(seconds=30)
ACTIVE_JOB_STATUSES = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")
TERMINAL_JOB_STATUSES = {"SUCCEEDED", "FAILED"}
OUTSTANDING_STATES = {"reserved", "submitted", "settling"}
FINAL_STATES = {"released", "expired"}
ALL_STATES = OUTSTANDING_STATES | FINAL_STATES

if LOCK_LEASE <= BROKER_WORKFLOW_TIMEOUT:
    raise RuntimeError("admission lease must outlive the entire broker workflow")


class LedgerError(RuntimeError):
    """Raised when cost admission cannot be proven safe."""


@dataclass(frozen=True)
class LedgerRecord:
    key: str
    etag: str
    entry: dict[str, Any]


@dataclass(frozen=True)
class AdmissionLease:
    key: str
    etag: str
    entry: dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise LedgerError(f"ledger {field} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerError(f"ledger {field} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise LedgerError(f"ledger {field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _money(value: Any, field: str) -> Decimal:
    if isinstance(value, bool):
        raise LedgerError(f"{field} must be a USD number")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise LedgerError(f"{field} must be a USD number") from exc
    if not amount.is_finite():
        raise LedgerError(f"{field} must be finite")
    return amount


def _money_text(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.0001")), "f")


def reservation_id(
    manifest: str, profile: str, source_revision: str, plan_digest: str
) -> str:
    payload = json.dumps(
        {
            "manifest": manifest,
            "plan_digest": plan_digest,
            "profile": profile,
            "source_revision": source_revision,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ledger_key(reservation: str) -> str:
    if len(reservation) != 64 or any(c not in "0123456789abcdef" for c in reservation):
        raise LedgerError("reservation ID must be a lowercase SHA-256 digest")
    return f"{LEDGER_PREFIX}{reservation}.json"


def validate_entry(entry: Mapping[str, Any], *, expected_key: str | None = None) -> None:
    required_strings = (
        "reservation_id",
        "manifest",
        "profile",
        "source_revision",
        "plan_digest",
        "job_name",
        "owner",
        "trigger",
        "state",
        "created_at",
        "updated_at",
    )
    if entry.get("schema_version") != SCHEMA_VERSION:
        raise LedgerError("ledger schema_version must be 1")
    for field in required_strings:
        if not isinstance(entry.get(field), str) or not entry[field]:
            raise LedgerError(f"ledger {field} must be a non-empty string")
    if entry["state"] not in ALL_STATES:
        raise LedgerError(f"unsupported ledger state {entry['state']!r}")
    reservation = entry["reservation_id"]
    if expected_key is not None and ledger_key(reservation) != expected_key:
        raise LedgerError("ledger key does not match reservation ID")
    expected_id = reservation_id(
        entry["manifest"],
        entry["profile"],
        entry["source_revision"],
        entry["plan_digest"],
    )
    if reservation != expected_id:
        raise LedgerError("ledger reservation identity is not canonical")
    amount = _money(entry.get("maximum_total_usd"), "maximum_total_usd")
    if amount <= 0 or amount > MAX_REQUEST_USD:
        raise LedgerError("ledger maximum_total_usd must be in (0, 5]")
    _money(entry.get("budget_actual_usd_at_reservation"), "budget actual")
    _parse_timestamp(entry["created_at"], "created_at")
    _parse_timestamp(entry["updated_at"], "updated_at")
    generation = entry.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise LedgerError("ledger generation must be a positive integer")
    job_id = entry.get("job_id")
    if job_id is not None and (not isinstance(job_id, str) or not job_id):
        raise LedgerError("ledger job_id must be null or a non-empty string")
    if entry["state"] in {"submitted", "settling", "released"} and not job_id:
        raise LedgerError(f"ledger state {entry['state']} requires a job_id")
    if entry["state"] == "settling":
        _parse_timestamp(entry.get("settle_after"), "settle_after")


def validate_lease(entry: Mapping[str, Any]) -> None:
    if entry.get("schema_version") != SCHEMA_VERSION:
        raise LedgerError("admission lease schema_version must be 1")
    if entry.get("state") not in {"held", "released"}:
        raise LedgerError("admission lease state must be held or released")
    for field in ("owner", "lease_id", "updated_at"):
        if not isinstance(entry.get(field), str) or not entry[field]:
            raise LedgerError(f"admission lease {field} must be a non-empty string")
    generation = entry.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise LedgerError("admission lease generation must be a positive integer")
    _parse_timestamp(entry["updated_at"], "lease updated_at")
    if entry["state"] == "held":
        _parse_timestamp(entry.get("acquired_at"), "lease acquired_at")
        _parse_timestamp(entry.get("expires_at"), "lease expires_at")
    else:
        _parse_timestamp(entry.get("released_at"), "lease released_at")


def new_reservation(
    *,
    manifest: str,
    profile: str,
    source_revision: str,
    plan_digest: str,
    maximum_total_usd: Decimal,
    budget_actual_usd: Decimal,
    job_name: str,
    owner: str,
    trigger: str,
    generation: int = 1,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or _utc_now()
    reservation = reservation_id(manifest, profile, source_revision, plan_digest)
    entry = {
        "schema_version": SCHEMA_VERSION,
        "reservation_id": reservation,
        "manifest": manifest,
        "profile": profile,
        "source_revision": source_revision,
        "plan_digest": plan_digest,
        "maximum_total_usd": float(maximum_total_usd),
        "budget_actual_usd_at_reservation": float(budget_actual_usd),
        "job_name": job_name,
        "job_id": None,
        "owner": owner,
        "trigger": trigger,
        "state": "reserved",
        "generation": generation,
        "created_at": _timestamp(now),
        "updated_at": _timestamp(now),
    }
    validate_entry(entry)
    return entry


def outstanding_total(records: Iterable[LedgerRecord]) -> Decimal:
    total = Decimal("0")
    for record in records:
        validate_entry(record.entry, expected_key=record.key)
        if record.entry["state"] in OUTSTANDING_STATES:
            total += _money(record.entry["maximum_total_usd"], "maximum_total_usd")
    return total


def projected_total(
    actual_usd: Decimal, outstanding_usd: Decimal, requested_usd: Decimal
) -> Decimal:
    if requested_usd <= 0 or requested_usd > MAX_REQUEST_USD:
        raise LedgerError("requested retry-inclusive maximum must be in (0, 5]")
    projected = actual_usd + outstanding_usd + requested_usd
    if projected > ADMISSION_LIMIT_USD:
        raise LedgerError(
            "budget admission refused: actual "
            f"${_money_text(actual_usd)} + outstanding "
            f"${_money_text(outstanding_usd)} + requested "
            f"${_money_text(requested_usd)} = ${_money_text(projected)} exceeds "
            f"the usable ${_money_text(ADMISSION_LIMIT_USD)} admission ceiling "
            f"(${_money_text(BUDGET_LIMIT_USD)} budget minus "
            f"${_money_text(SAFETY_RESERVE_USD)} safety reserve)"
        )
    return projected


def _run_aws(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["aws", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise LedgerError("AWS CLI is required for broker ledger operations") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise LedgerError(f"AWS CLI failed: {' '.join(args)}: {detail}") from exc


def _aws_json(*args: str) -> dict[str, Any]:
    completed = _run_aws([*args, "--output", "json"])
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LedgerError(f"AWS CLI returned invalid JSON for {' '.join(args)}") from exc
    if not isinstance(payload, dict):
        raise LedgerError(f"AWS CLI returned a non-object for {' '.join(args)}")
    return payload


def _aws_text(*args: str) -> str:
    return _run_aws([*args, "--output", "text"]).stdout.strip()


class AwsBrokerStore:
    """Small AWS CLI adapter used by the broker and mocked by unit tests."""

    def __init__(self, bucket: str, queue: str):
        if not bucket or not queue:
            raise LedgerError("artifact bucket and Batch queue are required")
        self.bucket = bucket
        self.queue = queue

    def _get_json_object(self, key: str) -> tuple[dict[str, Any], str]:
        with tempfile.TemporaryDirectory(prefix="ts-broker-state-") as directory:
            body = Path(directory) / "state.json"
            response = _aws_json(
                "s3api",
                "get-object",
                "--bucket",
                self.bucket,
                "--key",
                key,
                str(body),
            )
            try:
                payload = json.loads(body.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LedgerError(f"invalid broker state JSON at {key}") from exc
        if not isinstance(payload, dict):
            raise LedgerError(f"broker state object {key} must contain a JSON object")
        etag = response.get("ETag")
        if not isinstance(etag, str) or not etag:
            raise LedgerError(f"broker state object {key} has no ETag")
        return payload, etag

    def _put_json_object(
        self,
        key: str,
        payload: Mapping[str, Any],
        *,
        expected_etag: str | None,
    ) -> str:
        with tempfile.TemporaryDirectory(prefix="ts-broker-state-") as directory:
            body = Path(directory) / "state.json"
            body.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            arguments = [
                "s3api",
                "put-object",
                "--bucket",
                self.bucket,
                "--key",
                key,
                "--body",
                str(body),
                "--content-type",
                "application/json",
            ]
            if expected_etag is None:
                arguments.extend(("--if-none-match", "*"))
            else:
                arguments.extend(("--if-match", expected_etag))
            response = _aws_json(*arguments)
        etag = response.get("ETag")
        if not isinstance(etag, str) or not etag:
            raise LedgerError("conditional broker-state write returned no ETag")
        return etag

    def list_records(self) -> list[LedgerRecord]:
        response = _aws_json(
            "s3api",
            "list-objects-v2",
            "--bucket",
            self.bucket,
            "--prefix",
            LEDGER_PREFIX,
        )
        keys = sorted(
            item["Key"]
            for item in response.get("Contents", [])
            if isinstance(item, dict)
            and isinstance(item.get("Key"), str)
            and item["Key"] != ADMISSION_LOCK_KEY
            and item["Key"].endswith(".json")
        )
        return [self.get_record(key) for key in keys]

    def get_record(self, key: str) -> LedgerRecord:
        if not key.startswith(LEDGER_PREFIX) or not key.endswith(".json"):
            raise LedgerError("refusing to read outside the broker ledger prefix")
        entry, etag = self._get_json_object(key)
        validate_entry(entry, expected_key=key)
        return LedgerRecord(key=key, etag=etag, entry=entry)

    def put_record(
        self,
        key: str,
        entry: Mapping[str, Any],
        *,
        expected_etag: str | None,
    ) -> LedgerRecord:
        validate_entry(entry, expected_key=key)
        etag = self._put_json_object(key, entry, expected_etag=expected_etag)
        return LedgerRecord(key=key, etag=etag, entry=dict(entry))

    def get_admission_lease(self) -> AdmissionLease | None:
        response = _aws_json(
            "s3api",
            "list-objects-v2",
            "--bucket",
            self.bucket,
            "--prefix",
            ADMISSION_LOCK_KEY,
        )
        exists = any(
            isinstance(item, dict) and item.get("Key") == ADMISSION_LOCK_KEY
            for item in response.get("Contents", [])
        )
        if not exists:
            return None
        entry, etag = self._get_json_object(ADMISSION_LOCK_KEY)
        validate_lease(entry)
        return AdmissionLease(ADMISSION_LOCK_KEY, etag, entry)

    def put_admission_lease(
        self,
        entry: Mapping[str, Any],
        *,
        expected_etag: str | None,
        tolerate_conflict: bool = False,
    ) -> AdmissionLease | None:
        validate_lease(entry)
        try:
            etag = self._put_json_object(
                ADMISSION_LOCK_KEY, entry, expected_etag=expected_etag
            )
        except LedgerError as exc:
            conflict_markers = (
                "PreconditionFailed",
                "ConditionalRequestConflict",
                "status code: 409",
                "status code: 412",
                "(409)",
                "(412)",
            )
            if tolerate_conflict and any(marker in str(exc) for marker in conflict_markers):
                return None
            raise
        return AdmissionLease(ADMISSION_LOCK_KEY, etag, dict(entry))

    def describe_job(self, job_id: str) -> dict[str, Any] | None:
        response = _aws_json("batch", "describe-jobs", "--jobs", job_id)
        jobs = response.get("jobs", [])
        if not jobs:
            return None
        if len(jobs) != 1 or not isinstance(jobs[0], dict):
            raise LedgerError(f"Batch returned an ambiguous description for {job_id}")
        return jobs[0]

    def find_job(self, entry: Mapping[str, Any]) -> dict[str, Any] | None:
        # A non-SHARE ListJobs filter ignores jobStatus and searches all states.
        # Omit --max-items so the AWS CLI follows every service page.
        response = _aws_json(
            "batch",
            "list-jobs",
            "--job-queue",
            self.queue,
            "--filters",
            f"name=JOB_NAME,values={entry['job_name']}",
        )
        ids = list(
            dict.fromkeys(
                item["jobId"]
                for item in response.get("jobSummaryList", [])
                if isinstance(item, dict)
                and item.get("jobName") == entry["job_name"]
                and isinstance(item.get("jobId"), str)
            )
        )
        matches: list[dict[str, Any]] = []
        mismatched = False
        for job_id in ids:
            job = self.describe_job(job_id)
            if job is None or job.get("jobName") != entry["job_name"]:
                continue
            tags = job.get("tags", {})
            if (
                tags.get("PlanDigest") == entry["plan_digest"]
                and tags.get("SourceRevision") == entry["source_revision"]
            ):
                matches.append(job)
            else:
                mismatched = True
        if len(matches) > 1:
            raise LedgerError(f"multiple Batch jobs match {entry['reservation_id']}")
        if not matches and mismatched:
            raise LedgerError("deterministic Batch job name exists with mismatched tags")
        return matches[0] if matches else None

def _job_terminal_time(job: Mapping[str, Any], now: datetime) -> datetime:
    stopped_at = job.get("stoppedAt")
    if isinstance(stopped_at, (int, float)) and not isinstance(stopped_at, bool):
        return datetime.fromtimestamp(float(stopped_at) / 1000, tz=timezone.utc)
    return now


def _transition_for_job(
    entry: Mapping[str, Any], job: Mapping[str, Any], now: datetime
) -> dict[str, Any]:
    job_id = job.get("jobId")
    status = job.get("status")
    if not isinstance(job_id, str) or not job_id:
        raise LedgerError("Batch job has no jobId")
    if status not in ACTIVE_JOB_STATUSES and status not in TERMINAL_JOB_STATUSES:
        raise LedgerError(f"unsupported Batch job status {status!r}")
    updated = dict(entry)
    updated["job_id"] = job_id
    updated["batch_status"] = status
    updated["updated_at"] = _timestamp(now)
    if status in ACTIVE_JOB_STATUSES:
        updated["state"] = "submitted"
        updated.setdefault("submitted_at", _timestamp(now))
    else:
        terminal_at = _job_terminal_time(job, now)
        settle_after = terminal_at + SETTLEMENT_HOLD
        updated["state"] = "released" if now >= settle_after else "settling"
        updated["terminal_at"] = _timestamp(terminal_at)
        updated["settle_after"] = _timestamp(settle_after)
        if updated["state"] == "released":
            updated["released_at"] = _timestamp(now)
            updated["release_reason"] = "budget-settlement-hold-complete"
    validate_entry(updated)
    return updated


def reconcile_records(
    store: AwsBrokerStore,
    records: Iterable[LedgerRecord],
    *,
    current_owner: str,
    now: datetime | None = None,
) -> list[LedgerRecord]:
    """Recover jobs and expire only reservations proven abandoned."""
    now = now or _utc_now()
    reconciled: list[LedgerRecord] = []
    for record in records:
        entry = dict(record.entry)
        state = entry["state"]
        updated: dict[str, Any] | None = None
        if state == "reserved":
            job = store.find_job(entry)
            if job is not None:
                updated = _transition_for_job(entry, job, now)
            else:
                created = _parse_timestamp(entry["created_at"], "created_at")
                old_owner = entry["owner"] != current_owner
                if old_owner and now - created >= RESERVATION_GRACE:
                    updated = dict(entry)
                    updated.update(
                        {
                            "state": "expired",
                            "updated_at": _timestamp(now),
                            "released_at": _timestamp(now),
                            "release_reason": "abandoned-reservation-with-no-batch-job",
                        }
                    )
        elif state == "submitted":
            job_id = entry.get("job_id")
            job = store.describe_job(job_id) if isinstance(job_id, str) else None
            if job is None:
                # Batch normally retains terminal metadata for days. Missing
                # metadata is ambiguous, so keep the full amount reserved.
                updated = None
            else:
                candidate = _transition_for_job(entry, job, now)
                if candidate != entry:
                    updated = candidate
        elif state == "settling":
            settle_after = _parse_timestamp(entry["settle_after"], "settle_after")
            if now >= settle_after:
                updated = dict(entry)
                updated.update(
                    {
                        "state": "released",
                        "updated_at": _timestamp(now),
                        "released_at": _timestamp(now),
                        "release_reason": "budget-settlement-hold-complete",
                    }
                )
        if updated is not None:
            validate_entry(updated, expected_key=record.key)
            record = store.put_record(
                record.key, updated, expected_etag=record.etag
            )
        reconciled.append(record)
    return reconciled


def _held_lease(
    *,
    owner: str,
    lease_id: str,
    generation: int,
    now: datetime,
    lease_seconds: int,
) -> dict[str, Any]:
    entry = {
        "schema_version": SCHEMA_VERSION,
        "state": "held",
        "owner": owner,
        "lease_id": lease_id,
        "generation": generation,
        "acquired_at": _timestamp(now),
        "expires_at": _timestamp(now + timedelta(seconds=lease_seconds)),
        "updated_at": _timestamp(now),
    }
    validate_lease(entry)
    return entry


def acquire_admission_lease(
    store: AwsBrokerStore,
    *,
    owner: str,
    wait_seconds: float = LOCK_WAIT_SECONDS,
    lease_seconds: int = int(LOCK_LEASE.total_seconds()),
    poll_seconds: float = LOCK_POLL_SECONDS,
    now_fn=_utc_now,
    monotonic_fn=time.monotonic,
    sleep_fn=time.sleep,
) -> AdmissionLease:
    """Acquire the global aggregate-admission lease with bounded CAS retries."""
    if not owner:
        raise LedgerError("admission lease owner is required")
    if wait_seconds < 0 or lease_seconds <= 0 or poll_seconds < 0:
        raise LedgerError("admission lease timing values are invalid")
    deadline = monotonic_fn() + wait_seconds
    lease_id = secrets.token_hex(16)
    while True:
        now = now_fn()
        current = store.get_admission_lease()
        expected_etag: str | None = None
        generation = 1
        can_take = current is None
        if current is not None:
            entry = current.entry
            expires_at = (
                _parse_timestamp(entry.get("expires_at"), "lease expires_at")
                if entry["state"] == "held"
                else now
            )
            if (
                entry["state"] == "held"
                and entry["owner"] == owner
                and now < expires_at
            ):
                return current
            can_take = entry["state"] == "released" or now >= expires_at
            if can_take:
                expected_etag = current.etag
                generation = int(entry["generation"]) + 1
        if can_take:
            candidate = _held_lease(
                owner=owner,
                lease_id=lease_id,
                generation=generation,
                now=now,
                lease_seconds=lease_seconds,
            )
            acquired = store.put_admission_lease(
                candidate,
                expected_etag=expected_etag,
                tolerate_conflict=True,
            )
            if acquired is not None:
                return acquired
        if monotonic_fn() >= deadline:
            holder = current.entry["owner"] if current is not None else "unknown"
            raise LedgerError(
                f"timed out waiting for aggregate budget admission lease held by {holder}"
            )
        sleep_fn(poll_seconds)


def release_admission_lease(
    store: AwsBrokerStore,
    lease: AdmissionLease,
    *,
    now: datetime | None = None,
) -> AdmissionLease:
    """Release only the exact owner/lease generation acquired by this process."""
    now = now or _utc_now()
    current = store.get_admission_lease()
    if current is None:
        raise LedgerError("aggregate admission lease disappeared before release")
    expected = lease.entry
    observed = current.entry
    identity_fields = ("owner", "lease_id", "generation")
    if any(observed.get(field) != expected.get(field) for field in identity_fields):
        raise LedgerError("aggregate admission lease ownership changed before release")
    if observed["state"] == "released":
        return current
    released = dict(observed)
    released.update(
        {
            "state": "released",
            "released_at": _timestamp(now),
            "updated_at": _timestamp(now),
        }
    )
    result = store.put_admission_lease(
        released,
        expected_etag=current.etag,
        tolerate_conflict=False,
    )
    if result is None:  # pragma: no cover - strict writes never return None.
        raise LedgerError("aggregate admission lease release unexpectedly conflicted")
    return result


def assert_admission_lease_current(
    store: AwsBrokerStore,
    lease: AdmissionLease,
    *,
    now: datetime | None = None,
) -> AdmissionLease:
    """Fence a stale holder immediately before and after reservation write."""
    now = now or _utc_now()
    current = store.get_admission_lease()
    if current is None or current.entry["state"] != "held":
        raise LedgerError("aggregate admission lease is no longer held")
    identity_fields = ("owner", "lease_id", "generation")
    if any(
        current.entry.get(field) != lease.entry.get(field)
        for field in identity_fields
    ):
        raise LedgerError("aggregate admission lease ownership changed")
    expires_at = _parse_timestamp(current.entry["expires_at"], "lease expires_at")
    if expires_at - now < LOCK_MIN_REMAINING:
        raise LedgerError("aggregate admission lease expired or is too close to expiry")
    return current


@contextmanager
def held_admission_lease(store: AwsBrokerStore, *, owner: str):
    """Release on every path without masking a primary admission failure."""
    lease = acquire_admission_lease(store, owner=owner)
    try:
        yield lease
    except BaseException:
        try:
            release_admission_lease(store, lease)
        except LedgerError as release_error:
            print(
                f"warning: admission lease release also failed: {release_error}",
                file=os.sys.stderr,
            )
        raise
    else:
        release_admission_lease(store, lease)


def read_budget_actual(budget_name: str) -> Decimal:
    account_id = _aws_text("sts", "get-caller-identity", "--query", "Account")
    if not account_id or account_id == "None":
        raise LedgerError("AWS account ID could not be resolved")
    payload = _aws_json(
        "budgets",
        "describe-budget",
        "--account-id",
        account_id,
        "--budget-name",
        budget_name,
        "--show-filter-expression",
    )
    budget = payload.get("Budget")
    if not isinstance(budget, dict):
        raise LedgerError("AWS Budgets returned no Budget object")
    if budget.get("BudgetType") != "COST":
        raise LedgerError("budget must be a COST budget")
    if budget.get("TimeUnit") != "MONTHLY":
        raise LedgerError("budget must reset MONTHLY")
    if budget.get("Metrics") != ["NetUnblendedCost"]:
        raise LedgerError("budget metric must be exactly NetUnblendedCost")
    if budget.get("CostFilters") not in (None, {}):
        raise LedgerError("budget must cover the unfiltered account")
    if budget.get("FilterExpression") not in (None, {}):
        raise LedgerError("budget must not use a filter expression")
    if budget.get("BillingViewArn"):
        raise LedgerError("budget must not use a scoped billing view")
    cost_types = budget.get("CostTypes") or {}
    if not isinstance(cost_types, dict):
        raise LedgerError("budget CostTypes is malformed")
    include_fields = (
        "IncludeCredit",
        "IncludeDiscount",
        "IncludeOtherSubscription",
        "IncludeRecurring",
        "IncludeRefund",
        "IncludeSubscription",
        "IncludeSupport",
        "IncludeTax",
        "IncludeUpfront",
    )
    omitted_known = [
        field for field in include_fields if cost_types.get(field, True) is not True
    ]
    omitted_future = [
        field
        for field, value in cost_types.items()
        if field.startswith("Include") and field not in include_fields and value is not True
    ]
    if omitted_known or omitted_future:
        omitted = ", ".join(sorted((*omitted_known, *omitted_future)))
        raise LedgerError(f"budget must include every cost type; excluded: {omitted}")
    if (
        cost_types.get("UseBlended", False) is not False
        or cost_types.get("UseAmortized", False) is not False
    ):
        raise LedgerError("budget must use net unblended, non-amortized cost")
    limit = budget.get("BudgetLimit", {})
    amount = _money(limit.get("Amount"), "budget limit")
    if limit.get("Unit") != "USD" or amount != BUDGET_LIMIT_USD:
        raise LedgerError("expected an exact USD 20 budget limit")
    calculated = budget.get("CalculatedSpend")
    if not isinstance(calculated, dict):
        raise LedgerError("budget has no CalculatedSpend; refusing fail-open admission")
    actual_payload = calculated.get("ActualSpend")
    if not isinstance(actual_payload, dict) or "Amount" not in actual_payload:
        raise LedgerError("budget has no ActualSpend amount; refusing fail-open admission")
    actual = _money(actual_payload["Amount"], "budget actual spend")
    if actual_payload.get("Unit") != "USD":
        raise LedgerError("budget actual spend is not denominated in USD")
    return actual


def reserve_budget(
    store: AwsBrokerStore,
    *,
    budget_name: str,
    manifest: str,
    profile: str,
    source_revision: str,
    plan_digest: str,
    maximum_total_usd: Decimal,
    job_name: str,
    owner: str,
    trigger: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or _utc_now()
    with held_admission_lease(store, owner=owner) as lease:
        actual = read_budget_actual(budget_name)
        records = reconcile_records(
            store, store.list_records(), current_owner=owner, now=now
        )
        current_id = reservation_id(manifest, profile, source_revision, plan_digest)
        current_key = ledger_key(current_id)
        by_key = {record.key: record for record in records}
        existing = by_key.get(current_key)
        if existing and existing.entry["state"] in {
            "submitted",
            "settling",
            "released",
        }:
            total = outstanding_total(records)
            return {
                "should_submit": False,
                "existing_job_id": existing.entry.get("job_id") or "",
                "ledger_key": current_key,
                "reservation_id": current_id,
                "actual_usd": _money_text(actual),
                "outstanding_usd": _money_text(total),
                "projected_usd": _money_text(actual + total),
                "budget_limit_usd": _money_text(BUDGET_LIMIT_USD),
                "safety_reserve_usd": _money_text(SAFETY_RESERVE_USD),
                "usable_admission_usd": _money_text(ADMISSION_LIMIT_USD),
            }
        if existing and existing.entry["state"] == "reserved":
            raise LedgerError(
                "this request already has a live reservation; recovery found no Batch "
                "job and the 15-minute safety grace has not elapsed"
            )

        outstanding = outstanding_total(records)
        requested = _money(maximum_total_usd, "requested maximum")
        projected = projected_total(actual, outstanding, requested)
        generation = int(existing.entry["generation"]) + 1 if existing else 1
        reservation = new_reservation(
            manifest=manifest,
            profile=profile,
            source_revision=source_revision,
            plan_digest=plan_digest,
            maximum_total_usd=requested,
            budget_actual_usd=actual,
            job_name=job_name,
            owner=owner,
            trigger=trigger,
            generation=generation,
            now=now,
        )
        assert_admission_lease_current(store, lease)
        store.put_record(
            current_key,
            reservation,
            expected_etag=existing.etag if existing else None,
        )
        assert_admission_lease_current(store, lease)
        return {
            "should_submit": True,
            "existing_job_id": "",
            "ledger_key": current_key,
            "reservation_id": current_id,
            "actual_usd": _money_text(actual),
            "outstanding_usd": _money_text(outstanding),
            "projected_usd": _money_text(projected),
            "budget_limit_usd": _money_text(BUDGET_LIMIT_USD),
            "safety_reserve_usd": _money_text(SAFETY_RESERVE_USD),
            "usable_admission_usd": _money_text(ADMISSION_LIMIT_USD),
        }


def finalize_reservation(
    store: AwsBrokerStore,
    *,
    key: str,
    owner: str,
    job_id: str | None,
    recovery_attempts: int = 6,
    recovery_delay_seconds: float = 5,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Convert a reservation to submitted, or retain it when outcome is ambiguous."""
    now = now or _utc_now()
    record = store.get_record(key)
    entry = dict(record.entry)
    if entry["state"] in {"submitted", "settling", "released"}:
        return {
            "job_id": entry.get("job_id") or "",
            "state": entry["state"],
        }
    if entry["state"] != "reserved":
        return {"job_id": "", "state": entry["state"]}
    if entry["owner"] != owner:
        raise LedgerError("only the reservation owner can finalize this reservation")

    if job_id:
        job = {"jobId": job_id, "status": "SUBMITTED"}
    else:
        job = None
        for attempt in range(max(1, recovery_attempts)):
            job = store.find_job(entry)
            if job is not None:
                break
            if attempt + 1 < recovery_attempts:
                time.sleep(max(0, recovery_delay_seconds))
    if job is not None:
        updated = _transition_for_job(entry, job, now)
    else:
        # A successful SubmitJob call can become visible after this workflow's
        # bounded recovery loop.  Keep the full amount reserved.  A later
        # workflow may expire it only after the original 15-minute grace, a
        # different owner proves the first workflow ended, and another
        # deterministic Batch lookup still finds no job.
        return {"job_id": "", "state": "reserved"}
    record = store.put_record(key, updated, expected_etag=record.etag)
    return {
        "job_id": record.entry.get("job_id") or "",
        "state": record.entry["state"],
    }


def _write_result(payload: Mapping[str, Any], path: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(rendered, end="")
    else:
        path.write_text(rendered, encoding="utf-8")


def _append_github_output(payload: Mapping[str, Any], path: Path | None) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        for key, value in payload.items():
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            else:
                rendered = str(value)
            if "\n" in rendered or "\r" in rendered:
                raise LedgerError(f"GitHub output {key} contains a newline")
            handle.write(f"{key}={rendered}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)

    reserve = actions.add_parser("reserve")
    reserve.add_argument("--bucket", required=True)
    reserve.add_argument("--queue", required=True)
    reserve.add_argument("--budget-name", required=True)
    reserve.add_argument("--manifest", required=True)
    reserve.add_argument("--profile", required=True)
    reserve.add_argument("--source-revision", required=True)
    reserve.add_argument("--plan-digest", required=True)
    reserve.add_argument("--maximum-total-usd", required=True)
    reserve.add_argument("--job-name", required=True)
    reserve.add_argument("--owner", required=True)
    reserve.add_argument("--trigger", required=True)
    reserve.add_argument("--output", type=Path)
    reserve.add_argument("--github-output", type=Path)

    finalize = actions.add_parser("finalize")
    finalize.add_argument("--bucket", required=True)
    finalize.add_argument("--queue", required=True)
    finalize.add_argument("--ledger-key", required=True)
    finalize.add_argument("--owner", required=True)
    finalize.add_argument("--job-id")
    finalize.add_argument("--output", type=Path)
    finalize.add_argument("--github-output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store = AwsBrokerStore(args.bucket, args.queue)
    if args.action == "reserve":
        payload = reserve_budget(
            store,
            budget_name=args.budget_name,
            manifest=args.manifest,
            profile=args.profile,
            source_revision=args.source_revision,
            plan_digest=args.plan_digest,
            maximum_total_usd=_money(args.maximum_total_usd, "requested maximum"),
            job_name=args.job_name,
            owner=args.owner,
            trigger=args.trigger,
        )
    else:
        payload = finalize_reservation(
            store,
            key=args.ledger_key,
            owner=args.owner,
            job_id=args.job_id,
        )
    _write_result(payload, args.output)
    _append_github_output(payload, args.github_output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LedgerError as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        raise SystemExit(2) from exc
