from __future__ import annotations

import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

from infra.experiments.broker_ledger import (
    AdmissionLease,
    ADMISSION_LIMIT_USD,
    AwsBrokerStore,
    BUDGET_LIMIT_USD,
    LedgerError,
    LedgerRecord,
    SAFETY_RESERVE_USD,
    SETTLEMENT_HOLD,
    acquire_admission_lease,
    finalize_reservation,
    ledger_key,
    new_reservation,
    outstanding_total,
    projected_total,
    read_budget_actual,
    reconcile_records,
    release_admission_lease,
    reserve_budget,
)


NOW = datetime(2026, 7, 10, 16, 0, tzinfo=timezone.utc)


def reservation(
    *,
    manifest: str = "smoke",
    owner: str = "run:1",
    maximum: str = "1.5",
    created: datetime = NOW,
) -> dict:
    return new_reservation(
        manifest=manifest,
        profile="default",
        source_revision="a" * 40,
        plan_digest=(manifest.encode().hex() + "0" * 64)[:64],
        maximum_total_usd=Decimal(maximum),
        budget_actual_usd=Decimal("2"),
        job_name=f"ts-{manifest}-aaaaaaaaaaaa",
        owner=owner,
        trigger=f"tag:train-{manifest}-test",
        now=created,
    )


class FakeStore:
    def __init__(self, entries=()):
        self.records: dict[str, LedgerRecord] = {}
        self.jobs_by_name: dict[str, dict] = {}
        self.jobs_by_id: dict[str, dict] = {}
        self.writes: list[tuple[str, str | None, str]] = []
        self.lease: AdmissionLease | None = None
        self.lease_writes: list[tuple[str | None, str]] = []
        self.events: list[str] = []
        self.takeover_after_put = False
        for index, entry in enumerate(entries, start=1):
            key = ledger_key(entry["reservation_id"])
            self.records[key] = LedgerRecord(key, f'"etag-{index}"', deepcopy(entry))

    def list_records(self):
        self.events.append("list-records")
        return list(self.records.values())

    def get_record(self, key):
        return self.records[key]

    def put_record(self, key, entry, *, expected_etag):
        existing = self.records.get(key)
        if existing is None:
            if expected_etag is not None:
                raise AssertionError("new records require if-none-match")
        elif existing.etag != expected_etag:
            raise AssertionError("update did not use the current ETag")
        etag = f'"etag-write-{len(self.writes) + 1}"'
        record = LedgerRecord(key, etag, deepcopy(dict(entry)))
        self.records[key] = record
        self.writes.append((key, expected_etag, entry["state"]))
        self.events.append("put-record")
        if self.takeover_after_put and self.lease is not None:
            stolen = deepcopy(self.lease.entry)
            stolen.update(
                owner="other-run",
                lease_id="stolen-lease",
                generation=int(stolen["generation"]) + 1,
            )
            self.lease = AdmissionLease(
                self.lease.key, '"stolen-etag"', stolen
            )
        return record

    def get_admission_lease(self):
        return self.lease

    def put_admission_lease(
        self, entry, *, expected_etag, tolerate_conflict=False
    ):
        if self.lease is None:
            if expected_etag is not None:
                if tolerate_conflict:
                    return None
                raise AssertionError("missing lease cannot satisfy if-match")
        elif expected_etag != self.lease.etag:
            if tolerate_conflict:
                return None
            raise AssertionError("lease update did not use current ETag")
        etag = f'"lease-{len(self.lease_writes) + 1}"'
        self.lease = AdmissionLease(
            "broker-locks/budget-admission.json", etag, deepcopy(dict(entry))
        )
        self.lease_writes.append((expected_etag, entry["state"]))
        self.events.append(f"lease-{entry['state']}")
        return self.lease

    def find_job(self, entry):
        return deepcopy(self.jobs_by_name.get(entry["job_name"]))

    def describe_job(self, job_id):
        return deepcopy(self.jobs_by_id.get(job_id))

class BrokerLedgerTests(unittest.TestCase):
    @patch("infra.experiments.broker_ledger._aws_text", return_value="123456789012")
    @patch("infra.experiments.broker_ledger._aws_json")
    def test_budget_actual_is_required_and_must_be_usd(self, aws_json, _account):
        base = {
            "BudgetType": "COST",
            "TimeUnit": "MONTHLY",
            "Metrics": ["NetUnblendedCost"],
            "CostTypes": {
                "IncludeCredit": True,
                "IncludeRefund": True,
                "UseBlended": False,
                "UseAmortized": False,
            },
            "BudgetLimit": {"Amount": "20", "Unit": "USD"},
        }
        aws_json.return_value = {"Budget": base}
        with self.assertRaisesRegex(LedgerError, "no CalculatedSpend"):
            read_budget_actual("ts-net-out-of-pocket-20")

        aws_json.return_value = {
            "Budget": {
                **base,
                "CalculatedSpend": {"ActualSpend": {"Unit": "USD"}},
            }
        }
        with self.assertRaisesRegex(LedgerError, "no ActualSpend amount"):
            read_budget_actual("ts-net-out-of-pocket-20")

        aws_json.return_value = {
            "Budget": {
                **base,
                "CalculatedSpend": {
                    "ActualSpend": {"Amount": "1.25", "Unit": "EUR"}
                },
            }
        }
        with self.assertRaisesRegex(LedgerError, "not denominated in USD"):
            read_budget_actual("ts-net-out-of-pocket-20")

        aws_json.return_value = {
            "Budget": {
                **base,
                "CalculatedSpend": {
                    "ActualSpend": {"Amount": "1.25", "Unit": "USD"}
                },
            }
        }
        self.assertEqual(
            read_budget_actual("ts-net-out-of-pocket-20"), Decimal("1.25")
        )
        self.assertTrue(
            any(
                "--show-filter-expression" in call.args
                for call in aws_json.call_args_list
            )
        )

        filtered = deepcopy(base)
        filtered["FilterExpression"] = {
            "Dimensions": {"Key": "SERVICE", "Values": ["AmazonEC2"]}
        }
        filtered["CalculatedSpend"] = {
            "ActualSpend": {"Amount": "1.25", "Unit": "USD"}
        }
        aws_json.return_value = {"Budget": filtered}
        with self.assertRaisesRegex(LedgerError, "filter expression"):
            read_budget_actual("ts-net-out-of-pocket-20")

        omitted_tax = deepcopy(base)
        omitted_tax["CostTypes"]["IncludeTax"] = False
        omitted_tax["CalculatedSpend"] = {
            "ActualSpend": {"Amount": "1.25", "Unit": "USD"}
        }
        aws_json.return_value = {"Budget": omitted_tax}
        with self.assertRaisesRegex(LedgerError, "IncludeTax"):
            read_budget_actual("ts-net-out-of-pocket-20")

        drifted = deepcopy(base)
        drifted["Metrics"] = ["UnblendedCost"]
        drifted["CalculatedSpend"] = {
            "ActualSpend": {"Amount": "1.25", "Unit": "USD"}
        }
        aws_json.return_value = {"Budget": drifted}
        with self.assertRaisesRegex(LedgerError, "NetUnblendedCost"):
            read_budget_actual("ts-net-out-of-pocket-20")

    def test_admission_formula_counts_every_outstanding_state(self):
        entries = []
        for index, state in enumerate(("reserved", "submitted", "settling", "released")):
            entry = reservation(manifest=f"run{index}", maximum="2")
            entry["state"] = state
            if state in {"submitted", "settling", "released"}:
                entry["job_id"] = f"job-{index}"
            if state == "settling":
                entry["settle_after"] = (NOW + SETTLEMENT_HOLD).isoformat()
            entries.append(entry)
        records = FakeStore(entries).list_records()
        self.assertEqual(outstanding_total(records), Decimal("6.0"))
        self.assertEqual(
            ADMISSION_LIMIT_USD + SAFETY_RESERVE_USD,
            BUDGET_LIMIT_USD,
        )
        self.assertEqual(
            projected_total(Decimal("4"), Decimal("6"), Decimal("5")),
            ADMISSION_LIMIT_USD,
        )
        with self.assertRaisesRegex(
            LedgerError,
            r"actual \$5.*outstanding \$6.*usable \$15.*\$5.*safety reserve",
        ):
            projected_total(Decimal("5"), Decimal("6"), Decimal("5"))

    @patch(
        "infra.experiments.broker_ledger.read_budget_actual",
        return_value=Decimal("7"),
    )
    def test_reserve_is_conditional_and_refuses_projected_overspend(self, _budget):
        existing = reservation(manifest="existing", maximum="4")
        store = FakeStore([existing])
        with self.assertRaisesRegex(LedgerError, "exceeds"):
            reserve_budget(
                store,
                budget_name="ts-net-out-of-pocket-20",
                manifest="next",
                profile="default",
                source_revision="a" * 40,
                plan_digest="b" * 64,
                maximum_total_usd=Decimal("5"),
                job_name="ts-next-aaaaaaaaaaaa",
                owner="run:2",
                trigger="tag:train-run-next",
                now=NOW,
            )
        self.assertEqual(store.writes, [])

    @patch(
        "infra.experiments.broker_ledger.read_budget_actual",
        return_value=Decimal("3"),
    )
    def test_reserve_and_duplicate_request_are_idempotent(self, _budget):
        store = FakeStore()
        arguments = dict(
            budget_name="ts-net-out-of-pocket-20",
            manifest="smoke",
            profile="default",
            source_revision="a" * 40,
            plan_digest="b" * 64,
            maximum_total_usd=Decimal("1.5"),
            job_name="ts-smoke-aaaaaaaaaaaa",
            owner="run:2",
            trigger="tag:train-smoke-test",
            now=NOW,
        )
        admitted = reserve_budget(store, **arguments)
        self.assertTrue(admitted["should_submit"])
        self.assertEqual(admitted["projected_usd"], "4.5000")
        self.assertEqual(admitted["budget_limit_usd"], "20.0000")
        self.assertEqual(admitted["safety_reserve_usd"], "5.0000")
        self.assertEqual(admitted["usable_admission_usd"], "15.0000")
        self.assertEqual(store.writes[0][1], None)
        self.assertLess(store.events.index("lease-held"), store.events.index("list-records"))
        self.assertLess(store.events.index("put-record"), store.events.index("lease-released"))

        record = store.get_record(admitted["ledger_key"])
        submitted = deepcopy(record.entry)
        submitted.update(state="submitted", job_id="job-1")
        store.put_record(record.key, submitted, expected_etag=record.etag)
        duplicate = reserve_budget(store, **arguments)
        self.assertFalse(duplicate["should_submit"])
        self.assertEqual(duplicate["existing_job_id"], "job-1")

    def test_stranded_reservation_recovers_job_before_expiry(self):
        entry = reservation(owner="old-run", created=NOW - timedelta(hours=1))
        store = FakeStore([entry])
        job = {
            "jobId": "job-recovered",
            "jobName": entry["job_name"],
            "status": "RUNNING",
            "tags": {
                "PlanDigest": entry["plan_digest"],
                "SourceRevision": entry["source_revision"],
            },
        }
        store.jobs_by_name[entry["job_name"]] = job
        records = reconcile_records(store, store.list_records(), current_owner="new-run", now=NOW)
        self.assertEqual(records[0].entry["state"], "submitted")
        self.assertEqual(records[0].entry["job_id"], "job-recovered")

    @patch("infra.experiments.broker_ledger._aws_json")
    def test_recovery_uses_unbounded_server_side_job_name_filter(self, aws_json):
        aws_json.return_value = {"jobSummaryList": []}
        entry = reservation()
        store = AwsBrokerStore("artifact-bucket", "gpu-queue")
        self.assertIsNone(store.find_job(entry))
        arguments = list(aws_json.call_args.args)
        self.assertIn("--filters", arguments)
        self.assertIn(f"name=JOB_NAME,values={entry['job_name']}", arguments)
        self.assertNotIn("--job-status", arguments)
        self.assertNotIn("--max-items", arguments)

    def test_stranded_reservation_expires_only_after_safety_grace(self):
        stale = reservation(owner="old-run", created=NOW - timedelta(hours=1))
        fresh = reservation(
            manifest="fresh", owner="other-run", created=NOW - timedelta(minutes=5)
        )
        store = FakeStore([stale, fresh])
        records = reconcile_records(store, store.list_records(), current_owner="new-run", now=NOW)
        states = {record.entry["manifest"]: record.entry["state"] for record in records}
        self.assertEqual(states, {"smoke": "expired", "fresh": "reserved"})

    def test_terminal_jobs_hold_worst_case_until_budget_settlement_window(self):
        entry = reservation()
        entry.update(state="submitted", job_id="job-complete")
        stopped = NOW - timedelta(hours=1)
        job = {
            "jobId": "job-complete",
            "status": "SUCCEEDED",
            "stoppedAt": int(stopped.timestamp() * 1000),
        }
        store = FakeStore([entry])
        store.jobs_by_id["job-complete"] = job
        records = reconcile_records(store, store.list_records(), current_owner="new-run", now=NOW)
        self.assertEqual(records[0].entry["state"], "settling")
        self.assertEqual(outstanding_total(records), Decimal("1.5"))

        after_old_hold = NOW + timedelta(hours=72)
        records = reconcile_records(
            store, records, current_owner="new-run", now=after_old_hold
        )
        self.assertEqual(records[0].entry["state"], "settling")
        self.assertEqual(outstanding_total(records), Decimal("1.5"))

        later = stopped + SETTLEMENT_HOLD
        records = reconcile_records(store, records, current_owner="new-run", now=later)
        self.assertEqual(records[0].entry["state"], "released")
        self.assertEqual(outstanding_total(records), Decimal("0"))

    def test_finalize_converts_or_retains_ambiguous_reservation(self):
        store = FakeStore([reservation(owner="run:1")])
        key = next(iter(store.records))
        converted = finalize_reservation(
            store, key=key, owner="run:1", job_id="job-1", now=NOW
        )
        self.assertEqual(converted, {"job_id": "job-1", "state": "submitted"})

        second = reservation(manifest="second", owner="run:1")
        store = FakeStore([second])
        key = next(iter(store.records))
        retained = finalize_reservation(
            store,
            key=key,
            owner="run:1",
            job_id=None,
            recovery_attempts=1,
            recovery_delay_seconds=0,
            now=NOW,
        )
        self.assertEqual(retained, {"job_id": "", "state": "reserved"})
        self.assertEqual(store.get_record(key).entry["state"], "reserved")
        self.assertEqual(store.writes, [])

    def test_expired_lease_is_conditionally_taken_over_and_live_lease_blocks(self):
        store = FakeStore()
        first = acquire_admission_lease(
            store,
            owner="run:1",
            wait_seconds=0,
            lease_seconds=600,
            now_fn=lambda: NOW,
        )
        with self.assertRaisesRegex(LedgerError, "timed out waiting"):
            acquire_admission_lease(
                store,
                owner="run:2",
                wait_seconds=0,
                lease_seconds=600,
                now_fn=lambda: NOW + timedelta(minutes=5),
            )

        second = acquire_admission_lease(
            store,
            owner="run:2",
            wait_seconds=0,
            lease_seconds=600,
            now_fn=lambda: NOW + timedelta(minutes=11),
        )
        self.assertNotEqual(first.entry["lease_id"], second.entry["lease_id"])
        self.assertEqual(second.entry["generation"], 2)
        released = release_admission_lease(
            store, second, now=NOW + timedelta(minutes=12)
        )
        self.assertEqual(released.entry["state"], "released")

    @patch(
        "infra.experiments.broker_ledger.read_budget_actual",
        return_value=Decimal("0"),
    )
    def test_admission_allows_budgeted_work_to_join_the_batch_backlog(self, _budget):
        store = FakeStore()
        result = reserve_budget(
            store,
            budget_name="ts-net-out-of-pocket-20",
            manifest="smoke",
            profile="default",
            source_revision="a" * 40,
            plan_digest="b" * 64,
            maximum_total_usd=Decimal("1.5"),
            job_name="ts-smoke-aaaaaaaaaaaa",
            owner="run:2",
            trigger="tag:train-smoke-test",
            now=NOW,
        )
        self.assertTrue(result["should_submit"])

    @patch(
        "infra.experiments.broker_ledger.read_budget_actual",
        return_value=Decimal("0"),
    )
    def test_stale_holder_cannot_return_success_after_lease_takeover(self, _budget):
        store = FakeStore()
        store.takeover_after_put = True
        with self.assertRaisesRegex(LedgerError, "lease ownership changed"):
            reserve_budget(
                store,
                budget_name="ts-net-out-of-pocket-20",
                manifest="smoke",
                profile="default",
                source_revision="a" * 40,
                plan_digest="b" * 64,
                maximum_total_usd=Decimal("1.5"),
                job_name="ts-smoke-aaaaaaaaaaaa",
                owner="run:2",
                trigger="tag:train-smoke-test",
                now=NOW,
            )
        self.assertEqual(len(store.records), 1)
        self.assertEqual(store.lease.entry["owner"], "other-run")


if __name__ == "__main__":
    unittest.main()
