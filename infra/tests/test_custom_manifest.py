from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from infra.experiments.custom_manifest import (
    APPROVED_VARIANTS,
    MAX_TOTAL_USD,
    compile_custom_manifest,
    load_custom_spec,
    main,
    materialize_custom_manifest,
)
from infra.experiments.manifest import ManifestError, expand_runs, load_manifest


ENVIRONMENT = {
    "TS_UMAZE_DATASET_S3_URI": "s3://unit-test/datasets/point_maze_umaze.zip",
    "TS_UMAZE_DATASET_SHA256": "a" * 64,
    "TS_UMAZE_DATASET_VERSION_ID": "dataset-version-1",
    "TS_UMAZE_GOALS_S3_URI": "s3://unit-test/goals/umaze_fixed_v1.pkl",
    "TS_UMAZE_GOALS_SHA256": "b" * 64,
    "TS_UMAZE_GOALS_VERSION_ID": "goals-version-1",
    "TS_ARTIFACT_PREFIX": "s3://unit-test/jobs/source-revision",
}


def valid_spec(name: str = "safe_probe") -> dict:
    return {
        "schema_version": 1,
        "name": name,
        "variant": "normalized_acceleration",
        "rollouts": 50,
        "epochs": 10,
        "evaluation": "proxy",
        "goal_count": 50,
        "seeds": [
            {"data_seed": 10, "train_seed": 20, "planner_seed": 100},
        ],
        "limits": {"max_hours": 4, "max_usd": 2},
    }


class CustomManifestTests(unittest.TestCase):
    def write_spec(self, directory: str, spec: dict) -> Path:
        path = Path(directory) / f"{spec['name']}.yaml"
        path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        return path

    def test_compiler_emits_only_fixed_inputs_and_allowlisted_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_spec(directory, valid_spec())
            loaded = load_custom_spec(source, expected_name="safe_probe")
            manifest = compile_custom_manifest(loaded, expected_name="safe_probe")
            output = Path(directory) / "compiled.yaml"
            materialize_custom_manifest(source, output, expected_name="safe_probe")
            resolved = load_manifest(output, resolve=True, environment=ENVIRONMENT)

        self.assertEqual("custom_safe_probe", manifest["name"])
        self.assertEqual("${TS_UMAZE_DATASET_S3_URI}", manifest["dataset"]["uri"])
        self.assertEqual("${TS_UMAZE_DATASET_VERSION_ID}", manifest["dataset"]["version_id"])
        self.assertEqual(["train", "proxy"], manifest["stages"])
        self.assertEqual(1, len(manifest["variants"]))
        self.assertEqual(
            APPROVED_VARIANTS["normalized_acceleration"],
            manifest["variants"][0]["hydra_overrides"],
        )
        runs = expand_runs(resolved)
        self.assertEqual(1, len(runs))
        self.assertEqual("dataset-version-1", runs[0].dataset_version_id)
        self.assertEqual("goals-version-1", runs[0].goal_set_version_id)

    def test_unknown_fields_and_free_form_controls_are_rejected(self):
        for field, value in (
            ("command", "bash -c curl bad.example"),
            ("hydra_overrides", {"training.epochs": 999}),
            ("attempts", 9),
            ("image", "untrusted:latest"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                spec = valid_spec()
                spec[field] = value
                source = self.write_spec(directory, spec)
                with self.assertRaisesRegex(ManifestError, "unsupported fields"):
                    load_custom_spec(source)

    def test_duplicate_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "safe_probe.yaml"
            source.write_text(
                yaml.safe_dump(valid_spec(), sort_keys=False)
                + "variant: projector_only\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "duplicate key 'variant'"):
                load_custom_spec(source)

    def test_variant_and_tag_slug_are_allowlisted(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = valid_spec()
            spec["variant"] = "python -m arbitrary.module"
            source = self.write_spec(directory, spec)
            with self.assertRaisesRegex(ManifestError, "variant must be one of"):
                load_custom_spec(source)

        with tempfile.TemporaryDirectory() as directory:
            source = self.write_spec(directory, valid_spec())
            with self.assertRaisesRegex(ManifestError, "tag slug"):
                load_custom_spec(source, expected_name="different")

    def test_filename_must_match_name_and_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "different.yaml"
            source.write_text(yaml.safe_dump(valid_spec()), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "match the YAML filename"):
                load_custom_spec(source)

        with tempfile.TemporaryDirectory() as directory:
            real = self.write_spec(directory, valid_spec())
            link = Path(directory) / "linked.yaml"
            link.symlink_to(real)
            with self.assertRaisesRegex(ManifestError, "symbolic link"):
                load_custom_spec(link)

    def test_seed_and_total_retry_cost_limits_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = valid_spec()
            spec["seeds"] = [
                {"data_seed": i, "train_seed": i, "planner_seed": i}
                for i in range(4)
            ]
            source = self.write_spec(directory, spec)
            with self.assertRaisesRegex(ManifestError, "1 to 3"):
                load_custom_spec(source)

        with tempfile.TemporaryDirectory() as directory:
            spec = valid_spec()
            spec["seeds"] = [
                {"data_seed": i, "train_seed": i, "planner_seed": i}
                for i in range(3)
            ]
            spec["limits"] = {"max_hours": 12, "max_usd": 2.5}
            source = self.write_spec(directory, spec)
            with self.assertRaisesRegex(
                ManifestError, f"exceeds \\${MAX_TOTAL_USD:.2f}"
            ):
                load_custom_spec(source)

    def test_compilation_and_plan_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_spec(directory, valid_spec())
            first = Path(directory) / "first.yaml"
            second = Path(directory) / "second.yaml"
            materialize_custom_manifest(source, first)
            materialize_custom_manifest(source, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            plan_path = Path(directory) / "plan.json"
            with patch.dict(os.environ, ENVIRONMENT, clear=False):
                self.assertEqual(
                    0,
                    main(
                        [
                            "plan",
                            str(source),
                            "--expected-name",
                            "safe_probe",
                            "--budget-usd",
                            "5",
                            "--output",
                            str(plan_path),
                        ]
                    ),
                )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(1, plan["array_size"])
            self.assertEqual(3.2, plan["maximum_total_usd"])
            self.assertEqual(str(source), plan["custom_spec"])

    def test_example_request_validates(self):
        root = Path(__file__).resolve().parents[2]
        example = (
            root
            / "infra"
            / "experiments"
            / "manifests"
            / "custom"
            / "normacc_probe.yaml"
        )
        loaded = load_custom_spec(example)
        self.assertEqual("normacc_probe", loaded["name"])

    def test_broker_uses_approved_entrypoint_and_pre_submit_lock(self):
        root = Path(__file__).resolve().parents[2]
        wrapper = (root / ".github" / "workflows" / "aws-submit.yml").read_text(
            encoding="utf-8"
        )
        broker = (
            root / ".github" / "workflows" / "aws-paid-broker.yml"
        ).read_text(encoding="utf-8")
        ledger = (root / "infra" / "experiments" / "broker_ledger.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"train-run-*"', wrapper)
        self.assertIn(
            "^train-run-([a-z0-9][a-z0-9_]{0,31})-",
            broker,
        )
        self.assertIn('"infra.experiments.custom_manifest", "run"', broker)
        self.assertNotIn("bash -c", broker)
        self.assertIn("immutable tag for this source revision", broker)
        self.assertIn('resources.get("GPU") != "1"', broker)
        self.assertIn('get("attempts") != 2', broker)
        self.assertLess(
            broker.index("Validate the declarative custom request before AWS authentication"),
            broker.index("Assume the pinned experiment broker role"),
        )
        self.assertLess(
            broker.index("--if-none-match '*'"),
            broker.index('job_id="$(aws batch submit-job'),
        )
        self.assertLess(
            broker.index("Reserve aggregate budget with the S3 admission lease"),
            broker.index('job_id="$(aws batch submit-job'),
        )
        for status in ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"):
            self.assertIn(status, ledger)


if __name__ == "__main__":
    unittest.main()
