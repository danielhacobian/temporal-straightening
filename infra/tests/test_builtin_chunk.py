from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infra.experiments.builtin_chunk import (
    BUILTIN_FAMILIES,
    MAX_TOTAL_USD,
    compile_builtin_chunk,
    main,
    source_manifest_path,
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


class BuiltinChunkTests(unittest.TestCase):
    def selectors(self):
        for family in BUILTIN_FAMILIES:
            manifest = load_manifest(source_manifest_path(family))
            rollouts = manifest["dataset"]["rollouts"]
            if not isinstance(rollouts, list):
                rollouts = [rollouts]
            variants = [
                item["name"]
                for item in manifest["variants"]
                if item.get("enabled", True)
            ]
            for variant in variants:
                for seed_index in range(len(manifest["seed_sets"])):
                    for rollout_count in rollouts:
                        yield family, variant, seed_index, rollout_count

    def test_every_allowlisted_chunk_is_one_run_under_five_dollars(self):
        for family, variant, seed_index, rollouts in self.selectors():
            with self.subTest(
                family=family,
                variant=variant,
                seed_index=seed_index,
                rollouts=rollouts,
            ):
                compiled = compile_builtin_chunk(
                    family, variant, seed_index, rollouts
                )
                with patch.dict(os.environ, ENVIRONMENT, clear=False):
                    runs = expand_runs(
                        load_manifest_from_mapping(compiled, resolve=True)
                    )
                self.assertEqual(len(runs), 1)
                maximum = min(
                    runs[0].max_usd,
                    runs[0].max_hours * runs[0].max_hourly_usd,
                )
                self.assertLessEqual(maximum * 2, MAX_TOTAL_USD)

    def test_chunk_preserves_the_full_matrix_run_identity(self):
        for family, variant, seed_index, rollouts in self.selectors():
            with self.subTest(family=family, variant=variant, rollouts=rollouts):
                source = load_manifest(
                    source_manifest_path(family),
                    resolve=True,
                    environment=ENVIRONMENT,
                )
                source_seed = source["seed_sets"][seed_index]
                expected = next(
                    run
                    for run in expand_runs(source)
                    if run.variant == variant
                    and run.rollouts == rollouts
                    and run.data_seed == source_seed["data_seed"]
                    and run.train_seed == source_seed["train_seed"]
                    and run.planner_seed == source_seed["planner_seed"]
                )
                compiled = compile_builtin_chunk(
                    family, variant, seed_index, rollouts
                )
                actual = expand_runs(
                    load_manifest_from_mapping(compiled, resolve=True)
                )[0]
                self.assertEqual(actual.run_id, expected.run_id)
                self.assertEqual(actual.artifact_uri, expected.artifact_uri)

    def test_unknown_or_disabled_selectors_fail_closed(self):
        with self.assertRaisesRegex(ManifestError, "unknown built-in family"):
            compile_builtin_chunk("unknown", "projector_only", 0, 50)
        with self.assertRaisesRegex(ManifestError, "variant must be one of"):
            compile_builtin_chunk("screen", "r4_raw_acceleration_control", 0, 100)
        with self.assertRaisesRegex(ManifestError, "seed index"):
            compile_builtin_chunk("anchor", "projector_only", 3, 2000)
        with self.assertRaisesRegex(ManifestError, "rollouts must be one of"):
            compile_builtin_chunk("scale", "projector_only", 0, 4000)

    def test_plan_is_single_run_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, ENVIRONMENT, clear=False
        ):
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            arguments = [
                "plan",
                "--family",
                "screen",
                "--variant",
                "r2_normalized_acceleration",
                "--seed-index",
                "1",
                "--rollouts",
                "100",
                "--budget-usd",
                "5",
            ]
            self.assertEqual(main([*arguments, "--output", str(first)]), 0)
            self.assertEqual(main([*arguments, "--output", str(second)]), 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            plan = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(plan["array_size"], 1)
            self.assertLessEqual(plan["maximum_total_usd"], 5)


def load_manifest_from_mapping(mapping: dict, *, resolve: bool):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.yaml"
        import yaml

        path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
        return load_manifest(path, resolve=resolve, environment=ENVIRONMENT)


if __name__ == "__main__":
    unittest.main()
