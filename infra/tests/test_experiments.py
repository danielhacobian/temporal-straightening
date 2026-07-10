from __future__ import annotations

import copy
import io
import json
import os
import pickle
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch

from infra.experiments.manifest import (
    ManifestError,
    allowed_runtime_seconds,
    expand_runs,
    load_manifest,
    plan_digest,
    resolve_environment,
    validate_manifest,
)
from infra.experiments.paired_bootstrap import paired_bootstrap
from infra.experiments.prepare_subset import (
    canonical_training_indices,
    selected_indices,
)
from infra.experiments.prepare_goals import prepare as prepare_goals
from infra.experiments.proxy_signal import (
    MAZES,
    _find_checkpoint,
    shortest_grid_steps,
    spearman,
    summarize_group_rhos,
)
from infra.experiments.runner import (
    _extract_dataset,
    _restore,
    _s3_download,
    build_plan,
    build_stage_commands,
    execute_run,
    paths_for,
    verify_submission_identity,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "infra" / "experiments" / "manifests"
ENVIRONMENT = {
    "TS_UMAZE_DATASET_S3_URI": "s3://unit-test/datasets/point_maze.zip",
    "TS_UMAZE_DATASET_SHA256": "a" * 64,
    "TS_UMAZE_DATASET_VERSION_ID": "dataset-version-1",
    "TS_UMAZE_GOALS_S3_URI": "s3://unit-test/goals/fixed.pkl",
    "TS_UMAZE_GOALS_SHA256": "b" * 64,
    "TS_UMAZE_GOALS_VERSION_ID": "goals-version-1",
    "TS_ARTIFACT_PREFIX": "s3://unit-test/artifacts",
}


def resolved(name: str):
    return load_manifest(MANIFESTS / name, resolve=True, environment=ENVIRONMENT)


class ManifestTests(unittest.TestCase):
    def test_all_manifests_validate_statically_and_resolved(self):
        paths = sorted(MANIFESTS.glob("*.yaml"))
        self.assertEqual(5, len(paths))
        for path in paths:
            with self.subTest(path=path.name):
                validate_manifest(load_manifest(path))
                validate_manifest(
                    load_manifest(path, resolve=True, environment=ENVIRONMENT),
                    resolved=True,
                )

    def test_missing_environment_variable_is_explicit(self):
        manifest = load_manifest(MANIFESTS / "smoke.yaml")
        with self.assertRaisesRegex(ManifestError, "TS_UMAZE_DATASET_S3_URI"):
            resolve_environment(manifest, {})

    def test_anchor_expands_to_nine_or_six_runs(self):
        manifest = resolved("umaze_exact_anchor.yaml")
        full = expand_runs(manifest)
        minimal = expand_runs(manifest, "minimal")
        self.assertEqual(9, len(full))
        self.assertEqual(6, len(minimal))
        self.assertEqual({"projector_only", "projector_curvature"}, {r.variant for r in minimal})
        self.assertEqual({42}, {r.data_seed for r in minimal})
        self.assertEqual({0, 1, 2}, {r.train_seed for r in minimal})
        self.assertEqual({100}, {r.planner_seed for r in minimal})
        self.assertEqual(
            {1},
            {dict(run.train_overrides)["training.decoder_start_epoch"] for run in minimal},
        )

    def test_screen_skips_unimplemented_r4(self):
        manifest = resolved("screening_funnel.yaml")
        disabled = [v for v in manifest["variants"] if not v.get("enabled", True)]
        self.assertEqual(["r4_raw_acceleration_control"], [v["name"] for v in disabled])
        runs = expand_runs(manifest)
        self.assertEqual(18, len(runs))
        self.assertNotIn("r4_raw_acceleration_control", {run.variant for run in runs})

    def test_scaling_is_six_runs_at_requested_sizes(self):
        runs = expand_runs(resolved("scaling_trend.yaml"))
        self.assertEqual(6, len(runs))
        self.assertEqual({50, 200, 800}, {run.rollouts for run in runs})

    def test_duplicate_seed_triples_are_rejected(self):
        manifest = resolved("smoke.yaml")
        manifest["seed_sets"].append(copy.deepcopy(manifest["seed_sets"][0]))
        with self.assertRaisesRegex(ManifestError, "duplicate seed set"):
            validate_manifest(manifest, resolved=True)

    def test_plan_and_ids_are_deterministic(self):
        runs1 = expand_runs(resolved("finalists.yaml"))
        runs2 = expand_runs(resolved("finalists.yaml"))
        self.assertEqual([run.run_id for run in runs1], [run.run_id for run in runs2])
        self.assertEqual(plan_digest(runs1), plan_digest(runs2))
        self.assertEqual(
            build_stage_commands(runs1[0]), build_stage_commands(runs2[0])
        )


class RunnerTests(unittest.TestCase):
    @staticmethod
    def _write_dataset_zip(
        path: Path,
        episode_ids,
        *,
        omitted_core=(),
        extra_members=(),
    ):
        prefix = "release/point_maze"
        episode_ids = list(episode_ids)
        total = max(episode_ids, default=-1) + 1
        tensors = {
            "states.pth": torch.zeros(total, 1, 4),
            "actions.pth": torch.zeros(total, 1, 2),
            "seq_lengths.pth": torch.ones(total, dtype=torch.int64),
        }
        with zipfile.ZipFile(path, "w") as handle:
            for name in ("states.pth", "actions.pth", "seq_lengths.pth"):
                if name not in omitted_core:
                    payload = io.BytesIO()
                    torch.save(tensors[name], payload)
                    handle.writestr(f"{prefix}/{name}", payload.getvalue())
            handle.writestr(f"{prefix}/dataset_metadata.json", "{}")
            for episode_id in episode_ids:
                handle.writestr(
                    f"{prefix}/obses/episode_{episode_id:03d}.pth",
                    f"observation-{episode_id}",
                )
            for name, value in extra_members:
                handle.writestr(name, value)

    def test_commands_use_fixed_goals_separate_seeds_and_no_modal(self):
        run = expand_runs(resolved("finalists.yaml"))[0]
        commands = build_stage_commands(run)
        rendered = "\n".join(" ".join(item["argv"]) for item in commands)
        self.assertIn(f"training.seed={run.train_seed}", rendered)
        self.assertIn(f"+experiment_data_seed={run.data_seed}", rendered)
        self.assertIn(f"seed={run.planner_seed}", rendered)
        self.assertIn("goal_source=file", rendered)
        self.assertIn("/scratch/", rendered)
        self.assertNotIn("modal", rendered.lower())

    def test_hourly_and_cost_limits_are_enforced(self):
        run = expand_runs(resolved("smoke.yaml"))[0]
        self.assertEqual(3600, allowed_runtime_seconds(run, 0.40))
        with self.assertRaisesRegex(ManifestError, "exceeds"):
            allowed_runtime_seconds(run, 0.401)
        anchor = expand_runs(resolved("umaze_exact_anchor.yaml"), "minimal")[0]
        self.assertEqual(24 * 3600, allowed_runtime_seconds(anchor, 0.40))

    def test_dry_run_can_use_temporary_scratch(self):
        run = expand_runs(resolved("smoke.yaml"))[0]
        with tempfile.TemporaryDirectory() as directory:
            result = execute_run(run, scratch_root=Path(directory), dry_run=True)
        self.assertEqual(run.run_id, result["run"]["run_id"])
        self.assertEqual(
            ["train", "proxy", "plan"],
            [item["stage"] for item in result["commands"]],
        )

    def test_restore_syncs_artifacts_toward_local_run_root(self):
        run = expand_runs(resolved("smoke.yaml"))[0]
        with tempfile.TemporaryDirectory() as directory, patch(
            "infra.experiments.runner.subprocess.run"
        ) as mocked:
            local_run = Path(directory) / "local-run"
            _restore(run, local_run, timeout_seconds=30)
        argv = mocked.call_args.args[0]
        self.assertEqual(["aws", "s3", "sync"], argv[:3])
        self.assertEqual(run.artifact_uri, argv[3])
        self.assertEqual(str(local_run), argv[4])

    def test_canonical_download_pins_the_s3_object_version(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "infra.experiments.runner.subprocess.run"
        ) as mocked:
            destination = Path(directory) / "dataset.zip"
            _s3_download(
                "s3://unit-test/datasets/point_maze.zip",
                destination,
                30,
                version_id="dataset-version-1",
            )
        argv = mocked.call_args.args[0]
        self.assertEqual(["aws", "s3api", "get-object"], argv[:3])
        self.assertEqual("dataset-version-1", argv[argv.index("--version-id") + 1])
        self.assertEqual(str(destination), argv[-1])

    def test_dataset_archive_rejects_path_traversal(self):
        run = expand_runs(resolved("smoke.yaml"))[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escaped.txt", "no")
            destination = root / "extract"
            destination.mkdir()
            sentinel = destination / "previous-materialization"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                _extract_dataset(run, archive, destination)
            self.assertFalse((root / "escaped.txt").exists())
            self.assertEqual("preserve", sentinel.read_text(encoding="utf-8"))

    def test_dataset_archive_validates_irrelevant_symlinks_before_extraction(self):
        run = expand_runs(resolved("smoke.yaml"))[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe-link.zip"
            self._write_dataset_zip(archive, range(12))
            link = zipfile.ZipInfo("unrelated/late-symbolic-link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "a") as handle:
                handle.writestr(link, "/etc/passwd")
            destination = root / "extract"
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                _extract_dataset(run, archive, destination)
            self.assertFalse(destination.exists())

    def test_subset_zip_extracts_only_deterministically_selected_observations(self):
        run = expand_runs(resolved("smoke.yaml"))[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "point_maze.zip"
            self._write_dataset_zip(archive, range(20))
            destination = root / "extract"
            _extract_dataset(run, archive, destination)

            leaf = destination / "point_maze"
            observed = {
                path.name for path in (leaf / "obses").glob("episode_*.pth")
            }
            expected = {
                f"episode_{index:03d}.pth"
                for index in selected_indices(20, run.rollouts, run.data_seed)
            }
            self.assertEqual(expected, observed)
            self.assertEqual(run.rollouts, len(observed))
            for name in (
                "states.pth",
                "actions.pth",
                "seq_lengths.pth",
                "dataset_metadata.json",
            ):
                self.assertTrue((leaf / name).is_file(), name)
            self.assertFalse((destination / "release").exists())

    def test_exact_anchor_zip_extracts_all_two_thousand_observations(self):
        run = expand_runs(resolved("umaze_exact_anchor.yaml"), "minimal")[0]
        self.assertEqual(2000, run.rollouts)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "point_maze.zip"
            self._write_dataset_zip(archive, range(2000))
            destination = root / "extract"
            _extract_dataset(run, archive, destination)
            observed = list(
                (destination / "point_maze" / "obses").glob("episode_*.pth")
            )
            self.assertEqual(2000, len(observed))
            self.assertTrue(
                (destination / "point_maze" / "obses" / "episode_1999.pth").is_file()
            )

    def test_dataset_zip_layout_errors_fail_before_materialization(self):
        run = expand_runs(resolved("smoke.yaml"))[0]
        cases = (
            ("missing-core", range(12), ("actions.pth",), (), "exactly one"),
            ("noncontiguous", (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11), (), (), "contiguous"),
            (
                "malformed-observation",
                range(12),
                (),
                (("release/point_maze/obses/not-an-episode.pth", "bad"),),
                "malformed observation file",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, episodes, omitted, extras, error in cases:
                with self.subTest(name=name):
                    archive = root / f"{name}.zip"
                    self._write_dataset_zip(
                        archive,
                        episodes,
                        omitted_core=omitted,
                        extra_members=extras,
                    )
                    destination = root / f"extract-{name}"
                    with self.assertRaisesRegex((ValueError, FileNotFoundError), error):
                        _extract_dataset(run, archive, destination)
                    self.assertFalse(destination.exists())

    def test_plan_budget_is_visible(self):
        plan = build_plan(
            MANIFESTS / "umaze_exact_anchor.yaml",
            profile="minimal",
            environment=ENVIRONMENT,
        )
        self.assertEqual(6, plan["array_size"])
        self.assertEqual(57.6, plan["maximum_one_attempt_usd"])
        self.assertEqual(115.2, plan["maximum_total_usd"])
        self.assertEqual(2, plan["batch_max_attempts"])

    def test_batch_submission_requires_matching_plan_and_source(self):
        runs = expand_runs(resolved("smoke.yaml"))
        digest = plan_digest(runs)
        identity = verify_submission_identity(
            runs,
            {
                "AWS_BATCH_JOB_ID": "job-1",
                "EXPECTED_PLAN_DIGEST": digest,
                "EXPECTED_SOURCE_REVISION": "abc123",
                "SOURCE_REVISION": "abc123",
            },
        )
        self.assertEqual({"plan_digest": digest, "source_revision": "abc123"}, identity)
        with self.assertRaisesRegex(ManifestError, "does not match"):
            verify_submission_identity(
                runs,
                {
                    "AWS_BATCH_JOB_ID": "job-1",
                    "EXPECTED_PLAN_DIGEST": "0" * 64,
                    "EXPECTED_SOURCE_REVISION": "abc123",
                    "SOURCE_REVISION": "abc123",
                },
            )

    def test_completed_retry_is_not_overwritten_or_reuploaded(self):
        run = expand_runs(resolved("smoke.yaml"))[0]
        expected_status = {"run_id": run.run_id, "state": "complete"}
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)

            def restore_complete(_run, run_root, timeout_seconds):
                del timeout_seconds
                run_root.mkdir(parents=True, exist_ok=True)
                (run_root / "run_metadata.json").write_text(
                    json.dumps(
                        {
                            "submission_identity": {
                                "plan_digest": "plan-1",
                                "source_revision": "source-1",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                (run_root / "status.json").write_text(
                    json.dumps(expected_status), encoding="utf-8"
                )

            with patch("infra.experiments.runner.DEFAULT_SCRATCH", scratch), patch(
                "infra.experiments.runner._restore", side_effect=restore_complete
            ), patch("infra.experiments.runner._upload") as upload, patch.dict(
                os.environ,
                {
                    "VERIFIED_PLAN_DIGEST": "plan-1",
                    "VERIFIED_SOURCE_REVISION": "source-1",
                },
                clear=False,
            ):
                result = execute_run(run, scratch_root=scratch, repo_root=ROOT)
            self.assertEqual(expected_status, result)
            upload.assert_not_called()
            self.assertEqual(
                expected_status,
                json.loads((paths_for(run, scratch)["run"] / "status.json").read_text()),
            )


class HelperTests(unittest.TestCase):
    def test_latest_checkpoint_ignores_alias_when_numbered_epochs_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            checkpoints = model_dir / "checkpoints"
            checkpoints.mkdir()
            for name in ("model_2.pth", "model_10.pth", "model_latest.pth"):
                (checkpoints / name).touch()

            checkpoint, epoch = _find_checkpoint(model_dir, "latest")

        self.assertEqual("model_10.pth", checkpoint.name)
        self.assertEqual(10, epoch)

    def test_latest_checkpoint_alias_is_supported_without_numbered_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            checkpoints = model_dir / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "model_latest.pth").touch()

            checkpoint, epoch = _find_checkpoint(model_dir, "latest")

        self.assertEqual("model_latest.pth", checkpoint.name)
        self.assertEqual("latest", epoch)

    def test_seeded_subsets_are_stable_and_distinct(self):
        self.assertEqual(selected_indices(20, 5, 7), selected_indices(20, 5, 7))
        self.assertNotEqual(selected_indices(20, 5, 7), selected_indices(20, 5, 8))
        self.assertEqual(list(range(20)), selected_indices(20, 20, 7))

    def test_partial_subsets_never_sample_the_fixed_goal_validation_pool(self):
        total = 2000
        training_pool = set(canonical_training_indices(total))
        held_out_pool = set(range(total)) - training_pool
        self.assertEqual(1800, len(training_pool))
        self.assertEqual(200, len(held_out_pool))
        for rollouts in (50, 100, 200, 800):
            for data_seed in (0, 1, 2, 42):
                selected = set(selected_indices(total, rollouts, data_seed))
                self.assertTrue(selected <= training_pool)
                self.assertTrue(selected.isdisjoint(held_out_pool))

    def test_spearman_handles_ties(self):
        self.assertAlmostEqual(1.0, spearman([1, 2, 2, 4], [10, 20, 20, 40]))
        self.assertAlmostEqual(-1.0, spearman([1, 2, 3], [30, 20, 10]))

    def test_correlations_are_computed_within_groups_before_summary(self):
        summary = summarize_group_rhos(
            [
                {
                    "latent_distances": [1, 2, 3],
                    "shortest_path_steps": [1, 2, 3],
                },
                {
                    "latent_distances": [1, 2, 3],
                    "shortest_path_steps": [3, 2, 1],
                },
            ]
        )
        self.assertEqual(2, summary["groups"])
        self.assertAlmostEqual(0.0, summary["spearman_rho_mean"])
        self.assertAlmostEqual(0.5, summary["positive_group_fraction"])

    def test_goal_bundle_requires_and_trims_proxy_trajectories(self):
        bundle = {
            "obs_0": {"visual": [0, 1, 2], "proprio": [0, 1, 2]},
            "obs_g": {"visual": [0, 1, 2], "proprio": [0, 1, 2]},
            "state_0": [0, 1, 2],
            "state_g": [0, 1, 2],
            "gt_actions": [0, 1, 2],
            "goal_H": 25,
            "goal_ids": ["g0", "g1", "g2"],
            "proxy_trajectories": [{"id": 0}, {"id": 1}, {"id": 2}],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pkl"
            destination = Path(directory) / "out.pkl"
            with source.open("wb") as handle:
                pickle.dump(bundle, handle)
            result = prepare_goals(source, destination, 2)
            with destination.open("rb") as handle:
                trimmed = pickle.load(handle)
        self.assertEqual({"planner_goals": 2, "proxy_trajectories": 2}, result)
        self.assertEqual(2, len(trimmed["state_0"]))
        self.assertEqual(["g0", "g1"], trimmed["goal_ids"])
        self.assertEqual(2, len(trimmed["proxy_trajectories"]))

    def test_umaze_shortest_path_routes_around_wall(self):
        maze = MAZES["umaze"]
        self.assertEqual(6, shortest_grid_steps((1, 1), (3, 1), maze))

    def test_paired_bootstrap_resamples_crossed_seed_and_shared_goal_axes(self):
        baseline = {
            ("0", "a"): False,
            ("0", "b"): True,
            ("1", "a"): False,
            ("1", "b"): False,
        }
        candidate = {
            ("0", "a"): True,
            ("0", "b"): True,
            ("1", "a"): False,
            ("1", "b"): True,
        }
        result = paired_bootstrap(baseline, candidate, resamples=200, seed=7)
        self.assertEqual(["training_seed", "goal_id"], result["pairing_unit"])
        self.assertEqual(
            ["training_seed", "shared_goal_id"], result["bootstrap_axes"]
        )
        self.assertEqual(2, result["training_seeds"])
        self.assertEqual(2, result["paired_goals"])
        self.assertEqual(4, result["paired_cells"])
        self.assertAlmostEqual(0.5, result["candidate_minus_baseline"])
        self.assertLessEqual(result["ci_low"], result["candidate_minus_baseline"])
        self.assertGreaterEqual(result["ci_high"], result["candidate_minus_baseline"])

    def test_paired_bootstrap_rejects_an_incomplete_crossed_design(self):
        baseline = {("0", "a"): False, ("1", "a"): False, ("1", "b"): True}
        candidate = {("0", "a"): True, ("1", "a"): False, ("1", "b"): True}
        with self.assertRaisesRegex(ValueError, "complete crossed"):
            paired_bootstrap(baseline, candidate, resamples=100)


if __name__ == "__main__":
    unittest.main()
