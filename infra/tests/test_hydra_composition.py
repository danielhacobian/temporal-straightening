"""Composition tests for the exact Hydra commands emitted by the AWS runner."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from infra.experiments.manifest import expand_runs, load_manifest
from infra.experiments.runner import build_stage_commands


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "infra" / "experiments" / "manifests" / "finalists.yaml"
ENVIRONMENT = {
    "TS_UMAZE_DATASET_S3_URI": "s3://unit-test/datasets/point_maze.zip",
    "TS_UMAZE_DATASET_SHA256": "a" * 64,
    "TS_UMAZE_DATASET_VERSION_ID": "dataset-version-1",
    "TS_UMAZE_GOALS_S3_URI": "s3://unit-test/goals/fixed.pkl",
    "TS_UMAZE_GOALS_SHA256": "b" * 64,
    "TS_UMAZE_GOALS_VERSION_ID": "goals-version-1",
    "TS_ARTIFACT_PREFIX": "s3://unit-test/artifacts",
}
HYDRA_AVAILABLE = importlib.util.find_spec("hydra") is not None


@unittest.skipUnless(
    HYDRA_AVAILABLE,
    "Hydra is installed in the container and infrastructure CI environments",
)
class HydraCompositionTests(unittest.TestCase):
    def test_fixed_goal_plan_command_composes_with_repo_config(self) -> None:
        # Register the same output-path resolver imported by plan.py without
        # importing the planner's heavyweight model/environment dependencies.
        import custom_resolvers  # noqa: F401
        from hydra import compose, initialize_config_dir

        manifest = load_manifest(MANIFEST, resolve=True, environment=ENVIRONMENT)
        run = expand_runs(manifest)[0]
        command = next(
            item["argv"] for item in build_stage_commands(run) if item["stage"] == "plan"
        )
        config_flag = command.index("--config-name")
        config_name = command[config_flag + 1]
        overrides = command[config_flag + 2 :]

        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
            config = compose(config_name=config_name, overrides=overrides)

        self.assertEqual("file", config.goal_source)
        self.assertEqual(
            "/scratch/temporal-straightening/"
            f"{run.run_id}/inputs/fixed_goals.pkl",
            config.goal_file_path,
        )
        self.assertEqual(run.goal_count, config.n_evals)
        self.assertEqual(run.planner_seed, config.seed)


if __name__ == "__main__":
    unittest.main()
