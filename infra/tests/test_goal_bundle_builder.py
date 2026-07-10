from __future__ import annotations

import hashlib
import io
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from infra.experiments.build_goal_bundle import (
    assemble_planner_targets,
    build_bundle_from_archive,
    make_proxy_trajectory,
    normalize_and_group_actions,
    select_disjoint_proxy_segments,
    select_planner_segments,
    spatial_proxy_sample_indices,
    validation_episode_indices,
    write_bundle_atomic,
)


class GoalBundleBuilderTests(unittest.TestCase):
    @staticmethod
    def _tensor_bytes(value: torch.Tensor) -> bytes:
        payload = io.BytesIO()
        torch.save(value, payload)
        return payload.getvalue()

    def _write_tiny_point_maze_zip(self, path: Path) -> str:
        episode_count = 40
        positions = torch.tensor(
            [
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
                [2.0, 3.0],
                [1.0, 3.0],
                [1.0, 2.0],
                [1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        episode_states = torch.column_stack(
            [positions, torch.zeros(len(positions), 2, dtype=torch.float64)]
        )
        states = episode_states.unsqueeze(0).repeat(episode_count, 1, 1)
        actions = torch.linspace(
            -1.0, 1.0, episode_count * len(positions) * 2, dtype=torch.float64
        ).reshape(episode_count, len(positions), 2)
        lengths = torch.full((episode_count,), len(positions), dtype=torch.int64)
        prefix = "release/point_maze"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.writestr(f"{prefix}/states.pth", self._tensor_bytes(states))
            handle.writestr(f"{prefix}/actions.pth", self._tensor_bytes(actions))
            handle.writestr(
                f"{prefix}/seq_lengths.pth", self._tensor_bytes(lengths)
            )
            for episode_id in range(episode_count):
                if episode_id == 0:
                    # Episode zero is in the training split. A successful build
                    # proves the archive mode did not deserialize unselected data.
                    payload = b"intentionally unreadable unselected observation"
                else:
                    visuals = torch.zeros(
                        len(positions), 2, 3, 3, dtype=torch.uint8
                    )
                    visuals[:, :, :, 0] = episode_id
                    payload = self._tensor_bytes(visuals)
                handle.writestr(
                    f"{prefix}/obses/episode_{episode_id:03d}.pth", payload
                )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_validation_split_matches_seed_42_repository_split(self):
        self.assertEqual([16, 9], validation_episode_indices(20))

    def test_planner_segments_are_deterministic_valid_and_episode_distinct(self):
        lengths = [31, 50, 26, 25, 40, 80]
        validation = [0, 1, 2, 3, 4, 5]
        first = select_planner_segments(
            lengths, validation, count=4, horizon=25, selection_seed=123
        )
        second = select_planner_segments(
            lengths, validation, count=4, horizon=25, selection_seed=123
        )
        self.assertEqual(first, second)
        self.assertEqual(4, len({segment.episode_id for segment in first}))
        for segment in first:
            self.assertEqual(25, segment.horizon)
            self.assertGreaterEqual(segment.start, 0)
            self.assertLess(segment.goal_index, lengths[segment.episode_id])
        self.assertNotIn(3, {segment.episode_id for segment in first})

    def test_proxy_segments_are_deterministic_and_disjoint_from_planner(self):
        lengths = [7] * 40
        validation = validation_episode_indices(40)
        planner = select_planner_segments(
            lengths, validation, count=2, horizon=5, selection_seed=42
        )
        proxy = select_disjoint_proxy_segments(
            lengths,
            validation,
            planner,
            count=2,
            episode_length=7,
            selection_seed=43,
        )
        self.assertTrue(
            {item.episode_id for item in planner}.isdisjoint(
                {item.episode_id for item in proxy}
            )
        )

    def test_proxy_selection_skips_only_target_degenerate_fixed_segments(self):
        lengths = [7] * 40
        validation = validation_episode_indices(40)
        planner = select_planner_segments(
            lengths, validation, count=1, horizon=5, selection_seed=42
        )
        moving_positions = np.array(
            [
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
                [2.0, 3.0],
                [1.0, 3.0],
                [1.0, 2.0],
                [1.0, 1.0],
            ]
        )
        states = np.zeros((40, 7, 4), dtype=np.float64)
        states[:, :, :2] = moving_positions
        degenerate_id = next(index for index in validation if index != planner[0].episode_id)
        states[degenerate_id, :, :2] = np.array([3.0, 1.0])

        first = select_disjoint_proxy_segments(
            lengths,
            validation,
            planner,
            count=2,
            episode_length=7,
            selection_seed=43,
            states=states,
            max_samples=7,
        )
        second = select_disjoint_proxy_segments(
            lengths,
            validation,
            planner,
            count=2,
            episode_length=7,
            selection_seed=43,
            states=states,
            max_samples=7,
        )
        self.assertEqual(first, second)
        self.assertNotIn(degenerate_id, {item.episode_id for item in first})
        for segment in first:
            _, distances = spatial_proxy_sample_indices(
                states[segment.episode_id, segment.start : segment.goal_index + 1],
                max_samples=7,
            )
            self.assertGreaterEqual(len(set(distances)), 3)

    def test_normalized_actions_are_grouped_in_frameskip_order(self):
        actions = np.arange(10, dtype=np.float64).reshape(5, 2)
        grouped = normalize_and_group_actions(
            actions,
            np.array([1.0, 1.0]),
            np.array([2.0, 2.0]),
            frameskip=5,
        )
        self.assertEqual((1, 10), grouped.shape)
        self.assertEqual(np.float32, grouped.dtype)
        np.testing.assert_allclose(grouped[0], ((actions - 1.0) / 2.0).reshape(-1))

    def test_proxy_labels_are_grid_distance_not_remaining_time(self):
        # A valid path around the UMaze wall from lower-left to upper-left.
        positions = np.array(
            [
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
                [2.0, 3.0],
                [1.0, 3.0],
                [1.0, 2.0],
                [1.0, 1.0],
            ]
        )
        states = np.column_stack([positions, np.zeros((len(positions), 2))])
        indices, distances = spatial_proxy_sample_indices(states, max_samples=4)
        self.assertEqual([0, 2, 4, 6], indices)
        self.assertEqual([6, 4, 2, 0], distances)
        proxy = make_proxy_trajectory(
            np.zeros((len(states), 2, 3, 3), dtype=np.uint8),
            states.astype(np.float64),
            seed=1_000_000,
            max_samples=4,
        )
        self.assertEqual(np.float32, proxy["observations"]["proprio"].dtype)
        self.assertEqual([6, 4, 2, 0], proxy["shortest_path_steps"])

    def test_proxy_sampling_keeps_ties_and_backtracking(self):
        positions = np.array(
            [
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 1.0],  # backtrack to the same distance as frame zero
                [3.0, 3.0],
                [2.0, 3.0],
                [1.0, 3.0],
                [1.0, 2.0],
                [1.0, 1.0],
            ]
        )
        states = np.column_stack([positions, np.zeros((len(positions), 2))])
        indices, distances = spatial_proxy_sample_indices(states, max_samples=8)
        self.assertEqual(list(range(8)), indices)
        self.assertEqual(distances[0], distances[2])
        self.assertEqual(0, distances[-1])

    def test_planner_contract_has_raw_hwc_endpoints_and_grouped_actions(self):
        records = []
        for index in range(2):
            states = np.arange(24, dtype=np.float64).reshape(6, 4) + index
            records.append(
                {
                    "goal_id": f"goal-{index}",
                    "source": {"episode_id": index},
                    "observations": {
                        "visual": np.full((6, 2, 3, 3), index, dtype=np.uint8),
                        "proprio": states,
                    },
                    "states": states,
                    "actions": np.arange(10, dtype=np.float64).reshape(5, 2),
                }
            )
        bundle = assemble_planner_targets(
            records,
            action_mean=np.zeros(2),
            action_std=np.ones(2),
            frameskip=5,
            goal_horizon=5,
        )
        self.assertEqual((2, 1, 2, 3, 3), bundle["obs_0"]["visual"].shape)
        self.assertEqual((2, 1, 2, 3, 3), bundle["obs_g"]["visual"].shape)
        self.assertEqual((2, 1, 4), bundle["obs_0"]["proprio"].shape)
        self.assertEqual((2, 1, 10), bundle["gt_actions"].shape)
        self.assertEqual(["goal-0", "goal-1"], bundle["goal_ids"])

    def test_pickle_and_sha256_sidecar_are_published_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "umaze_fixed_50.pkl"
            digest = write_bundle_atomic({"goal_ids": ["one"]}, destination)
            self.assertEqual(
                hashlib.sha256(destination.read_bytes()).hexdigest(), digest
            )
            self.assertEqual(
                f"{digest}  {destination.name}\n",
                Path(f"{destination}.sha256").read_text(encoding="utf-8"),
            )
            with destination.open("rb") as handle:
                self.assertEqual({"goal_ids": ["one"]}, pickle.load(handle))
            self.assertEqual([], list(Path(temporary).glob("*.tmp")))

    def test_archive_mode_reads_only_disjoint_selected_members_without_mujoco(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "released.zip"
            archive_sha256 = self._write_tiny_point_maze_zip(archive)
            output = root / "goals.pkl"
            result = build_bundle_from_archive(
                archive,
                output,
                expected_dataset_sha256=archive_sha256,
                dataset_version_id="version-1",
                planner_count=2,
                proxy_count=2,
                goal_horizon=5,
                frameskip=5,
                proxy_episode_length=7,
                proxy_max_samples=7,
            )
            with output.open("rb") as handle:
                bundle = pickle.load(handle)

        self.assertEqual("archive_no_mujoco", result["mode"])
        self.assertEqual("archive_no_mujoco", bundle["metadata"]["mode"])
        self.assertEqual(archive_sha256, bundle["metadata"]["dataset"]["archive_sha256"])
        self.assertEqual("version-1", bundle["metadata"]["dataset"]["version_id"])
        planner_ids = set(bundle["metadata"]["planner"]["episode_ids"])
        proxy_ids = set(bundle["metadata"]["proxy"]["episode_ids"])
        self.assertTrue(planner_ids.isdisjoint(proxy_ids))
        self.assertEqual(
            4,
            len(
                bundle["metadata"]["dataset"][
                    "selected_observation_member_sha256"
                ]
            ),
        )
        self.assertEqual((2, 1, 2, 3, 3), bundle["obs_0"]["visual"].shape)
        self.assertEqual(np.uint8, bundle["obs_0"]["visual"].dtype)
        self.assertEqual(2, len(bundle["proxy_trajectories"]))
        self.assertTrue(
            all(
                item["source"]["temporal_sampling"]
                == "evenly_spaced_before_bfs_labelling"
                for item in bundle["proxy_trajectories"]
            )
        )

    def test_archive_mode_rejects_wrong_release_checksum_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "released.zip"
            self._write_tiny_point_maze_zip(archive)
            output = root / "goals.pkl"
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                build_bundle_from_archive(
                    archive,
                    output,
                    expected_dataset_sha256="0" * 64,
                    planner_count=2,
                    proxy_count=2,
                    goal_horizon=5,
                    frameskip=5,
                    proxy_episode_length=7,
                    proxy_max_samples=7,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
