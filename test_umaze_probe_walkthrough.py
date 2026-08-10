import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.umaze_probe_walkthrough import (
    align_representation,
    build_motion_targets,
    episode_group_split,
    fit_probe,
    load_activation_cache,
    mask_slow_directions,
    regression_scores,
    residualize_against_position,
    save_activation_cache,
    spatial_holdout_split,
)


class UMazeProbeWalkthroughTests(unittest.TestCase):
    def setUp(self):
        position = np.array(
            [
                [[0, 0], [1, 0], [3, 0], [6, 0]],
                [[0, 1], [0, 2], [0, 4], [0, 7]],
            ],
            dtype=float,
        )
        velocity = np.concatenate([np.zeros((2, 1, 2)), np.diff(position, axis=1)], axis=1)
        self.states = np.concatenate([position, velocity], axis=-1)
        self.targets = build_motion_targets(self.states, frameskip=1)
        self.rep = np.concatenate([position, position**2], axis=-1)

    def test_temporal_alignment(self):
        feature, label, context = align_representation(
            self.rep, self.targets, "velocity", "delta"
        )
        self.assertEqual(feature.shape, (2, 3, 4))
        np.testing.assert_allclose(label, np.diff(self.states[..., :2], axis=1))
        np.testing.assert_allclose(
            context, 0.5 * (self.states[:, :-1, :2] + self.states[:, 1:, :2])
        )

    def test_second_difference_alignment(self):
        feature, label, _ = align_representation(
            self.rep, self.targets, "acceleration", "second_delta"
        )
        self.assertEqual(feature.shape, (2, 2, 4))
        self.assertEqual(label.shape, (2, 2, 2))

    def test_episode_split_has_no_episode_overlap(self):
        choices = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]
        train, test = episode_group_split(choices, test_fraction=0.34, seed=2)
        episodes = np.asarray(choices)[:, 0]
        self.assertFalse(set(episodes[train]) & set(episodes[test]))

    def test_spatial_holdout_has_buffer(self):
        position = np.column_stack([np.zeros(10), np.arange(10)])
        train, test, config = spatial_holdout_split(position, buffer_fraction=0.1)
        self.assertLess(position[train, 1].max(), config["boundary"])
        self.assertGreaterEqual(position[test, 1].min(), config["boundary"])

    def test_ridge_probe_recovers_linear_target(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(20, 3, 4))
        labels = features @ np.array([[1.0], [-2.0], [0.5], [3.0]])
        truth, prediction, _ = fit_probe(features, labels, np.arange(15), np.arange(15, 20), 1e-4)
        self.assertGreater(regression_scores(truth, prediction)["r2"], 0.999)

    def test_position_residual_removes_linear_shortcut(self):
        position = self.states[..., :2]
        labels = 2 * position[..., :1]
        residual = residualize_against_position(labels, position, np.array([0]), ridge=1e-8)
        self.assertLess(np.nanmax(np.abs(residual)), 1e-5)

    def test_activation_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            save_activation_cache(
                path,
                {"dino/0/cls": self.rep},
                self.states,
                np.zeros((2, 4, 2)),
                [(0, 0), (1, 0)],
                {"checkpoint": "demo"},
            )
            reps, states, _, choices, metadata = load_activation_cache(path)
            np.testing.assert_allclose(reps["dino/0/cls"], self.rep)
            np.testing.assert_allclose(states, self.states)
            np.testing.assert_array_equal(choices, [[0, 0], [1, 0]])
            self.assertEqual(metadata["checkpoint"], "demo")

    def test_direction_mask_uses_training_cutoff(self):
        labels = np.ones((2, 4, 2))
        magnitude = np.array([[0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5]])
        masked, cutoff = mask_slow_directions(labels, magnitude, np.array([0]), 0.25)
        self.assertAlmostEqual(cutoff, 0.75)
        self.assertTrue(np.isnan(masked[0, 0]).all())
        self.assertTrue(np.isnan(masked[1, 0]).all())
        self.assertTrue(np.isfinite(masked[1, 1]).all())


if __name__ == "__main__":
    unittest.main()
