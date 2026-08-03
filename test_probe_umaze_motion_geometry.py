import unittest

import numpy as np

from scripts.probe_umaze_motion_geometry import (
    angular_mae_degrees,
    circular_harmonics,
    effective_rank,
    physical_targets,
    principal_angle_degrees,
    unit_vectors,
)


class MotionGeometryProbeTests(unittest.TestCase):
    def test_unit_vectors_keep_magnitude(self):
        direction, magnitude = unit_vectors(np.array([[3.0, 4.0], [0.0, 0.0]]))
        np.testing.assert_allclose(magnitude, [5.0, 0.0])
        np.testing.assert_allclose(direction[0], [0.6, 0.8])
        np.testing.assert_allclose(direction[1], [0.0, 0.0])

    def test_angle_error_wraps_at_pi(self):
        target = np.array([[np.cos(np.pi - 0.05), np.sin(np.pi - 0.05)]])
        prediction = np.array([[np.cos(-np.pi + 0.05), np.sin(-np.pi + 0.05)]])
        self.assertAlmostEqual(
            angular_mae_degrees(target, prediction, np.array([True])),
            np.degrees(0.1),
        )

    def test_physical_targets_use_state_velocity(self):
        states = np.array([[[0, 0, 1, 0], [1, 0, 3, 0], [2, 0, 3, 4]]], dtype=float)
        targets = physical_targets(states, frameskip=2)
        np.testing.assert_allclose(targets["speed"], [[1, 3, 5]])
        np.testing.assert_allclose(targets["acceleration_xy"][0, 1:], [[1, 0], [0, 2]])

    def test_harmonics_and_effective_rank(self):
        directions = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(circular_harmonics(directions, 3).shape, (2, 6))
        rank, d90, _ = effective_rank(np.eye(4))
        self.assertAlmostEqual(rank, 4.0)
        self.assertEqual(d90, 4)

    def test_principal_angle(self):
        self.assertAlmostEqual(principal_angle_degrees(np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])), 90.0)


if __name__ == "__main__":
    unittest.main()
