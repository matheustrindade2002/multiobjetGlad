import unittest

import numpy as np

from selection import (
    get_pareto_frontier,
    get_robust_expected_confidence,
    get_typicality_and_delta,
)


class TypicalityTests(unittest.TestCase):
    def test_excludes_self_from_nearest_neighbors(self):
        embeddings = np.array([[0.0], [1.0], [3.0]])

        tau, delta = get_typicality_and_delta(embeddings, k_knn=1)

        np.testing.assert_allclose(tau, [1.0, 1.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(delta, 1.0 - tau)

    def test_single_point_is_central_by_convention(self):
        tau, delta = get_typicality_and_delta(np.array([[2.0, 4.0]]), 15)

        np.testing.assert_array_equal(tau, [1.0])
        np.testing.assert_array_equal(delta, [0.0])

    def test_rejects_invalid_k(self):
        with self.assertRaises(ValueError):
            get_typicality_and_delta(np.array([[0.0], [1.0]]), 0)


class BootstrapTests(unittest.TestCase):
    def test_is_reproducible_with_seeded_generator(self):
        tau = np.linspace(0.0, 1.0, 12)
        confidence = np.linspace(0.55, 0.95, 12)

        first = get_robust_expected_confidence(
            tau, confidence, 20, rng=np.random.default_rng(42)
        )
        second = get_robust_expected_confidence(
            tau, confidence, 20, rng=np.random.default_rng(42)
        )

        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])

    def test_rejects_zero_bootstraps(self):
        with self.assertRaises(ValueError):
            get_robust_expected_confidence(
                np.array([0.0]), np.array([0.8]), 0
            )


class ParetoTests(unittest.TestCase):
    def test_returns_only_non_dominated_points(self):
        objective_a = [3.0, 2.0, 1.0, 1.0]
        objective_b = [1.0, 2.0, 3.0, 1.0]

        frontier = get_pareto_frontier(objective_a, objective_b)

        np.testing.assert_array_equal(frontier, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
