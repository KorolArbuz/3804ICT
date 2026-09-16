import unittest
import numpy as np
from src.evaluation.diagnostics import full_sort_details


class NumericalDiagnosticTests(unittest.TestCase):
    def test_zero_only_manual_and_finite_native_weights_differ(self):
        _, chosen, manual, native, reconstructed = full_sort_details(
            np.array([[0.0], [0.0], [1.0]]), np.array([0, 1, 1]), np.array([0.0]), 3
        )
        self.assertEqual(manual, 0.5)
        self.assertGreater(reconstructed, 0.5)
        self.assertLess(reconstructed, 1)
        np.testing.assert_array_equal(chosen, [0, 1, 2])

    def test_full_sort_tie_prefers_first_row_and_native_includes_boundary(self):
        _, chosen, manual, native, reconstructed = full_sort_details(
            np.array([[-1.0], [1.0], [2.0]]), np.array([0, 1, 1]), np.array([0.0]), 1
        )
        np.testing.assert_array_equal(chosen, [0])
        np.testing.assert_array_equal(native, [0, 1])
        self.assertEqual(manual, 0)
        self.assertEqual(reconstructed, 0.5)


if __name__ == "__main__":
    unittest.main()
