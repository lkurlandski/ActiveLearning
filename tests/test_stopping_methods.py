"""Tests for stopping methods.
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import

import numpy as np
from scipy import sparse

from active_learning import stopping_methods


# class TestManager(unittest.TestCase):
#     def test_check_stopped(self):

#         sp_1 = stopping_methods.StabilizingPredictions(windows=3, threshold=0.66)
#         sp_2 = stopping_methods.StabilizingPredictions(windows=3, threshold=0.67)
#         sp_3 = stopping_methods.StabilizingPredictions(windows=3, threshold=1.0)

#         mg = stopping_methods.Manager([sp_1, sp_2, sp_3])

#         stopped = mg.check_stopped(stop_set_predictions=[1, 2, 3], other_random_args=None)
#         stopped = mg.check_stopped(stop_set_predictions=[3, 1, 2], other_random_args=None)

#         for i in range(10):
#             stopped = mg.check_stopped(stop_set_predictions=[3, 2, 1], other_random_args=None)

#             if i in (0, 1):
#                 self.assertEqual(stopped, [])
#             elif i == 2:
#                 self.assertEqual(len(stopped), 1)
#                 self.assertEqual(repr(stopped[0]), repr(sp_1))
#             elif i > 2:
#                 self.assertEqual(len(stopped), 2)
#                 self.assertEqual(repr(stopped[0]), repr(sp_1))
#                 self.assertEqual(repr(stopped[1]), repr(sp_2))

#     def test_update_results(self):

#         sp_1 = stopping_methods.StabilizingPredictions(windows=3, threshold=0.66)
#         sp_2 = stopping_methods.StabilizingPredictions(windows=3, threshold=0.67)
#         sp_3 = stopping_methods.StabilizingPredictions(
#             windows=3, threshold=1.0
#         )  # should never stop

#         mg = stopping_methods.Manager([sp_1, sp_2, sp_3])

#         mg.update_results(annotations=10, iteration=1, accuracy=0.40)
#         for m in mg.stopping_methods:
#             self.assertEqual(m.results, dict(annotations=10, iteration=1, accuracy=0.40))

#         mg.update_results(iteration=2, annotations=20, accuracy=0.80)
#         for m in mg.stopping_methods:
#             self.assertEqual(m.results, dict(annotations=20, iteration=2, accuracy=0.80))


# class TestStabilizingPredictions(unittest.TestCase):
# def test_check_stopped(self):

#     m = stopping_methods.StabilizingPredictions(windows=3, threshold=0.67)
#     self.assertFalse(m.stopped)

#     preds = [1, 2, 3]
#     m.check_stopped(preds)
#     self.assertFalse(m.stopped)
#     self.assertTrue(np.isnan(m.kappas[0]))
#     self.assertEqual(m.previous_stop_set_predictions, preds)

#     preds = [3, 1, 2]
#     m.check_stopped(preds)
#     self.assertFalse(m.stopped)
#     self.assertEqual(m.kappas[-1], -0.5)
#     self.assertEqual(m.previous_stop_set_predictions, preds)

#     preds = [3, 2, 1]
#     m.check_stopped(preds)
#     self.assertFalse(m.stopped)
#     self.assertEqual(m.kappas[-1], 0)
#     self.assertEqual(m.previous_stop_set_predictions, preds)

#     preds = [3, 2, 1]
#     m.check_stopped(preds)
#     self.assertFalse(m.stopped)
#     self.assertEqual(m.kappas[-1], 1)
#     self.assertEqual(m.previous_stop_set_predictions, preds)

#     preds = [3, 2, 1]
#     m.check_stopped(preds)
#     self.assertFalse(m.stopped)
#     self.assertEqual(m.kappas[-1], 1)
#     self.assertEqual(m.previous_stop_set_predictions, preds)

#     preds = [3, 2, 1]
#     m.check_stopped(preds)
#     self.assertTrue(m.stopped)
#     self.assertEqual(m.kappas[-1], 1)
#     self.assertEqual(m.previous_stop_set_predictions, preds)

# def test_update_results(self):

#     m = stopping_methods.StabilizingPredictions(windows=3, threshold=0.99)

#     m.update_results(annotations=10, iteration=1, accuracy=0.40)
#     self.assertEqual(m.results, dict(annotations=10, iteration=1, accuracy=0.40))

#     m.update_results(iteration=2, annotations=20, accuracy=0.80)
#     self.assertEqual(m.results, dict(annotations=20, iteration=2, accuracy=0.80))

#     m.stopped = True
#     m.update_results(iteration=3, annotations=30, accuracy=0.85)
#     self.assertEqual(m.results, dict(annotations=20, iteration=2, accuracy=0.80))


class TestStabilizingPredictions:
    def test_get_cohen_kappa_score1(self):
        y1 = None
        y2 = np.array([0, 1, 2, 0, 2, 1])
        kappa = stopping_methods.StabilizingPredictions.get_cohen_kappa_score(y1, y2)
        assert np.isnan(kappa)

    def test_get_cohen_kappa_score2(self):
        y1 = np.array([0, 1, 2, 0, 2, 1])
        y2 = np.array([0, 1, 2, 0, 2, 1])
        kappa = stopping_methods.StabilizingPredictions.get_cohen_kappa_score(y1, y2)
        assert kappa == 1

    def test_get_cohen_kappa_score3(self):
        y1 = np.array(
            [
                [0, 1, 0, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0],
            ]
        )
        y2 = np.array(
            [
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 0],
            ]
        )
        kappa = stopping_methods.StabilizingPredictions.get_cohen_kappa_score(y1, y2)

    def test_get_cohen_kappa_score4(self):
        y1 = sparse.csr_matrix(
            np.array(
                [
                    [0, 1, 0, 1, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0],
                ]
            )
        )
        y2 = sparse.csr_matrix(
            np.array(
                [
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [1, 0, 0, 0, 0],
                ]
            )
        )
        kappa = stopping_methods.StabilizingPredictions.get_cohen_kappa_score(y1, y2)
