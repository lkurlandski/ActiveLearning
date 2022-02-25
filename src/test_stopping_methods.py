from pprint import pprint
import sys
import unittest

import numpy as np

from src import stopping_methods

class TestStoppingMethod(unittest.TestCase):

    pass

class TestManager(unittest.TestCase):

    def test_check_stopped(self):

        sp_1 = stopping_methods.StabilizingPredictions(windows=3, threshold=.66)
        sp_2 = stopping_methods.StabilizingPredictions(windows=3, threshold=.67)
        sp_3 = stopping_methods.StabilizingPredictions(windows=3, threshold=1.0) # should never stop

        mg = stopping_methods.Manager([sp_1, sp_2, sp_3])

        stopped = mg.check_stopped(stop_set_predictions=[1,2,3], other_random_args=None)
        stopped = mg.check_stopped(stop_set_predictions=[3,1,2], other_random_args=None)

        for i in range(10):
            stopped = mg.check_stopped(stop_set_predictions=[3,2,1], other_random_args=None)

            if i == 0 or i == 1:
                self.assertEqual(stopped, [])
            elif i == 2:
                self.assertEqual(len(stopped), 1)
                self.assertEqual(repr(stopped[0]), repr(sp_1))
            elif i > 2:
                self.assertEqual(len(stopped), 2)
                self.assertEqual(repr(stopped[0]), repr(sp_1))
                self.assertEqual(repr(stopped[1]), repr(sp_2))

    def test_update_results(self):

        sp_1 = stopping_methods.StabilizingPredictions(windows=3, threshold=.66)
        sp_2 = stopping_methods.StabilizingPredictions(windows=3, threshold=.67)
        sp_3 = stopping_methods.StabilizingPredictions(windows=3, threshold=1.0) # should never stop

        mg = stopping_methods.Manager([sp_1, sp_2, sp_3])

        mg.update_results(annotations=10, iteration=1, accuracy=.40)
        for m in mg.stopping_methods:
            self.assertEqual(m.results, dict(annotations=10, iteration=1, accuracy=.40))

        mg.update_results(iteration=2, annotations=20, accuracy=.80)
        for m in mg.stopping_methods:
            self.assertEqual(m.results, dict(annotations=20, iteration=2, accuracy=.80))

class TestStabilizingPredictions(unittest.TestCase):

    def test_check_stopped(self):

        m = stopping_methods.StabilizingPredictions(windows=3, threshold=0.67)
        self.assertFalse(m.stopped)

        preds = [1,2,3]
        m.check_stopped(preds)
        self.assertFalse(m.stopped)
        self.assertTrue(np.isnan(m.kappas[0]))
        self.assertEqual(m.previous_stop_set_predictions, preds)

        preds = [3,1,2]
        m.check_stopped(preds)
        self.assertFalse(m.stopped)
        self.assertEqual(m.kappas[-1], -.5)
        self.assertEqual(m.previous_stop_set_predictions, preds)

        preds = [3,2,1]
        m.check_stopped(preds)
        self.assertFalse(m.stopped)
        self.assertEqual(m.kappas[-1], 0)
        self.assertEqual(m.previous_stop_set_predictions, preds)

        preds = [3,2,1]
        m.check_stopped(preds)
        self.assertFalse(m.stopped)
        self.assertEqual(m.kappas[-1], 1)
        self.assertEqual(m.previous_stop_set_predictions, preds)

        preds = [3,2,1]
        m.check_stopped(preds)
        self.assertFalse(m.stopped)
        self.assertEqual(m.kappas[-1], 1)
        self.assertEqual(m.previous_stop_set_predictions, preds)

        preds = [3,2,1]
        m.check_stopped(preds)
        self.assertTrue(m.stopped)
        self.assertEqual(m.kappas[-1], 1)
        self.assertEqual(m.previous_stop_set_predictions, preds)

    def test_update_results(self):

        m = stopping_methods.StabilizingPredictions(windows=3, threshold=0.99)

        m.update_results(annotations=10, iteration=1, accuracy=.40)
        self.assertEqual(m.results, dict(annotations=10, iteration=1, accuracy=.40))

        m.update_results(iteration=2, annotations=20, accuracy=.80)
        self.assertEqual(m.results, dict(annotations=20, iteration=2, accuracy=.80))

        m.stopped = True
        m.update_results(iteration=3, annotations=30, accuracy=.85)
        self.assertEqual(m.results, dict(annotations=20, iteration=2, accuracy=.80))
