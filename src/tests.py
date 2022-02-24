from pprint import pprint
import sys
import unittest

import numpy as np

from src import stopping_methods

class TestStoppingMethod(unittest.TestCase):

    pass

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