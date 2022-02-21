"""Protoype stopping algorithms, designed to be computed during learning instead of after it.
"""

from abc import ABC, abstractmethod
import itertools
import statistics

import numpy as np
from sklearn.metrics import cohen_kappa_score

class StoppingMethod(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def evaluate_stopping(self):
        pass

    def results_to_json(self):
        pass

class StabilizingPredictions(StoppingMethod):

    def __init__(self, windows, thresholds):

        self.results = {i : None for i in itertools.product(windows, thresholds)}
        self.previous_stop_set_predictions = None
        self.kappas = [np.NaN]

    def evaluate_stopping(self, stop_set_predictions, iteration):

        self.stop_set_predictions = stop_set_predictions
        self.update_kappas()
        for i, j in self.results:
            if statistics.mean(self.kappas[-i:-1]) > j:
                self.results[(i, j)] = iteration

    def update_kappas(self):

        kappa = cohen_kappa_score(self.previous_stop_set_predictions, self.stop_set_predictions)
        self.kappas.append(kappa)
        self.previous_stop_set_predictions = self.stop_set_predictions

if __name__ == "__main__":
    s = StabilizingPredictions([2,3,4], [.97,.98,.99])
    print(s.combos)
