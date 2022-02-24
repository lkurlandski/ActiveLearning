"""Protoype stopping algorithms, designed to be computed during learning instead of after it.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from pprint import pprint
import statistics
from typing import List

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

class StoppingMethod(ABC):

    def __init__(self, name):
        
        self.stopped = False
        self.results = None
        self.name = name

    def __str__(self):

        return self.name \
            + '\n' + ''.join('-' for _ in range(len(self.name))) + '\n' \
            + str(self.get_hyperparameters_dict())

    def __repr__(self):

        return (self.name + str(self.get_hyperparameters_dict())).replace(' ', '').replace("'","")

    @abstractmethod
    def get_hyperparameters_dict(self):
        pass

    @abstractmethod
    def check_stopped(self, **kwargs):
        pass

    def update_results(self, **results):

        self.results = dict(results)

class Manager:

    def __init__(self, stopping_methods:List[StoppingMethod]):

        self.stopping_methods = stopping_methods

    def check_stopped(self, **kwargs):

        for m in self.stopping_methods:
            m.check_stopped(kwargs)

        return [m for m in self.stopping_methods if m.stopped]

    def results_to_dataframe(self):
        
        return pd.DataFrame(
            {repr(m) : pd.Series(m.results) for m in self.stopping_methods}
        )

    def results_to_csv(self, path, **pandas_to_csv_kwargs):

        pd.to_csv(path, self.results_to_dataframe(), pandas_to_csv_kwargs)

    def results_to_json(self):

        pass

class StabilizingPredictions(StoppingMethod):

    def __init__(self, windows=3, threshold=0.99, previous_stop_set_predictions=None):

        super().__init__("StabilizingPredictions")

        self.windows = windows
        self.threshold = threshold
        self.kappas = [] if previous_stop_set_predictions is None else [np.NaN]
        self.previous_stop_set_predictions = previous_stop_set_predictions

    def get_hyperparameters_dict(self):

        return {
            'threshold': self.threshold,
            'windows': self.windows,
        }

    def check_stopped(self, stop_set_predictions, **kwargs):

        self.stop_set_predictions = stop_set_predictions
        self.update_kappas()

        if len(self.kappas[1:]) < self.windows:
            return

        if statistics.mean(self.kappas[-self.windows:]) > self.threshold:
            self.stopped = True

    def update_kappas(self):

        kappa = np.NaN if self.previous_stop_set_predictions is None else \
            cohen_kappa_score(self.previous_stop_set_predictions, self.stop_set_predictions)
        self.kappas.append(kappa)
        self.previous_stop_set_predictions = self.stop_set_predictions

class ClassificationChange:

    pass

class OverallUncertainty:

    pass

class OracleAccuracyMCS:

    pass

class PerformanceConvergence:

    pass

def tests():
    
    stopping_method = StabilizingPredictions(windows=3, threshold=0.99)
    print(str(stopping_method))
    print(repr(stopping_method))
    #stopping_method_manager = Manager()

if __name__ == "__main__":
    tests()
