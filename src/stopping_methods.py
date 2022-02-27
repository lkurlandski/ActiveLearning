"""Protoype stopping algorithms, designed to be computed during learning instead of after it.
"""

from abc import ABC, abstractmethod
from pprint import pprint                                           # pylint: disable=unused-import
import statistics
import sys                                                          # pylint: disable=unused-import
from typing import Any, Dict, List

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

class StoppingMethod(ABC):
    """Abstract stopping method to provide uniform behavior for different stopping criterion.
    """

    def __init__(self, name:str):
        """Abstract stopping method to provide uniform behavior for different stopping criterion.

        Parameters
        ----------
        name : str
            Name of the subclass stopping criterion
        """

        self.stopped:bool = False
        self.results:Dict[str, Any] = None
        self.name = name

    def __str__(self) -> str:

        return self.name \
            + '\n' + ''.join('-' for _ in range(len(self.name))) + '\n' \
            + str(self.get_hyperparameters_dict())

    def __repr__(self) -> str:

        return self.name.replace(' ', '') + str(self.get_hyperparameters_dict())\
            .replace(' ', '').replace("'","").replace(':','=').replace('{', '(').replace('}', ')')

    @abstractmethod
    def get_hyperparameters_dict(self) -> Dict[str, Any]:
        """Get the hyperparameters for the relevant StoppingMethod instance.

        Returns
        -------
        Dict[str, Any]
            Hyperparameter names and their values
        """

    @abstractmethod
    def check_stopped(self, **kwargs) -> None:
        """Determine if this StoppingMethod instance has stopped and update its stopped attribute.
        """

    def update_results(self, **results) -> None:
        """Update this StoppingMethod instance with the current status of the AL experiment.

        User should provide an update containing useful information like the model's performance
            on a test set, the number of labels annotated by an orcale etc.
        """

        if not self.stopped:
            self.results = dict(results)

class StabilizingPredictions(StoppingMethod):
    """Stablizing Predictions stopping method.
    """

    stop_set_predictions:np.ndarray

    def __init__(self,
            windows : int = 3,
            threshold : float = 0.99,
            initial_stop_set_predictions:np.ndarray = None
        ):
        """Stablizing Predictions stopping method.

        Parameters
        ----------
        windows : int, optional
            The number of kappas to compute the mean of, by default 3
        threshold : float, optional
            The threshold the mean kappas must exceed, by default 0.99
        intial_stop_set_predictions : np.ndarray, optional
            The very first stop set predictions, if the user computes them, by default None

        The first time this method is evaluated, there will not be two sets of model
            predictions to compute the kappa agreement on. The user can "skip" this phase and
            essentially start one iteration later in the AL process by providing the previous
            set of predictions.
        """

        super().__init__("Stabilizing Predictions")

        self.windows = windows
        self.threshold = threshold
        self.kappas = [] if initial_stop_set_predictions is None else [np.NaN]
        self.previous_stop_set_predictions = initial_stop_set_predictions

    def get_hyperparameters_dict(self) -> Dict[str, float]:

        return {
            'threshold': self.threshold,
            'windows': self.windows,
        }

    def check_stopped(self, stop_set_predictions:np.ndarray, **kwargs) -> None:
        """Determine if this instance has stopped and update its stopped attribute.

        Parameters
        ----------
        stop_set_predictions : np.ndarray
            Predictions on a stop set of training data.
        """

        self.stop_set_predictions = stop_set_predictions
        self.update_kappas()

        if len(self.kappas[1:]) < self.windows:
            return

        if statistics.mean(self.kappas[-self.windows:]) > self.threshold:
            self.stopped = True

    def update_kappas(self) -> None:
        """Update this instances' kappa list by computing agreement between the sets of predictions.
        """

        kappa = np.NaN if self.previous_stop_set_predictions is None else \
            cohen_kappa_score(self.previous_stop_set_predictions, self.stop_set_predictions)
        self.kappas.append(kappa)
        self.previous_stop_set_predictions = self.stop_set_predictions

# TODO: implement
#class ClassificationChange:
#    pass

#class OverallUncertainty:
#    pass

#class OracleAccuracyMCS:
#    pass

#class PerformanceConvergence:
#    pass

class Manager:
    """Manage multiple instances of the StoppingMethod class.
    """

    def __init__(self, stopping_methods:List[StoppingMethod]):
        """Manage multiple instances of the StoppingMethod class.

        Parameters
        ----------
        stopping_methods : List[StoppingMethod]
            A list of StoppingMethod instances to manage and evaluate.
        """

        self.stopping_methods = stopping_methods

    def stopping_condition_met(self, stopping_condition:StoppingMethod) -> bool:
        """Determine if a particular StoppingMethod's stopping_condition has been met.

        Parameters
        ----------
        stopping_condition : StoppingMethod
            The StoppingMethod to check

        Returns
        -------
        bool
            Whether the StoppingMethod stopping conditions says to stop or not
        """

        if repr(stopping_condition) in {repr(m) for m in self.stopping_methods if m.stopped}:
            return True

        return False

    def check_stopped(self, **kwargs):
        """Determine if every StoppingMethod instance has stopped and update its stopped attribute.

        Parameters
        ----------
            Named arguments which should be supplied to the individual instances of StoppingMethod

        The arguments for every StoppingMethod instance's check_stopped() method are needed.
        """

        for m in self.stopping_methods:
            m.check_stopped(**kwargs)

    def update_results(self, **results):
        """Update the results of every StoppingMethod instance.

        Parameters
        ----------
            Named arguments which should be supplied to the individual instances of StoppingMethod
        """

        for m in self.stopping_methods:
            m.update_results(**results)

    def results_to_dataframe(self) -> pd.DataFrame:
        """Return the results of every StoppingMethod instance formatted as a dataframe.

        Returns
        -------
        pd.DataFrame
            Representation of data
        """

        return pd.DataFrame(
            {repr(m) : pd.Series(m.results) for m in self.stopping_methods}
        )

    def results_to_dict(self) -> Dict[str, Any]:
        """Return the results of every StoppingMethod instance formatted as a dict.

        Returns
        -------
        Dict[str, Any]
            Representation of data
        """

        return {repr(m) : m.results for m in self.stopping_methods}

def test():
    """Testing.
    """

    stopping_method = StabilizingPredictions(windows=3, threshold=0.99)
    print(str(stopping_method))
    print(repr(stopping_method))

if __name__ == "__main__":
    test()
