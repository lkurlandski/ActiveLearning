"""Protoype stopping algorithms, designed to be computed during learning instead of after it.

TODO
----
- Bring into its own repository/module.

FIXME
-----
-
"""

from abc import ABC, abstractmethod
from pprint import pprint  # pylint: disable=unused-import
import random
import statistics
import sys  # pylint: disable=unused-import
from typing import Any, Dict, List
import warnings

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics import cohen_kappa_score


class StoppingMethod(ABC):
    """Abstract stopping method to provide uniform behavior for different stopping criterion."""

    def __init__(self, name: str):
        """Abstract stopping method to provide uniform behavior for different stopping criterion.

        Parameters
        ----------
        name : str
            Name of the subclass stopping criterion
        """

        self.stopped: bool = False
        self.results: Dict[str, Any] = None
        self.name = name

    def __str__(self) -> str:

        return (
            self.name
            + "\n"
            + "".join("-" for _ in range(len(self.name)))
            + "\n"
            + str(self.get_hyperparameters_dict())
        )

    def __repr__(self) -> str:

        return self.name.replace(" ", "") + str(self.get_hyperparameters_dict()).replace(
            " ", ""
        ).replace("'", "").replace(":", "=").replace("{", "(").replace("}", ")")

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
        """Determine if this StoppingMethod instance has stopped and update its stopped attribute."""

    def update_results(self, **results) -> None:
        """Update this StoppingMethod instance with the current status of the AL experiment.

        User should provide an update containing useful information like the model's performance
            on a test set, the number of labels annotated by an orcale etc.
        """

        if not self.stopped:
            self.results = dict(results)


class StabilizingPredictions(StoppingMethod):
    """Stablizing Predictions stopping method."""

    prv_preds: np.ndarray
    preds: np.ndarray

    def __init__(self, windows: int = 3, threshold: float = 0.99):
        """Stablizing Predictions stopping method.

        Parameters
        ----------
        windows : int, optional
            The number of kappas to compute the mean of, by default 3
        threshold : float, optional
            The threshold the mean kappas must exceed, by default 0.99
        """

        super().__init__("Stabilizing Predictions")

        self.windows = windows
        self.threshold = threshold
        self.kappas = []
        self.prv_preds = None

    def get_hyperparameters_dict(self) -> Dict[str, float]:

        return {
            "threshold": self.threshold,
            "windows": self.windows,
        }

    def check_stopped(self, preds: np.ndarray, **kwargs) -> None:
        """Determine if this instance has stopped and update its stopped attribute.

        Parameters
        ----------
        preds : np.ndarray
            Predictions on a stop set of training data.
        """

        self.preds = preds
        self.update_kappas()

        if len(self.kappas[1:]) < self.windows:
            return

        if statistics.mean(self.kappas[-self.windows :]) > self.threshold:
            self.stopped = True

    def update_kappas(self) -> None:
        """Update this instances' kappa list by computing agreement between the sets of predictions.

        In the case of a multilabel classification problem, our objective is to cast the model's
            two-dimensional multilabel prediction vector into a one-dimensional single label
            prediction vector. The most logical way to do this is as follows:
            - if both models think example x belongs in class c, then the single label should be c.
            - else, the single label should be any of their predicted classes.
        """

        kappa = self.get_cohen_kappa_score(self.prv_preds, self.preds)
        self.kappas.append(kappa)
        self.prv_preds = self.preds

    @staticmethod
    def get_cohen_kappa_score(y1, y2):
        y1 = y1.toarray() if sparse.issparse(y1) else y1
        y2 = y2.toarray() if sparse.issparse(y2) else y2

        if y1 is None or y2 is None:
            return np.NaN

        if y1.ndim == 1 and y2.ndim == 1:
            return cohen_kappa_score(y1, y2)

        if y1.ndim != 2 or y2.ndim != 2:
            raise ValueError("Unexpected dimensionality of predictions: {(y1.ndim, y2.ndim)}.")

        warnings.warn(
            "WARNING: Cohen's Kappa statistic is not formally defined for multilabel agreement.\n"
            "Computing a naive multilable Cohen's Kappa statistic, but recommend you use another"
            " agreement metric, such as Krippendorff's Alpha."
        )

        y1_ = [None] * y1.shape[0]
        y2_ = [None] * y2.shape[0]
        for i, (prev_pred, pred) in enumerate(zip(y1, y2)):
            for c, (c1, c2) in enumerate(zip(prev_pred, pred)):
                if (c1 == 1 and c2 == 1) or (c1 == 0 and c2 == 0):
                    y1_[i] = c
                    y2_[i] = c
                    break
                if c1 == 1:
                    y1_[i] = c
                if c2 == 1:
                    y2_[i] = c
                if y1_[i] is None:
                    y1_[1] = random.choice([i for i in range(len(prev_pred)) if i != y2_[i]])
                if y2_[i] is None:
                    y2_[1] = random.choice([i for i in range(len(pred)) if i != y1_[i]])

        return cohen_kappa_score(y1_, y2_)


# TODO: implement
# class ClassificationChange:
#    pass

# class OverallUncertainty:
#    pass

# class OracleAccuracyMCS:
#    pass

# class PerformanceConvergence:
#    pass


class Manager:
    """Manage multiple instances of the StoppingMethod class."""

    def __init__(self, stopping_methods: List[StoppingMethod]):
        """Manage multiple instances of the StoppingMethod class.

        Parameters
        ----------
        stopping_methods : List[StoppingMethod]
            A list of StoppingMethod instances to manage and evaluate.
        """

        self.stopping_methods = stopping_methods

    def stopping_condition_met(self, stopping_condition: StoppingMethod) -> bool:
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

        return pd.DataFrame({repr(m): pd.Series(m.results) for m in self.stopping_methods})

    def results_to_dict(self) -> Dict[str, Any]:
        """Return the results of every StoppingMethod instance formatted as a dict.

        Returns
        -------
        Dict[str, Any]
            Representation of data
        """

        return {repr(m): m.results for m in self.stopping_methods}


def test():
    """Testing."""

    stopping_method = StabilizingPredictions(windows=3, threshold=0.99)
    print(str(stopping_method))
    print(repr(stopping_method))


if __name__ == "__main__":
    test()
