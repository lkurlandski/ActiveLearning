"""
"""

from __future__ import annotations
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Dict

import numpy as np

from active_learning.stopping_criteria.base import StoppingCriteria


class ContradictoryInformation(StoppingCriteria):
    def __init__(self, windows: int = 3):

        self.windows = windows
        self.conf_scores = []

    def get_hyperparameters_dict(self) -> Dict[str, float]:
        return {"windows": self.windows}

    def update(self, confidences: np.ndarray, *args, **kwargs) -> ContradictoryInformation:

        conf = np.sum(confidences)
        self.conf_scores.append(conf)

        return self

    def is_stop(self) -> bool:

        if len(self.conf_scores) < self.windows:
            return False

        for prev, curr in zip(
            self.conf_scores[-self.windows - 1 :], self.conf_scores[-self.windows :]
        ):
            if not curr < prev:
                return False

        return True


if __name__ == "__main__":
    pass
