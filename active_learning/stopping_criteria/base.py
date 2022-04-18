"""Base stopping method abstract class.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Dict


class StoppingCriteria(ABC):

    def __init__(self):
        self.has_stopped = False

    def __str__(self):
        return type(self).__name__ + "(" + ", ".join([f"{k}={v}" for k, v in self.get_hyperparams().items()]) + ")"

    @abstractmethod
    def get_hyperparams(self) -> Dict[str, Any]:
        ...
