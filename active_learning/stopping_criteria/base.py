"""Base stopping method abstract class.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Dict


class StoppingCriteria(ABC):

    @abstractmethod
    def get_hyperparams(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def update(self, **kwargs) -> StoppingCriteria:
        ...

    @abstractmethod
    def is_stop(self) -> bool:
        ...
