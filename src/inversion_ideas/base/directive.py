"""
Base class for directives.
"""

from abc import ABC, abstractmethod

from ..typing import Model


class Directive(ABC):
    """
    Abstract class for directives.
    """

    @abstractmethod
    def __call__(self, model: Model, iteration: int):
        pass
