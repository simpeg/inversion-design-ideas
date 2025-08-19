"""
Base class for directives.
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Directive(ABC):
    """
    Abstract class for directives.
    """

    @abstractmethod
    def __call__(self, model: npt.NDArray[np.float64], iteration: int):
        pass
