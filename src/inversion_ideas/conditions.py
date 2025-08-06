"""
Functions and callable classes that define conditions.

Use these objects as stopping criteria for inversions.
"""
from collections.abc import Callable

from .base import Condition


class CustomCondition(Condition):
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, model) -> bool:
        return self.func(model)

    @classmethod
    def create(cls, func: Callable):
        return cls(func)


class ChiTarget(Condition):
    """
    Stopping criteria for when chi factor meets the target.

    Parameters
    ----------
    data_misfit : Objective
        Data misfit term to be evaluated.
    chi_target : float
        Target for the chi factor.
    """

    def __init__(self, data_misfit, chi_target=1.0):
        self.data_misfit = data_misfit
        self.chi_target = chi_target

    def __call__(self, model) -> bool:
        """
        Check if condition has been met.
        """
        chi = self.data_misfit(model) / self.data_misfit.n_data
        return chi < self.chi_target
