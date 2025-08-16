"""
Functions and callable classes that define conditions.

Use these objects as stopping criteria for inversions.
"""
from collections.abc import Callable

import numpy as np

from .base import Condition, Objective


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
        return float(chi) < self.chi_target


class ModelChanged(Condition):
    r"""
    Stopping criteria for when model didn't changed above threshold.

    Parameters
    ----------
    threshold : float, optional
        Threshold below which the model will be considered of not changing enough.

    Notes
    -----
    The stopping criteria evaluates:

    .. math::

        \frac{
            \lVert \mathbf{m} - \mathbf{m}_\text{prev} \rVert_2
        }{
            \lVert \mathbf{m}_\text{old} \rVert_2
        }
        < \delta

    where :math:`\mathbf{m}` is the current model, :math:`\mathbf{m}_\text{prev}` is the
    previous model in the inversion, :math:`\lVert \cdot \rVert_2` represents an
    :math:`l_2` norm, and :math:`\delta` is the threshold whose value is given by
    ``threshold``.

    When called, if the inequality holds, the stopping criteria will return ``True``,
    and ``False`` otherwise.
    """

    def __init__(self, threshold: float = 1e-3):
        self.threshold = threshold

    def __call__(self, model) -> bool:
        if not hasattr(self, "previous"):
            return False
        den = np.linalg.norm(self.previous)
        if den == 0.0:
            return False
        num = np.linalg.norm(model - self.previous)
        return float(num / den) < self.threshold

    def update(self, model):
        self.previous = model


class ObjectiveChanged(Condition):
    r"""
    Stopping criteria for when an objective function didn't changed above threshold.

    Parameters
    ----------
    objective_function : float
        Threshold below which the model will be considered of not changing enough.
    threshold : float, optional
        Threshold below which the model will be considered of not changing enough.

    Notes
    -----
    The stopping criteria evaluates:

    .. math::

        \frac{
             \phi(\mathbf{m}) - \phi(\mathbf{m}_\text{old})
        }{
             \phi(\mathbf{m}_\text{old})
        }
        < \delta

    where :math:`\phi`, is the objective function, :math:`\mathbf{m}` is the current
    model, :math:`\mathbf{m}_\text{old}` is the previous model in the inversion,
    :math:`\lVert \cdot \rVert_2` represents an :math:`l_2` norm, and :math:`\delta` is
    the threshold whose value is given by ``threshold``.

    When called, if the inequality holds, the stopping criteria will return ``True``,
    and ``False`` otherwise.
    """

    def __init__(self, objective_function: Objective, threshold: float = 1e-3):
        self.objective_function = objective_function
        self.threshold = threshold

    def __call__(self, model) -> bool:
        if not hasattr(self, "previous"):
            return False
        new = self.objective_function(model)
        return float((new - self.previous) / self.previous) < self.threshold

    def update(self, model):
        self.previous = self.objective_function(model)
