"""
Functions and callable classes that define conditions.

Use these objects as stopping criteria for inversions.
"""

from collections.abc import Callable

import numpy as np
from rich.tree import Tree

from inversion_ideas.data_misfit import DataMisfit

from .base import Condition, Objective
from .typing import Model


class CustomCondition(Condition):
    """
    Define a custom :class:`~inversion_ideas.base.Condition` object through a function.

    Parameters
    ----------
    func : Callable
        Function to use as the condition. It should take a model as only argument
        and return a bool to evaluate whether the condition is valid or not for that
        particular model.
    """

    def __init__(self, func: Callable[[Model], bool]):
        self.func = func

    def __call__(self, model) -> bool:
        return self.func(model)

    @classmethod
    def create(cls, func: Callable[[Model], bool]):
        """
        Create a ``CustomCondition`` object directly from a function.

        Parameters
        ----------
        func : Callable
            Function to use as the condition. It should take a model as only argument
            and return a bool to evaluate whether the condition is valid or not for that
            particular model.

        Returns
        -------
        CustomCondition

        Examples
        --------
        >>> import numpy as np
        >>>
        >>> def is_all_positive(model):
        ...     return np.all(model > 0)
        >>>
        >>> condition = CustomCondition.create(is_all_positive)
        >>> model = np.array([1, 2, 3])
        >>> print(condition(model))
        True

        >>> model = np.array([-1, 2, 3])
        >>> print(condition(model))
        False
        """
        return cls(func)


class ChiTarget(Condition):
    """
    Stopping criteria for when chi factor meets the target.

    Parameters
    ----------
    data_misfit : DataMisfit
        Data misfit term to be evaluated.
    chi_target : float
        Target for the chi factor.
    """

    def __init__(self, data_misfit: DataMisfit, chi_target=1.0):
        self.data_misfit = data_misfit
        self.chi_target = chi_target

    def __call__(self, model) -> bool:
        """
        Check if condition has been met.
        """
        chi = self.data_misfit.chi_factor(model)
        return float(chi) < self.chi_target

    def info(self, model) -> Tree:
        tree = super().info(model)
        tree.add("Condition: chi < chi_target")
        tree.add(f"chi        = {self.data_misfit.chi_factor(model):.2e}")
        tree.add(f"chi_target = {self.chi_target:.2e}")
        return tree


class ModelChanged(Condition):
    r"""
    Stopping criteria for when model didn't changed above tolerance.

    Parameters
    ----------
    rtol : float, optional
        Relative tolerance below which the model will be considered of not changing
        enough.
    atol : float, optional
        Absolute tolerance below which the model will be considered of not changing
        enough.

    Notes
    -----
    The stopping criteria evaluates:

    .. math::

        \frac{
            \lVert \mathbf{m} - \mathbf{m}_\text{prev} \rVert_2
        }{
            \lVert \mathbf{m}_\text{old} \rVert_2
        }
        \le \delta_r,

    and

    .. math::

        \lVert \mathbf{m} - \mathbf{m}_\text{prev} \rVert_2 \le \delta_a,

    where :math:`\mathbf{m}` is the current model, :math:`\mathbf{m}_\text{prev}` is the
    previous model in the inversion, :math:`\lVert \cdot \rVert_2` represents an
    :math:`l_2` norm, and :math:`\delta_r` and :math:`\delta_a` are the relative and
    absolute tolerances whose values are given
    by ``rtol`` and ``atol``, respectively.

    When called, if any of those inequalities hold, the stopping criteria will return
    ``True``, and ``False`` otherwise.
    """

    def __init__(self, rtol: float = 1e-3, atol: float = 0.0):
        self.rtol = rtol
        self.atol = atol

    def __call__(self, model) -> bool:
        if not hasattr(self, "previous"):
            return False
        diff = float(np.linalg.norm(model - self.previous))
        previous = float(np.linalg.norm(self.previous))
        return diff <= max(previous * self.rtol, self.atol)

    def update(self, model):
        """
        Cache model as the ``previous`` one.
        """
        self.previous = model

    def info(self, model) -> Tree:
        tree = super().info(model)
        diff = float(np.linalg.norm(model - self.previous))
        previous = float(np.linalg.norm(self.previous))
        tree.add("Condition: |m - m_prev| <= max(|m_prev| * rtol, atol)")
        tree.add(f"|m - m_prev| = {diff:.2e}")
        tree.add(f"|m_prev|     = {previous:.2e}")
        tree.add(f"rtol         = {self.rtol:.2e}")
        tree.add(f"atol         = {self.atol:.2e}")
        return tree

    def initialize(self):
        """
        Initialize condition and clean ``previous`` attribute.
        """
        attr = "previous"
        if hasattr(self, attr):
            delattr(self, attr)


class ObjectiveChanged(Condition):
    r"""
    Stopping criteria for when an objective function didn't changed above a tolerance.

    Parameters
    ----------
    objective_function : Objective
        Objective function that will be evaluated.
    rtol : float, optional
        Relative tolerance below which the model will be considered of not changing
        enough.
    atol : float, optional
        Absolute tolerance below which the model will be considered of not changing
        enough.

    Notes
    -----
    The stopping criteria evaluates:

    .. math::

        \frac{
             | \phi(\mathbf{m}) - \phi(\mathbf{m}_\text{old}) |
        }{
             | \phi(\mathbf{m}_\text{old}) |
        }
        \le \delta_r,

    and

    .. math::

        | \phi(\mathbf{m}) - \phi(\mathbf{m}_\text{old}) | \le \delta_a,

    where :math:`\phi`, is the objective function, :math:`\mathbf{m}` is the current
    model, :math:`\mathbf{m}_\text{old}` is the previous model in the inversion,
    and :math:`\delta_r` and :math:`\delta_a` are the relative and absolute tolerances
    whose values are given by ``rtol`` and ``atol``, respectively.

    When called, if any of those inequalities hold, the stopping criteria will return
    ``True``, and ``False`` otherwise.
    """

    def __init__(self, objective_function: Objective, rtol: float = 1e-3, atol=0.0):
        self.objective_function = objective_function
        self.rtol = rtol
        self.atol = atol

    def __call__(self, model) -> bool:
        if not hasattr(self, "previous"):
            return False
        diff = abs(self.objective_function(model) - self.previous)
        previous = abs(self.previous)
        return diff <= max(previous * self.rtol, self.atol)

    def update(self, model):
        """
        Cache value of objective function with model as the ``previous`` one.
        """
        self.previous: float = float(self.objective_function(model))

    def info(self, model) -> Tree:
        tree = super().info(model)
        diff = abs(self.objective_function(model) - self.previous)
        previous = abs(self.previous)
        tree.add("Condition: |φ(m) - φ(m_prev)| <= max(|φ(m_prev)| * rtol, atol)")
        tree.add(f"|φ(m) - φ(m_prev)| = {diff:.2e}")
        tree.add(f"|φ(m_prev)|        = {previous:.2e}")
        tree.add(f"rtol               = {self.rtol:.2e}")
        tree.add(f"atol               = {self.atol:.2e}")
        return tree

    def initialize(self):
        """
        Initialize condition and clean ``previous`` attribute.
        """
        attr = "previous"
        if hasattr(self, attr):
            delattr(self, attr)
