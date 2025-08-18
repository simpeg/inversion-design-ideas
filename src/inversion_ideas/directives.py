"""
Directives to modify the objective function between iterations of an inversion.
"""

import numpy as np
import numpy.typing as npt

from .base import Directive, Scaled


class MultiplierCooler(Directive):
    r"""
    Cool the multiplier of an objective function.

    Parameters
    ----------
    scaled_objective : Scaled
        Scaled objective function whose multiplier will be cooled.
    cooling_factor : float
        Factor by which the multiplier will be cooled.
    cooling_rate : int, optional
        Cool down the multiplier every ``cooling_rate`` call to this directive.

    Notes
    -----
    Given a scaled objective function :math:`\phi(\mathbf{m}) = \alpha
    \varphi(\mathbf{m})`, and a cooling factor :math:`k`, this directive will *cool* the
    multiplier `\alpha` by dividing it by :math:`k` on every ``cooling_rate`` call to
    the directive.
    """

    def __init__(
        self, scaled_objective: Scaled, cooling_factor: float, cooling_rate: int = 1
    ):
        if not hasattr(scaled_objective, "multiplier"):
            msg = "Invalid 'scaled_objective': it must have a `multiplier` attribute."
            raise TypeError(msg)
        self.regularization = scaled_objective
        self.cooling_factor = cooling_factor
        self.cooling_rate = cooling_rate

    def __call__(self, model: npt.NDArray[np.float64], iteration: int):  # noqa: ARG002
        """
        Cool the multiplier.
        """
        if iteration % self.cooling_rate == 0:
            self.regularization.multiplier /= self.cooling_factor
