"""
Directives to modify the objective function between iterations of an inversion.
"""

import numpy as np
import numpy.typing as npt

from inversion_ideas.utils import get_sensitivity_weights

from .base import Directive, Objective, Scaled, Simulation


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


class UpdateSensitivityWeights(Directive):
    """
    Update sensitivity weights on regularizations.

    .. note::

        The regularizations on which this directive can operate need to have
        a ``cell_weights`` attribute, and it must be a dictionary.

    Parameters
    ----------
    *args : Objective
        Regularizations to which the sensitivity weights will be updated.
    simulation : Simulation
        Simulation used to get the jacobian matrix that will be used while updating the
        sensitivity weights.
    weights_key : str, optional
        Key used to store the sensitivity weights on the regularization's
        ``cell_weights`` dictionary. Only the weights under this key will be updated.
    **kwargs
        Extra arguments passed to the
        :func:`inversion_ideas.utils.get_sensitivity_weights` function.

    See also
    --------
    inversion_ideas.utils.get_sensitivity_weights
    """

    def __init__(
        self, *args, simulation: Simulation, weights_key: str = "sensitivity", **kwargs
    ):
        self.regularizations = args
        self.weights_key = weights_key
        self.simulation = simulation
        self.kwargs = kwargs

        # Sanity checks for cell_weights in regularizations
        for regularization in self.regularizations:
            self._check_cell_weights(regularization)

    def __call__(self, model: npt.NDArray[np.float64], iteration: int):  # noqa: ARG002
        """
        Update sensitivity weights.
        """
        for regularization in self.regularizations:
            # Check cell_weights in regularization
            self._check_cell_weights(regularization)

            # Compute the jacobian
            jacobian = self.simulation.jacobian(model)
            self._check_jacobian_type(jacobian)
            new_sensitivity_weights = get_sensitivity_weights(jacobian, **self.kwargs)

            # Update the sensitivity weights
            regularization.cell_weights[self.weights_key] = new_sensitivity_weights

    def _check_jacobian_type(self, jacobian):
        """Check if jacobian is a dense array."""
        if not isinstance(jacobian, np.ndarray):
            msg = (
                "Cannot compute sensitivity weights for simulation "
                f"{self.simulation} : its jacobian is a {type(jacobian)}. "
                "It must be a dense array."
            )
            raise TypeError(msg)

    def _check_cell_weights(self, regularization: Objective):
        """Sanity checks for cell_weights in regularization."""
        # Check if regularization have cell_weights attribute
        if not hasattr(regularization, "cell_weights"):
            msg = (
                "Missing `cell-weights` attribute in regularization "
                f"'{regularization}'."
            )
            raise AttributeError(msg)

        if not isinstance(regularization.cell_weights, dict):
            msg = (
                f"Invalid `cell_weights` attribute of type '{type(regularization)}' "
                f"for the '{regularization}'. It must be a dictionary."
            )
            raise TypeError(msg)
        if self.weights_key not in regularization.cell_weights:
            msg = (
                f"Missing '{self.weights_key}' weights in "
                f"{regularization}.cell_weights. "
            )
            raise KeyError(msg)
