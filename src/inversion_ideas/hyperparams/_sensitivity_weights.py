"""
Custom class for sensitivity weights.
"""

import numpy as np
import numpy.typing as npt

from ..base import Simulation, WrappedArray
from ..typing import Model
from ..utils import get_sensitivity_weights


class SensitivityWeights(WrappedArray):
    """
    Updatable sensitivity weights.

    This class wraps a sensitivity weights array that can be updated by calling the
    :meth:`~inversion_ideas.hyperparams.SensitivityWeights.update` method, passing a
    given ``model`` as argument.

    Parameters
    ----------
    simulation : Simulation
        Simulation used to get the jacobian matrix that will be used while updating the
        sensitivity weights.
    initial_model : (n_params) array
        Initial model used to initialize the sensitivity weights.
    data_weights : (n_data,) array or None, optional
        Array with data weights used to compute the sensitivty weights.
        Can use the :attr:`inversion_ideas.DataMisfit.weights` property.
    volumes : (n_params) array
        Array with the volumes of the active cells. Sensitivity weights are
        divided by the volumes to account for sensitivity changes due to cell sizes.
    vmin : float or None, optional
        Minimum value used for clipping.
    """

    def __init__(
        self,
        simulation: Simulation,
        initial_model: Model,
        *,
        data_weights: npt.NDArray[np.float64] | None = None,
        volumes: npt.NDArray[np.float64] | None = None,
        vmin: float | None = 1e-12,
    ):
        self.simulation = simulation
        self.data_weights = data_weights
        self.volumes = volumes
        self.vmin = vmin
        self._initial_model = initial_model.copy()
        super().__init__(self._compute_sensitivity_weights(self.initial_model))

    @property
    def initial_model(self) -> Model:
        """Model used to compute the initial sensitivity weights."""
        return self._initial_model

    def update(self, model, *args):  # ruff: ignore[ARG002]
        """
        Update sensitivity weights for the given model.

        Parameters
        ----------
        model : (n_params) array
            Array with model parameters that will be used to update the sensitivity
            weights.
        *args :
            Any extra argument will be ignored. They are kept to guarantee compatibility
            with the ``update`` method interface.
        """
        self.array = self._compute_sensitivity_weights(model)

    def _compute_sensitivity_weights(self, model):
        """
        Compute sensitivity weights for a given model.
        """
        jacobian = self.simulation.jacobian(model)
        sensitivity_weights = get_sensitivity_weights(
            jacobian,
            data_weights=self.data_weights,
            volumes=self.volumes,
            vmin=self.vmin,
        )
        return sensitivity_weights
