"""
Wrap SimPEG simulations to work with this new inversion framework.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator

from inversion_ideas.base import Simulation


def wrap_simulation(simulation, *, store_jacobian=False):
    """
    Wrap a SimPEG's simulation.

    Parameters
    ----------
    simulation : object
        Instance of a SimPEG simulation.
    store_jacobian : bool, optional
        Whether to store the jacobian matrix as a dense or sparse matrix.
        If False, the ``jacobian`` method will return
        a :class:`~scipy.sparse.linalg.LinearOperator` that calls the ``Jvec`` and
        ``Jtvec`` methods of the SimPEG simulation.
        Default to False.

    Returns
    -------
    WrappedSimulation
    """
    return WrappedSimulation(simulation, store_jacobian=store_jacobian)


class WrappedSimulation(Simulation):
    """
    Wrapper of SimPEG's simulations.

    This class is meant to be used within the new framework.

    Parameters
    ----------
    simulation : object
        Instance of a SimPEG simulation.
    store_jacobian : bool, optional
        Whether to store the jacobian matrix as a dense or sparse matrix.
        If False, the ``jacobian`` method will return
        a :class:`~scipy.sparse.linalg.LinearOperator` that calls the ``Jvec`` and
        ``Jtvec`` methods of the SimPEG simulation.
        Default to False.
    """

    def __init__(self, simulation, *, store_jacobian=False):
        has_getJ = hasattr(simulation, "getJ") and callable(simulation.getJ)
        if store_jacobian and not has_getJ:
            msg = (
                "Not possible to set `store_jacobian` to True when wrapping the "
                f"`{type(simulation).__name__}`: the simulation doesn't have a "
                "`getJ` method to build the jacobian matrix."
            )
            raise TypeError(msg)

        self.simulation = simulation
        self.store_jacobian = store_jacobian

    @property
    def n_params(self) -> int:
        """
        Number of model parameters.
        """
        # Potential field simulations have nC attribute with number of parameters
        if hasattr(self.simulation, "nC"):
            return self.simulation.nC

        # Cover other type of simulations
        if hasattr(self.simulation, "model") and self.simulation.model is not None:
            return len(self.simulation.model)

        msg = f"Cannot obtain number of parameters for simulation '{self.simulation}'."
        raise AttributeError(msg)

    @property
    def n_data(self) -> int:
        """
        Number of data values.
        """
        return self.simulation.survey.nD

    def __call__(self, model) -> npt.NDArray[np.float64]:
        """
        Evaluate simulation for a given model.
        """
        return self.simulation.dpred(model)

    def jacobian(self, model) -> npt.NDArray[np.float64] | LinearOperator:
        """
        Jacobian matrix for a given model.
        """
        if self.store_jacobian:
            jac = self.simulation.getJ(model)
        else:
            jac = LinearOperator(
                shape=(self.n_data, self.n_params),
                dtype=np.float64,
                matvec=lambda v: self.simulation.Jvec(model, v),
                rmatvec=lambda v: self.simulation.Jtvec(model, v),
            )
        return jac
