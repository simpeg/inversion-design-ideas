import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator
from inversion_ideas.base import Simulation


class GravitySimulation(Simulation):
    """
    Wrapper of SimPEG's gravity simulation.

    This class is meant to be used within the new framework.
    """

    def __init__(self, simulation):
        self.simulation = simulation

    @property
    def n_params(self) -> int:
        """
        Number of model parameters.
        """
        return self.simulation.nC

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
        jac = LinearOperator(
            shape=(self.n_data, self.n_params),
            dtype=np.float64,
            matvec=lambda v: self.simulation.Jvec(model, v),
            rmatvec=lambda v: self.simulation.Jtvec(model, v),
        )
        return jac
