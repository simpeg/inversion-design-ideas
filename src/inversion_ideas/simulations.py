"""
Wrap SimPEG simulations to work with this new inversion framework.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator
from simpeg.base.pde_simulation import BasePDESimulation

from ._utils import array_to_str, compute_hash, hash_to_str
from .base import Simulation
from .decorators import cache_on_model
from .typing import Model
from .utils import get_logger


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

    Allows to use SimPEG simulations in the new framework, extending them to be
    compatible with the :class:`inversion_ideas.base.Simulation` interface.

    .. important::

        This class is meant to be a glue between current SimPEG simulations and the
        new framework. The ultimate goal is to make SimPEG simulations compatible with
        this framework. After that, this class will become obsolete and we'll be able
        to remove it.

    Parameters
    ----------
    simulation : object
        Instance of a SimPEG simulation.
    store_jacobian : bool, optional
        Whether to store the jacobian matrix as a dense or sparse matrix.
        If False, the :meth:`~inversion_ideas.WrappedSimulation.jacobian` method will
        return a :class:`~scipy.sparse.linalg.LinearOperator` that calls the ``Jvec``
        and ``Jtvec`` methods of the SimPEG simulation.
        Default to False.
    cache_fields : bool, optional
        Whether to cache fields within the simulation or not.
        If True, the fields in PDE-based SimPEG simualtions will be cached within the
        :class:`~inversion_ideas.WrappedSimulation` object to avoid recomputing them
        when the object gets called, or an operation with the Jacobian matrix is
        performed.
        Default to True.
    cache : bool, optional
        Whether to cache the last result of the
        :meth:`~inversion_ideas.WrappedSimulation.__call__` and
        :meth:`~inversion_ideas.WrappedSimulation.jacobian` methods.
        Default to True.
    """

    def __init__(
        self, simulation, *, store_jacobian=False, cache_fields=True, cache=True
    ):
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
        self.cache_fields = cache_fields
        self.cache = cache

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

    @cache_on_model
    def __call__(self, model: Model) -> npt.NDArray[np.float64]:
        """
        Evaluate simulation for a given model.

        Parameters
        ----------
        model : (n_params) array
            Array with model parameters.

        Returns
        -------
        (n_data) array
            Array with predicted data for the given model.
        """
        fields = self._get_fields(model)
        return self.simulation.dpred(model, f=fields)

    @cache_on_model
    def jacobian(self, model: Model) -> npt.NDArray[np.float64] | LinearOperator:
        """
        Jacobian matrix for a given model.

        Parameters
        ----------
        model : (n_params) array
            Array with model parameters.

        Returns
        -------
        (n_data, n_params) array or LinearOperator
            2D dense or sparse array or a :class:`~scipy.sparse.linalg.LinearOperator`
            that represents the Jacobian matrix: the matrix of first derivatives of the
            forward model with respect to each model parameter.
            If ``store_jacobian`` is True, a dense or sparse array will be returned.
            Otherwise, a :class:`~scipy.sparse.linalg.LinearOperator` will be returned.
        """
        fields = self._get_fields(model)
        if self.store_jacobian:
            jac = self.simulation.getJ(model, f=fields)
        else:
            jac = LinearOperator(
                shape=(self.n_data, self.n_params),
                dtype=np.float64,
                matvec=lambda v: self.simulation.Jvec(model, v, f=fields),
                rmatvec=lambda v: self.simulation.Jtvec(model, v, f=fields),
            )
        return jac

    @property
    def _is_pde_simulation(self):
        """Whether the SimPEG simulation is a PDE simulation or not."""
        return isinstance(self.simulation, BasePDESimulation)

    def _get_fields(self, model):
        """
        Return fields computed for a given model.

        If ``cache_fields`` is True, this method will cache the fields based on the
        model hash. By using it we can avoid recomputing the fields for the
        same model whenever we are calling ``dpred``, ``Jvec``, or ``Jtvec``.

        Parameters
        ----------
        model : (n_params) array

        Returns
        -------
        fields : simpeg.fields.Fields or None
            Computed fields objects. Return None if the ``simulation`` is not a PDE
            simulation (like integral gravity and magnetic simulations).
        """
        # Return None for non PDE simulations
        if not self._is_pde_simulation:
            return None

        # Do not cache fields if cache_fields is False
        if not self.cache_fields:
            return self.simulation.fields(model)

        model_hash = compute_hash(model)
        if hasattr(self, cache_attr := "_cached_fields"):
            cached_hash, cached_fields = getattr(self, cache_attr)
            if cached_hash.digest() == model_hash.digest():
                # -- Debug log --
                msg = (
                    f"{type(self).__name__}: reusing cached fields in '{self}' for "
                    f" model {array_to_str(model)} with hash "
                    f"'{hash_to_str(model_hash)}'."
                )
                get_logger().debug(msg)
                # ---
                return cached_fields

        # Compute new fields and cache them
        fields = self.simulation.fields(model)
        setattr(self, cache_attr, (model_hash, fields))
        # -- Debug log --
        msg = (
            f"{type(self).__name__}: computed and cached fields in '{self}' "
            f"for model {array_to_str(model)} with hash "
            f"'{hash_to_str(model_hash)}'."
        )
        get_logger().debug(msg)
        # ---
        return fields
