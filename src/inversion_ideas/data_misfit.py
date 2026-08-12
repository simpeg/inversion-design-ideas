"""
Class to represent a data misfit term.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import dia_array, diags_array
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from inversion_ideas.utils import get_logger

from .base import Objective, Simulation
from .operators import get_diagonal
from .typing import Model, SimulationProtocol, SparseArray


class DataMisfit(Objective):
    r"""
    L2 data misfit.

    Parameters
    ----------
    data : (n_data) array
        Array with observed data values.
    uncertainty : (n_data) array
        Array with data uncertainty.
    simulation : Simulation
        Instance of Simulation.
    build_hessian : bool, optional
        If True, the :meth:`~inversion_ideas.DataMisfit.hessian` method will build the
        Hessian matrix and allocate it in memory.
        If False, the :meth:`~inversion_ideas.DataMisfit.hessian` method will return a
        :class:`~scipy.sparse.linalg.LinearOperator` that represents the Hessian matrix.
        Default to False.

        .. warning::

            Hessian matrices are usually very large. Use ``build_hessian=True`` only if
            you need to build it.

    estimate_hessian_diagonal : bool, optional
        If True, the :meth:`~inversion_ideas.DataMisfit.hessian_diagonal` method will
        estimate the diagonal of the Hessian, even if the Jacobian of the
        ``simulation`` is a :class:`~scipy.sparse.linalg.LinearOperator`.
        If False, an error will be raised when calling the
        :meth:`~inversion_ideas.DataMisfit.hessian_diagonal` method
        in case the Jacobian of the ``simulation`` is a
        :class:`~scipy.sparse.linalg.LinearOperator`.

        .. important::

            Estimating the diagonal of a :class:`~scipy.sparse.linalg.LinearOperator`
            require a high number of dot products between the operator and multiple unit
            vectors. This might lead to very expensive computations. Enable
            ``estimate_hessian_diagonal`` only if you really need to.

    Notes
    -----
    The L2 data misfit objective function is defined as:

    .. math::

        \phi_d(\mathbf{m}) =
        \sum\limits_{i=1}^N
        \frac{
            \left\lvert f_i(\mathbf{m}) - d_i^\text{obs} \right\rvert^2
        }{
            \epsilon_i^2
        }

    where :math:`\mathbf{m}` is the model vector, :math:`d_i^\text{obs}` is the
    :math:`i`-th observed datum, :math:`f_i(\mathbf{m})` is the forward modelling
    function for the :math:`i`-th datum, and :math:`\epsilon_i` is the uncertainty of
    the :math:`i`-th datum.

    The data misfit term can be expressed in terms of weights :math:`w_i
    = 1 / \epsilon_i^2`:

    .. math::

        \phi_d(\mathbf{m}) =
        \sum\limits_{i=1}^N
        w_i \left\lvert f_i(\mathbf{m}) - d_i^\text{obs} \right\rvert^2

    And also in matrix form:

    .. math::

        \phi_d(\mathbf{m}) =
        \left\lVert
        \mathbf{W} \left[ \mathbf{f}(\mathbf{m}) - \mathbf{d}^\text{obs} \right]
        \right\rVert^2

    where :math:`\mathbf{W}` is a diagonal matrix with the square root of the weights

    .. math::

        \mathbf{W} =
        \begin{bmatrix}
            \sqrt{w_1} & & 0 \\
            & \ddots &  \\
            0 & & \sqrt{w_N} \\
        \end{bmatrix},

    :math:`\mathbf{d}^\text{obs}` is the vector of observed data

    .. math::

        \mathbf{d}^\text{obs} =
        \begin{bmatrix}
        d_1^\text{obs} \\
        \vdots \\
        d_N^\text{obs} \\
        \end{bmatrix},

    and :math:`\mathbf{f}(\mathbf{m})` is the forward modelling vector

    .. math::

        \mathbf{f}(\mathbf{m}) =
        \begin{bmatrix}
            f_1(\mathbf{m}) \\
            \vdots \\
            f_N(\mathbf{m}) \\
        \end{bmatrix}.

    """

    def __init__(
        self,
        data: npt.NDArray[np.float64],
        uncertainty: npt.NDArray[np.float64],
        simulation: Simulation,
        *,
        build_hessian=False,
        estimate_hessian_diagonal=False,
    ):
        # Validate inputs
        if data.ndim != 1:
            msg = (
                f"Invalid `data` array with {data.ndim} dimensions. "
                "It must be a 1D array."
            )
            raise ValueError(msg)
        if uncertainty.ndim != 1:
            msg = (
                f"Invalid `uncertainty` array with {uncertainty.ndim} dimensions. "
                "It must be a 1D array."
            )
            raise ValueError(msg)
        if not isinstance(simulation, SimulationProtocol):
            msg = (
                "Invalid `simulation` argument of type "
                f"'{type(simulation).__name__}'. "
                "It must be a child of `inversion_ideas.base.Simulation` or "
                "a custom object that implements its interface."
            )
            raise TypeError(msg)
        if not (simulation.n_data == data.size == uncertainty.size):
            msg = (
                f"Invalid `data` and `uncertainty` arguments with {data.size} and "
                f"{uncertainty.size} elements, respectively, and `simulation` "
                f"argument with {simulation.n_data} 'n_params'. "
            )
            raise ValueError(msg)
        if np.any(np.isnan(data)):
            msg = "Invalid `data` array with NaN values."
            raise ValueError(msg)
        if np.any(np.isnan(uncertainty)):
            msg = "Invalid `uncertainty` array with NaN values."
            raise ValueError(msg)

        self.data = data
        self.uncertainty = uncertainty
        self.simulation = simulation
        self.build_hessian = build_hessian
        self.estimate_hessian_diagonal = estimate_hessian_diagonal
        self.set_name("d")

    def __call__(self, model: Model) -> float:
        r"""
        Evaluate the data misfit function.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        float
            Value of the data misfit for the given model.

        Notes
        -----
        Evaluates the data misfit objective function defined as:

        .. math::

            \phi_d(\mathbf{m}) =
            \sum\limits_{i=1}^N
            \frac{
                \left\lvert f_i(\mathbf{m}) - d_i^\text{obs} \right\rvert^2
            }{
                \epsilon_i^2
            }

        where :math:`\mathbf{m}` is the model vector, :math:`d_i^\text{obs}` is the
        :math:`i`-th observed datum, :math:`f_i(\mathbf{m})` is the forward modelling
        function for the :math:`i`-th datum, and :math:`\epsilon_i` is the uncertainty of
        the :math:`i`-th datum.
        """
        residual = self.residual(model)
        weights_matrix = self.weights_matrix
        return residual.T @ weights_matrix.T @ weights_matrix @ residual

    def gradient(self, model: Model) -> npt.NDArray[np.float64]:
        r"""
        Gradient vector of the data misfit function.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        (n_params,) array
            Gradient vector of the data misfit for the given model.

        Notes
        -----
        Computes the gradient of the data misfit as:

        .. math::

            \nabla\phi_d(\mathbf{m}) =
            2
            \mathbf{J}^\text{T}
            \mathbf{W}^\text{T}
            \mathbf{W}
            \mathbf{J}
            \left[ \mathbf{f}(\mathbf{m}) - \mathbf{d}^\text{obs} \right],

        where :math:`\mathbf{J}` is the Jacobian matrix of the ``simulation`` (the forward model),
        :math:`\mathbf{W}` is a diagonal matrix with the square root of the weights,
        :math:`\mathbf{d}^\text{obs}` is the vector of observed data, and
        :math:`\mathbf{f}(\mathbf{m})` is the forward modelling vector.
        """
        jac = self.simulation.jacobian(model)
        weights_matrix = self.weights_matrix
        return 2 * jac.T @ (weights_matrix.T @ weights_matrix @ self.residual(model))

    def hessian(
        self, model: Model
    ) -> npt.NDArray[np.float64] | SparseArray | LinearOperator:
        r"""
        Evaluate the hessian of the data misfit function for a given model.

        .. important::

            If ``build_hessian`` is set to True, this method will attempt to return a 2D dense or sparse array. If it's False, it'll return a
            :class:`~scipy.sparse.linalg.LinearOperator`.

        .. warning::

            Hessian matrices are usually very large. Use ``build_hessian=True`` only if
            you need to build it.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        (n_params, n_params) array or :class:`~scipy.sparse.linalg.LinearOperator`
            2D array or :class:`~scipy.sparse.linalg.LinearOperator` that represents
            the Hessian matrix of the objective funciton, or an approximate version
            of it.

        Notes
        -----
        Computes the Hessian matrix of the data misfit as:

        .. math::

            \bar{\bar{\nabla}} \phi_d(\mathbf{m}) =
            2 \mathbf{J}^\text{T} \mathbf{W}^\text{T} \mathbf{W} \mathbf{J},

        where :math:`\mathbf{J}` is the Jacobian matrix of the ``simulation`` (the forward model), and
        :math:`\mathbf{W}` is a diagonal matrix with the square root of the weights.
        """
        jac = self.simulation.jacobian(model)

        if self.build_hessian and isinstance(jac, LinearOperator):
            msg = (
                f"Cannot build Hessian for DataMisfit '{self}' since the Jacobian "
                f"of {self.simulation} is a LinearOperator. "
                f"Set `build_hessian` to False in '{self}', or adjust your "
                "simulation to return a dense or sparse Jacobian matrix."
            )
            raise TypeError(msg)

        if not self.build_hessian:
            jac = aslinearoperator(jac)
        weights_matrix = aslinearoperator(self.weights_matrix)
        return 2 * jac.T @ weights_matrix.T @ weights_matrix @ jac

    def hessian_diagonal(self, model: Model) -> npt.NDArray[np.float64]:
        """
        Get the main diagonal of the Hessian.

        .. important::

            If the Jacobian of the ``simulation`` is
            a :class:`~scipy.sparse.linalg.LinearOperator`, the diagonal of the Hessian
            will be estimated only if ``estimate_hessian_diagonal`` is True.
            Diagonal estimations can be expensive computational tasks for large
            problems. Make sure you to enable diagonal estimations only if you need it.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        (n_params,) array
            Array containing the diagonal of the Hessian.
        """
        if self.build_hessian:
            return self.hessian(model).diagonal()

        jac = self.simulation.jacobian(model)
        if isinstance(jac, LinearOperator):
            if not self.estimate_hessian_diagonal:
                msg = (
                    f"Cannot estimate diagonal of Hessian for '{self}' since it "
                    "has `estimate_hessian_diagonal` set to False. "
                    "Set `estimate_hessian_diagonal` to True to allow diagonal "
                    "estimation, or make sure your simulation returns a dense or "
                    "sparse Jacobian matrix."
                )
                raise AttributeError(msg)

            # Repeat hessian implementation here to avoid recomputing the jacobian
            weights_matrix = aslinearoperator(self.weights_matrix)
            hessian = 2 * jac.T @ weights_matrix.T @ weights_matrix @ jac

            # -- Debug --
            get_logger().debug(
                f"Estimating diagonal of the Hessian matrix of '{self}'."
            )
            # ---

            # TODO: Extend algorithms for estimating the diagonal. Add keyword arguments
            #       to the constructor of the DataMisfit to choose method and set
            #       parameters.

            # Compute the diagonal.
            diagonal = get_diagonal(hessian)
        else:
            diagonal = 2 * np.einsum("i,ij,ij->j", self.weights, jac, jac)
        return diagonal

    @property
    def n_params(self):
        """
        Number of model parameters.
        """
        return self.simulation.n_params

    @property
    def n_data(self):
        """
        Number of data values.
        """
        return self.data.size

    def residual(self, model: Model):
        r"""
        Residual vector.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        (n_data) array
            Array with residual vector.

        Notes
        -----
        Residual vector defined as:

        .. math::

            \mathbf{r} = \mathbf{f}(\mathbf{m}) - \mathbf{d}

        where :math:`\mathbf{d}` is the vector with observed data, :math:`\mathbf{f}`
        is the forward model, and :math:`\mathbf{m}` is the model vector.
        """
        return self.simulation(model) - self.data

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """
        Data weights: 1D array with the square of the inverse of the uncertainties.
        """
        return 1 / self.uncertainty**2

    @property
    def weights_matrix(self) -> dia_array:
        """
        Diagonal matrix with the square root of the regularization weights.
        """
        return diags_array(1 / self.uncertainty)

    def chi_factor(self, model: Model):
        r"""
        Compute the chi factor for the given model.

        Parameters
        ----------
        model : (n_params,) array
            Array with model values.

        Returns
        -------
        float
            Chi factor for the given model.

        Notes
        -----
        The chi factor of a data misfit term :math:`\phi_d(\mathbf{m})` is defined as a
        function of the model vector :math:`\mathbf{m}` as follows:

        .. math::

            \chi(\mathbf{m})
            = \frac{\phi_d(\mathbf{m})}{N}
            = \frac{1}{N}
              \sum\limits_{i=1}^N
              \frac{
                  \left\lvert f_i(\mathbf{m}) - d_i^\text{obs} \right\rvert^2
              }{
                  \epsilon_i^2
              }

        References
        ----------
        - https://www.eoas.ubc.ca/courses/eosc350/content/tutorials/glossary.htm
        - https://giftoolscookbook.readthedocs.io/en/latest/content/fundamentals/Beta.html#chi-factor
        """
        return self(model) / self.n_data
