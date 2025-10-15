"""
Functions to easily build commonly used objects in inversions.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from .data_misfit import DataMisfit
from .base import Minimizer, Objective
from .conditions import ChiTarget, ObjectiveChanged
from .directives import IRLS, MultiplierCooler
from .inversion import Inversion
from .inversion_log import Column
from .preconditioners import JacobiPreconditioner
from .typing import Model, Preconditioner


def create_inversion(
    data_misfit: Objective,
    model_norm: Objective,
    *,
    starting_beta: float,
    initial_model: npt.NDArray[np.float64],
    minimizer: Minimizer | Callable[[Objective, Model], Model],
    beta_cooling_factor: float = 2.0,
    beta_cooling_rate: int = 1,
    chi_target: float = 1.0,
    max_iterations: int | None = None,
    cache_models: bool = True,
    preconditioner: Preconditioner | Callable[[Model], Preconditioner] | None = None,
) -> Inversion:
    r"""
    Create inversion of the form :math:`\phi_d + \beta \phi_m`.

    Build an inversion with a beta cooling schedule and a stopping criteria for a chi
    factor target.

    Parameters
    ----------
    data_misfit : Objective
        Data misfit term :math:`\phi_d`.
    model_norm : Objective
        Model norm :math:`\phi_m`.
    starting_beta : float
        Starting value for the trade-off parameter :math:`\beta`.
    initial_model : (n_params) array
        Initial model to use in the inversion.
    minimizer : Minimizer
        Instance of :class:`Minimizer` used to minimize the objective function during
        the inversion.
    beta_cooling_factor : float, optional
        Cooling factor for the trade-off parameter :math:`\beta`. Every
        ``beta_cooling_rate`` iterations, the :math:`\beta` will be _cooled down_ by
        dividing it by the ``beta_cooling_factor``.
    beta_cooling_rate : int, optional
        Cooling rate for the trade-off parameter :math:`\beta`. The trade-off parameter
        will be cooled down every ``beta_cooling_rate`` iterations.
    chi_target : float, optional
        Target for the chi factor. The inversion will finish after the data misfit
        reaches a :math:`\chi` factor lower or equal to ``chi_target``.
    max_iterations : int, optional
        Max amount of iterations that will be performed. If ``None``, then there will be
        no limit on the total amount of iterations.
    cache_models : bool, optional
        Whether to cache models after each iteration in the inversion.
    preconditioner : {"jacobi"} or 2d array or sparray or LinearOperator or callable or None, optional
        Preconditioner that will be passed to the ``minimizer`` on every call during the
        inversion. The preconditioner can be a predefined 2d array, a sparse array or
        a LinearOperator. Alternatively, it can be a callable that takes the ``model``
        as argument and returns a preconditioner matrix (same types listed before). If
        ``"jacobi"``, a default Jacobi preconditioner that will get updated on every
        iteration will be defined for the inversion. If None, no preconditioner will be
        passed.

    Returns
    -------
    Inversion
    """
    # Define objective function
    regularization = starting_beta * model_norm
    objective_function = data_misfit + regularization

    # Define directives
    directives = [
        MultiplierCooler(
            regularization,
            cooling_factor=beta_cooling_factor,
            cooling_rate=beta_cooling_rate,
        ),
    ]

    # Stopping criteria
    stopping_criteria = ChiTarget(data_misfit, chi_target=chi_target)

    # Preconditioner
    minimizer_kwargs = {}
    if preconditioner is not None:
        if isinstance(preconditioner, str):
            if preconditioner == "jacobi":
                preconditioner = JacobiPreconditioner(objective_function)
            else:
                msg = f"Invalid preconditioner '{preconditioner}'."
                raise ValueError(msg)
        minimizer_kwargs["preconditioner"] = preconditioner

    # Define inversion
    inversion = Inversion(
        objective_function,
        initial_model,
        minimizer,
        directives=directives,
        stopping_criteria=stopping_criteria,
        cache_models=cache_models,
        max_iterations=max_iterations,
        log=True,
        minimizer_kwargs=minimizer_kwargs,
    )
    return inversion


def create_sparse_inversion(
    data_misfit: DataMisfit,
    model_norm: Objective,
    *,
    starting_beta: float,
    initial_model: Model,
    minimizer: Minimizer | Callable[[Objective, Model], Model],
    beta_cooling_factor: float = 2.0,
    data_misfit_rtol=1e-1,
    chi_l2_target: float = 1.0,
    model_norm_rtol: float = 1e-3,
    max_iterations: int | None = None,
    cache_models: bool = True,
    preconditioner: Preconditioner | Callable[[Model], Preconditioner] | None = None,
) -> Inversion:
    r"""
    Create sparse norm inversion of the form: :math:`\phi_d + \beta \phi_m`.

    Build an inversion where :math:`\phi_m` is a sparse norm regularization.
    An IRLS algorithm will be applied, split in two stages.
    The inversion will stop when the following inequality holds:

    .. math::

        \frac{|\phi_m^{(k)} - \phi_m^{(k-1)}|}{|\phi_m^{(k-1)}|} <  \eta_{\phi_m}

    where :math:`\eta_{\phi_m}` is the ``model_norm_rtol``.

    Parameters
    ----------
    data_misfit : Objective
        Data misfit term :math:`\phi_d`.
    model_norm : Objective
        Model norm :math:`\phi_m`. It can be a single objective function term or a combo
        containing multiple ones. At least one of them should be a sparse regularization
        term.
    starting_beta : float
        Starting value for the trade-off parameter :math:`\beta`.
    initial_model : (n_params) array
        Initial model to use in the inversion.
    minimizer : Minimizer
        Instance of :class:`Minimizer` used to minimize the objective function during
        the inversion.
    beta_cooling_factor : float, optional
        Cooling factor for the trade-off parameter :math:`\beta`. Every
        ``beta_cooling_rate`` iterations, the :math:`\beta` will be _cooled down_ by
        dividing it by the ``beta_cooling_factor``.
    data_misfit_rtol : float, optional
        Tolerance for the data misfit. This value is used to determine whether to cool
        down the IRLS threshold or beta. See eq. 21 in Fournier and Oldenburg (2019).
    chi_l2_target : float, optional
        Chi factor target for the stage one (the L2 inversion). Once this chi target is
        reached, the second stage starts.
    model_norm_rtol : float, optional
        Tolerance for the model norm. This value is used to determine if the inversion
        should stop. See eq. 22 in Fournier and Oldenburg (2019).
    max_iterations : int, optional
        Max amount of iterations that will be performed. If ``None``, then there will be
        no limit on the total amount of iterations.
    cache_models : bool, optional
        Whether to cache models after each iteration in the inversion.
    preconditioner : {"jacobi"} or 2d array or sparray or LinearOperator or callable or None, optional
        Preconditioner that will be passed to the ``minimizer`` on every call during the
        inversion. The preconditioner can be a predefined 2d array, a sparse array or
        a LinearOperator. Alternatively, it can be a callable that takes the ``model``
        as argument and returns a preconditioner matrix (same types listed before). If
        ``"jacobi"``, a default Jacobi preconditioner that will get updated on every
        iteration will be defined for the inversion. If None, no preconditioner will be
        passed.

    Returns
    -------
    Inversion
    """
    # Define objective function
    regularization = starting_beta * model_norm
    objective_function = data_misfit + regularization

    # Define IRLS directive
    directives = [
        IRLS(
            regularization,
            data_misfit=data_misfit,
            chi_l2_target=chi_l2_target,
            beta_cooling_factor=beta_cooling_factor,
            data_misfit_rtol=data_misfit_rtol,
        )
    ]

    # Stopping criteria
    smallness_not_changing = ObjectiveChanged(model_norm, rtol=model_norm_rtol)

    # Preconditioner
    minimizer_kwargs = {}
    if preconditioner is not None:
        if isinstance(preconditioner, str):
            if preconditioner == "jacobi":
                preconditioner = JacobiPreconditioner(objective_function)
            else:
                msg = f"Invalid preconditioner '{preconditioner}'."
                raise ValueError(msg)
        minimizer_kwargs["preconditioner"] = preconditioner

    # Define inversion
    inversion = Inversion(
        objective_function,
        initial_model,
        minimizer,
        directives=directives,
        stopping_criteria=smallness_not_changing,
        cache_models=cache_models,
        max_iterations=max_iterations,
        log=True,
        minimizer_kwargs=minimizer_kwargs,
    )

    # Add extra columns to log
    if inversion.log is not None:
        # TODO: fix this in case that model norm is a combo
        inversion.log.add_column(
            "IRLS", lambda _, __: "active" if model_norm.irls else "inactive"
        )
        inversion.log.add_column(
            "IRLS threshold",
            Column(
                title="ε",
                callable=lambda _, __: model_norm.threshold,
                fmt=None,
            ),
        )
        inversion.log.add_column(
            "model_norm_relative_diff",
            Column(
                title=r"|φm_(k) - φm_(k-1)|/|φm_(k-1)|",
                callable=lambda _, model: smallness_not_changing.ratio(model),
                fmt=None,
            ),
        )
    return inversion
