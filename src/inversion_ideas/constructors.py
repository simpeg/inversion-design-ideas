"""
Functions to easily build commonly used objects in inversions.
"""

from typing import Callable
from inversion_ideas.utils import get_jacobi_preconditioner
from .conditions import ChiTarget
from .directives import MultiplierCooler
from .inversion import Inversion


def create_inversion(
    data_misfit,
    model_norm,
    *,
    starting_beta,
    initial_model,
    optimizer,
    beta_cooling_factor=2.0,
    beta_cooling_rate=1,
    chi_target=1.0,
    cache_models=True,
    preconditioner="jacobi",
    update_preconditioner=True,
) -> Inversion:
    r"""
    Create inversion of the form :math:`\phi_d + \beta \phi_m`.
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
    kwargs = {}
    if preconditioner is not None:
        if isinstance(preconditioner, str):
            if preconditioner == "jacobi":
                if update_preconditioner:
                    preconditioner = (  # noqa: E731
                        lambda model: get_jacobi_preconditioner(
                            objective_function, model
                        )
                    )
                else:
                    preconditioner = get_jacobi_preconditioner(
                        objective_function, initial_model
                    )
            else:
                msg = f"Invalid preconditioner '{preconditioner}'."
                raise ValueError(msg)

        if update_preconditioner and not isinstance(preconditioner, Callable):
            msg = (
                f"Invalid preconditioner '{preconditioner}'"
                "When setting `update_preconditioner` to True, "
                "the `preconditioner` should be a callable."
            )
            raise TypeError(msg)
        if not update_preconditioner and isinstance(preconditioner, Callable):
            msg = (
                f"Invalid preconditioner '{preconditioner}'."
                f"Cannot set `update_preconditioner` to False and pass "
                "`preconditioner` as a Callable."
            )
            raise TypeError(msg)

        kwargs["preconditioner"] = preconditioner

    # Define inversion
    inversion = Inversion(
        objective_function,
        initial_model,
        optimizer,
        directives=directives,
        stopping_criteria=stopping_criteria,
        cache_models=cache_models,
        log=True,
        **kwargs,
    )
    return inversion
