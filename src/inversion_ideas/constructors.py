"""
Functions to easily build commonly used objects in inversions.
"""

from .conditions import ChiTarget
from .directives import MultiplierCooler
from .inversion import Inversion, InversionLogRich


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

    inversion = Inversion(
        objective_function,
        initial_model,
        optimizer,
        directives=directives,
        stopping_criteria=stopping_criteria,
        cache_models=cache_models,
        log=True,
    )
    return inversion
