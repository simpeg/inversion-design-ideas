"""
Functions to easily build commonly used objects in inversions.
"""

from .base import Combo
from .conditions import ChiTarget
from .directives import MultiplierCooler
from .inversion import Inversion, InversionLog, InversionLogRich


def create_standard_log(phi: Combo):
    r"""
    Define standard log for an inversion like :math:`\phi = \phi_d + \beta \phi_m`.
    """
    assert len(phi) == 2
    data_misfit = phi[0]
    regularization = phi[1]
    assert hasattr(regularization, "multiplier")

    columns = {
        "iter": lambda iteration, _: iteration,
        "beta": lambda _, __: regularization.multiplier,
        "phi_d": lambda _, model: data_misfit(model),
        "phi_m": lambda _, model: regularization.function(model),
        "beta * phi_m": lambda _, model: regularization(model),
        "phi": lambda _, model: phi(model),
        "chi": lambda _, model: data_misfit(model) / data_misfit.n_data,
    }
    return InversionLogRich(columns)


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

    # Inversion log
    columns = {
        "iter": lambda iteration, _: iteration,
        "beta": lambda _, __: regularization.multiplier,
        "phi_d": lambda _, model: data_misfit(model),
        "phi_m": lambda _, model: regularization.function(model),
        "phi": lambda _, model: objective_function(model),
        "chi": lambda _, model: data_misfit(model) / data_misfit.n_data,
    }
    inversion_log = InversionLog(columns)

    inversion = Inversion(
        objective_function,
        initial_model,
        optimizer,
        directives=directives,
        stopping_criteria=stopping_criteria,
        cache_models=cache_models,
        log=inversion_log,
    )
    return inversion
