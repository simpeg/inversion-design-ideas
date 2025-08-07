"""
Common inversion recipes.
"""

from .constructors import create_standard_log
from .directives import MultiplierCooler
from .inversion import Inversion


def simple_inversion(
    data_misfit,
    model_norm,
    *,
    initial_model,
    minimizer,
    stopping_criteria,
    starting_beta,
    beta_cooling_factor,
    cache_models=True,
    **kwargs,
):
    # Objective function
    regularization = starting_beta * model_norm
    phi = data_misfit + regularization

    # Beta cooler
    beta_cooler = MultiplierCooler(cooling_factor=beta_cooling_factor)

    # Inversion log
    inversion_log = create_standard_log(phi)

    # Define inversion
    kwargs["cache_models"] = cache_models
    inversion = Inversion(
        phi,
        initial_model,
        minimizer,
        stopping_criteria=stopping_criteria,
        log=inversion_log,
        **kwargs,
    )

    with inversion.log.show_live() as live:
        for model in inversion:
            # Cool down beta
            beta_cooler(regularization)

            # Refresh table
            live.refresh()

            yield model
