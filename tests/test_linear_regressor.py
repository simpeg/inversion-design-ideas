"""
Test the :class:`inversion_ideas.LinearRegressor` simulation.
"""

import numpy as np
import pytest

from inversion_ideas import LinearRegressor

from .utils import derivative_test


@pytest.mark.parametrize("build_jacobian", [True, False], ids=["dense", "linop"])
def test_jacobian(build_jacobian):
    """Test the ``jacobian`` method of the ``LinearRegressor``."""
    n_data, n_params = 10, 15
    rng = np.random.default_rng(seed=41423)
    linear_regressor = LinearRegressor.create_random(
        n_data, n_params, seed=rng, build_jacobian=build_jacobian
    )
    model = rng.uniform(low=-1, high=1, size=n_params)
    delta_m = rng.normal(size=n_params)

    # Since the simulation is linear, the derivative test should hold up to precision
    # error, so let's use a very small relative tolerance
    rtol = 1e-12

    # Perform derivative test
    derivative_test(
        linear_regressor, linear_regressor.jacobian, model, delta_m, rtol=rtol
    )
