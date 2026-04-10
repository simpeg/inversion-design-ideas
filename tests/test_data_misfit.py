"""
Test the ``DataMisfit`` class.
"""

import numpy as np

from inversion_ideas import DataMisfit

from .utils import LinearRegressor


class TestDataMisfit:
    """
    Test the DataMisfit class.
    """

    n_params = 10
    n_data = 25

    def test_hessian_diagonal(self):
        """
        Test the ``hessian_diagonal`` method.
        """
        # Generate some random data
        rng = np.random.default_rng(seed=42)
        data = rng.uniform(size=self.n_data)
        uncertainties = 1e-2 * np.ones(self.n_data)

        # Build linear regressor
        shape = (self.n_data, self.n_params)
        X = rng.uniform(size=self.n_data * self.n_params).reshape(shape)
        simulation = LinearRegressor(X)

        # Define data misfit. Store full hessian for the test.
        data_misfit = DataMisfit(data, uncertainties, simulation, build_hessian=True)

        # Compare the true diagonal of the hessian with the one returned by
        # hessian_diagonal.
        model = rng.uniform(size=self.n_params)
        np.testing.assert_allclose(
            data_misfit.hessian(model).diagonal(), data_misfit.hessian_diagonal(model)
        )
