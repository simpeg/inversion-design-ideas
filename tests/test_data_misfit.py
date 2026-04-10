"""
Test the ``DataMisfit`` class.
"""

import numpy as np
import scipy.sparse as sp

from inversion_ideas import DataMisfit

from .utils import LinearRegressor, assert_allclose_linear_operators


class TestDataMisfit:
    """
    Test the DataMisfit class.
    """

    n_params = 10
    n_data = 25

    def test_hessian_approx(self):
        """
        Test the ``hessian_approx`` method.
        """
        # Generate some random data
        rng = np.random.default_rng(seed=42)
        data = rng.uniform(size=self.n_data)
        uncertainties = 1e-2 * np.ones(self.n_data)

        # Build linear regressor
        shape = (self.n_data, self.n_params)
        X = rng.uniform(size=self.n_data * self.n_params).reshape(shape)
        simulation = LinearRegressor(X)

        # Define data misfit
        data_misfit = DataMisfit(data, uncertainties, simulation)

        # Get approximated hessian
        model = rng.uniform(size=self.n_params)
        hessian_approx = data_misfit.hessian_approx(model)

        # Compare with expected one
        full_hessian = DataMisfit(
            data, uncertainties, simulation, build_hessian=True
        ).hessian(model)
        expected = sp.diags_array(full_hessian.diagonal())
        assert_allclose_linear_operators(hessian_approx, expected, to_dense=True)
