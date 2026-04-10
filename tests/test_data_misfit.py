"""
Test the ``DataMisfit`` class.
"""

import re

import numpy as np
import pytest
import scipy.sparse as sp

from inversion_ideas import DataMisfit

from .utils import LinearRegressor, assert_allclose_linear_operators


class TestDataMisfit:
    """
    Test the DataMisfit class.
    """

    n_params = 10
    n_data = 25
    rng = np.random.default_rng(seed=42)

    @pytest.fixture
    def true_model(self):
        return self.rng.uniform(size=10)

    @pytest.fixture
    def data_and_uncertainties(self, regressor_matrix, true_model):
        """Synthetic data and uncertainties."""
        synthetic_data = regressor_matrix @ true_model
        std = 1e-2 * np.max(np.abs(synthetic_data))
        noise = self.rng.normal(scale=std, size=synthetic_data.size)
        synthetic_data += noise
        uncertainties = np.full_like(synthetic_data, fill_value=std)
        return synthetic_data, uncertainties

    @pytest.fixture
    def regressor_matrix(self):
        shape = (self.n_data, self.n_params)
        return self.rng.uniform(size=self.n_data * self.n_params).reshape(shape)

    @pytest.mark.parametrize(
        "jacobian_as_linop", [False, True], ids=["dense-jac", "linop-jac"]
    )
    def test_hessian_approx(
        self, data_and_uncertainties, regressor_matrix, jacobian_as_linop
    ):
        """
        Test the ``hessian_approx`` method.
        """
        data, uncertainties = data_and_uncertainties

        # Define data misfit
        simulation = LinearRegressor(regressor_matrix, linop=jacobian_as_linop)
        data_misfit = DataMisfit(data, uncertainties, simulation)

        # Get approximated hessian
        model = self.rng.uniform(size=self.n_params)
        hessian_approx = data_misfit.hessian_approx(model)

        # Compare with expected one
        full_hessian = DataMisfit(
            data,
            uncertainties,
            simulation=LinearRegressor(regressor_matrix),
            build_hessian=True,
        ).hessian(model)
        expected = sp.diags_array(full_hessian.diagonal())
        assert_allclose_linear_operators(hessian_approx, expected, to_dense=True)

    def test_hessian_approx_with_dense_hessian(
        self, data_and_uncertainties, regressor_matrix
    ):
        """
        Test that ``hessian_approx`` returns the Hessian if ``build_hessian``.
        """
        data, uncertainties = data_and_uncertainties

        # Define data misfit
        simulation = LinearRegressor(regressor_matrix)
        data_misfit = DataMisfit(data, uncertainties, simulation, build_hessian=True)

        # Test if hessian and hessian_approx are the same
        model = self.rng.uniform(size=self.n_params)
        np.testing.assert_equal(
            data_misfit.hessian(model), data_misfit.hessian_approx(model)
        )

    def test_hessian_error(self, data_and_uncertainties, regressor_matrix):
        """
        Test error if `build_hessian` is True and Jacobian is a linear operator.
        """
        data, uncertainties = data_and_uncertainties
        simulation = LinearRegressor(regressor_matrix, linop=True)
        data_misfit = DataMisfit(data, uncertainties, simulation, build_hessian=True)

        model = self.rng.uniform(size=self.n_params)
        msg = re.escape("Cannot build Hessian for DataMisfit")
        with pytest.raises(TypeError, match=msg):
            data_misfit.hessian(model)

    @pytest.mark.parametrize(
        "jacobian_as_linop", [False, True], ids=["dense-jac", "linop-jac"]
    )
    def test_hessian(self, data_and_uncertainties, regressor_matrix, jacobian_as_linop):
        """
        Compare dense Hessian vs Hessian as LinearOperator.
        """
        data, uncertainties = data_and_uncertainties

        # Define a baseline data misfit term: dense Jacobian, build the full hessian.
        data_misfit = DataMisfit(
            data,
            uncertainties,
            simulation=LinearRegressor(regressor_matrix),
            build_hessian=True,
        )

        # Define a test data misfit: do not build the hessian.
        data_misfit_test = DataMisfit(
            data,
            uncertainties,
            simulation=LinearRegressor(regressor_matrix, linop=jacobian_as_linop),
            build_hessian=False,
        )

        model = self.rng.uniform(size=self.n_params)
        assert_allclose_linear_operators(
            data_misfit.hessian(model), data_misfit_test.hessian(model)
        )
