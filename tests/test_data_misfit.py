"""
Test the ``DataMisfit`` class.
"""

import re

import numpy as np
import pytest

from inversion_ideas import DataMisfit

from .utils import LinearRegressor, assert_allclose_linear_operators


class TestDataMisfit:
    """
    Test the DataMisfit class.

    Use a linear regressor as simulation to quickly test things out.
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
        return self.rng.uniform(size=shape)

    @pytest.mark.parametrize(
        "jacobian_as_linop", [False, True], ids=["dense-jac", "linop-jac"]
    )
    def test_hessian_diagonal(
        self, data_and_uncertainties, regressor_matrix, jacobian_as_linop
    ):
        """
        Test the ``hessian_diagonal`` method.
        """
        data, uncertainties = data_and_uncertainties

        # Define data misfit
        simulation = LinearRegressor(regressor_matrix, linop=jacobian_as_linop)
        data_misfit = DataMisfit(
            data,
            uncertainties,
            simulation,
            # Enable estimation of hessian diagonal if jacobian is a linop
            estimate_hessian_diagonal=jacobian_as_linop,
        )

        # Get diagonal of the hessian
        model = self.rng.uniform(size=self.n_params)
        hessian_diagonal = data_misfit.hessian_diagonal(model)

        # Compare with expected one
        expected = (
            DataMisfit(
                data,
                uncertainties,
                simulation=LinearRegressor(regressor_matrix),
                build_hessian=True,
            )
            .hessian(model)
            .diagonal()
        )

        np.testing.assert_allclose(hessian_diagonal, expected)

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


class TestSanityChecks:
    """Test sanity checks for arguments of ``DataMisfit``."""

    rng = np.random.default_rng(seed=42)
    n_data = 25
    n_params = 30

    @pytest.fixture
    def regressor_matrix(self):
        shape = (self.n_data, self.n_params)
        return self.rng.uniform(size=shape)

    def test_data_not_1d_array(self, regressor_matrix):
        data_2d = self.rng.uniform(size=self.n_data).reshape((5, 5))
        uncertainty = self.rng.uniform(size=self.n_data)
        simulation = LinearRegressor(regressor_matrix)
        msg = re.escape(
            "Invalid `data` array with 2 dimensions. It must be a 1D array."
        )
        with pytest.raises(ValueError, match=msg):
            DataMisfit(data_2d, uncertainty, simulation)

    def test_uncertainty_not_1d_array(self, regressor_matrix):
        data = self.rng.uniform(size=self.n_data)
        uncertainty_2d = self.rng.uniform(size=self.n_data).reshape((5, 5))
        simulation = LinearRegressor(regressor_matrix)
        msg = re.escape(
            "Invalid `uncertainty` array with 2 dimensions. It must be a 1D array."
        )
        with pytest.raises(ValueError, match=msg):
            DataMisfit(data, uncertainty_2d, simulation)

    @pytest.mark.parametrize("offending", ["data", "uncertainty", "simulation"])
    def test_wrong_size(self, offending, regressor_matrix):
        if offending == "simulation":
            x = self.rng.uniform(size=(self.n_data + 1, self.n_params))
            simulation = LinearRegressor(x)
            data = self.rng.uniform(size=self.n_data)
            uncertainty = self.rng.uniform(size=self.n_data)
        elif offending == "data":
            data = self.rng.uniform(size=self.n_data + 1)
            uncertainty = self.rng.uniform(size=self.n_data)
            simulation = LinearRegressor(regressor_matrix)
        elif offending == "uncertainty":
            data = self.rng.uniform(size=self.n_data)
            uncertainty = self.rng.uniform(size=self.n_data + 1)
            simulation = LinearRegressor(regressor_matrix)
        else:
            raise ValueError()
        msg = re.escape(
            f"Invalid `data` and `uncertainty` arguments with {data.size} and "
            f"{uncertainty.size} elements, respectively, and `simulation` "
            f"argument with {simulation.n_data} 'n_params'. "
        )
        with pytest.raises(ValueError, match=msg):
            DataMisfit(data, uncertainty, simulation)

    @pytest.mark.parametrize("offending", ["data", "uncertainty"])
    def test_nans(self, offending, regressor_matrix):
        data = self.rng.uniform(size=self.n_data)
        uncertainty = self.rng.uniform(size=self.n_data)
        simulation = LinearRegressor(regressor_matrix)
        if offending == "data":
            data[5] = np.nan
        elif offending == "uncertainty":
            uncertainty[5] = np.nan
        else:
            raise ValueError()
        msg = re.escape(f"Invalid `{offending}` array with NaN values.")
        with pytest.raises(ValueError, match=msg):
            DataMisfit(data, uncertainty, simulation)
