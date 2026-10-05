"""
Test the ``DataMisfit`` class.
"""

import re

import numpy as np
import pytest

from inversion_ideas import DataMisfit, LinearRegressor

from .utils import (
    assert_allclose_linear_operators,
    derivative_convergence_test,
    derivative_test,
)


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
        "build_jacobian", [True, False], ids=["dense-jac", "linop-jac"]
    )
    def test_hessian_diagonal(
        self, data_and_uncertainties, regressor_matrix, build_jacobian
    ):
        """
        Test the ``hessian_diagonal`` method.
        """
        data, uncertainties = data_and_uncertainties

        # Define data misfit
        simulation = LinearRegressor(regressor_matrix, build_jacobian=build_jacobian)
        data_misfit = DataMisfit(
            data,
            uncertainties,
            simulation,
            # Enable estimation of hessian diagonal if jacobian is a linop
            estimate_hessian_diagonal=True,
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
        simulation = LinearRegressor(regressor_matrix, build_jacobian=False)
        data_misfit = DataMisfit(data, uncertainties, simulation, build_hessian=True)

        model = self.rng.uniform(size=self.n_params)
        msg = re.escape("Cannot build Hessian for DataMisfit")
        with pytest.raises(TypeError, match=msg):
            data_misfit.hessian(model)

    @pytest.mark.parametrize(
        "build_jacobian", [True, False], ids=["dense-jac", "linop-jac"]
    )
    def test_hessian(self, data_and_uncertainties, regressor_matrix, build_jacobian):
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
            simulation=LinearRegressor(regressor_matrix, build_jacobian=build_jacobian),
            build_hessian=False,
        )

        model = self.rng.uniform(size=self.n_params)
        assert_allclose_linear_operators(
            data_misfit.hessian(model), data_misfit_test.hessian(model)
        )

    @pytest.mark.parametrize("order", [1, 2], ids=["first-order", "second-order"])
    def test_derivative(self, data_and_uncertainties, regressor_matrix, order):
        """
        Test gradient and hessian by comparison with Taylor series expansion.
        """
        data, uncertainties = data_and_uncertainties
        data_misfit = DataMisfit(
            data,
            uncertainties,
            simulation=LinearRegressor(regressor_matrix),
            build_hessian=True,
        )
        rng = np.random.default_rng(seed=12312)
        model = rng.uniform(low=-1.0, high=1.0, size=self.n_params)

        # Define whether to test the gradient or the Hessian
        if order == 1:
            delta_m = rng.normal(scale=1e-4, size=self.n_params)
            function, derivative = data_misfit, data_misfit.gradient
        elif order == 2:
            delta_m = rng.normal(size=self.n_params)
            function, derivative = data_misfit.gradient, data_misfit.hessian
        else:
            raise ValueError()

        # Perform derivative test
        derivative_test(function, derivative, model, delta_m)

    def test_derivative_convergence(self, data_and_uncertainties, regressor_matrix):
        """
        Test gradient through a convergence test of Taylor series approximation.
        """
        data, uncertainties = data_and_uncertainties
        data_misfit = DataMisfit(
            data,
            uncertainties,
            simulation=LinearRegressor(regressor_matrix),
            build_hessian=True,
        )
        rng = np.random.default_rng(seed=12312)
        model = rng.uniform(low=-1.0, high=1.0, size=self.n_params)
        delta_m = rng.normal(size=self.n_params)
        derivative_convergence_test(data_misfit, data_misfit.gradient, model, delta_m)


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

    @pytest.mark.parametrize("offending_arg", ["data", "uncertainty", "simulation"])
    def test_wrong_size(self, offending_arg, regressor_matrix):
        if offending_arg == "simulation":
            x = self.rng.uniform(size=(self.n_data + 1, self.n_params))
            simulation = LinearRegressor(x)
            data = self.rng.uniform(size=self.n_data)
            uncertainty = self.rng.uniform(size=self.n_data)
        elif offending_arg == "data":
            data = self.rng.uniform(size=self.n_data + 1)
            uncertainty = self.rng.uniform(size=self.n_data)
            simulation = LinearRegressor(regressor_matrix)
        elif offending_arg == "uncertainty":
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

    @pytest.mark.parametrize("invalid_value", [np.nan, np.inf, "both"])
    @pytest.mark.parametrize("offending_arg", ["data", "uncertainty"])
    def test_nans_or_infs(self, invalid_value, offending_arg, regressor_matrix):
        data = self.rng.uniform(size=self.n_data)
        uncertainty = self.rng.uniform(size=self.n_data)
        simulation = LinearRegressor(regressor_matrix)

        # Contaminate offending argument with nan, inf, or both
        array = data if offending_arg == "data" else uncertainty
        if invalid_value == "both":
            array[4], array[5] = np.nan, np.inf
        else:
            array[5] = invalid_value

        msg = re.escape(f"Invalid `{offending_arg}` array with NaN values.")
        with pytest.raises(ValueError, match=msg):
            DataMisfit(data, uncertainty, simulation)

    def test_invalid_simulation(self):
        class NonSimulation:
            """
            Dummy class that doesn't implement the full interface of a Simulation.
            """

            @property
            def n_params(self):
                return 30

            @property
            def n_data(self):
                return 25

            def __call__(self, model):
                pass

        data = self.rng.uniform(size=self.n_data)
        uncertainty = self.rng.uniform(size=self.n_data)
        simulation = NonSimulation()

        msg = re.escape("Invalid `simulation` argument of type 'NonSimulation'.")
        with pytest.raises(TypeError, match=msg):
            DataMisfit(data, uncertainty, simulation)
