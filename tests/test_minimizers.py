"""
Test minimizer classes and functions.
"""

import numpy as np
import pytest

from inversion_ideas import DataMisfit, SimpleSmallness
from inversion_ideas.minimize import (
    GaussNewtonConjugateGradient,
    backtracking_line_search,
)

from .utils import Dummy, NonLinearRegressor


class TestGaussNewtonConjugateGradient:
    """
    Test the ``GaussNewtonConjugateGradient`` minimizer.
    """

    n_data = 15
    n_params = 10

    @pytest.fixture
    def true_model(self):
        true_model = np.linspace(1, self.n_params, self.n_params, dtype=np.float64)
        return true_model

    @pytest.fixture
    def simulation(self):
        rng = np.random.default_rng(seed=19723)
        a_matrix = rng.uniform(low=-1.0, high=1.0, size=(self.n_data, self.n_params))
        b_matrix = np.zeros((self.n_data, self.n_params))
        simulation = NonLinearRegressor(a_matrix, b_matrix)
        return simulation

    def get_objective_function(self, simulation, true_model):
        std = 0.1
        rng = np.random.default_rng(seed=19882)
        data = simulation(true_model) + rng.normal(scale=std, size=self.n_data)
        uncertainties = np.full_like(data, fill_value=std)
        data_misfit = DataMisfit(data, uncertainties, simulation)
        smallness = SimpleSmallness(self.n_params)
        phi = data_misfit + 10 * smallness
        return phi

    def test_minimizer(self, simulation, true_model):
        phi = self.get_objective_function(simulation, true_model)
        minimizer = GaussNewtonConjugateGradient()
        initial_model = np.ones(self.n_params)
        models = list(minimizer(phi, initial_model))

        # Check that we run more than one gauss-newton iteration
        assert len(models) >= 2

        # Check the inverted model is close enough to the true model
        inverted_model = models[-1]
        np.testing.assert_allclose(inverted_model, true_model, rtol=0.02)

    @pytest.mark.parametrize("argument", ["rtol", "atol"])
    def test_invalid_cg_kwargs(self, argument):
        """
        Test error after passing 'rtol' or 'atol' through 'cg_kwargs'.
        """
        cg_kwargs = {argument: 0.1}
        with pytest.raises(KeyError, match=f"Cannot pass '{argument}'"):
            GaussNewtonConjugateGradient(cg_kwargs=cg_kwargs)


class TestBacktrackingLineSearch:
    """Test the ``backtracking_line_search`` function."""

    def test_invalid_contraction_factor(self):
        phi = Dummy(3)
        model = np.zeros(phi.n_params)
        search_direction = np.ones(phi.n_params)
        with pytest.raises(ValueError, match="Invalid contraction_factor value of"):
            backtracking_line_search(
                phi, model, search_direction, contraction_factor=1.5
            )

    def test_invalid_c_factor(self):
        phi = Dummy(3)
        model = np.zeros(phi.n_params)
        search_direction = np.ones(phi.n_params)
        with pytest.raises(ValueError, match="Invalid c_factor value of"):
            backtracking_line_search(phi, model, search_direction, c_factor=1.5)
