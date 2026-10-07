"""
Test custom hyperparameter objects.
"""

import numpy as np
import pytest

from inversion_ideas.hyperparams import CooledMultiplier, SensitivityWeights
from inversion_ideas.utils import get_sensitivity_weights

from .utils import NonLinearRegressor


class TestCooledMultiplier:
    """Test the ``CooledMultiplier`` class."""

    def test_initial_value(self):
        initial, cooling_factor = 10.0, 2.0
        multiplier = CooledMultiplier(initial, cooling_factor=cooling_factor)
        assert multiplier.initial_value == initial
        # Test error when trying to modify the initial_value property
        with pytest.raises(AttributeError, match="has no setter"):
            multiplier.initial_value = 20.0

    @pytest.mark.parametrize("mutable", [True, False])
    def test_immutability(self, mutable):
        initial, cooling_factor = 10.0, 2.0
        multiplier = CooledMultiplier(
            initial, cooling_factor=cooling_factor, mutable=mutable
        )
        if mutable:
            multiplier.value = 30.0
            assert multiplier.value == 30.0
            multiplier.update()
            assert multiplier.value == 15.0
        else:
            with pytest.raises(TypeError):
                multiplier.value = 30.0

    @pytest.mark.parametrize("cooling_factor", [2.0, 3.0])
    def test_update(self, cooling_factor):
        """Test the update method."""
        initial = 10.0
        multiplier = CooledMultiplier(initial, cooling_factor=cooling_factor)
        multiplier.update()
        assert multiplier == initial / cooling_factor
        assert multiplier.value == initial / cooling_factor

    @pytest.mark.parametrize("cooling_factor", [1.5, 2.0])
    def test_multiple_updates(self, cooling_factor):
        """Test multiple calls to the update method."""
        initial = 10.0
        multiplier = CooledMultiplier(initial, cooling_factor=cooling_factor)
        n = 8
        for _ in range(n):
            multiplier.update()
        np.testing.assert_allclose(float(multiplier), initial / cooling_factor**n)

    def test_cooling_rate(self):
        """Test cooling_rate behaviour."""
        initial, cooling_factor = 10.0, 2.0
        cooling_rate = 2
        multiplier = CooledMultiplier(
            initial, cooling_factor=cooling_factor, cooling_rate=cooling_rate
        )
        # Test the first call: should cool down
        multiplier.update()
        assert multiplier == initial / cooling_factor
        assert multiplier.value == initial / cooling_factor

        # Test the second call: should NOT cool down
        multiplier.update()
        assert multiplier == initial / cooling_factor
        assert multiplier.value == initial / cooling_factor

        # Test the third call: should cool down
        multiplier.update()
        assert multiplier == initial / cooling_factor**2
        assert multiplier.value == initial / cooling_factor**2

    def test_invalid_cooling_factor_type(self):
        class NonReal: ...  # pragma: nocover

        cooling_factor = NonReal()
        with pytest.raises(
            TypeError, match=f"Invalid cooling_factor '{cooling_factor}'"
        ):
            CooledMultiplier(2.0, cooling_factor=cooling_factor)

    def test_invalid_cooling_rate_type(self):
        cooling_rate = 3.14
        with pytest.raises(TypeError, match=f"Invalid cooling_rate '{cooling_rate}'"):
            CooledMultiplier(2.0, cooling_factor=2.3, cooling_rate=cooling_rate)

    def test_invalid_cooling_rate_value(self):
        cooling_rate = -3
        with pytest.raises(ValueError, match=f"Invalid cooling_rate '{cooling_rate}'"):
            CooledMultiplier(2.0, cooling_factor=2.3, cooling_rate=cooling_rate)

    def test_reset(self):
        initial, factor = 2.0, 2.0
        multiplier = CooledMultiplier(initial, cooling_factor=factor)

        # Check that the multiplier doesn't have a _counter attribute yet
        assert not hasattr(multiplier, "_counter")

        # Update twice and check for expected values
        for _ in range(2):
            multiplier.update()
        assert float(multiplier) == initial / factor**2
        assert hasattr(multiplier, "_counter")
        assert multiplier.counter == 2

        # Reset it and check for expected values
        multiplier.reset()
        assert hasattr(multiplier, "_counter")
        assert multiplier.counter == 0
        assert float(multiplier) == initial


class TestSensitivityWeights:
    """
    Test the :class:`~inversion_ideas.hyperparams.SensitivityWeights` hyperparameter.
    """

    n_params = 5
    n_data = 3

    @pytest.fixture
    def simulation(self):
        """Non-linear simulation for the tests."""
        return NonLinearRegressor.create_random(self.n_data, self.n_params, seed=4141)

    @pytest.fixture
    def kwargs(self):
        """Extra keyword arguments for the sensitivity weights function."""
        data_weights = 0.1 * np.ones(self.n_data)
        volumes = np.linspace(1, self.n_params + 1, self.n_params, dtype=np.float64)
        vmin = 1e-2
        kwargs = {"data_weights": data_weights, "volumes": volumes, "vmin": vmin}
        return kwargs

    def test_initialization(self, simulation, kwargs):
        """Test initialization of sensitivity weights."""
        # Define a SensitivityWeights object
        initial_model = np.ones(self.n_params)
        sensitivity_weights = SensitivityWeights(simulation, initial_model, **kwargs)
        # Check if they match the expected sensitivity weights
        expected = get_sensitivity_weights(simulation.jacobian(initial_model), **kwargs)
        np.testing.assert_allclose(sensitivity_weights, expected)

    def test_update(self, simulation, kwargs):
        """Test update of sensitivity weights."""
        # Define a SensitivityWeights object
        initial_model = np.ones(self.n_params)
        sensitivity_weights = SensitivityWeights(simulation, initial_model, **kwargs)
        # Update the weights with a new model
        model = np.random.default_rng(seed=4124).uniform(
            low=-1.0, high=1.0, size=self.n_params
        )
        sensitivity_weights.update(model)
        # Check if they match the expected sensitivity weights
        expected = get_sensitivity_weights(simulation.jacobian(model), **kwargs)
        np.testing.assert_allclose(sensitivity_weights, expected)
