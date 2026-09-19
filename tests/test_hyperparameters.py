"""
Test custom hyperparameter objects.
"""

import pytest

from inversion_ideas.hyperparams import CooledMultiplier


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

    def test_update(self):
        """Test the update method."""
        initial, cooling_factor = 10.0, 2.0
        multiplier = CooledMultiplier(initial, cooling_factor=cooling_factor)
        multiplier.update()
        assert multiplier == initial / cooling_factor
        assert multiplier.value == initial / cooling_factor

    def test_multiple_updates(self):
        """Test multiple calls to the update method."""
        initial, cooling_factor = 10.0, 2.0
        multiplier = CooledMultiplier(initial, cooling_factor=cooling_factor)
        n = 8
        for _ in range(n):
            multiplier.update()
        assert multiplier == initial / cooling_factor**n
        assert multiplier.value == initial / cooling_factor**n
