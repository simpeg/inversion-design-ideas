"""
Test the base hyperparameter classes.
"""

import re
from math import ceil, floor, trunc

import numpy as np
import pytest

from inversion_ideas.base import Multiplier

from ..utils import Dummy


class TestMultiplier:
    """Test the ``Multiplier`` class."""

    @pytest.mark.parametrize("value", [10.0, -10.0, 0, 2, -2])
    @pytest.mark.parametrize("mutable", [True, False])
    def test_init(self, value, mutable):
        """Test the ``__init__`` method."""
        multiplier = Multiplier(value, mutable=mutable)
        assert multiplier.value == value
        assert multiplier.mutable == mutable

    @pytest.mark.parametrize("invalid_value", [1j, np.array([1.0, 2.0]), [1.0]])
    def test_invalid_value(self, invalid_value):
        """Test error after invalid value argument."""
        msg = re.escape("Invalid 'value' argument of type")
        with pytest.raises(TypeError, match=msg):
            Multiplier(invalid_value)

    @pytest.mark.parametrize("invalid_mutable", [1, 0, np.array([1.0, 2.0]), [1.0]])
    def test_invalid_mutable(self, invalid_mutable):
        """Test error after invalid mutable argument."""
        msg = re.escape("Invalid 'mutable' argument of type")
        with pytest.raises(TypeError, match=msg):
            Multiplier(10.0, mutable=invalid_mutable)

    def test_str(self):
        """Test the ``__str__`` method."""
        value = 10.0
        multiplier = Multiplier(value)
        assert str(multiplier) == "Multiplier(10.0)"

    def test_add(self):
        """Test the ``__add__`` and ``__radd__`` methods."""
        value = 10.0
        other = 2.5
        multiplier = Multiplier(value)
        assert multiplier + other == value + other
        assert isinstance(multiplier + other, float)
        assert other + multiplier == value + other
        assert isinstance(other + multiplier, float)

    def test_mul(self):
        """Test the ``__mul__`` and ``__rmul__`` methods."""
        value = 10.0
        other = 2.5
        multiplier = Multiplier(value)
        assert multiplier * other == value * other
        assert isinstance(multiplier * other, float)
        assert other * multiplier == value * other
        assert isinstance(other * multiplier, float)

    @pytest.mark.parametrize("rmul", [False, True], ids=["mul", "rmul"])
    def test_mul_vs_objective(self, rmul):
        """Test ``__mul__`` and ``__rmul__`` when multiplying by objective function."""
        dummy = Dummy(n_params=3)
        value = 10.0
        multiplier = Multiplier(value)
        scaled = dummy * multiplier if rmul else multiplier * dummy

        # Make sure that the multiplier of the Scaled object is the same Multiplier
        assert scaled.multiplier is multiplier

        # Test the common methods of the objective function
        model = np.array([1.0, 2.0, 3.0])
        assert value * dummy(model) == scaled(model)
        np.testing.assert_allclose(
            value * dummy.gradient(model), scaled.gradient(model)
        )
        np.testing.assert_allclose(value * dummy.hessian(model), scaled.hessian(model))

    def test_truediv(self):
        """Test the ``__truediv__`` and ``__rtruediv__`` methods."""
        value = 10.0
        other = 2.5
        multiplier = Multiplier(value)
        assert multiplier / other == value / other
        assert isinstance(multiplier / other, float)
        assert other / multiplier == other / value
        assert isinstance(other / multiplier, float)

    def test_truediv_vs_objective(self):
        """Test error on ``__truediv__`` and ``__rtruediv__`` when dividing by objective function."""
        value = 10.0
        multiplier = Multiplier(value)
        dummy = Dummy(n_params=3)
        with pytest.raises(TypeError, match="True division is not supported between"):
            multiplier / dummy
        with pytest.raises(
            TypeError, match="True division is not implemented for objective functions"
        ):
            dummy / multiplier

    def test_floordiv(self):
        """Test the ``__floordiv__`` and ``__rfloordiv__`` methods."""
        value = 10.0
        other = 2.0
        multiplier = Multiplier(value)
        assert multiplier // other == value // other
        assert isinstance(multiplier // other, float)
        assert other // multiplier == other // value
        assert isinstance(other // multiplier, float)

    def test_floordiv_vs_objective(self):
        """Test error on ``__floordiv__`` when dividing by objective function."""
        value = 10.0
        multiplier = Multiplier(value)
        dummy = Dummy(n_params=3)
        with pytest.raises(TypeError, match="unsupported operand type"):
            multiplier // dummy
        with pytest.raises(TypeError, match="Floor division is not implemented"):
            dummy // multiplier

    def test_pow(self):
        """Test the ``__pow__`` and ``__rpow__`` methods."""
        value = 3.0
        other = 2.5
        multiplier = Multiplier(value)
        assert multiplier**other == value**other
        assert isinstance(multiplier**other, float)
        assert other**multiplier == other**value
        assert isinstance(other**multiplier, float)

    @pytest.mark.parametrize("value", [10.0, -10.0, 0.0])
    def test_abs(self, value):
        """Test the ``__abs__`` method."""
        multiplier = Multiplier(value)
        assert abs(multiplier) == abs(value)

    @pytest.mark.parametrize("value", [10.2, -10.3, 0.0, 2.8, -2.8, 3, -3])
    def test_ceil(self, value):
        """Test the ``__ceil__`` method."""
        multiplier = Multiplier(value)
        assert ceil(multiplier) == ceil(value)

    @pytest.mark.parametrize("value", [10.2, -10.3, 0.0, 2.8, -2.8, 3, -3])
    def test_floor(self, value):
        """Test the ``__floor__`` method."""
        multiplier = Multiplier(value)
        assert floor(multiplier) == floor(value)

    @pytest.mark.parametrize("value", [10.2, -10.3, 0.0, 2.8, -2.8, 3, -3])
    def test_eq(self, value):
        """Test the ``__eq__`` method."""
        multiplier = Multiplier(value)
        assert multiplier == value

    @pytest.mark.parametrize("value", [10.2, -10.3, 0.0, 2.8, -2.8, 3, -3])
    def test_float(self, value):
        """Test the ``__float__`` method."""
        multiplier = Multiplier(value)
        assert float(multiplier) == float(value)

    def test_inequalities(self):
        """Test the inequality methods."""
        multiplier = Multiplier(10.0)
        assert multiplier <= 20.0
        assert multiplier >= 5.0
        assert multiplier < 20.0
        assert multiplier > 5.0
        assert multiplier <= 10.0
        assert multiplier >= 10.0
        assert 20.0 >= multiplier  # ruff: ignore[SIM300]
        assert 5.0 <= multiplier  # ruff: ignore[SIM300]
        assert 20 > multiplier  # ruff: ignore[SIM300]
        assert 5.0 < multiplier  # ruff: ignore[SIM300]
        assert 10.0 <= multiplier  # ruff: ignore[SIM300]
        assert 10.0 >= multiplier  # ruff: ignore[SIM300]

    def test_mod(self):
        """Test the ``__mod__`` and ``__rmod__`` methods."""
        value = 10.0
        multiplier = Multiplier(value)
        assert multiplier % 3.0 == value % 3.0
        assert isinstance(multiplier % 3.0, float)
        assert 40.0 % multiplier == 40.0 % value
        assert isinstance(40.0 % multiplier, float)

    def test_neg(self):
        """Test the ``__neg__`` method."""
        value = 10.0
        multiplier = Multiplier(value)
        assert -multiplier == -value
        assert isinstance(-multiplier, float)

    def test_round(self):
        """Test the ``__round__`` method."""
        value = 10.5
        multiplier = Multiplier(value)
        assert round(multiplier) == round(value)
        assert isinstance(round(multiplier), int)

    def test_trunc(self):
        """Test the ``__trunc__`` method."""
        value = 10.5
        multiplier = Multiplier(value)
        assert trunc(multiplier) == trunc(value)
        assert isinstance(trunc(multiplier), int)


class TestMultiplierMutability:
    """Test the mutability of ``Multiplier`` class."""

    @pytest.mark.parametrize("mutable", [True, False])
    def test_value_setter(self, mutable):
        """Test the value setter when mutable is True or False."""
        multiplier = Multiplier(10.0, mutable=mutable)
        new_value = 2.5
        if mutable:
            multiplier.value = new_value
            assert multiplier.value == new_value
        else:
            with pytest.raises(TypeError, match="Cannot modify the value of"):
                multiplier.value = new_value

    @pytest.mark.parametrize("mutable", [True, False])
    def test_iadd(self, mutable):
        """Test the ``__iadd__`` method when mutable is True or False."""
        value = 10.0
        multiplier = Multiplier(value, mutable=mutable)
        other = 2.5
        if mutable:
            multiplier += other
            assert multiplier.value == value + other
        else:
            with pytest.raises(TypeError, match="Cannot modify the value of"):
                multiplier += other

    @pytest.mark.parametrize("mutable", [True, False])
    def test_imul(self, mutable):
        """Test the ``__imul__`` method when mutable is True or False."""
        value = 10.0
        multiplier = Multiplier(value, mutable=mutable)
        other = 2.5
        if mutable:
            multiplier *= other
            assert multiplier.value == value * other
        else:
            with pytest.raises(TypeError, match="Cannot modify the value of"):
                multiplier *= other

    @pytest.mark.parametrize("mutable", [True, False])
    def test_itruediv(self, mutable):
        """Test the ``__itruediv__`` method when mutable is True or False."""
        value = 10.0
        multiplier = Multiplier(value, mutable=mutable)
        other = 2.5
        if mutable:
            multiplier /= other
            assert multiplier.value == value / other
        else:
            with pytest.raises(TypeError, match="Cannot modify the value of"):
                multiplier /= other

    @pytest.mark.parametrize("mutable", [True, False])
    def test_ifloordiv(self, mutable):
        """Test the ``__ifloordiv__`` method when mutable is True or False."""
        value = 10.0
        multiplier = Multiplier(value, mutable=mutable)
        other = 2.5
        if mutable:
            multiplier //= other
            assert multiplier.value == value // other
        else:
            with pytest.raises(TypeError, match="Cannot modify the value of"):
                multiplier //= other


class TestMultiplierVsMultiplier:
    """
    Test arithmetic operations between multipliers.
    """

    a_value = 10.2
    b_value = -30.8

    @pytest.fixture
    def a(self):
        return Multiplier(self.a_value)

    @pytest.fixture
    def b(self):
        return Multiplier(self.b_value)

    def test_add(self, a, b):
        result = a + b
        assert isinstance(result, float)
        assert result == self.a_value + self.b_value

    def test_diff(self, a, b):
        result = a - b
        assert isinstance(result, float)
        assert result == self.a_value - self.b_value

    def test_mul(self, a, b):
        result = a * b
        assert isinstance(result, float)
        assert result == self.a_value * self.b_value

    def test_truediv(self, a, b):
        result = a / b
        assert isinstance(result, float)
        assert result == self.a_value / self.b_value

    def test_floordiv(self, a, b):
        result = a // b
        assert isinstance(result, float)
        assert result == self.a_value // self.b_value

    def test_eq(self, a, b):
        result = a == b
        assert isinstance(result, bool)
        assert result is (self.a_value == self.b_value)

    def test_gt(self, a, b):
        result = a > b
        assert isinstance(result, bool)
        assert result is (self.a_value > self.b_value)

    def test_ge(self, a, b):
        result = a >= b
        assert isinstance(result, bool)
        assert result is (self.a_value >= self.b_value)

    def test_lt(self, a, b):
        result = a < b
        assert isinstance(result, bool)
        assert result is (self.a_value < self.b_value)

    def test_le(self, a, b):
        result = a <= b
        assert isinstance(result, bool)
        assert result is (self.a_value <= self.b_value)

    def test_ne(self, a, b):
        result = a != b
        assert isinstance(result, bool)
        assert result is (self.a_value != self.b_value)
