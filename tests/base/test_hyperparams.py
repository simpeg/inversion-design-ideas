"""
Test the base hyperparameter classes.
"""

import re
from math import ceil, floor, trunc
from numbers import Real

import numpy as np
import pytest

from inversion_ideas.base import Multiplier, WrappedArray

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
        """Test addition."""
        value = 10.0
        other = 2.5
        multiplier = Multiplier(value)
        assert multiplier + other == value + other
        assert isinstance(multiplier + other, float)
        assert other + multiplier == value + other
        assert isinstance(other + multiplier, float)

    def test_mul(self):
        """Test the multiplication by float."""
        value = 10.0
        other = 2.5
        multiplier = Multiplier(value)
        assert multiplier * other == value * other
        assert isinstance(multiplier * other, float)
        assert other * multiplier == value * other
        assert isinstance(other * multiplier, float)

    @pytest.mark.parametrize("rmul", [False, True], ids=["mul", "rmul"])
    def test_mul_vs_objective(self, rmul):
        """Test multiplication by objective function."""
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

    @pytest.mark.parametrize("other_type", [float, Dummy])
    @pytest.mark.parametrize("rmul", [False, True], ids=["mul", "rmul"])
    def test_mul_dunder(self, other_type, rmul):
        """Test the ``__mul__`` and ``__rmul__`` methods directly."""
        value = 10.0
        multiplier = Multiplier(value)
        dunder = multiplier.__rmul__ if rmul else multiplier.__mul__
        if other_type is float:
            other = -45.0
            assert dunder(other) == value * other
        elif other_type is Dummy:
            phi = Dummy(3)
            assert dunder(phi) is NotImplemented
        else:
            raise TypeError()

    def test_truediv(self):
        """Test division by float."""
        value = 10.0
        other = 2.5
        multiplier = Multiplier(value)
        assert multiplier / other == value / other
        assert isinstance(multiplier / other, float)
        assert other / multiplier == other / value
        assert isinstance(other / multiplier, float)

    def test_truediv_vs_objective(self):
        """Test error on division by objective function."""
        value = 10.0
        multiplier = Multiplier(value)
        dummy = Dummy(n_params=3)
        with pytest.raises(TypeError, match="True division is not supported between"):
            multiplier / dummy
        with pytest.raises(
            TypeError, match="True division is not implemented for objective functions"
        ):
            dummy / multiplier

    @pytest.mark.parametrize("other_type", [float, Dummy])
    @pytest.mark.parametrize("rtruediv", [False, True], ids=["mul", "rmul"])
    def test_truediv_dunder(self, other_type, rtruediv):
        """Test ``__truediv__`` and ``__rtruediv__`` dunder methods."""
        value = 10.0
        multiplier = Multiplier(value)
        if other_type is float:
            other = -45.0
            if rtruediv:
                expected = other / value
                assert expected == multiplier.__rtruediv__(other)
            else:
                expected = value / other
                assert expected == multiplier.__truediv__(other)
        elif other_type is Dummy:
            phi = Dummy(3)
            dunder = multiplier.__rtruediv__ if rtruediv else multiplier.__truediv__
            with pytest.raises(TypeError, match="True division is not supported"):
                dunder(phi)

    def test_floordiv(self):
        """Test the floor division."""
        value = 10.0
        other = 2.0
        multiplier = Multiplier(value)
        assert multiplier // other == value // other
        assert isinstance(multiplier // other, float)
        assert other // multiplier == other // value
        assert isinstance(other // multiplier, float)

    def test_floordiv_vs_objective(self):
        """Test error on floor division when dividing by objective function."""
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

    def test_pos(self):
        """Test the ``__pos__`` method."""
        value = 10.0
        multiplier = Multiplier(value)
        assert multiplier.__pos__() == value.__pos__()
        assert isinstance(multiplier.__pos__(), float)

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


class TestWrappedArray:
    """Test the :class:`inversion_ideas.base.WrappedArray` class."""

    size = 30

    @pytest.fixture
    def array(self):
        rng = np.random.default_rng(seed=4141)
        return rng.uniform(size=self.size)

    @pytest.fixture
    def other_array(self):
        rng = np.random.default_rng(seed=4884)
        return rng.uniform(size=self.size)

    def test_len(self, array):
        wrapped_array = WrappedArray(array)
        assert isinstance(len(wrapped_array), int)
        assert len(array) == len(wrapped_array)

    def test_size(self, array):
        wrapped_array = WrappedArray(array)
        assert isinstance(wrapped_array.size, int)
        assert array.size == wrapped_array.size

    def test_shape(self, array):
        wrapped_array = WrappedArray(array)
        assert isinstance(wrapped_array.shape, tuple)
        assert array.shape == wrapped_array.shape

    def test_ndim(self, array):
        wrapped_array = WrappedArray(array)
        assert isinstance(wrapped_array.ndim, int)
        assert array.ndim == wrapped_array.ndim

    def test_dtype(self, array):
        wrapped_array = WrappedArray(array)
        assert wrapped_array.dtype is array.dtype

    @pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
    def test_add(self, array, other_array, right):
        wrapped_array = WrappedArray(array)
        expected = array + other_array
        result = other_array + wrapped_array if right else wrapped_array + other_array
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(expected, result, strict=True)

    @pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
    def test_add_dunder(self, array, other_array, right):
        """Test the ``__add__`` and ``__radd__`` methods."""
        wrapped_array = WrappedArray(array)
        expected = array + other_array
        dunder = wrapped_array.__radd__ if right else wrapped_array.__add__
        result = dunder(other_array)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(expected, result, strict=True)

    @pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
    def test_mul(self, array, other_array, right):
        wrapped_array = WrappedArray(array)
        expected = array * other_array
        result = other_array * wrapped_array if right else wrapped_array * other_array
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(expected, result, strict=True)

    @pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
    def test_mul_dunder(self, array, other_array, right):
        """Test the ``__mul__`` and ``__rmul__`` methods."""
        wrapped_array = WrappedArray(array)
        expected = array * other_array
        dunder = wrapped_array.__rmul__ if right else wrapped_array.__mul__
        result = dunder(other_array)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(expected, result, strict=True)

    @pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
    def test_truediv(self, array, other_array, right):
        wrapped_array = WrappedArray(array)
        if right:
            expected = other_array / array
            result = other_array / wrapped_array
        else:
            expected = array / other_array
            result = wrapped_array / other_array
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(expected, result, strict=True)

    @pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
    def test_truediv_dunder(self, array, other_array, right):
        """Test the ``__truediv__`` and ``__rtruediv__`` methods."""
        wrapped_array = WrappedArray(array)
        expected = other_array / array if right else array / other_array
        dunder = wrapped_array.__rtruediv__ if right else wrapped_array.__truediv__
        result = dunder(other_array)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(expected, result, strict=True)

    @pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
    def test_floordiv(self, array, other_array, right):
        wrapped_array = WrappedArray(array)
        if right:
            expected = other_array // array
            result = other_array // wrapped_array
        else:
            expected = array // other_array
            result = wrapped_array // other_array
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(expected, result, strict=True)

    @pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
    def test_floordiv_dunder(self, array, other_array, right):
        """Test the ``__floordiv__`` and ``__rfloordiv__`` methods."""
        wrapped_array = WrappedArray(array)
        expected = other_array // array if right else array // other_array
        dunder = wrapped_array.__rfloordiv__ if right else wrapped_array.__floordiv__
        result = dunder(other_array)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(expected, result, strict=True)

    @pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
    def test_matmul(self, array, other_array, right):
        wrapped_array = WrappedArray(array)
        if right:
            expected = other_array @ array
            result = other_array @ wrapped_array
        else:
            expected = array @ other_array
            result = wrapped_array @ other_array
        assert isinstance(result, Real)
        np.testing.assert_allclose(expected, result, strict=True)

    @pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
    def test_matmul_dunder(self, array, other_array, right):
        """Test the ``__matmul__`` and ``__rmatmul__`` methods."""
        wrapped_array = WrappedArray(array)
        expected = other_array @ array if right else array @ other_array
        dunder = wrapped_array.__rmatmul__ if right else wrapped_array.__matmul__
        result = dunder(other_array)
        assert isinstance(result, Real)
        np.testing.assert_allclose(expected, result, strict=True)

    def test_equality(self, array, other_array):
        wrapped_array = WrappedArray(array)
        assert (wrapped_array == array).all()
        assert (wrapped_array == wrapped_array.copy()).all()
        assert (wrapped_array != other_array).all()
        assert (wrapped_array != WrappedArray(other_array)).all()
        assert (array == wrapped_array).all()
        assert (wrapped_array.copy() == wrapped_array).all()
        assert (other_array != wrapped_array).all()
        assert (WrappedArray(other_array) != wrapped_array).all()

    @pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
    @pytest.mark.parametrize("operator", ["<", "<=", ">", ">=", "!="])
    def test_inequalities(self, array, other_array, operator, right):
        wrapped_array = WrappedArray(array)
        if operator == "<":
            expected = array < other_array if not right else other_array < array
            result = (
                wrapped_array < other_array
                if not right
                else other_array < wrapped_array
            )
        elif operator == ">":
            expected = array > other_array if not right else other_array > array
            result = (
                wrapped_array > other_array
                if not right
                else other_array > wrapped_array
            )
        elif operator == "<=":
            expected = array <= other_array if not right else other_array <= array
            result = (
                wrapped_array <= other_array
                if not right
                else other_array <= wrapped_array
            )
        elif operator == ">=":
            expected = array >= other_array if not right else other_array >= array
            result = (
                wrapped_array >= other_array
                if not right
                else other_array >= wrapped_array
            )
        elif operator == "!=":
            expected = array != other_array if not right else other_array != array
            result = (
                wrapped_array != other_array
                if not right
                else other_array != wrapped_array
            )
        else:
            raise ValueError()
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(expected, result, strict=True)

    @pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
    def test_and(self, right):
        rng = np.random.default_rng(seed=1212)
        array = rng.choice([True, False], size=self.size)
        other_array = rng.choice([True, False], size=self.size)
        wrapped_array = WrappedArray(array)
        expected = array & other_array if not right else other_array & array
        result = (
            wrapped_array & other_array if not right else other_array & wrapped_array
        )
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(expected, result, strict=True)

    @pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
    def test_or(self, right):
        rng = np.random.default_rng(seed=1212)
        array = rng.choice([True, False], size=self.size)
        other_array = rng.choice([True, False], size=self.size)
        wrapped_array = WrappedArray(array)
        expected = array | other_array if not right else other_array | array
        result = (
            wrapped_array | other_array if not right else other_array | wrapped_array
        )
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(expected, result, strict=True)

    def test_abs(self, array):
        result = abs(WrappedArray(array))
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, abs(array), strict=True)

    def test_neg(self, array):
        result = -WrappedArray(array)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, -array, strict=True)

    def test_bool(self, array):
        wrapped_array = WrappedArray(array)
        match = "The truth value of an array with more than one element is ambiguous"
        with pytest.raises(ValueError, match=match):
            bool(wrapped_array)
        with pytest.raises(ValueError, match=match):
            not wrapped_array  # ruff: ignore[B018]

    @pytest.mark.parametrize("index", [0, -1, slice(0, 3), slice(0, 5, 2)])
    def test_getitem(self, array, index):
        wrapped_array = WrappedArray(array)
        np.testing.assert_allclose(wrapped_array[index], array[index], strict=True)

    @pytest.mark.parametrize("index", [0, -1, slice(0, 3), slice(0, 5, 2)])
    def test_setitem(self, array, index):
        original = array.copy()
        wrapped = WrappedArray(array)
        wrapped[index] = 100.0
        np.testing.assert_allclose(wrapped[index], 100.0)
        np.testing.assert_allclose(wrapped.array[index], 100.0)
        # Since we are storing a reference to the array, a modification to it changes
        # the value of the array itself.
        np.testing.assert_allclose(array[index], 100.0)
        # Compare the full array
        original[index] = 100.0
        np.testing.assert_allclose(wrapped.array, original, strict=True)

    def test_contains(self, array):
        wrapped_array = WrappedArray(array)
        assert array[3] in wrapped_array
        assert 100.0 not in wrapped_array

    def test_min(self, array):
        wrapped_array = WrappedArray(array)
        np.testing.assert_allclose(wrapped_array.min(), array.min(), strict=True)

    def test_max(self, array):
        wrapped_array = WrappedArray(array)
        np.testing.assert_allclose(wrapped_array.max(), array.max(), strict=True)

    def test_copy(self, array):
        wrapped_array = WrappedArray(array)
        copied = wrapped_array.copy()
        assert not (wrapped_array is copied)
        assert not (wrapped_array.array is copied.array)
        assert array is wrapped_array.array
        assert not (array is copied.array)

    def test_transpose(self, array):
        wrapped = WrappedArray(array)
        np.testing.assert_allclose(wrapped.T, array.T, strict=True)
        np.testing.assert_allclose(wrapped.transpose(), array.transpose(), strict=True)


class TestWrappedArrayVsWrappedArray:
    """Test operations between two :class:`inversion_ideas.base.WrappedArray`."""

    size = 11

    @pytest.fixture
    def array_a(self):
        rng = np.random.default_rng(seed=4141)
        return rng.uniform(size=self.size)

    @pytest.fixture
    def array_b(self):
        rng = np.random.default_rng(seed=1512)
        return rng.uniform(size=self.size)

    def test_add(self, array_a, array_b):
        wrapped_a, wrapped_b = WrappedArray(array_a), WrappedArray(array_b)
        assert isinstance(wrapped_a + wrapped_b, np.ndarray)
        np.testing.assert_allclose(wrapped_a + wrapped_b, array_a + array_b)

    def test_mul(self, array_a, array_b):
        wrapped_a, wrapped_b = WrappedArray(array_a), WrappedArray(array_b)
        assert isinstance(wrapped_a * wrapped_b, np.ndarray)
        np.testing.assert_allclose(wrapped_a * wrapped_b, array_a * array_b)

    def test_matmul(self, array_a, array_b):
        wrapped_a, wrapped_b = WrappedArray(array_a), WrappedArray(array_b)
        assert isinstance(wrapped_a @ wrapped_b, Real)
        np.testing.assert_allclose(wrapped_a @ wrapped_b, array_a @ array_b)

    def test_truediv(self, array_a, array_b):
        wrapped_a, wrapped_b = WrappedArray(array_a), WrappedArray(array_b)
        assert isinstance(wrapped_a / wrapped_b, np.ndarray)
        np.testing.assert_allclose(wrapped_a / wrapped_b, array_a / array_b)

    def test_floordiv(self, array_a, array_b):
        wrapped_a, wrapped_b = WrappedArray(array_a), WrappedArray(array_b)
        assert isinstance(wrapped_a // wrapped_b, np.ndarray)
        np.testing.assert_allclose(wrapped_a // wrapped_b, array_a // array_b)

    def test_eq(self, array_a):
        wrapped = WrappedArray(array_a)
        copied = wrapped.copy()
        assert not (wrapped is copied)
        assert isinstance(wrapped == copied, np.ndarray)
        assert (wrapped == copied).all()

    @pytest.mark.parametrize("operator", ["<", "<=", ">", ">=", "!="])
    def test_inequalities(self, array_a, array_b, operator):
        wrapped_a, wrapped_b = WrappedArray(array_a), WrappedArray(array_b)
        if operator == "<":
            expected = array_a < array_b
            result = wrapped_a < wrapped_b
        elif operator == "<=":
            expected = array_a <= array_b
            result = wrapped_a <= wrapped_b
        elif operator == ">":
            expected = array_a > array_b
            result = wrapped_a > wrapped_b
        elif operator == ">=":
            expected = array_a >= array_b
            result = wrapped_a >= wrapped_b
        elif operator == "!=":
            expected = array_a != array_b
            result = wrapped_a != wrapped_b
        else:
            raise ValueError()
        np.testing.assert_allclose(expected, result)


class TestWrappedArrayNDim:
    """
    Test n-dimensional wrapped arrays.
    """

    shape = (3, 4)

    @pytest.fixture
    def array(self):
        rng = np.random.default_rng(seed=4141)
        return rng.uniform(size=self.shape)

    def test_transpose(self, array):
        wrapped = WrappedArray(array)
        np.testing.assert_allclose(wrapped.T, array.T, strict=True)
        np.testing.assert_allclose(wrapped.transpose(), array.transpose(), strict=True)

    def test_matmul(self, array):
        wrapped = WrappedArray(array)
        v = np.arange(self.shape[1], dtype=np.float64)
        np.testing.assert_allclose(wrapped @ v, array @ v, strict=True)
