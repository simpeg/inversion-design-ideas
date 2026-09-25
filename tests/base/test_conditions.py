"""
Test conditions.
"""

import numpy as np
import pytest

from inversion_ideas.base import Condition
from inversion_ideas.typing import Model


class Even(Condition):
    """
    Simple condition that checks if model is even.
    """

    def __call__(self, model: Model) -> bool:
        return bool(np.all((model % 2) == 0))


class Positive(Condition):
    """
    Simple condition that checks if model is positive.
    """

    def __call__(self, model: Model) -> bool:
        return bool(np.all(model > 0))


class TestMixin:
    """
    Test mixin operations.
    """

    @pytest.fixture(params=("class", "function"))
    def is_even(self, request):
        """
        Return a condition that checks if model is even.

        Parametrize it to be either a function or a :class:`Condition`.
        """
        if request.param == "function":

            def is_even(model: Model) -> bool:
                return bool(np.all((model % 2) == 0))

            return is_even
        return Even()

    def test_positive(self):
        is_positive = Positive()
        assert is_positive(np.array([1.0]))
        assert is_positive(np.array([10.0]))
        assert not is_positive(np.array([0.0]))
        assert not is_positive(np.array([-2.0]))

    def test_even(self, is_even):
        assert not is_even(np.array([1.0]))
        assert is_even(np.array([2.0]))
        assert not is_even(np.array([3.0]))
        assert is_even(np.array([4.0]))
        assert is_even(np.array([0.0]))
        assert not is_even(np.array([-1.0]))
        assert is_even(np.array([-2.0]))
        assert not is_even(np.array([-3.0]))
        assert is_even(np.array([-4.0]))

    def test_and(self, is_even):
        is_positive = Positive()
        condition = is_even & is_positive
        assert not condition(np.array([1.0]))
        assert condition(np.array([2.0]))
        assert not condition(np.array([3.0]))
        assert condition(np.array([4.0]))
        assert not condition(np.array([0.0]))
        assert not condition(np.array([-1.0]))
        assert not condition(np.array([-2.0]))
        assert not condition(np.array([-3.0]))
        assert not condition(np.array([-4.0]))

    def test_or(self, is_even):
        is_positive = Positive()
        condition = is_even | is_positive
        assert condition(np.array([1.0]))
        assert condition(np.array([2.0]))
        assert condition(np.array([3.0]))
        assert condition(np.array([4.0]))
        assert condition(np.array([0.0]))
        assert not condition(np.array([-1.0]))
        assert condition(np.array([-2.0]))
        assert not condition(np.array([-3.0]))
        assert condition(np.array([-4.0]))

    def test_xor(self, is_even):
        is_positive = Positive()
        condition = is_even ^ is_positive
        assert condition(np.array([1.0]))
        assert not condition(np.array([2.0]))
        assert condition(np.array([3.0]))
        assert not condition(np.array([4.0]))
        assert condition(np.array([0.0]))
        assert not condition(np.array([-1.0]))
        assert condition(np.array([-2.0]))
        assert not condition(np.array([-3.0]))
        assert condition(np.array([-4.0]))


class GreaterThan(Condition):
    """
    Check if model is greater than certain value for all elements in the model.
    """

    def __init__(self, value: Model):
        self.value = value

    def __call__(self, model: Model) -> bool:
        return bool(np.all(model > self.value))

    def update(self, model: Model):
        self.value = model

    def initialize(self):
        self.value = np.array([0])


class TestUpdateMixin:
    """
    Test updating conditions in mixins.
    """

    def test_greater_than(self):
        condition = GreaterThan(np.array([2]))
        assert condition(np.array([3]))
        assert not condition(np.array([2]))
        assert not condition(np.array([1]))

    def test_update(self):
        condition = GreaterThan(np.array([2]))
        new_value = np.array([3])
        condition.update(new_value)
        assert condition.value == new_value

    @pytest.mark.parametrize("operation", ["and", "or", "xor"])
    def test_update_mixin(self, operation):
        condition_a = GreaterThan(np.array([2]))
        condition_b = GreaterThan(np.array([3]))
        match operation:
            case "and":
                condition = condition_a & condition_b
            case "or":
                condition = condition_a | condition_b
            case "xor":
                condition = condition_a ^ condition_b
            case _:
                msg = f"{operation}"
                raise ValueError(msg)
        new_value = 4
        condition.update(np.array([new_value]))
        assert (condition_a.value == new_value).all()
        assert (condition_b.value == new_value).all()

    @pytest.mark.parametrize("operation", ["and", "or", "xor"])
    def test_update_mixin_with_function(self, operation):
        """
        Test if update works in case a condition is a function.
        """

        def is_even(model) -> bool:
            return bool(np.all((model % 2) == 0))

        condition_a = GreaterThan(np.array([2]))
        match operation:
            case "and":
                condition = condition_a & is_even
            case "or":
                condition = condition_a | is_even
            case "xor":
                condition = condition_a ^ is_even
            case _:
                msg = f"{operation}"
                raise ValueError(msg)
        new_value = np.array([4])
        condition.update(new_value)
        assert condition_a.value == new_value
        assert condition.condition_b is is_even


class TestInitializeMixin:
    """
    Test initializing conditions in mixins.
    """

    def test_initialize(self):
        condition = GreaterThan(np.array([2]))
        condition.initialize()
        assert (condition.value == np.array([0])).all()

    @pytest.mark.parametrize("operation", ["and", "or", "xor"])
    def test_initialize_mixin(self, operation):
        condition_a = GreaterThan(np.array([2]))
        condition_b = GreaterThan(np.array([3]))
        match operation:
            case "and":
                condition = condition_a & condition_b
            case "or":
                condition = condition_a | condition_b
            case "xor":
                condition = condition_a ^ condition_b
            case _:
                msg = f"{operation}"
                raise ValueError(msg)
        condition.initialize()
        assert (condition_a.value == np.array([0])).all()
        assert (condition_b.value == np.array([0])).all()

    @pytest.mark.parametrize("operation", ["and", "or", "xor"])
    def test_initialize_mixin_with_function(self, operation):
        """
        Test if initialize works in case a condition is a function.
        """

        def is_even(model) -> bool:
            return bool(np.all((model % 2) == 0))

        condition_a = GreaterThan(np.array([2]))
        match operation:
            case "and":
                condition = condition_a & is_even
            case "or":
                condition = condition_a | is_even
            case "xor":
                condition = condition_a ^ is_even
            case _:
                msg = f"{operation}"
                raise ValueError(msg)
        condition.initialize()
        assert (condition_a.value == np.array([0])).all()
        assert condition.condition_b is is_even


class TestInplaceErrors:
    """
    Test errors on inplace operators of conditions.
    """

    def test_iand(self):
        condition_a, condition_b = Even(), Positive()
        with pytest.raises(
            TypeError, match="Inplace AND binary operation is not supported"
        ):
            condition_a &= condition_b

    def test_ior(self):
        condition_a, condition_b = Even(), Positive()
        with pytest.raises(
            TypeError, match="Inplace OR binary operation is not supported"
        ):
            condition_a |= condition_b

    def test_ixor(self):
        condition_a, condition_b = Even(), Positive()
        with pytest.raises(
            TypeError, match="Inplace XOR binary operation is not supported"
        ):
            condition_a ^= condition_b


class TestInfo:
    """
    Simple tests to check if the ``info()`` method works without failing.
    """

    def test_info_condition(self):
        """
        Test ``info()`` method of the ``Condition`` base class.
        """
        even = Even()
        model = np.array([1])
        even.info(model)

    @pytest.mark.parametrize("operation", ["and", "or", "xor"])
    def test_info_combo(self, operation):
        """
        Test ``info()`` method of the combo condition classes.
        """
        condition_a = GreaterThan(np.array([2]))
        condition_b = Even()
        match operation:
            case "and":
                combo = condition_a & condition_b
            case "or":
                combo = condition_a | condition_b
            case "xor":
                combo = condition_a ^ condition_b
            case _:
                msg = f"{operation}"
                raise ValueError(msg)
        model = np.array([1])
        combo.info(model)

    @pytest.mark.parametrize("operation", ["and", "or", "xor"])
    def test_info_combo_with_function(self, operation):
        """
        Test ``info()`` method of a combo with condition and function.
        """

        def is_even(model: Model) -> bool:
            return bool(np.all((model % 2) == 0))

        condition = GreaterThan(np.array([2]))
        match operation:
            case "and":
                combo = condition & is_even
            case "or":
                combo = condition | is_even
            case "xor":
                combo = condition ^ is_even
            case _:
                msg = f"{operation}"
                raise ValueError(msg)
        model = np.array([1])
        combo.info(model)
