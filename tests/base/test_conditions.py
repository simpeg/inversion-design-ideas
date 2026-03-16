"""
Test conditions.
"""
import pytest
import numpy as np
from inversion_ideas.base import Condition


class Even(Condition):
    """
    Simple condition that checks if model is even.
    """

    def __call__(self, model) -> bool:
        return bool(np.all((model % 2) == 0))


class Positive(Condition):
    """
    Simple condition that checks if model is positive.
    """

    def __call__(self, model) -> bool:
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

            def is_even(model) -> bool:
                return bool(np.all((model % 2) == 0))

            return is_even
        return Even()

    def test_positive(self):
        is_positive = Positive()
        assert is_positive(1.0)
        assert is_positive(10.0)
        assert not is_positive(0.0)
        assert not is_positive(-2.0)

    def test_even(self, is_even):
        assert not is_even(1.0)
        assert is_even(2.0)
        assert not is_even(3.0)
        assert is_even(4.0)
        assert is_even(0.0)
        assert not is_even(-1.0)
        assert is_even(-2.0)
        assert not is_even(-3.0)
        assert is_even(-4.0)

    def test_and(self, is_even):
        is_even = Even()
        is_positive = Positive()
        condition = is_even & is_positive
        assert not condition(1.0)
        assert condition(2.0)
        assert not condition(3.0)
        assert condition(4.0)
        assert not condition(0.0)
        assert not condition(-1.0)
        assert not condition(-2.0)
        assert not condition(-3.0)
        assert not condition(-4.0)

    def test_or(self, is_even):
        is_even = Even()
        is_positive = Positive()
        condition = is_even | is_positive
        assert condition(1.0)
        assert condition(2.0)
        assert condition(3.0)
        assert condition(4.0)
        assert condition(0.0)
        assert not condition(-1.0)
        assert condition(-2.0)
        assert not condition(-3.0)
        assert condition(-4.0)

    def test_xor(self, is_even):
        is_even = Even()
        is_positive = Positive()
        condition = is_even ^ is_positive
        assert condition(1.0)
        assert not condition(2.0)
        assert condition(3.0)
        assert not condition(4.0)
        assert condition(0.0)
        assert not condition(-1.0)
        assert condition(-2.0)
        assert not condition(-3.0)
        assert condition(-4.0)


class GreaterThan(Condition):
    def __init__(self, value):
        self.value = value

    def __call__(self, model) -> bool:
        return bool(np.all(model > self.value))

    def update(self, model):
        self.value = model

    def initialize(self):
        self.value = None


class UpdateMixin:
    """
    Test updating conditions in mixins.
    """

    def test_greater_than(self):
        condition = GreaterThan(2)
        assert condition(3)
        assert not condition(2)
        assert not condition(1)

    def test_update(self):
        condition = GreaterThan(2)
        new_value = 3
        condition.update(new_value)
        assert condition.value == new_value

    @pytest.mark.parametrize("operation", ["and", "or", "xor"])
    def test_update_mixin(self, operation):
        condition_a = GreaterThan(2)
        condition_b = GreaterThan(3)
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
        condition.update(new_value)
        assert condition_a.value == new_value
        assert condition_b.value == new_value

    @pytest.mark.parametrize("operation", ["and", "or", "xor"])
    def test_update_mixin_with_function(self, operation):
        """
        Test if update works in case a condition is a function.
        """

        def is_even(model) -> bool:
            return bool(np.all((model % 2) == 0))

        condition_a = GreaterThan(2)
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
        new_value = 4
        condition.update(new_value)
        assert condition_a.value == new_value
        assert condition.condition_b is is_even


class InitializeMixin:
    """
    Test initializing conditions in mixins.
    """

    def test_initialize(self):
        condition = GreaterThan(2)
        condition.initialize()
        assert condition.value is None

    @pytest.mark.parametrize("operation", ["and", "or", "xor"])
    def test_initialize_mixin(self, operation):
        condition_a = GreaterThan(2)
        condition_b = GreaterThan(3)
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
        assert condition_a.value is None
        assert condition_b.value is None

    @pytest.mark.parametrize("operation", ["and", "or", "xor"])
    def test_initialize_mixin_with_function(self, operation):
        """
        Test if initialize works in case a condition is a function.
        """

        def is_even(model) -> bool:
            return bool(np.all((model % 2) == 0))

        condition_a = GreaterThan(2)
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
        assert condition_a.value is None
        assert condition.condition_b is is_even
