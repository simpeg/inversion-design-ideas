"""
Base classes for defining conditions.

Conditions are callable objects that return either a bool when either a certain
condition is met or not. They are use to define abstract objects like stopping criteria
for inversions. We can use binary operators (and, or, xor) to logically group multiple
conditions together.
"""
from abc import ABC, abstractmethod


class Condition(ABC):
    """
    Base abstract class for conditions.
    """

    @abstractmethod
    def __call__(self, model) -> bool:
        ...

    def update(self, model):  # noqa: B027
        """
        Update the condition.
        """
        # This is not an abstract method. Children classes can choose to override it if
        # necessary. The base class implements it to provide a common interface, even
        # for those children that don't implement it.

    def __and__(self, other) -> "LogicalAnd":
        return LogicalAnd(self, other)

    def __or__(self, other) -> "LogicalOr":
        return LogicalOr(self, other)

    def __xor__(self, other) -> "LogicalXor":
        return LogicalXor(self, other)

    def __iand__(self, other):
        msg = "Inplace AND binary operation is not supported for conditions."
        raise TypeError(msg)

    def __ior__(self, other):
        msg = "Inplace OR binary operation is not supported for conditions."
        raise TypeError(msg)

    def __ixor__(self, other):
        msg = "Inplace XOR binary operation is not supported for conditions."
        raise TypeError(msg)


class LogicalAnd(Condition):
    """
    Mixin condition for the AND operation between two other conditions.
    """

    def __init__(self, condition_a, condition_b):
        self.condition_a = condition_a
        self.condition_b = condition_b

    def __call__(self, model) -> bool:
        return self.condition_a(model) and self.condition_b(model)

    def update(self, model):
        """
        Update the underlying conditions.
        """
        self.condition_a.update(model)
        self.condition_b.update(model)


class LogicalOr(Condition):
    """
    Mixin condition for the OR operation between two other conditions.
    """

    def __init__(self, condition_a, condition_b):
        self.condition_a = condition_a
        self.condition_b = condition_b

    def __call__(self, model) -> bool:
        return self.condition_a(model) or self.condition_b(model)

    def update(self, model):
        """
        Update the underlying conditions.
        """
        self.condition_a.update(model)
        self.condition_b.update(model)


class LogicalXor(Condition):
    """
    Mixin condition for the XOR operation between two other conditions.
    """

    def __init__(self, condition_a, condition_b):
        self.condition_a = condition_a
        self.condition_b = condition_b

    def __call__(self, model) -> bool:
        return self.condition_a(model) ^ self.condition_b(model)

    def update(self, model):
        """
        Update the underlying conditions.
        """
        self.condition_a.update(model)
        self.condition_b.update(model)
