"""
Base classes for defining conditions.

Conditions are callable objects that return either a bool when either a certain
condition is met or not. They are use to define abstract objects like stopping criterion
for inversions. We can use binary operators (and, or, xor) to logically group multiple
conditions together.
"""

from abc import ABC, abstractmethod

from rich.panel import Panel
from rich.tree import Tree

from ..typing import Model


def _get_info_title(condition, model) -> str:
    """
    Generate title for condition's information.
    """
    status = condition(model)
    checkbox = "x" if status else " "
    color = "green" if status else "red"
    text = rf"[bold {color}]\[{checkbox}] {type(condition).__name__}[/bold {color}]"
    return text


class Condition(ABC):
    """
    Base abstract class for conditions.

    .. important::

        This class is not meant to be instantiated. Use it to create child classes in
        order to implement custom condition objects.
    """

    @abstractmethod
    def __call__(self, model: Model) -> bool:
        """
        Evaluate the condition on a given model.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        bool
            Whether the condition is True or False.
        """

    def update(self, model: Model) -> None:  # noqa: B027
        """
        Update the condition.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
        """
        # This is not an abstract method. Children classes can choose to override it if
        # necessary. The base class implements it to provide a common interface, even
        # for those children that don't implement it.

    def initialize(self):  # noqa: B027
        """
        Initialize the condition.
        """
        # This is not an abstract method. Children classes can choose to override it if
        # necessary. The base class implements it to provide a common interface, even
        # for those children that don't implement it.

    def info(self, model: Model) -> Tree:
        """
        Display information about the condition for a given model.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        rich.tree.Tree
            :class:`rick.tree.Tree` object containing information about the condition.
        """
        return Tree(_get_info_title(self, model))

    def __and__(self, other) -> "LogicalAnd":
        return LogicalAnd(self, other)

    def __rand__(self, other) -> "LogicalAnd":
        return LogicalAnd(other, self)

    def __or__(self, other) -> "LogicalOr":
        return LogicalOr(self, other)

    def __ror__(self, other) -> "LogicalOr":
        return LogicalOr(other, self)

    def __xor__(self, other) -> "LogicalXor":
        return LogicalXor(self, other)

    def __rxor__(self, other) -> "LogicalXor":
        return LogicalXor(other, self)

    def __iand__(self, other):
        msg = "Inplace AND binary operation is not supported for conditions."
        raise TypeError(msg)

    def __ior__(self, other):
        msg = "Inplace OR binary operation is not supported for conditions."
        raise TypeError(msg)

    def __ixor__(self, other):
        msg = "Inplace XOR binary operation is not supported for conditions."
        raise TypeError(msg)


class _Mixin(ABC):
    """
    Mixin class for combo conditions.

    .. important::

        This class is not meant to be instantiated.
        This is a mixin class intended to provide implementation of certain methods
        for the condition classes that group together two other conditions.

    """

    def __init__(self, condition_a, condition_b):
        self.condition_a = condition_a
        self.condition_b = condition_b

    @abstractmethod
    def __call__(self, model: Model) -> bool: ...

    def update(self, model: Model):
        for condition in (self.condition_a, self.condition_b):
            if hasattr(condition, "update"):
                condition.update(model)

    def info(self, model: Model) -> Tree:
        status = self(model)
        checkbox = "x" if status else " "
        color = "green" if status else "red"
        text = rf"[bold {color}]\[{checkbox}] {type(self).__name__}[/bold {color}]"
        tree = Tree(text, guide_style=color)
        for condition in (self.condition_a, self.condition_b):
            if hasattr(condition, "info"):
                subtree = condition.info(model)
                if isinstance(condition, _Mixin):
                    tree.add(subtree)
                else:
                    color = "green" if condition(model) else "red"
                    tree.add(Panel(subtree, border_style=color))
            else:
                raise NotImplementedError()
        return tree

    def initialize(self):
        for condition in (self.condition_a, self.condition_b):
            if hasattr(condition, "initialize"):
                condition.initialize()


class LogicalAnd(_Mixin, Condition):
    """
    Combo condition for the AND operation between two other conditions.

    .. important::

        This class is not meant to be instantiated. It should only be used when
        combining two conditions through the ``&`` (AND) operator.

    """

    def __call__(self, model: Model) -> bool:
        return self.condition_a(model) and self.condition_b(model)


class LogicalOr(_Mixin, Condition):
    """
    Combo condition for the OR operation between two other conditions.

    .. important::

        This class is not meant to be instantiated. It should only be used when
        combining two conditions through the ``|`` (OR) operator.

    """

    def __call__(self, model: Model) -> bool:
        return self.condition_a(model) or self.condition_b(model)


class LogicalXor(_Mixin, Condition):
    """
    Combo condition for the XOR operation between two other conditions.

    .. important::

        This class is not meant to be instantiated. It should only be used when
        combining two conditions through the ``^`` (XOR) operator.

    """

    def __call__(self, model: Model) -> bool:
        return self.condition_a(model) ^ self.condition_b(model)
