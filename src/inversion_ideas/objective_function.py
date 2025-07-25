"""
Classes to represent objective functions.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


class Objective(ABC):
    """
    Abstract representation of an objective function.
    """

    _base_str = "φ"
    _base_latex = r"\phi"
    name = None

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        """
        Number of model parameters.
        """

    @abstractmethod
    def __call__(self, model: NDArray) -> float:
        """
        Evaluate the objective function for a given model.
        """

    @abstractmethod
    def gradient(self, model: NDArray) -> NDArray[np.float64]:
        """
        Evaluate the gradient of the objective function for a given model.
        """

    @abstractmethod
    def hessian(self, model: NDArray) -> NDArray[np.float64]:
        """
        Evaluate the hessian of the objective function for a given model.
        """

    def set_name(self, value):
        """
        Set name for the objective function.
        """
        if not (isinstance(value, str) or value is None):
            msg = (
                f"Invalid name '{value}' of type {type(value)}. "
                "Please provide a string or None."
            )
            raise TypeError(msg)
        self.name = value

    def __repr__(self):
        repr_ = f"{self._base_str}"
        if self.name is not None:
            repr_ += f"{self.name}"
        return f"{repr_}(m)"

    def _repr_latex_(self):
        repr_ = f"{self._base_latex}"
        if self.name is not None:
            repr_ += rf"_{{{self.name}}}"
        return f"${repr_} (m)$"

    def __add__(self, other) -> "Combo":
        return Combo([self, other])

    def __radd__(self, other) -> "Combo":
        return Combo([other, self])

    def __mul__(self, value) -> "Scaled":
        return Scaled(value, self)

    def __rmul__(self, value):
        return self.__mul__(value)

    def __div__(self, denominator):
        return self * (1.0 / denominator)

    def __truediv__(self, denominator):
        return self * (1.0 / denominator)


class Scaled(Objective):
    """
    Scaled objective function.
    """

    def __init__(self, multiplier, function):
        self.multiplier = multiplier
        self.function = function

    @property
    def n_params(self) -> int:
        """
        Number of model parameters.
        """
        return self.function.n_params

    def __call__(self, model):
        """
        Evaluate the objective function.
        """
        return self.multiplier * self.function(model)

    def gradient(self, model: NDArray) -> NDArray[np.float64]:
        """
        Evaluate the gradient of the objective function for a given model.
        """
        return self.multiplier * self.function.gradient(model)

    def hessian(self, model: NDArray) -> NDArray[np.float64]:
        """
        Evaluate the hessian of the objective function for a given model.
        """
        return self.multiplier * self.function.hessian(model)

    def __repr__(self):
        fmt = ".2e" if np.abs(self.multiplier) > 1e3 else ".2f"
        phi_repr = f"{self.function}"
        if isinstance(self.function, Iterable):
            phi_repr = f"[{phi_repr}]"
        return f"{self.multiplier:{fmt}} {phi_repr}"

    def _repr_latex_(self):
        fmt = (
            ".2e"
            if np.abs(self.multiplier) > 1e2 or np.abs(self.multiplier) < 1e-2
            else ".2f"
        )
        multiplier_str = f"{self.multiplier:{fmt}}"
        if "e" in multiplier_str:
            base, exp = multiplier_str.split("e")
            exp = exp.replace("+", "")
            exp = str(int(exp))
            multiplier_str = rf"{base} \cdot 10^{{{exp}}}"
        phi_str = self.function._repr_latex_().strip("$")
        # Add parenthesis in case that the function is a collection
        if isinstance(self.function, Iterable):
            phi_str = f"[ {phi_str} ]"
        return rf"${multiplier_str} \, {phi_str}$"


class Combo(Objective):
    """
    Sum of objective functions.
    """

    def __init__(self, functions):
        _get_n_params(functions)  # check if functions have the same n_params
        self._functions = functions

    def __iter__(self):
        return (f for f in self.functions)

    def __getitem__(self, index):
        return self.functions[index]

    @property
    def functions(self) -> list[Objective]:
        """
        List of objective functions in the sum.
        """
        return self._functions

    @property
    def n_params(self) -> int:
        """
        Number of model parameters.
        """
        return _get_n_params(self.functions)

    def __call__(self, model):
        """
        Evaluate the objective function.
        """
        return sum(f(model) for f in self.functions)

    def gradient(self, model: NDArray) -> NDArray[np.float64]:
        """
        Evaluate the gradient of the objective function for a given model.
        """
        return sum(f.gradient(model) for f in self.functions)

    def hessian(self, model: NDArray) -> NDArray[np.float64]:
        """
        Evaluate the hessian of the objective function for a given model.
        """
        return sum(f.hessian(model) for f in self.functions)

    def __repr__(self):
        return " + ".join(repr(f) for f in self.functions)

    def _repr_latex_(self):
        return " + ".join(f._repr_latex_() for f in self.functions)


def _unpack_combo(functions: list) -> list:
    """
    Unpack combo objective functions.
    """
    unpacked = []
    for f in functions:
        if isinstance(f, Iterable):
            unpacked.extend(_unpack_combo(f))
        else:
            unpacked.append(f)
    return unpacked


def _get_n_params(functions: list) -> int:
    """
    Get number of parameters of a list of objective functions.

    Parameters
    ----------
    functions : list of Objective
        List of objective functions.

    Returns
    -------
    int
        Number of parameters of every objective function in the list.

    Raises
    ------
    ValueError
        If any of the objective functions in the list have different number of
        parameters.
    """
    n_params_list = [f.n_params for f in functions]
    n_params = n_params_list[0]
    if not all(p == n_params for p in n_params_list):
        msg = "Invalid objective functions with different n_params."
        raise ValueError(msg)
    return n_params
