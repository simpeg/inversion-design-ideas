"""
Classes to represent objective functions.
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from copy import copy
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator, aslinearoperator


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
    def __call__(self, model: npt.NDArray) -> float:
        """
        Evaluate the objective function for a given model.
        """

    @abstractmethod
    def gradient(self, model: npt.NDArray) -> npt.NDArray[np.float64]:
        """
        Evaluate the gradient of the objective function for a given model.
        """

    @abstractmethod
    def hessian(
        self, model: npt.NDArray
    ) -> npt.NDArray[np.float64] | sparray | LinearOperator:
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

    def __iadd__(self, other) -> "Combo":  # noqa: PYI034
        msg = "Inplace addition is not implemented for this class."
        raise TypeError(msg)

    def __imul__(self, other) -> "Scaled":  # noqa: PYI034
        msg = "Inplace multiplication is not implemented for this class."
        raise TypeError(msg)


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

    def gradient(self, model: npt.NDArray) -> npt.NDArray[np.float64]:
        """
        Evaluate the gradient of the objective function for a given model.
        """
        return self.multiplier * self.function.gradient(model)

    def hessian(
        self, model: npt.NDArray
    ) -> npt.NDArray[np.float64] | sparray | LinearOperator:
        """
        Evaluate the hessian of the objective function for a given model.
        """
        return self.multiplier * self.function.hessian(model)

    def __repr__(self):
        fmt = ".2e" if np.abs(self.multiplier) > 1e3 else ".2f"
        phi_repr = f"{self.function}"
        # Add brackets in case that the function has a multiplier or is a Combo
        if isinstance(self.function, Iterable) or hasattr(self.function, "multiplier"):
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
        # Add brackets in case that the function has a multiplier or is a Combo
        if isinstance(self.function, Iterable) or hasattr(self.function, "multiplier"):
            phi_str = f"[ {phi_str} ]"
        return rf"${multiplier_str} \, {phi_str}$"

    def __imul__(self, value) -> Self:
        self.multiplier *= value
        return self


class Combo(Objective):
    """
    Sum of objective functions.
    """

    def __init__(self, functions):
        _get_n_params(functions)  # check if functions have the same n_params
        self._functions = functions

    def __iter__(self):
        return (f for f in self.functions)

    def __len__(self):
        return len(self._functions)

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

    def gradient(self, model: npt.NDArray) -> npt.NDArray[np.float64]:
        """
        Evaluate the gradient of the objective function for a given model.
        """
        return sum(f.gradient(model) for f in self.functions)

    def hessian(
        self, model: npt.NDArray
    ) -> npt.NDArray[np.float64] | sparray | LinearOperator:
        """
        Evaluate the hessian of the objective function for a given model.
        """
        return _sum(f.hessian(model) for f in self.functions)

    def flatten(self) -> "Combo":
        """
        Create a new flattened combo.

        Create a new ``Combo`` object by unpacking nested ``Combo``s in the current one.
        """
        return Combo(_unpack_combo(self.functions))

    def __repr__(self):
        functions = []
        for function in self.functions:
            function_str = repr(function)
            if isinstance(function, Iterable):
                function_str = f"[ {function_str} ]"
            functions.append(function_str)
        return " + ".join(functions)

    def _repr_latex_(self):
        functions = []
        for function in self.functions:
            function_str = function._repr_latex_().strip("$")
            if isinstance(function, Iterable):
                function_str = f"[ {function_str} ]"
            functions.append(function_str)
        phi_str = " + ".join(functions)
        return f"$ {phi_str} $"

    def __iadd__(self, other) -> Self:
        if other.n_params != self.n_params:
            msg = (
                f"Trying to add objective function '{other}' with invalid "
                f"n_params ({other.n_params}) different from the one of "
                f"'{self}' ({self.n_params})."
            )
            raise ValueError(msg)
        self._functions.append(other)
        return self


def _unpack_combo(functions: Iterable) -> list:
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


def _sum(
    operators: Iterator[npt.NDArray | sparray | LinearOperator],
) -> npt.NDArray | sparray | LinearOperator:
    """
    Sum objects within an iterator.

    This function supports summing together ``LinearOperators`` with Numpy arrays and
    sparse arrays.
    """
    if not operators:
        msg = "Invalid empty 'operators' array when summing."
        raise ValueError(msg)

    result = copy(next(operators))
    for operator in operators:
        if isinstance(operator, LinearOperator) or isinstance(result, LinearOperator):
            result = aslinearoperator(result)
            result += aslinearoperator(operator)
        else:
            result += operator
    return result
