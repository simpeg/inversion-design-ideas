"""
Classes to represent objective functions.
"""

import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from copy import copy
from numbers import Integral, Real
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from ..typing import HasDiagonal, Model, SparseArray

FLOAT_TO_STR_PRECISION = 3


class Objective(ABC):
    """
    Abstract representation of an objective function.
    """

    _base_str = "φ"
    _base_latex = r"\phi"

    @abstractmethod
    def __init__(self):
        pass  # pragma: no cover

    @property
    @abstractmethod
    def n_params(self) -> int:
        """
        Number of model parameters.
        """

    @abstractmethod
    def __call__(self, model: Model) -> float:
        """
        Evaluate the objective function for a given model.
        """

    @abstractmethod
    def gradient(self, model: Model) -> npt.NDArray[np.float64]:
        """
        Evaluate the gradient of the objective function for a given model.
        """

    @abstractmethod
    def hessian(
        self, model: Model
    ) -> npt.NDArray[np.float64] | SparseArray | LinearOperator:
        """
        Evaluate the hessian of the objective function for a given model.
        """

    def hessian_diagonal(self, model: Model) -> npt.NDArray[np.float64]:
        """
        Get the main diagonal of the Hessian.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        (n_params,) array
            Array containing the diagonal of the Hessian.
        """
        hessian = self.hessian(model)
        if not isinstance(hessian, HasDiagonal):
            msg = (
                f"Cannot get 'hessian_diagonal' for objective function '{self}', "
                f"since its Hessian of type {type(hessian).__name__} doesn't implement "
                "a `diagonal` method."
            )
            raise TypeError(msg)
        return hessian.diagonal()

    def hessian_approx(self, model: Model) -> npt.NDArray[np.float64] | SparseArray:
        """
        Approximated version of the Hessian.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        (n_params, n_params) array or sparse array
            2D array that approximates the Hessian of the objective function.
        """
        hessian = self.hessian(model)
        if isinstance(hessian, LinearOperator):
            msg = (
                f"Cannot build a 'hessian_approx' for objective function '{self}', "
                "since its Hessian is a LinearOperator."
            )
            raise TypeError(msg)
        return hessian

    @property
    def name(self) -> str | None:
        """
        Name of the objective function.
        """
        return getattr(self, "_name", None)

    @name.setter
    def name(self, value: str | None):
        """
        Setter for the name property.
        """
        if not isinstance(value, str | None):
            msg = (
                f"Invalid name '{value}' of type '{type(value).__name__}'. "
                "Please provide a string or None."
            )
            raise TypeError(msg)
        self._name = value

    def set_name(self, value: str | None):
        """
        Set name for the objective function.
        """
        self.name = value
        return self  # return self so we can pipe this method

    def __repr__(self):
        title = f"{self._base_str}"
        if self.name is not None:
            title += f"{self.name}"
        return f"{title}(m)"

    def _repr_latex_(self):
        repr_ = f"{self._base_latex}"
        if self.name is not None:
            repr_ += rf"_{{{self.name}}}"
        return f"${repr_} (m)$"

    def info(self):
        """Get information about the objective function."""
        type_ = type(self)
        class_name = type_.__name__
        info = f"{class_name}\n"
        info += "-" * len(class_name) + "\n"
        info += f" • Class: {type_.__module__}.{type_.__name__}\n"
        info += f" • Memory address: {hex(id(self))}\n"
        info += f" • String representation: {self}\n"
        info += f" • name: {self.name}\n"
        info += f" • n_params: {self.n_params}"
        sys.stdout.write(info + "\n")

    def __add__(self, other) -> "Combo | Self":
        # Allow to add a zero to the objective function.
        # This is needed to add objective functions with the sum() function.
        if isinstance(other, Integral):
            if other != 0:
                msg = (
                    f"Cannot add objective function '{self}' with '{other}'."
                    "Objective functions cannot be added to integers other than zero."
                )
                raise ValueError(msg)
            return self
        return Combo([self, other])

    def __radd__(self, other) -> "Combo | Self":
        # Allow to add a zero to the objective function.
        # This is needed to add objective functions with the sum() function.
        if isinstance(other, Integral):
            if other != 0:
                msg = (
                    f"Cannot add objective function '{self}' with '{other}'."
                    "Objective functions cannot be added to integers other than zero."
                )
                raise ValueError(msg)
            return self
        return Combo([other, self])

    def __mul__(self, value: Real) -> "Scaled":
        return Scaled(value, self)

    def __rmul__(self, value):
        return self.__mul__(value)

    def __truediv__(self, denominator: Real):
        return self * (1.0 / denominator)  # type: ignore[operator]

    def __floordiv__(self, denominator):
        msg = "Floor division is not implemented for objective functions."
        raise TypeError(msg)

    def __iadd__(self, other) -> Self:
        msg = "Inplace addition is not implemented for this class."
        raise TypeError(msg)

    def __imul__(self, other: Real) -> Self:
        msg = "Inplace multiplication is not implemented for this class."
        raise TypeError(msg)

    def __itruediv__(self, value: Real) -> Self:
        msg = "Inplace division is not implemented for this class."
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

    def __call__(self, model: Model):
        """
        Evaluate the objective function.
        """
        return self.multiplier * self.function(model)

    def gradient(self, model: Model) -> npt.NDArray[np.float64]:
        """
        Evaluate the gradient of the objective function for a given model.
        """
        return self.multiplier * self.function.gradient(model)

    def hessian(
        self, model: Model
    ) -> npt.NDArray[np.float64] | SparseArray | LinearOperator:
        """
        Evaluate the hessian of the objective function for a given model.
        """
        return self.multiplier * self.function.hessian(model)

    def hessian_approx(self, model: Model) -> npt.NDArray[np.float64] | SparseArray:
        return self.multiplier * self.function.hessian_approx(model)

    def hessian_diagonal(self, model: Model) -> npt.NDArray[np.float64]:
        return self.multiplier * self.function.hessian_diagonal(model)

    def info(self):
        type_ = type(self)
        class_name = type_.__name__
        info = f"{class_name}\n"
        info += "-" * len(class_name) + "\n"
        info += f" • Class: {type_.__module__}.{type_.__name__}\n"
        info += f" • Memory address: {hex(id(self))}\n"
        info += f" • String representation: {self}\n"
        info += f" • n_params: {self.n_params}\n"
        info += f" • multiplier: {self.multiplier}\n"
        info += f" • function: {self.function} ({type(self.function).__name__})"
        sys.stdout.write(info + "\n")

    def __repr__(self):
        multiplier = _float_to_str(self.multiplier)
        phi_repr = f"{self.function}"
        # Add brackets in case that the function has a multiplier or is a Combo
        if isinstance(self.function, Iterable) or hasattr(self.function, "multiplier"):
            phi_repr = f"[{phi_repr}]"
        return f"{multiplier:} {phi_repr}"

    def _repr_latex_(self):
        multiplier = _float_to_str(self.multiplier)
        if "e" in multiplier:
            base, exp = multiplier.split("e")
            exp = exp.replace("+", "")
            exp = str(int(exp))
            multiplier = rf"{base} \cdot 10^{{{exp}}}"
        phi_str = self.function._repr_latex_().strip("$")
        # Add brackets in case that the function has a multiplier or is a Combo
        if isinstance(self.function, Iterable) or hasattr(self.function, "multiplier"):
            phi_str = f"[{phi_str}]"
        return rf"${multiplier} \, {phi_str}$"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Scaled):
            return False
        return self.multiplier == other.multiplier and self.function == other.function

    def __hash__(self):
        return hash((self.multiplier, self.function))

    def __imul__(self, value: Real) -> Self:
        self.multiplier *= value
        return self

    def __itruediv__(self, value: Real) -> Self:
        self.multiplier /= value
        return self


class Combo(Objective):
    """
    Sum of objective functions.
    """

    # Combo behaves like a list and therefore it's not hashable
    __hash__ = None

    def __init__(self, functions: list[Objective]):
        if not isinstance(functions, Sequence):
            msg = (
                f"Invalid 'functions' argument of type '{type(functions).__name__}'. "
                "It must be a sequence of `Objective` functions."
            )
            raise TypeError(msg)
        if not functions:
            msg = (
                "Invalid empty 'functions' argument. "
                "The list of objective functions must contain at least one function."
            )
            raise ValueError(msg)

        # Call the _get_n_params function to check if functions have the same n_params
        _get_n_params(functions)

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
        if not self._functions:
            msg = "Invalid empty `Combo` without functions."
            raise ValueError(msg)
        return self._functions

    @property
    def n_params(self) -> int:
        """
        Number of model parameters.
        """
        return _get_n_params(self.functions)

    def __call__(self, model: Model):
        """
        Evaluate the objective function.
        """
        return sum(f(model) for f in self.functions)

    def gradient(self, model: Model) -> npt.NDArray[np.float64]:
        """
        Evaluate the gradient of the objective function for a given model.
        """
        return sum(f.gradient(model) for f in self.functions)  # type: ignore[return-value]

    def hessian(
        self, model: Model
    ) -> npt.NDArray[np.float64] | SparseArray | LinearOperator:
        """
        Evaluate the hessian of the objective function for a given model.
        """
        return _sum_operators(f.hessian(model) for f in self.functions)

    def hessian_approx(self, model: Model) -> npt.NDArray[np.float64] | SparseArray:
        return _sum_operators(f.hessian_approx(model) for f in self.functions)

    def hessian_diagonal(self, model: Model) -> npt.NDArray[np.float64]:
        return sum(
            (f.hessian_approx(model) for f in self.functions),
            start=np.zeros(self.n_params, dtype=np.float64),
        )

    def flatten(self) -> "Combo":
        """
        Create a new flattened combo.

        Create a new ``Combo`` object by unpacking nested ``Combo``s in the current one.
        """
        return Combo(_unpack_combo(self.functions))

    def contains(self, objective) -> bool:
        """
        Check if the ``Combo`` contains the given objective function, recursively.
        """
        return _contains(self, objective)

    def info(self):
        """Get information about the combo objective function."""
        type_ = type(self)
        class_name = type_.__name__
        info = f"{class_name}\n"
        info += "-" * len(class_name) + "\n"
        info += f" • Class: {type_.__module__}.{type_.__name__}\n"
        info += f" • Memory address: {hex(id(self))}\n"
        info += f" • String representation: {self}\n"
        info += f" • n_params: {self.n_params}\n"
        info += f" • size: {len(self)}\n"
        info += " • functions:\n"
        for i, function in enumerate(self):
            info += (
                f"   {i:2d}) {function}: "
                f"{type(function).__name__} at {hex(id(function))}\n"
            )
        sys.stdout.write(info)

    def __repr__(self):
        functions = []
        for function in self.functions:
            function_str = repr(function)
            if isinstance(function, Iterable):
                function_str = f"[{function_str}]"
            functions.append(function_str)
        return " + ".join(functions)

    def _repr_latex_(self):
        functions = []
        for function in self.functions:
            function_str = function._repr_latex_().strip("$")
            if isinstance(function, Iterable):
                function_str = f"[{function_str}]"
            functions.append(function_str)
        phi_str = " + ".join(functions)
        return f"${phi_str}$"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Combo):
            return False
        if len(self) != len(other):
            return False
        for self_term, other_term in zip(self, other, strict=True):
            if self_term != other_term:
                return False
        return True

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


def _contains(combo: Combo, objective: Objective) -> bool:
    """
    Check if combo contains a given objective function, recursively.
    """
    for f in combo.functions:
        if f is objective:
            return True
        if isinstance(f, Combo) and _contains(f, objective):
            return True
        if isinstance(f, Scaled):
            if f.function is objective:
                return True
            if isinstance(f.function, Combo) and _contains(f.function, objective):
                return True
    return False


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
        functions_str = ", ".join(str(f) for f in functions)
        n_params_str = ", ".join(str(n) for n in n_params_list)
        msg = (
            f"Invalid objective functions {functions_str} with different n_params: "
            f"{n_params_str}, respectively."
        )
        raise ValueError(msg)
    return n_params


def _sum_operators(
    operators: Iterable[npt.NDArray | SparseArray | LinearOperator],
) -> npt.NDArray | SparseArray | LinearOperator:
    """
    Sum linear operators within an iterator.

    This function supports summing together
    :class:`~scipy.sparse.linalg.LinearOperator`s with Numpy arrays and sparse arrays.

    Parameters
    ----------
    operators : iterable
        Iterable containing a mixed collection of Numpy arrays, sparse arrays and
        :class:`~scipy.sparse.linalg.LinearOperator`s.

    Returns
    -------
    array or sparse array or LinearOperator

    Raises
    ------
    TypeError : if any operator is a sparse matrix.
    ValueError : if ``operators`` is empty.
    """
    if not isinstance(operators, Iterator):
        operators = iter(operators)

    # Define result as a copy of the first element in the iterator (if any).
    try:
        result = next(operators)
    except StopIteration as err:
        msg = "Invalid empty 'operators' iterator when summing."
        raise ValueError(msg) from err
    else:
        _raise_if_sparse_matrix(result)
        result = copy(result)

    # Sum over operators in the iterator
    for operator in operators:
        _raise_if_sparse_matrix(operator)
        if isinstance(operator, LinearOperator) or isinstance(result, LinearOperator):
            result = aslinearoperator(result)  # type: ignore[arg-type]
            result += aslinearoperator(operator)  # type: ignore[arg-type]
        else:
            result += operator  # type: ignore[operator]
    return result


def _raise_if_sparse_matrix(operator):
    """Raise TypeError if operator is a sparse matrix."""
    if isinstance(operator, spmatrix):
        msg = (
            f"Invalid sparse matrix '{operator}' when summing multiple operators. "
            "Make sure to use sparse arrays instead "
            "(https://docs.scipy.org/doc/scipy/reference/"
            "sparse.migration_to_sparray.html)."
        )
        raise TypeError(msg)


def _float_to_str(number: float, precision: int = FLOAT_TO_STR_PRECISION) -> str:
    """
    Format float to string.

    Formats a floating point number into string.

    Parameters
    ----------
    number : float
        Floating point number to represent as a string.
    precision : int
        Decimal point precision for positional and scientific representation. The
        ``precision`` is used to choose between a positional representation (e.g. 1.013)
        and a scientific notation. If the absolute value of the number is between
        ``10**(-precision)`` and ``10**precision``, then the positional representation
        will be used, otherwise the scientific notation will be chosen.
        It must be a positive integer.

    Returns
    -------
    str
        String representation of the floating point number.
    """
    if precision <= 0:
        msg = f"Invalid precision value '{precision}'. It must be a positive integer."
        raise ValueError(msg)
    if number == 0.0:
        return "0."
    min_bound, max_bound = 10 ** (-precision), 10**precision
    if min_bound <= np.abs(number) <= max_bound:
        return np.format_float_positional(number, precision=precision)
    return np.format_float_scientific(number, precision=precision)
