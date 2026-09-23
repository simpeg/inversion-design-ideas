"""
Utilities for the base submodule.

Objects in this submodule are meant to be private.
"""

import numpy as np

FLOAT_TO_STR_PRECISION = 3


def float_to_str(number: float, precision: int = FLOAT_TO_STR_PRECISION) -> str:
    r"""
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

    Examples
    --------
    >>> float_to_str(1.0)
    '1.'

    >>> float_to_str(-1.0)
    '-1.'

    >>> float_to_str(1e3)
    '1000.'

    >>> float_to_str(2e3)
    '2.e+03'

    >>> float_to_str(0.002)
    '0.002'

    >>> float_to_str(0.0002)
    '2.e-04'
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


def float_to_latex(number: float, precision: int = FLOAT_TO_STR_PRECISION) -> str:
    r"""
    Format a float to a LaTeX string.

    Formats a floating point number into string in LaTeX mathematical form.

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
        String representation of the floating point number in LaTeX math form.

    Examples
    --------
    >>> float_to_latex(1.0)
    '1.'

    >>> float_to_latex(-1.0)
    '-1.'

    >>> float_to_latex(1e3)
    '1000.'

    >>> float_to_latex(2e3)
    '2. \\cdot 10^{3}'

    >>> float_to_latex(0.002)
    '0.002'

    >>> float_to_latex(0.0002)
    '2. \\cdot 10^{-4}'
    """
    multiplier = float_to_str(number, precision=precision)
    if "e" in multiplier:
        base, exp = multiplier.split("e")
        exp = exp.replace("+", "")
        exp = str(int(exp))
        multiplier = rf"{base} \cdot 10^{{{exp}}}"
    return multiplier
