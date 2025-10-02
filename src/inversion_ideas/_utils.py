"""
Code utilities.

Objects in this submodule are meant to be private.
"""

from collections.abc import Callable, Iterable, Iterator
from copy import copy

import numpy as np
import numpy.typing as npt

from inversion_ideas.base.objective_function import Objective, Scaled


def prod_arrays(arrays: Iterator[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """
    Compute product of arrays within an iterator.

    Parameters
    ----------
    arrays : Iterator
        Iterator with arrays.
    """
    if not arrays:
        msg = "Invalid empty 'arrays' array when summing."
        raise ValueError(msg)

    result = copy(next(arrays))
    for array in arrays:
        result *= array
    return result


def extract_from_combo(
    objective: Objective, condition: Callable[[Objective], bool]
) -> list[Objective]:
    """
    Extract objective functions within a Combo objective function recursively.

    .. important::

        Scaled objective functions are not going to be included in the extracted list,
        but their underlying functions are going to be considered.

    Parameters
    ----------
    objective : Objective
        Objective function to explore.
    condition : Callable
        Condition that each objective function must satisfy to be included in the
        returned list.

    Returns
    -------
    list of Objective
        List of extracted objective functions within the ``objective`` that satisfy the
        ``condition``.
    """
    if not isinstance(objective, Iterable):
        if isinstance(objective, Scaled):
            extracted = extract_from_combo(objective.function, condition)
        else:
            extracted = [objective] if condition(objective) else []
        return extracted

    extracted = []
    for reg in objective:
        extracted += extract_from_combo(reg, condition)

    return extracted
