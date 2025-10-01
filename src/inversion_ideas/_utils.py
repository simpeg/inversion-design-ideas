"""
Code utilities.

Objects in this submodule are meant to be private.
"""
from collections.abc import Iterator
from copy import copy

import numpy as np
import numpy.typing as npt


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
