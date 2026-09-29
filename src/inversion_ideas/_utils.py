"""
Code utilities.

Objects in this submodule are meant to be private.
"""

from collections.abc import Callable, Iterable, Iterator
from copy import copy

import numpy as np
import numpy.typing as npt

from .base.objective_function import Objective, Scaled

try:
    import xxhash
except ImportError:
    import hashlib

    HASHING_FUNCTION = hashlib.sha256
else:
    HASHING_FUNCTION = xxhash.xxh32


def compute_hash(*args, **kwargs):
    """
    Compute hash of inputs using the default hashing function.

    If ``xxhash`` is installed, then :func:`xxhash.xx32` will be used to compute the hash. Otherwise, :func:`hashlib.sha256` will be used instead.

    Parameters
    ----------
    *args :
        Positional arguments will be passed to the hashing function.
    **kwargs :
        Keyword arguments will be passed to the hashing function.
    """
    return HASHING_FUNCTION(*args, **kwargs)


def hash_to_str(hash):
    """
    Convert hash to string in the form of ``f"{hash.name}:{hash.hexdigest()}"``.

    Parameters
    ----------
    hash : hash-like object
        Hash-like object with ``name`` property and ``hexdigest`` method.

    Returns
    -------
    str
    """
    return f"{hash.name}:{hash.hexdigest()}"


def array_to_str(array: npt.NDArray, *, single_line=True, threshold=10, **kwargs):
    """
    Represent Numpy arrays as strings.

    Use this function to simplify printouts like debug lines.

    Parameters
    ----------
    array : array
        Numpy array to represent as string.
    single_line : bool, optional
        Whether to show the array in a single line or allow Numpy to break lines.
    threshold : int, optional
        Total number of array elements which trigger summarization rather than full
        repr.
    kwargs : dict
        Extra keyword arguments passed to :func:`numpy.printoptions`.

    Returns
    -------
    str
    """
    kwargs["threshold"] = threshold
    with np.printoptions(**kwargs):
        string = f"{array}"
        if single_line:
            string = string.replace("\n", "")
        return string


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
