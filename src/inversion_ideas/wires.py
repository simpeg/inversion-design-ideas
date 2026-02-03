"""
Model wiring.
"""

from numbers import Integral

import numpy as np
import numpy.typing as npt
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator

from .operators import BlockColumnMatrix
from .typing import Model


class Wires:

    def __init__(self, **kwargs):
        self._slices = {}

        current_index: int = 0
        for key, value in kwargs.items():
            if not isinstance(key, str):
                # TODO: Add msg
                raise TypeError()

            if isinstance(value, Integral):
                slice_ = slice(current_index, current_index + value)
                current_index += int(value)
            else:
                # TODO: Add msg
                raise TypeError()

            self._slices[key] = ModelSlice(name=key, slice=slice_, wires=self)

        self._size = sum([slice_.size for slice_ in self._slices.values()])

    @property
    def size(self) -> int:
        """
        Total size of the model that can be sliced.
        """
        return self._size

    def keys(self):
        """
        Return keys in the wiring.
        """
        return self._slices.keys()

    def __getattr__(self, value: str) -> "ModelSlice":
        if value not in self._slices:
            # TODO: Add msg
            raise AttributeError()
        return self._slices[value]

    def __getitem__(self, value: str) -> "ModelSlice":
        return self._slices[value]

    @classmethod
    def create_from(cls, model_dict: dict[str, Model]):
        """
        Create a ``Wires`` object from a model dictionary.
        """
        kwargs = {key: array.size for key, array in model_dict.items()}
        return cls(**kwargs)

    def array_to_dict(self, model: Model):
        """
        Transform a model array into a dictionary.
        """
        if model.size != self.size:
            # TODO: add msg
            raise ValueError()
        model_dict = {key: model[self[key].slice] for key in self.keys()}
        return model_dict

    def dict_to_array(self, model_dict: dict[str, Model]):
        """
        Ravel a model dictionary into a single 1D array.
        """
        # TODO: we should run sanity checks to ensure that the model_dict is compatible
        # with this wiring.
        model = np.hstack([model_dict[key] for key in self.keys()])
        return model


class ModelSlice:
    """
    Class used to slice a model.

    .. important::

        Don't instantiate this class. Use the ``Wires`` class instead.
    """

    def __init__(self, name: str, slice: slice, wires: Wires):
        if not isinstance(name, str):
            # TODO: Add msg
            raise TypeError()
        self._name = name
        self._slice = slice
        self._wires = wires

    @property
    def name(self) -> str:
        return self._name

    @property
    def slice(self) -> slice:
        return self._slice

    @property
    def size(self) -> int:
        return self._slice.stop - self._slice.start

    @property
    def full_size(self) -> int:
        return self.wires.size

    @property
    def wires(self) -> Wires:
        return self._wires

    def extract(self, model: Model):
        if model.size != self.wires.size:
            # TODO: add msg
            raise ValueError()
        return model[self.slice]

    def expand_array(self, array: npt.NDArray) -> npt.NDArray:
        """
        Expand a 1D array with zeros outside the model slice.
        """
        if not isinstance(array, np.ndarray):
            msg = f"Invalid array of type {type(array).__name__}."
            raise TypeError(msg)
        if array.ndim != 1:
            msg = f"Invalid array with {array.ndim} dimensions. It must be a 1D array."
            raise ValueError(msg)
        out = np.zeros(self.full_size, dtype=array.dtype)
        out[self.slice] = array
        return out

    def expand_matrix(
        self, matrix: npt.NDArray | sparray | LinearOperator
    ) -> BlockColumnMatrix:
        """
        Expand a matrix that operates on the model slice into a block matrix.
        """
        if matrix.ndim != 2:
            msg = (
                f"Invalid matrix with {matrix.ndim} dimensions. "
                "It must be a 2D matrix or LinearOperator."
            )
            raise ValueError(msg)
        if matrix.shape[1] != self.size:
            # TODO: Complete the error message
            msg = f"Invalid matrix with shape '{matrix.shape}'."
            raise ValueError(msg)
        return BlockColumnMatrix(
            block=matrix,
            index_start=self.slice.start,
            n_cols=self.full_size,
        )
