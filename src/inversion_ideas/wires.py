"""
Model wiring.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from numbers import Integral

import numpy as np
import numpy.typing as npt
from scipy.sparse import dia_array, diags_array
from scipy.sparse.linalg import LinearOperator

from .operators import BlockSquareMatrix
from .typing import Model, SparseArray


class Wires:
    """
    Collection of model slices.

    The ``Wires`` have capabilities for handling models with multiple physical
    properties.
    """

    def __init__(self, **kwargs):
        self._slices: dict[str, ModelSlice] = {}

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

    def __getitem__(self, value: str | Sequence[str]) -> "ModelSlice | MultiSlice":
        if not isinstance(value, str):
            if not isinstance(value, Sequence):
                # TODO: add msg
                raise TypeError()
            slices = {name: self._slices[name].slice for name in value}
            return MultiSlice(slices=slices, wires=self)
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


class _BaseModelSlice(ABC):
    """
    Base class for a ModelSlice.
    """

    @property
    @abstractmethod
    def size(self) -> int: ...

    @property
    @abstractmethod
    def full_size(self) -> int: ...

    @property
    @abstractmethod
    def wires(self) -> Wires: ...

    @property
    @abstractmethod
    def slice_matrix(self) -> dia_array[np.float64]: ...

    def expand_array(self, array: npt.NDArray) -> npt.NDArray:
        """
        Expand a 1D array by filling it with extra zeros.

        Parameters
        ----------
        array : (n,) array
            Array that will be filled. It must have the same number of elements as the
            model slice.

        Returns
        -------
        array : (m,) array
            Array filled with zeros on elements outside of the model slice.
        """
        return self.slice_matrix.T @ array

    def expand_matrix(
        self, matrix: npt.NDArray | LinearOperator | SparseArray
    ) -> "BlockSquareMatrix":
        """
        Expand a square matrix into a ``BlockSquareMatrix`` filling blocks with zeros.

        Parameters
        ----------
        matrix : array, sparse array or LinearOperator
            Square matrix that will be expanded

        Returns
        -------
        block_square_matrix : BlockSquareMatrix
            LinearOperator that represents the matrix filled with zeros outside of the
            block.
        """
        return BlockSquareMatrix(block=matrix, slice_matrix=self.slice_matrix)


class ModelSlice(_BaseModelSlice):
    """
    Class used to slice a model.

    .. important::

        Don't instantiate this class. Use the ``Wires`` class instead.
    """

    def __init__(self, name: str, slice: slice, wires: Wires):
        # TODO: Check that the slice has slice.start < slice.stop, no negative numbers,
        # no None as start/stop, and step == 1.
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

    @property
    def slice_matrix(self) -> dia_array[np.float64]:
        """
        Return a matrix that can be used to slice a model array.
        """
        ones = np.ones(self.size)
        shape = (self.size, self.full_size)
        return diags_array(
            ones, offsets=self.slice.start, shape=shape, dtype=np.float64
        )

    def extract(self, model: Model):
        if model.size != self.wires.size:
            # TODO: add msg
            raise ValueError()
        return model[self.slice]


class MultiSlice(_BaseModelSlice):
    """
    Multiple slices.

    .. important::

        Don't instantiate this class. Use the ``Wires`` class instead.
    """

    def __init__(self, slices: dict[str, slice], wires: Wires):
        # TODO: Check that the slices have slice.start < slice.stop, no negative
        # numbers, no None as start/stop, and step == 1.
        # TODO: check that keys in slices dict are all strings
        self._slices = slices
        self._wires = wires

    @property
    def names(self) -> list[str]:
        """
        Slices names.
        """
        return list(self.slices.keys())

    @property
    def slices(self) -> dict[str, slice]:
        return self._slices

    @property
    def wires(self) -> Wires:
        return self._wires

    @property
    def size(self) -> int:
        return sum(s.stop - s.start for s in self.slices.values())

    @property
    def full_size(self) -> int:
        return self.wires.size

    @property
    def slice_matrix(self) -> dia_array[np.float64]:
        """
        Return a matrix that can be used to slice a model array.
        """
        if not self.slices:
            # Add msg and choose right error type for empty slices
            raise ValueError()

        shape = (self.size, self.full_size)
        s = dia_array(shape)
        row = 0
        for slice_ in self.slices.values():
            offset = slice_.start - row
            index_start = row if offset >= 0 else row + offset
            length = slice_.stop - slice_.start
            diagonal = np.zeros(self.size, dtype=np.float64)
            diagonal[index_start : index_start + length] = 1.0
            s += diags_array(diagonal, offsets=offset, shape=shape, dtype=np.float64)
            row += length
        return s

    def extract(self, model: Model):
        if model.size != self.wires.size:
            # TODO: add msg
            raise ValueError()
        parts = [model[s] for s in self.slices.values()]
        return np.hstack(parts)
