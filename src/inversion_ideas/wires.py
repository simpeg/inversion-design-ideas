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

    Parameters
    ----------
    kwargs : dict
        Keyword arguments. Each key represents the label of a portion of the model, and
        its value should be an integer with the number of elements of that portion of
        the model.

    Examples
    --------
    Define a :class:`~inversion_ideas.Wires` object with two physical properties
    (density and susceptibility), that hold different amount of model values for each.

    >>> import numpy as np
    >>> wires = Wires(density=3, susceptibility=4)
    >>> wires.size
    7
    >>> wires.keys()
    dict_keys(['density', 'susceptibility'])

    Slice a model using the wires
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Define a model with 3 values for density and 4 for susceptibility:

    >>> model = np.array([0.1, 0.2, -0.1, 1e-3, 1e-2, 1e-1, 0.3])

    Slice the model to get the density values:

    >>> wires.density.extract(model)
    array([ 0.1,  0.2, -0.1])

    Slice the model to get the susceptibility values:

    >>> wires.susceptibility.extract(model)
    array([0.001, 0.01 , 0.1  , 0.3  ])

    Convert a model to a *model dictionary*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    >>> wires.array_to_dict(model)
    {'density': array([ 0.1,  0.2, -0.1]), 'susceptibility': array([0.001, 0.01 , 0.1  , 0.3  ])}


    Define a :class:`~inversion_ideas.Wires` object from a model dictionary
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    >>> model_dict = {
    ...     "conductivity": np.array([1e-4, 1e-2, 1e-1]),
    ...     "density": np.array([-0.3, 0.15, 0.01]),
    ... }
    >>> wires = Wires.from_dict(model_dict)
    >>> wires.keys()
    dict_keys(['conductivity', 'density'])

    We can use this ``Wires`` object to convert the model dictionary into an array:

    >>> model = wires.dict_to_array(model_dict)
    >>> model
    array([ 1.0e-04,  1.0e-02,  1.0e-01, -3.0e-01,  1.5e-01,  1.0e-02])

    Defining multiple model slices
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Let's say we have a model for the parameters of a sphere:

    >>> model_sphere = {
    ...     "radius": np.array([10.0]),
    ...     "center": np.array([0.0, 0.0, -15.0]),
    ...     "density": np.array([0.2]),
    ...     "susceptibility": np.array([1e-2]),
    ...     "conductivity": np.array([1e-3]),
    ... }

    We can create a ``Wires`` object from it:

    >>> wires = Wires.from_dict(model_sphere)

    We can access each individual ``ModelSlice`` as an attribute:

    >>> wires.center  # doctest: +SKIP

    Or by indexing it:

    >>> wires["center"]  # doctest: +SKIP

    And we can use these to extract pieces of the model array:

    >>> model = wires.dict_to_array(model_sphere)
    >>> wires["center"].extract(model)
    array([  0.,   0., -15.])

    We can also extract multiple slices:

    >>> wires["radius", "center", "density"].extract(model)
    array([ 10. ,   0. ,   0. , -15. ,   0.2])

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
    def from_dict(cls, model_dict: dict[str, Model]):
        """
        Create a ``Wires`` object from a model dictionary.

        Parameters
        ----------
        model_dict : dict
            Dictionary with string labels and 1d arrays as values.

        Returns
        -------
        Wires
        """
        kwargs = {key: array.size for key, array in model_dict.items()}
        return cls(**kwargs)

    def array_to_dict(self, model: Model):
        """
        Transform a model array into a dictionary.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        dict
            Dictionary with string labels and 1d arrays as values.
            Each pair of key-value in the dictionary represents a portion of the model
            array.
        """
        if model.size != self.size:
            # TODO: add msg
            raise ValueError()
        model_dict = {key: model[self[key].slice] for key in self.keys()}
        return model_dict

    def dict_to_array(self, model_dict: dict[str, Model]):
        """
        Ravel a model dictionary into a single 1D array.

        Parameters
        ----------
        model_dict : dict
            Dictionary with string labels and 1d arrays as values.

        Returns
        -------
        array
            1D array with model values.
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
