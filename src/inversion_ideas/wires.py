"""
Model wiring.
"""

from numbers import Integral

import numpy as np
import numpy.typing as npt
from scipy.sparse import dia_array, diags_array, sparray
from scipy.sparse.linalg import LinearOperator

from .typing import Model


class Wires:
    """
    Collection of model slices.

    The ``Wires`` have capabilities for handling models with multiple physical
    properties.
    """

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
        self, matrix: npt.NDArray | LinearOperator | sparray
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


class BlockSquareMatrix(LinearOperator):
    r"""
    Operator that represents a square matrix with a non-zero block surrounded by zeros.

    Use this class to represent hessian matrices that are built from smaller matrices
    that operate only on a subset of the entire model. Only a block of that hessian will
    be non-zero (the one related to the relevant model elements for that objective
    function), while the rest of the matrix will be filled of zeros.

    Parameters
    ----------
    block : array, sparse array or LinearOperator
        Square block matrix.
    slice_matrix : dia_array
        Diagonal array to represent the slicing matrix.

    Notes
    -----
    This ``LinearOperator`` represents square matrices like:

    .. math::

        \mathbf{H} = \begin{bmatrix}
            0 & 0          & 0\\
            0 & \mathbf{B} & 0\\
            0 & 0          & 0
            \end{bmatrix}

    where :math:`\mathbf{B}` is a non-zero square block matrix that sits in the diagonal
    of :math:`\mathbf{H}`. The matrix :math:`\mathbf{H}` can be built as:

    .. math::

        \mathbf{H} = \mathbf{s}^T \mathbf{B} \mathbf{s}

    where :math:`\mathbf{s}` is the *slicing matrix*.
    """

    def __init__(
        self,
        block: npt.NDArray | LinearOperator | sparray,
        slice_matrix: dia_array,
    ):
        # TODO:
        # - [ ] raise error if the block plus the offset doesn't fit in the shape
        # - [x] raise error if the block is not square
        # - [x] raise error if the shape is not square
        if block.shape[0] != block.shape[1]:
            msg = (
                f"Invalid block matrix '{block}' with shape '{block.shape}'. "
                "It must be a square matrix."
            )
            raise ValueError(msg)

        if slice_matrix.shape[0] != block.shape[1]:
            msg = (
                f"Invalid block '{block}' and slice_matrix '{slice_matrix}' with "
                f"shapes '{block.shape}' and {slice_matrix.shape}."
            )
            raise ValueError(msg)

        if slice_matrix.shape[1] <= block.shape[1]:
            # TODO: improve msg
            msg = "block is larger than slice matrix"
            raise ValueError(msg)

        _, full_size = slice_matrix.shape
        shape = (full_size, full_size)
        super().__init__(shape=shape, dtype=block.dtype)

        self._block = block
        self._slice_matrix = slice_matrix

    @property
    def block(self):
        return self._block

    @property
    def slice_matrix(self) -> dia_array:
        return self._slice_matrix

    def _matvec(self, x):
        """
        Dot product between the matrix and a vector.
        """
        result = self.slice_matrix.T @ (self.block @ (self.slice_matrix @ x))
        return result

    def _rmatvec(self, x):
        """
        Dot product between the transposed matrix and a vector.
        """
        result = self.slice_matrix @ (self.block.T @ (self.slice_matrix.T @ x))
        return result

    def diagonal(self):
        """
        Diagonal of the matrix.
        """
        # TODO: check if the block has a diagonal method (implement a protocol for it)
        diagonal = self.block.diagonal()
        return self.slice_matrix.T @ diagonal
