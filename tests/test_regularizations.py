"""
Test regularization classes.
"""

import pytest
from discretize.tensor_mesh import TensorMesh
from scipy.sparse import sparray

from inversion_ideas import Flatness


class TestBugfixFlatness:
    """
    Test bugfix: check the `_cell_gradient` returns a sparse array and not a matrix.
    """

    @pytest.fixture
    def mesh(self):
        hx = [(1.0, 5)]
        h = [hx, hx, hx]
        return TensorMesh(h=h)

    @pytest.mark.parametrize("direction", ["x", "y", "z"])
    def test_cell_gradient_type(self, mesh, direction):
        flatness = Flatness(mesh, direction=direction)
        assert isinstance(flatness._cell_gradient, sparray)
