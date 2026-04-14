"""
Test regularization classes.
"""

import numpy as np
import pytest
from discretize.tensor_mesh import TensorMesh
from scipy.sparse import dia_array, sparray

from inversion_ideas import Flatness, Smallness


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


class TestSmallness:
    """
    Test the :class:`inversion_ideas.Smallness` regularization class.
    """

    @pytest.fixture
    def mesh(self):
        hx = [(1.0, 10)]
        h = [hx, hx, hx]
        return TensorMesh(h, origin="CCN")

    @pytest.fixture
    def active_cells(self, mesh: TensorMesh):
        active_cells = np.ones(mesh.n_cells, dtype=bool)
        _, _, z = mesh.cell_centers.T
        active_cells[z > -1.0] = False
        assert not active_cells.all()
        return active_cells

    def test_smallness(self, mesh, active_cells):
        n_active = active_cells.sum()
        cell_weights = np.full(n_active, fill_value=0.1)
        reference_model = np.full(n_active, 1e-8)
        smallness = Smallness(
            mesh,
            active_cells=active_cells,
            cell_weights=cell_weights,
            reference_model=reference_model,
        )

        model = np.random.default_rng(seed=12312).uniform(size=n_active)

        # Test call
        result = smallness(model)
        assert np.isscalar(result)
        expected = np.sum(
            mesh.cell_volumes[active_cells]
            * cell_weights
            * (model - reference_model) ** 2
        )
        np.testing.assert_allclose(result, expected)

        # Test gradient
        gradient = smallness.gradient(model)
        expected = (
            2
            * mesh.cell_volumes[active_cells]
            * cell_weights
            * (model - reference_model)
        )
        assert gradient.size == model.size
        np.testing.assert_allclose(gradient, expected)

        # Test hessian
        hessian = smallness.hessian(model)
        assert hessian.shape == (model.size, model.size)
        assert isinstance(hessian, dia_array)
        assert hessian.offsets == 0  # should be a diagonal matrix (only main diag)
        expected_diagonal = 2 * mesh.cell_volumes[active_cells] * cell_weights
        np.testing.assert_allclose(hessian.diagonal(), expected_diagonal)
