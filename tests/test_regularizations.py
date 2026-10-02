"""
Test regularization classes.
"""

import numpy as np
import pytest
from discretize.tensor_mesh import TensorMesh
from scipy.sparse import dia_array, sparray

from inversion_ideas import Flatness, Smallness

from .utils import derivative_convergence_test, derivative_test


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


class MeshBasedTest:
    """
    Base class for mesh-based regularizations.
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


class TestSmallness(MeshBasedTest):
    """
    Test the :class:`inversion_ideas.Smallness` regularization class.
    """

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

    @pytest.mark.parametrize("order", [1, 2], ids=["first-order", "second-order"])
    def test_derivative(self, mesh, active_cells, order):
        """
        Test gradient and hessian by comparison with Taylor series expansion.
        """
        n_active = active_cells.sum()
        cell_weights = np.full(n_active, fill_value=0.1)
        reference_model = np.full(n_active, 1e-8)
        smallness = Smallness(
            mesh,
            active_cells=active_cells,
            cell_weights=cell_weights,
            reference_model=reference_model,
        )
        rng = np.random.default_rng(seed=12312)
        model = rng.uniform(low=-1.0, high=1.0, size=n_active)

        # Define whether to test the gradient or the Hessian
        if order == 1:
            delta_m = rng.normal(scale=1e-4, size=n_active)
            function, derivative = smallness, smallness.gradient
        elif order == 2:
            delta_m = rng.normal(size=n_active)
            function, derivative = smallness.gradient, smallness.hessian
        else:
            raise ValueError()

        # Perform derivative test
        derivative_test(function, derivative, model, delta_m)

    def test_derivative_convergence(self, mesh, active_cells):
        """
        Test gradient through a convergence test of Taylor series approximation.
        """
        n_active = active_cells.sum()
        cell_weights = np.full(n_active, fill_value=0.1)
        reference_model = np.full(n_active, 1e-8)
        smallness = Smallness(
            mesh,
            active_cells=active_cells,
            cell_weights=cell_weights,
            reference_model=reference_model,
        )
        rng = np.random.default_rng(seed=12312)
        model = rng.uniform(low=-1.0, high=1.0, size=n_active)
        delta_m = rng.normal(size=n_active)
        derivative_convergence_test(smallness, smallness.gradient, model, delta_m)


@pytest.mark.parametrize("direction", ["x", "y", "z"])
class TestFlatness(MeshBasedTest):
    """
    Test the :class:`inversion_ideas.Flatness` regularization class.
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

    @pytest.mark.parametrize("order", [1, 2], ids=["first-order", "second-order"])
    def test_derivative(self, mesh, active_cells, direction, order):
        """
        Test gradient and hessian by comparison with Taylor series expansion.
        """
        n_active = active_cells.sum()
        cell_weights = np.full(n_active, fill_value=0.1)
        reference_model = np.full(n_active, 1e-8)
        flatness = Flatness(
            mesh,
            direction=direction,
            active_cells=active_cells,
            cell_weights=cell_weights,
            reference_model=reference_model,
        )
        rng = np.random.default_rng(seed=12312)
        model = rng.uniform(low=-1.0, high=1.0, size=n_active)

        # Define whether to test the gradient or the Hessian
        if order == 1:
            delta_m = rng.normal(scale=1e-4, size=n_active)
            function, derivative = flatness, flatness.gradient
        elif order == 2:
            delta_m = rng.normal(size=n_active)
            function, derivative = flatness.gradient, flatness.hessian
        else:
            raise ValueError()

        # Perform derivative test
        derivative_test(function, derivative, model, delta_m)

    def test_derivative_convergence(self, mesh, active_cells, direction):
        """
        Test gradient through a convergence test of Taylor series approximation.
        """
        n_active = active_cells.sum()
        cell_weights = np.full(n_active, fill_value=0.1)
        reference_model = np.full(n_active, 1e-8)
        flatness = Flatness(
            mesh,
            direction=direction,
            active_cells=active_cells,
            cell_weights=cell_weights,
            reference_model=reference_model,
        )
        rng = np.random.default_rng(seed=12312)
        model = rng.uniform(low=-1.0, high=1.0, size=n_active)
        delta_m = rng.normal(size=n_active)
        derivative_convergence_test(flatness, flatness.gradient, model, delta_m)
