"""
Test regularization classes.
"""

import numpy as np
import pytest
from discretize.tensor_mesh import TensorMesh
from scipy.sparse import dia_array, sparray

from inversion_ideas import Flatness, Smallness
from inversion_ideas.base import WrappedArray
from inversion_ideas.regularization._mesh_based import _MeshBasedRegularization

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
            raise ValueError()  # pragma: nocover

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


class MockRegularization(_MeshBasedRegularization):
    """Mock mesh-based regularization to test the methods of the base class."""

    def __init__(self, active_cells):
        self.active_cells = active_cells

    def __call__(self, model):
        raise NotImplementedError  # pragma: nocover

    def gradient(self, model):
        raise NotImplementedError  # pragma: nocover

    def hessian(self, model):
        raise NotImplementedError  # pragma: nocover


class TestMeshBasedRegularization:
    """Test the ``_MeshBasedRegularization`` base class."""

    active_cells = np.array([True, True, False, False])
    n_active = active_cells.sum()

    @pytest.mark.parametrize(
        "wrapped_array", [False, True], ids=["array", "wrapped-array"]
    )
    def test_cell_weights_array(self, wrapped_array):
        """Test cell_weights setter with an array or array-like object."""
        cell_weights = np.ones(self.n_active)
        if wrapped_array:
            cell_weights = WrappedArray(cell_weights)
        reg = MockRegularization(self.active_cells)
        reg.cell_weights = cell_weights
        assert reg.cell_weights is cell_weights

    @pytest.mark.parametrize(
        "wrapped_array", [False, True], ids=["array", "wrapped-array"]
    )
    def test_cell_weights_dictionary_arrays(self, wrapped_array):
        """Test cell_weights setter with a dictionary."""
        weights_a = np.ones(self.n_active)
        weights_b = 2 * np.ones(self.n_active)
        if wrapped_array:
            weights_b = WrappedArray(weights_b)
        cell_weights = {"a": weights_a, "b": weights_b}

        reg = MockRegularization(self.active_cells)
        reg.cell_weights = cell_weights
        assert reg.cell_weights is cell_weights

    def test_cell_weights_invalid_type(self):
        """Test cell_weights error after passing an object of invalid type."""

        class Blah: ...  # pragma: nocover

        cell_weights = Blah()
        reg = MockRegularization(self.active_cells)
        with pytest.raises(TypeError, match="Invalid cell_weights of type"):
            reg.cell_weights = cell_weights

    def test_cell_weights_invalid_type_in_dict(self):
        """Test cell_weights error after passing an object of invalid type in dict."""

        class Blah: ...  # pragma: nocover

        cell_weights = {"a": np.ones(self.n_active), "b": Blah()}
        reg = MockRegularization(self.active_cells)
        with pytest.raises(TypeError, match=r"Invalid cell_weights array 'b' of type"):
            reg.cell_weights = cell_weights

    @pytest.mark.parametrize(
        "wrapped_array", [False, True], ids=["array", "wrapped-array"]
    )
    def test_cell_weights_invalid_size(self, wrapped_array):
        """Test cell_weights error after passing array with wrong size."""
        cell_weights = np.ones(self.n_active + 2)
        if wrapped_array:
            cell_weights = WrappedArray(cell_weights)
        reg = MockRegularization(self.active_cells)
        with pytest.raises(
            ValueError, match=r"Invalid cell_weights array with '[0-9]+' elements"
        ):
            reg.cell_weights = cell_weights

    @pytest.mark.parametrize(
        "wrapped_array", [False, True], ids=["array", "wrapped-array"]
    )
    def test_cell_weights_invalid_size_in_dict(self, wrapped_array):
        """Test cell_weights error after passing array within dict with wrong size."""
        weights_a = np.ones(self.n_active)
        weights_b = 2 * np.ones(self.n_active + 2)
        if wrapped_array:
            weights_b = WrappedArray(weights_b)
        cell_weights = {"a": weights_a, "b": weights_b}

        reg = MockRegularization(self.active_cells)
        with pytest.raises(
            ValueError, match=r"Invalid cell_weights array 'b' with '[0-9]+' elements"
        ):
            reg.cell_weights = cell_weights

    @pytest.mark.parametrize(
        "wrapped_array", [False, True], ids=["array", "wrapped-array"]
    )
    def test_cell_weights_invalid_dims(self, wrapped_array):
        """Test cell_weights error after passing array with wrong dimensions."""
        cell_weights = np.ones((1, self.n_active))
        if wrapped_array:
            cell_weights = WrappedArray(cell_weights)
        reg = MockRegularization(self.active_cells)
        with pytest.raises(
            ValueError, match=r"Invalid cell_weights array with '[0-9]+' dimensions"
        ):
            reg.cell_weights = cell_weights

    @pytest.mark.parametrize(
        "wrapped_array", [False, True], ids=["array", "wrapped-array"]
    )
    def test_cell_weights_invalid_dims_in_dict(self, wrapped_array):
        """Test cell_weights error after passing array with wrong dimensions in dict."""
        weights_a = np.ones(self.n_active)
        weights_b = 2 * np.ones((1, self.n_active))
        if wrapped_array:
            weights_b = WrappedArray(weights_b)
        cell_weights = {"a": weights_a, "b": weights_b}

        reg = MockRegularization(self.active_cells)
        with pytest.raises(
            ValueError, match=r"Invalid cell_weights array 'b' with '[0-9]+' dimensions"
        ):
            reg.cell_weights = cell_weights


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
            raise ValueError()  # pragma: nocover

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
