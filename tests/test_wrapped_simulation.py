"""
Test the ``WrappedSimulation`` class.
"""

import discretize
import numpy as np
import pytest
import simpeg
from scipy.sparse.linalg import LinearOperator
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static.utils import generate_dcip_sources_line
from simpeg.potential_fields import gravity

from inversion_ideas.simulations import WrappedSimulation


def get_dc_simulation(store_jacobian=False):
    """
    Build a simple SimPEG's DC simulation.
    """
    # Define a dummy mesh (doesn't need to generate accurate results for this)
    dh, n_cells = 4.0, 11
    hx = [(dh, n_cells)]
    mesh = discretize.TensorMesh([hx, hx], origin="CC")

    # Define active cells
    _, z = mesh.cell_centers.T
    active_cells = z <= 0
    n_active = np.sum(active_cells)

    # Define survey
    topo_2d = np.vstack((mesh.cell_centers_x, np.zeros_like(mesh.cell_centers_x))).T
    sources = generate_dcip_sources_line(
        survey_type="dipole-dipole",
        data_type="volt",
        dimension_type="2D",
        end_points=(-20.0, 20.0),
        topo=topo_2d,
        num_rx_per_src=10,
        station_spacing=5.0,
    )
    survey = dc.Survey(sources)

    # Define SimPEG simulation
    log_conductivity_map = simpeg.maps.InjectActiveCells(
        mesh, active_cells, 1e-8
    ) * simpeg.maps.ExpMap(nP=n_active)

    simulation_simpeg = dc.Simulation2DNodal(
        mesh,
        survey=survey,
        sigmaMap=log_conductivity_map,
        storeJ=store_jacobian,
        solver=simpeg.utils.get_default_solver(),
    )
    return simulation_simpeg


def get_gravity_simulation():
    dh, n_cells = 4.0, 11
    hx = [(dh, n_cells)]
    mesh = discretize.TensorMesh([hx, hx, hx], origin="CCN")
    active_cells = np.ones(mesh.n_cells, dtype=bool)
    n_active = np.sum(active_cells)

    receiver_location = np.array([[0, 0, 10]])
    receiver = gravity.receivers.Point(receiver_location, components="gz")
    source = gravity.SourceField(receiver_list=[receiver])
    survey = gravity.Survey(source)

    model_map = simpeg.maps.IdentityMap(nP=n_active)
    simulation_simpeg = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=active_cells,
        store_sensitivities="ram",
        engine="choclo",
    )
    return simulation_simpeg


class TestWrappedLinearSimulation:
    """
    Test (no) cached fields in non-PDE simulations.
    """

    def test_fields_as_none(self):
        """Test if fields are always None in non-PDE simulations."""
        simulation_simpeg = get_gravity_simulation()
        n_params = simulation_simpeg.rhoMap.nP
        simulation = WrappedSimulation(simulation_simpeg)
        model = np.random.default_rng(seed=41).uniform(size=n_params)
        assert simulation._get_fields(model) is None
        assert not hasattr(simulation, "_cached_fields")


class TestWrappedPDESimulation:
    """
    Test cached fields and jacobians in PDE ``WrappedSimulation``.
    """

    def test_cached_fields(self):
        """Test if fields with same model are correctly cached."""
        simulation_simpeg = get_dc_simulation()
        n_params = simulation_simpeg.sigmaMap.nP
        simulation = WrappedSimulation(simulation_simpeg)
        model = np.random.default_rng(seed=41).uniform(size=n_params)
        fields_1 = simulation._get_fields(model)
        fields_2 = simulation._get_fields(model)
        # Check that the fields returned the second time are the same ones
        assert fields_1 is fields_2
        _, cached_fields = simulation._cached_fields
        assert cached_fields is fields_1

    def test_cache_new_fields(self):
        """Test if new fields are computed and cached upon different model."""
        simulation_simpeg = get_dc_simulation()
        n_params = simulation_simpeg.sigmaMap.nP
        simulation = WrappedSimulation(simulation_simpeg)
        rng = np.random.default_rng(seed=41)
        model_1, model_2 = rng.uniform(size=n_params), rng.uniform(size=n_params)
        fields_1 = simulation._get_fields(model_1)
        fields_2 = simulation._get_fields(model_2)
        # Check the fields are not the same ones
        assert fields_1 is not fields_2
        # Check that the cached fields are the last ones
        _, cached_fields = simulation._cached_fields
        assert cached_fields is fields_2

    @pytest.mark.parametrize(
        "store_jacobian", [True, False], ids=["store_j", "no_store_j"]
    )
    def test_cached_jacobian(self, store_jacobian):
        """
        Test @cache_on_model decorator in ``WrappedSimulation.jacobian``.
        """
        simulation_simpeg = get_dc_simulation()
        n_params = simulation_simpeg.sigmaMap.nP
        simulation = WrappedSimulation(simulation_simpeg, store_jacobian=store_jacobian)
        rng = np.random.default_rng(seed=41)
        model = rng.uniform(size=n_params)
        jacobian_1 = simulation.jacobian(model)
        jacobian_2 = simulation.jacobian(model)
        # Check type
        if store_jacobian:
            assert not isinstance(jacobian_1, LinearOperator)
            assert not isinstance(jacobian_2, LinearOperator)
        else:
            assert isinstance(jacobian_1, LinearOperator)
            assert isinstance(jacobian_2, LinearOperator)
        # Check the two objects are the same
        assert jacobian_1 is jacobian_2

    def test_two_jacobians_same_fields(self):
        """Test if jacobians for same models are defined with same fields."""
        simulation_simpeg = get_dc_simulation()
        n_params = simulation_simpeg.sigmaMap.nP
        # Set cache to False so we are not caching the output of jacobian
        simulation = WrappedSimulation(simulation_simpeg, cache=False)
        rng = np.random.default_rng(seed=41)
        model = rng.uniform(size=n_params)
        jacobian_1 = simulation.jacobian(model)
        jacobian_2 = simulation.jacobian(model)
        # Check they are both LinearOperators
        assert isinstance(jacobian_1, LinearOperator)
        assert isinstance(jacobian_2, LinearOperator)
        # Check that the two jacobians are different LinearOperators
        assert jacobian_1 is not jacobian_2
        # But they should represent the same operator
        vector = rng.uniform(size=n_params)
        np.testing.assert_allclose(jacobian_1 @ vector, jacobian_2 @ vector)
        vector = rng.uniform(size=jacobian_1.shape[0])
        np.testing.assert_allclose(jacobian_1.T @ vector, jacobian_2.T @ vector)

    def test_two_jacobians_different_fields(self):
        """Test if jacobians for different models are defined with different fields."""
        simulation_simpeg = get_dc_simulation()
        n_params = simulation_simpeg.sigmaMap.nP
        # Set cache to False so we are not caching the output of jacobian
        simulation = WrappedSimulation(simulation_simpeg, cache=False)
        rng = np.random.default_rng(seed=41)
        model_1, model_2 = rng.uniform(size=n_params), rng.uniform(size=n_params)
        jacobian_1 = simulation.jacobian(model_1)
        jacobian_2 = simulation.jacobian(model_2)
        # Check they are both LinearOperators
        assert isinstance(jacobian_1, LinearOperator)
        assert isinstance(jacobian_2, LinearOperator)
        # Since they were defined with different models, they should represent
        # different operators. So the dot products with the same vectors should
        # not be equal.
        vector = rng.uniform(size=n_params)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(jacobian_1 @ vector, jacobian_2 @ vector)
        vector = rng.uniform(size=jacobian_1.shape[0])
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(jacobian_1.T @ vector, jacobian_2.T @ vector)

    def test_dense_jacobians_same_fields(self):
        """
        Test if two jacobians with same fields are the same.
        """
        simulation_simpeg = get_dc_simulation()
        n_params = simulation_simpeg.sigmaMap.nP
        simulation = WrappedSimulation(
            simulation_simpeg,
            store_jacobian=True,
            cache=False,  # do not cache the output of jacobian method
        )
        rng = np.random.default_rng(seed=41)
        model = rng.uniform(size=n_params)
        jacobian_1 = simulation.jacobian(model)
        jacobian_2 = simulation.jacobian(model)
        # Check they are not LinearOperators
        assert not isinstance(jacobian_1, LinearOperator)
        assert not isinstance(jacobian_2, LinearOperator)
        # Check they are the same matrices
        np.testing.assert_allclose(jacobian_1, jacobian_2)

    def test_dense_jacobians_different_fields(self):
        """
        Test if two jacobians with different fields are not the same.
        """
        simulation_simpeg = get_dc_simulation()
        n_params = simulation_simpeg.sigmaMap.nP
        simulation = WrappedSimulation(
            simulation_simpeg,
            store_jacobian=True,
            cache=False,  # do not cache the output of jacobian method
        )
        rng = np.random.default_rng(seed=41)
        model_1, model_2 = rng.uniform(size=n_params), rng.uniform(size=n_params)
        jacobian_1 = simulation.jacobian(model_1)
        jacobian_2 = simulation.jacobian(model_2)
        # Check they are not LinearOperators
        assert not isinstance(jacobian_1, LinearOperator)
        assert not isinstance(jacobian_2, LinearOperator)
        # Check they are not the same matrices
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(jacobian_1, jacobian_2)
