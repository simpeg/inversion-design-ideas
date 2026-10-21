.. _simulations-simpeg:

Plugging in SimPEG simulations
------------------------------

.. warning::

    Plugging in SimPEG simulations into the new inversion framwork is still in
    experimental stages. If you run into any issue following these instructions,
    please `open an Issue <https://github.com/simpeg/inversion-design-ideas/issues/new>`__
    to report it.

SimPEG simulations are not yet fully compatible with the new inversion framework, but we can *wrap* them in a custom :class:`~inversion_ideas.WrappedSimulation` class that will expose the right interface for it to work with the new inversion framework.

For example, let's define a SimPEG's gravity simulation using the integral formulation:

.. jupyter-execute::

    import numpy as np
    import bordado as bd
    import discretize
    import simpeg
    from simpeg.potential_fields.gravity import (
        Point,
        Simulation3DIntegral,
        SourceField,
        Survey,
    )
    import inversion_ideas as ii

    # Define a mesh
    h = [(5.0, 40)]
    mesh = discretize.TensorMesh(h=[h, h, h], origin="CCN")
    mesh

    # Define a set of receivers in a regular grid at zeroth height
    region = (-100, 100, -100, 100)
    shape = (11, 11)
    height = 0
    coordinates = bd.grid_coordinates(region, shape=shape, non_dimensional_coords=height)

    # Define the SimPEG gravity survey
    locations = np.vstack(tuple(c.ravel() for c in coordinates)).T
    receivers = Point(locations, components="gz")
    source = SourceField(receiver_list=[receivers])
    survey = Survey(source)

    # Define the SimPEG simulation
    simulation_simpeg = Simulation3DIntegral(
        mesh, survey=survey, rhoMap=simpeg.maps.IdentityMap(mesh), engine="choclo"
    )
    simulation_simpeg

Such simulation can be evaluated through the ``dpred`` method by passing a given ``model``.
Let's consider a model that contains two rectangular blocks with different densities:

.. jupyter-execute::

    model = np.zeros(mesh.n_cells)

    # Define block through their west-south-lower and east-north-upper vertices
    blocks = [
        [(-60, -10, -70), (-40, 10, -50)],
        [(40, -10, -70), (60, 10, -50)],
    ]
    densities = [-0.2, 0.2]  # in g/cc

    for (p0, p1), density in zip(blocks, densities, strict=True):
        model = simpeg.utils.model_builder.add_block(
            mesh.cell_centers, model, p0, p1, density
        )

.. jupyter-execute::

    import matplotlib.pyplot as plt

    (tmp,) = mesh.plot_slice(model, normal="Y", slice_loc=0)
    plt.colorbar(tmp, label="densities [g/cc]")
    plt.show()

And get the predicted gravity data for that model:

.. jupyter-execute::

    gz = simulation_simpeg.dpred(model)

.. jupyter-execute::

    maxabs = np.abs(gz).max()
    easting, northing, _ = coordinates
    plt.pcolormesh(easting, northing, gz.reshape(shape), vmin=-maxabs, vmax=maxabs)
    plt.colorbar(label="gz [mgal]")
    plt.gca().set_aspect("equal")
    plt.show()

Let's now wrap this simulation to make it compatible within the new inversion framework:

.. jupyter-execute::

    simulation = ii.WrappedSimulation(simulation_simpeg)
    simulation

We can use the :meth:`inversion_ideas.WrappedSimulation.__call__` method to evaluate it:

.. jupyter-execute::

    gz = simulation(model)

    maxabs = np.abs(gz).max()
    easting, northing, _ = coordinates
    plt.pcolormesh(easting, northing, gz.reshape(shape), vmin=-maxabs, vmax=maxabs)
    plt.colorbar(label="gz [mgal]")
    plt.gca().set_aspect("equal")
    plt.show()


And the :meth:`inversion_ideas.WrappedSimulation.jacobian` to compute its Jacobian matrix:

.. jupyter-execute::

    jacobian = simulation.jacobian(model)
    jacobian

.. hint::

    By default the :class:`~inversion_ideas.WrappedSimulation` will not build the
    Jacobian matrix, but create a  :class:`~scipy.sparse.linalg.LinearOperator`
    that connects to the ``Jvec`` and ``Jtvec`` methods of the SimPEG
    simulation.

    For some SimPEG simulations we can force the :class:`~inversion_ideas.WrappedSimulation` to build the Jacobian matrix and store it in memory by setting ``store_jacobian=True``:

    .. jupyter-execute::

        sim = ii.WrappedSimulation(simulation_simpeg, store_jacobian=True)
        sim

    .. jupyter-execute::

        sim.jacobian(model)
