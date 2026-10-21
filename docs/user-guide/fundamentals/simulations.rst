.. _simulations:

Simulations
===========

.. toctree::
   :hidden:
   :maxdepth: 1

   simulations-simpeg

Generally speaking, any inversion require at least one *forward modelling function*, i.e. a function that can predict data based on a given model.
It is usually represented by a function :math:`\mathbf{f}: \mathbb{R}^M \rightarrow \mathbb{R}^N`, and it's usually incorporated to the inversion through the data misfit term :math:`\phi_\text{d}(\mathbf{m})`:

.. math::

     \phi_\text{d}(\mathbf{m}) =
     \sum\limits_{i=1}^N
     \frac{
         \left\lvert f_i(\mathbf{m}) - d_i^\text{obs} \right\rvert^2
     }{
         \epsilon_i^2
     }

where :math:`\mathbf{m}` is the model vector, :math:`d_i^\text{obs}` is the
:math:`i`-th observed datum, :math:`f_i(\mathbf{m})` is the :math:`i`-th element
of the :math:`\mathbf{f}(\mathbf{m})` vector, and :math:`\epsilon_i` is the
uncertainty of the :math:`i`-th datum.

The inversion will require the ability to evaluate that function :math:`\mathbf{f}` on any model vector :math:`\mathbf{m}`, and also its Jacobian matrix :math:`\mathbf{J}_\mathbf{f}(\mathbf{m})`, i.e. the matrix of first derivatives:

.. math::

    \mathbf{J}_\mathbf{f}(\mathbf{m}) =
    \begin{bmatrix}
       \frac{\partial f_1(\mathbf{m})}{\partial m_1} & \dots  & \frac{\partial f_1(\mathbf{m})}{\partial m_M} \\
       \vdots                            & \ddots & \vdots \\
       \frac{\partial f_N(\mathbf{m})}{\partial m_1} & \dots  & \frac{\partial f_N(\mathbf{m})}{\partial m_M}
    \end{bmatrix}

where :math:`\frac{\partial f_i(\mathbf{m})}{\partial m_j}` is the derivative of the :math:`i`-th element of the forward modelling function with respect to the :math:`j`-th element of the model vector evaluated on a given model :math:`\mathbf{m}`.


In the inversion framework the forward modelling functions are represented through :class:`~inversion_ideas.base.Simulation` classes. These classes always provide two attributes:

* :attr:`~inversion_ideas.base.Simulation.n_params`: the number of elements in the expected ``model`` array.
* :attr:`~inversion_ideas.base.Simulation.n_data`: the number of elements of the array with predicted data.

and two methods:

* :meth:`~inversion_ideas.base.Simulation.__call__`: evaluates the function :math:`\mathbf{f}` on the provided ``model`` array (with ``n_params`` elements). It will always return an array with ``n_data`` elements.
* :meth:`~inversion_ideas.base.Simulation.jacobian`: returns the Jacobian matrix of the function :math:`\mathbf{f}` evaluated on the provided ``model`` array (with ``n_params`` elements). It can either return a dense or sparse 2D array (with shape ``(n_data, n_params)``), or a :class:`~scipy.sparse.linalg.LinearOperator` object.


For example, let's define a :class:`inversion_ideas.LinearRegressor` simulation with a random matrix:

.. jupyter-execute::

    import inversion_ideas as ii

    n_data = 5
    n_params = 10
    simulation = ii.LinearRegressor.create_random(n_data, n_params, seed=42)

We can access the ``n_params`` and ``n_data`` attributes of the simulation:

.. jupyter-execute::

    simulation.n_data, simulation.n_params

We can call the ``simulation`` with a given model vector to evaluate the forward modelling function:

.. jupyter-execute::

    import numpy as np

    rng = np.random.default_rng(seed=43)
    model = rng.uniform(low=-1.0, high=1.0, size=n_params)

    simulation(model)

We can also get its Jacobian matrix for the same ``model``:

.. jupyter-execute::

    jacobian = simulation.jacobian(model)
    jacobian

.. jupyter-execute::

    jacobian.shape


Alternatively, we could configure the :class:`~inversion_ideas.LinearRegressor` to return a
:class:`~scipy.sparse.linalg.LinearOperator` as a Jacobian. In this example, we need to pass ``build_jacobian=False`` as argument while creating the simulation:

.. jupyter-execute::

    simulation = ii.LinearRegressor.create_random(n_data, n_params, seed=42, build_jacobian=False)

.. hint::

    The ``build_jacobian`` option in :class:`~inversion_ideas.LinearRegressor` forces its :meth:`~inversion_ideas.LinearRegressor.jacobian` method to return a
    :class:`~scipy.sparse.linalg.LinearOperator` for learning and testing purposes, but it will still store the whole matrix in memory under the hood.

Let's now generate its Jacobian matrix for a given ``model``:

.. jupyter-execute::

    jacobian = simulation.jacobian(model)
    jacobian

As we can see it no longer returns a dense matrix, but a ``MatrixLinearOperator``, which is a child class of  :class:`~scipy.sparse.linalg.LinearOperator`.
We can use this object to compute dot products with any other array, as if it were a regular matrix:

.. jupyter-execute::

    vector = rng.random(size=n_params)
    jacobian @ vector

We can also compute the dot product of its transpose against a vector:

.. jupyter-execute::

    vector = rng.random(size=n_data)
    jacobian.T @ vector

.. important::

    The Jacobian matrix of any simulation will contain ``n_data * n_params`` elements. For large problems with high ``n_params`` and/or high ``n_data``, those matrices can grow to big that they don't fit in memory.
    In those cases it's preferrable to work with :class:`~scipy.sparse.linalg.LinearOperator` objects that are capable of computing dot products on the fly, without the need to store all the matrix elements in memory.

    The capability of each simulation my depend on its implementation, and users are encouraged to read the specific documentation of each one of them.
