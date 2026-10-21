.. _preconditioners:

Preconditioners
===============

Some optimization methods, like the conjugate gradient, benefit from using a *preconditioner* matrix.

Some optimization methods, like the conjugate gradient, can be accelerated by transforming the linear system they need to iteratively solve in such a way that improves the eigenvalue distribution of the matrix :math:`\mathbf{A}`.
One way to achieve this is to perform a change of variables through a *preconditioning* matrix (see [NocedalWright1999]_, page 118).

We can pass a preconditioner to most of the minimizers of the new framework, and it offers some functions to generate appropriate preconditioners for a given objective function.

Preconditioning the conjugate gradient
--------------------------------------

The :func:`minimize.conjugate_gradient <inversion_ideas.minimize.conjugate_gradient>`
function through the ``preconditioner`` argument.

For example, consider an objective function ``phi``.
We can use the :func:`~inversion_ideas.get_jacobi_preconditioner` to obtain a suitable preconditioner matrix for it:

.. jupyter-execute::
   :hide-code:

   import numpy as np
   import inversion_ideas as ii

   # Define a linear regressor as a simulation
   n_data, n_params = 3, 5
   simulation = ii.simulations.LinearRegressor.create_random(n_data, n_params, seed=42)

   # Create some synthetic data and uncertainties
   true_model = np.array([-5., 2., -3., 4., 1.])
   std = 1e-2
   data = simulation(true_model) + np.random.default_rng(seed=42).normal(size=n_data, scale=std)
   uncertainties = std * np.ones_like(data)

   # Define a data misfit function
   data_misfit = ii.DataMisfit(data, uncertainties, simulation)

   # Define smallness
   smallness = ii.SimpleSmallness(n_params)

   # Define objective function
   model = np.zeros(n_params)
   beta = ii.utils.estimate_tradeoff_parameter(data_misfit, smallness, model, seed=42)
   phi = data_misfit + beta * smallness


.. jupyter-execute::

   import inversion_ideas as ii

   initial_model = np.zeros(n_params)
   preconditioner = ii.get_jacobi_preconditioner(phi, initial_model)
   preconditioner

.. jupyter-execute::

   preconditioner.toarray()

Now let's minimize it using :func:`~inversion_ideas.minimize.conjugate_gradient` and use that matrix as a preconditioner:

.. jupyter-execute::

   inverted_model = ii.minimize.conjugate_gradient(
      phi, initial_model, preconditioner=preconditioner
   )


Preconditioning the Gauss-Newton conjugate gradient
---------------------------------------------------

When minimizing non-linear functions we'll some non-linear minimizer like :class:`minimize.GaussNewtonConjugateGradient <inversion_ideas.minimize.GaussNewtonConjugateGradient>`. When calling this object we can also pass a ``preconditioner`` object:

.. jupyter-execute::

   # Define the minimizer
   minimizer = ii.minimize.GaussNewtonConjugateGradient()

   # Run it using the `run()` method, and pass the preconditioner
   inverted_model = minimizer.run(
      phi, initial_model, preconditioner=preconditioner
   )

The preconditioner will be passed by the
:class:`minimize.GaussNewtonConjugateGradient <inversion_ideas.minimize.GaussNewtonConjugateGradient>` to the conjugate gradient algorithm that runs once per Gauss-Newton iteration.

The issue of this approach is that the linear system that gets passed to the conjugate gradient function is different for each iteration: the Hessian matrix depends on the current model in the Gauss-Newton step, and because the problem is non-linear, that Hessian will be different for each iteration.
Trying to use the same preconditioner for each one of those systems might be unproductive. Instead, it's better to calculate a new preconditioner for each Gauss-Newton step.

We can achieve that by using a *dynamic preconditioner* object, like the
:class:`inversion_ideas.JacobiPreconditioner`, and pass it to the minimizer as the ``preconditioner`` argument.
The :class:`minimize.GaussNewtonConjugateGradient <inversion_ideas.minimize.GaussNewtonConjugateGradient>` will run the ``update`` method of the passed precondtioner object before running the conjugate gradient algorithm, and it'll pass it as argument to it.

For example:

.. jupyter-execute::

   # Define a dynamic Jacobi preconditioner
   preconditioner = ii.JacobiPreconditioner(phi)
   preconditioner

.. note::

   These dynamic preconditioner are :class:`~scipy.sparse.linalg.LinearOperator` objects that under the hood they hold the preconditioning matrix after it gets updated or recomputed.

We can pass it to the minimizer when running it:

.. jupyter-execute::

   # Run it using the `run()` method, and pass the preconditioner
   inverted_model = minimizer.run(
      phi, initial_model, preconditioner=preconditioner
   )

.. seealso::

   Find more **dynamic preconditioners** in the :ref:`API Reference <api>`.


References
----------

.. [NocedalWright1999] Nocedal, J., & Wright, S. J. (1999). Numerical optimization. Springer.
