.. _minimizer:

Minimizer
=========

Once we defined the :ref:`objective function <objective-function>` for our inversion, we need to find a model vector that would minimize it.
To do so we need to make use of a **minimizer**.

In the new framework, a **minimizer** can be a function that takes the objective
function (and other arguments, like an initial model) and returns the *inverted
model* or *recovered model*, i.e. the model that minimize that objective function.

Linear minimizers
-----------------

If our inversion is linear (e.g. potential field inversions), we can make use of linear minimizers, like :func:`inversion_ideas.conjugate_gradient` to minimize the objective function.
The :func:`~inversion_ideas.conjugate_gradient` function will ask the objective function and an initial model from which it'll start the conjugate gradient algorithm, and it will return the recovered model after convergence has been achieved.

For example, let's consider the :class:`~inversion_ideas.LinearRegressor` simulation and the ``data_misfit`` object we built when we introduced the :ref:`objective function <objective-function>`:

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

.. jupyter-execute::

   data_misfit = ii.DataMisfit(data, uncertainties, simulation)
   data_misfit

where the ``true_model`` was:

.. jupyter-execute::

   true_model


Let's define an objective function by adding a :class:`~inversion_ideas.TikhonovZero` regularization:

.. jupyter-execute::

   beta = 30.0
   phi = data_misfit + beta * ii.TikhonovZero(n_params)
   phi

And use the  :func:`~inversion_ideas.conjugate_gradient` to minimize it by setting an initial model full of zeros:

.. jupyter-execute::

   initial_model = np.zeros(n_params, dtype=np.float64)
   model = ii.conjugate_gradient(phi, initial_model)
   model

We were able to recover a model that is close to the true model used to generate the synthetic data.

We can make use of the :meth:`inversion_ideas.DataMisfit.chi_factor` method to see if we are over- or under-fitting the data.

- :math:`\chi \approx 1.0` would mean that we are recovering the data within the known uncertainties.
- :math:`\chi > 1.0` would mean that we are under-fitting the data.
- :math:`\chi < 1.0` would mean that we are over-fitting the data.

.. jupyter-execute::

   data_misfit.chi_factor(model)


Non-linear minimizers
---------------------

If our inversion is non-linear (DC-IP, EM inversions, etc.), we'll need to use a suitable minimizer for non-linear inversion problems, like the Gauss-Newton method.

The :class:`inversion_ideas.GaussNewtonConjugateGradient` object implements a Gauss-Newton algorithm using Conjugate Gradient to find search directions, and performing a backtracking line search.

We can use it to minimize the non-linear objective function through the :meth:`inversion_ideas.GaussNewtonConjugateGradient.run` method:

.. code:: python

   minimizer = GaussNewtonConjugateGradient()
   inverted_model = minimizer.run(phi, initial_model)


Alternatively we can iterate over it after calling the minimizer:

.. code:: python

   minimizer = GaussNewtonConjugateGradient()

   for model in minimizer(phi, initial_model):
      # The model is the one obtained after each Gauss-Newton iteration
      ...
      # We can perform stuff here, like plot the model, print information,
      # save stuff to disk, etc.
      ...

   # The inverted model is the last value of the `model` variable
   inverted_model = model.copy()


Third-party minimizers
----------------------

It's also possible to plug our :class:`~inversion_ideas.base.Objective` function objects into third-party minimizers, likes the ones available in :mod:`scipy`.

For example, we could use :func:`scipy.optimize.minimize` to minimize our objective function using a conjugate gradient algorithm:

.. jupyter-execute::

   from scipy.optimize import minimize

   # Pass the phi.gradient method as the `jac` argument
   result = minimize(phi, x0=initial_model, method="CG", jac=phi.gradient)

   inverted_model = result.x
   inverted_model
