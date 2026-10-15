.. _inversion_class:

The :class:`~inversion_ideas.Inversion` class
=============================================

Writing full inversions using ``for`` loops -like we've seen in :ref:`beta-cooling`- could be cumbersome.
An alternative would be to pack the whole inversion process into a single :class:`inversion_ideas.Inversion` object.
Such objects will group together all the pieces required to run the inversion, and provide an easy way to run all iterations, either through the :meth:`~inversion_ideas.Inversion.run` method, or by iterating over it.

In order to define any :class:`~inversion_ideas.Inversion` object we need the following pieces:

- the **objective function** we'll minimize (in its initial stage),
- an **initial model** from which the inversion will start,
- the **minimizer** we'll use in each iteration, and
- a **stopping criterion** that the inversion use to decide if it should finish or continue.

We can optionally pass a set of **directives**: these are functions that will be called after each iteration.
We can use these directives to modify the objective function and update
hyperparameters, like cooling down the trade-off parameter.

For example, an :class:`~inversion_ideas.Inversion` object can be defined as:

.. code:: python

   inversion = ii.Inversion(
      phi,
      initial_model,
      minimizer=minimizer,
      directives=directives,
      stopping_criterion=stopping_criterion,
   )

Once defined, we can run the :class:`~inversion_ideas.Inversion` through the :meth:`~inversion_ideas.Inversion.run` method:

.. code:: python

   inverted_model = inversion.run()


Or we can iterate over it:

.. code:: python

   for model in inversion:
      # Print information
      print(f"model: {model}")
      print(f"chi factor: {data_misfit.chi_factor(model)}")

      # Plot results
      ...

      # Save stuff to disk
      ...

This allows us to easily insert any custom code in between iterations.
The ``for`` loop will finish once the ``stopping_criterion`` has been met.

.. hint::

   The ``model`` variable will store the inverted model for each one of the
   iterations in the inversion. The inverted model will be the last value of it.

In general, the :class:`~inversion_ideas.Inversion` will perform the following
tasks per iteration:

1. Check if the stopping condition is met and finish the process if that's the case.
2. Minimize the objective function and update its :attr:`~inversion_ideas.Inversion.model` attribute.
3. Call each one of the functions in the ``directives`` list.

Beta-cooling with :class:`~inversion_ideas.Inversion`
-----------------------------------------------------

Let's define an :class:`~inversion_ideas.Inversion` object that implements the same beta-cooling schedule we've seen in :ref:`beta-cooling`.

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

   # Define a smallness regularization
   smallness = ii.SimpleSmallness(data_misfit.n_params)


Let's start by defining our objective function with a :class:`~inversion_ideas.hyperparams.CooledMultiplier` as our trade-off parameter:

.. jupyter-execute::

   beta = ii.hyperparams.CooledMultiplier(200.0, cooling_factor=2.0)
   phi = data_misfit + beta * smallness
   phi

And the remaining pieces to build our inversion:

.. jupyter-execute::

   # Define a stopping criterion
   chi_target = ii.conditions.ChiTarget(data_misfit, chi_target=1.0)

   # Define an initial model
   initial_model = np.ones(data_misfit.n_params)

Finally, let's create our :class:`~inversion_ideas.Inversion` object. We'll add the :meth:`~inversion_ideas.hyperparams.CooledMultiplier.beta` method of ``beta`` as the sole directive for this

.. jupyter-execute::

   # Define the inversion
   inversion = ii.Inversion(
      phi,
      initial_model,
      minimizer=ii.conjugate_gradient,
      directives=[beta.update],
      stopping_criterion=chi_target,
   )


We can run it using the :meth:`~inversion_ideas.Inversion.run` method:

.. jupyter-execute::

   inverted_model = inversion.run()


Alternatively, we can iterate over the :class:`~inversion_ideas.Inversion` object, and insert some custom code in between iterations:

.. jupyter-execute::

   # Define objective function
   beta = ii.hyperparams.CooledMultiplier(200.0, cooling_factor=2.0)
   phi = data_misfit + beta * smallness

   # Define a stopping criterion
   chi_target = ii.conditions.ChiTarget(data_misfit, chi_target=1.0)

   # Define an initial model
   initial_model = np.ones(data_misfit.n_params)

   # Define the inversion
   inversion = ii.Inversion(
      phi,
      initial_model,
      minimizer=ii.conjugate_gradient,
      directives=[beta.update],
      stopping_criterion=chi_target,
   )

   # Iterate over the inversion.
   # The inversion will return the inverted model obtained after each iteration.
   for model in inversion:
      # Print information
      print(f"model: {model}")
      print(f"chi factor: {data_misfit.chi_factor(model)}")


The inversion log
-----------------
The :class:`~inversion_ideas.Inversion` keep by default an *inversion log*
through which we can access to some details about the whole inversion process,
both during the inversion is running and after the inversion has finished.

.. jupyter-execute::

   inversion.log.table


We can convert it to a :class:`pandas.DataFrame`:

.. jupyter-execute::

   df = inversion.log.to_pandas()
   df

And use it to generate some convergence plots:

.. jupyter-execute::

   import matplotlib.pyplot as plt

   fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6.4, 8))

   ax1.plot(df.index, df.beta, "-o")
   ax1.set_ylabel(r"$\beta$")

   ax2.plot(df.index, df.phi_d, "-o")
   ax2.set_yscale("log")
   ax2.set_ylabel(r"$\phi_\text{d}$")

   ax3.plot(df.index, df.phi_m, "-o")
   ax3.set_ylabel(r"$\phi_\text{m}$")

   for ax in (ax1, ax2, ax3):
      ax.grid(True)

   plt.show()


.. hint::

   We can also access the inverted model for each iteration through the
   :attr:`~inversion_ideas.Inversion.models` attribute. Models will be cached
   if the ``cache_models`` argument is set to ``True`` when intantiating the
   :class:`~inversion_ideas.Inversion` object.

   .. jupyter-execute::

      inversion.models
