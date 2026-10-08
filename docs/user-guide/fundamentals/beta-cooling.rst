.. _beta-cooling:

Beta-cooling schedule
=====================

One common example is the *beta cooling schedule* that consists in defining an
objective function in the form of

.. math::

   \phi(\mathbf{m}) = \phi_\text{d}(\mathbf{m}) + \beta_0 \phi_\text{m}(\mathbf{m}),

where :math:`\phi_\text{d}(\mathbf{m})` is the data misfit term,
:math:`\phi_\text{m}(\mathbf{m})` is the model norm, and :math:`\beta_0` is an
initial value for the trade-off parameter, which is usually chosen to be a high value.
We proceed to minimize the objective function to obtain an inverted model, and then
*cool down* the trade-off parameter by dividing it by a constant factor.
Then we carry another minimization step, and another cooling step, and so on until a certain condition is met, usually that the data is fitted up to their uncertainties.


Beta-cooling: simple
--------------------
We can achieve that in the new inversion framework with just a ``for`` loop.
For example, consider a :class:`~inversion_ideas.DataMisfit` object that uses a
:class:`~inversion_ideas.LinearRegressor` simulation, like the one we defined in
:ref:`objective-function`:

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

.. jupyter-execute::

   data_misfit

Let's define a :class:`~inversion_ideas.SimpleSmallness` regularization:

.. jupyter-execute::

   smallness = ii.SimpleSmallness(data_misfit.n_params)
   smallness

And create our objective function with a high value for our starting trade-off parameter ``beta``:

.. jupyter-execute::

   beta_0 = 200.0
   regularization = beta_0 * smallness
   phi = data_misfit + regularization
   phi

.. note::

   A *high value* for the trade-off parameter depends on the type of problem we
   are facing. For one problem a value of 200.0 might be high enough, but for other
   it might need to be several order of magnitudes larger.

Since we can access the value of the trade-off parameter through ``regularization.multiplier``, we can also change it.
Let's use this capability to implement a simple beta-cooling schedule:

.. jupyter-execute::

   # Define an initial model
   model = np.zeros(data_misfit.n_params)

   # Perform a beta cooling schedule
   while True:

      # Minimize the objective function using a conjugate gradient minimizer.
      # Overwrite the model variable with the minimized model.
      model = ii.conjugate_gradient(phi, model)

      # Define a stopping criterion:
      #   finish the inversion if the chi factor is below 1.
      if data_misfit.chi_factor(model) <= 1.0:
         break

      # Cool down beta using a factor of 2
      regularization.multiplier /= 2.0

The final inverted model will be stored in the ``model`` variable:

.. jupyter-execute::

   inverted_model = model.copy()
   inverted_model

We can also see the final value of the trade-off parameter by displaying the objective function:

.. jupyter-execute::

   phi


Beta-cooling: with custom objects
---------------------------------

The new framework ships some objects that would make writing this inversion in a slicker way.
First, instead of defining our tradeoff parameter as a ``float`` and manually cooling it down, we can use a :class:`inversion_ideas.hyperparams.CooledMultiplier`.
This object acts like a float, but it has an
:meth:`~inversion_ideas.hyperparams.CooledMultiplier.update` method that can automatically perform the cooling step.

Following the previous example, we can define our objective function as:

.. jupyter-execute::

   # Define trade-off parameter as a hyperparameter object.
   # Set its initial value to 200.0 and the factor that will be used to cool it down to 2.0.
   beta = ii.hyperparams.CooledMultiplier(200.0, cooling_factor=2.0)

   # Use this beta as any other number
   phi = data_misfit + beta * smallness
   phi

.. seealso::

   Learn more about custom hyperparameter objects in **TODO**


We can also define a stopping criterion to define when our inversion should stop.
In this case we can make use of :class:`inversion_ideas.conditions.ChiTarget` to check if the :math:`\chi` factor for the ``data_misfit`` falls below a desired threshold:

.. jupyter-execute::

   # Define a stopping criterion
   chi_target = ii.conditions.ChiTarget(data_misfit, chi_target=1.0)


And finally re-write our inversion in a slicker way:

.. jupyter-execute::

   # Define an initial model
   model = np.zeros(data_misfit.n_params)

   # Perform a beta cooling schedule
   while True:

      # Minimize the objective function
      model = ii.conjugate_gradient(phi, model)

      # Finish the inversion if the stopping criterion is met
      if chi_target(model):
         break

      # Cool down beta
      beta.update()

   # Define inverted model as last model in the iterations
   inverted_model = model.copy()
   inverted_model


We can also explore the custom ``beta`` hyperparameter and see how its value changed:

.. jupyter-execute::

   print(beta)


.. hint::

   We can easily insert any custom code in between iterations.
   For example we could print information about the inversion process:

   .. jupyter-execute::

      # Define objective function
      beta = ii.hyperparams.CooledMultiplier(200.0, cooling_factor=2.0)
      phi = data_misfit + beta * smallness

      # Define an initial model
      model = np.zeros(data_misfit.n_params)

      # Perform a beta cooling schedule
      iteration = 0
      while True:

         # Minimize the objective function
         model = ii.conjugate_gradient(phi, model)

         # Print information
         iteration += 1
         print(f"Iteration: {iteration:2d}")
         print("-------------")
         print(f"- model: {model}")
         print(f"- beta: {beta}")
         print(f"- chi factor: {data_misfit.chi_factor(model)}")
         print("")

         # Finish the inversion if the stopping criterion is met
         if chi_target(model):
            break

         # Cool down beta
         beta.update()
