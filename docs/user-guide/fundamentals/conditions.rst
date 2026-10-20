.. _conditions:

Conditions
==========

When we covered :ref:`inversion_class` we made use of the ``stopping_criterion`` argument that the :class:`~inversion_ideas.Inversion` uses to determine if it should finish or continue with the next iteration.

The ``stopping_criterion`` can be a simple function (or any *callable*) that takes the ``model`` as argument and returns a bool variable (``True`` or ``False``).

For example, the following function that checks if a given model is close to zero would be a valid argument:

.. code:: python

   def is_close_to_zero(model):
      if np.all(np.abs(model) <= 1e-13):
         return True
      return False

But in most situations, the stopping criteria for an inversion will not only depend on the current model, but on the values of certain functions when they are evaluated on the current model.

For such cases, the new framework offers a :class:`inversion_ideas.base.Condition` class that implements a ``__call__`` method that takes the ``model`` as argument and returns either ``True`` or ``False`` depending if the desired conditions is met or not.
For example, the :class:`inversion_ideas.conditions.ChiTarget` holds a
:class:`~inversion_ideas.DataMisfit` object, and everytime its
:meth:`inversion_ideas.conditions.ChiTarget.__call__` method gets called, it will evaluate the data misfit term, get its :math:`\chi` factor, and return ``True`` or ``False`` if it's below or equal to 1.

Consider the same ``data_misfit`` object we created in :ref:`objective-function`:

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

And define a :class:`inversion_ideas.conditions.ChiTarget` object for it:

.. jupyter-execute::

   import inversion_ideas as ii

   condition = ii.conditions.ChiTarget(data_misfit)
   condition

If we evaluate it with any model that doesn't fit the data, the :math:`\chi` factor will be higher than one and therefore the condition will be ``False``:

.. jupyter-execute::

   import numpy as np

   rng = np.random.default_rng(seed=4149)
   model = rng.random(size=data_misfit.n_params)
   condition(model)

.. jupyter-execute::
   :hide-code:

   assert not condition(model)

.. jupyter-execute::

   data_misfit.chi_factor(model)

But if we evaluate it with a model that fits the data, it will return ``True``:

.. jupyter-execute::

   condition(true_model)

.. jupyter-execute::
   :hide-code:

   assert condition(true_model)

.. jupyter-execute::

   data_misfit.chi_factor(true_model)


Combining conditions
--------------------

The :class:`~inversion_ideas.Inversion` takes a single callable as the ``stopping_criterion`` argument.
But in some situations we might want the inversion to stop when multiple conditions are satisfied at the same time, or when at least one of them is.
For those cases, we can combine :class:`~inversion_ideas.base.Condition` objects together through the logical binary operators ``&`` (AND), ``|`` (OR), and ``^`` (XOR).

For example, consider this very simple :class:`~inversion_ideas.base.Condition` classes:
.. note::

   These classes are presented for learning purposes, not intended to be used in real-world inversions.

.. jupyter-execute::

   import numpy as np
   import inversion_ideas as ii

   class IsPositive(ii.base.Condition):
      """Simple condition that checks if every value in the model is positive."""

      def __call__(self, model) -> bool:
         return bool(np.all(model > 0))

   class IsEven(ii.base.Condition):
      """Simple condition that checks if every value in the model is even."""

      def __call__(self, model) -> bool:
         return bool(np.all(model % 2 == 0))

Let's create one instance of each:

.. jupyter-execute::

   is_positive = IsPositive()
   is_even = IsEven()

And see how they work with a given model:

.. jupyter-execute::

   is_positive(np.array([1., 2., 3.]))

.. jupyter-execute::

   is_positive(np.array([1., -2., 3.]))

.. jupyter-execute::

   is_even(np.array([2., 4., 6.]))

.. jupyter-execute::

   is_even(np.array([1., 4., 6.]))

The ``&`` (AND) operator
~~~~~~~~~~~~~~~~~~~~~~~~

Let's combine them through the ``&`` (AND) operator:

.. jupyter-execute::

   condition = is_positive & is_even

This condition will be ``True`` only if all values in the model are positive and even:

.. jupyter-execute::

   condition(np.array([2., 4., 6.]))

But it will fail if at least one of them doesn't satisfy both conditions:

.. jupyter-execute::

   # A single odd number in the model
   condition(np.array([1., 4., 6.]))

.. jupyter-execute::

   # A single non-positive number in the model
   condition(np.array([2., -4., 6.]))

.. jupyter-execute::

   # A non-positive and an odd number in the model
   condition(np.array([2., -4., 7.]))

The ``~`` (OR) operator
~~~~~~~~~~~~~~~~~~~~~~~

Analougsly, we can combine them with an ``|`` (OR) operator:

.. jupyter-execute::

   condition = is_positive | is_even

If at least one of the two conditions is satisfied, the ``condition`` will return ``True``:

.. jupyter-execute::

   # All positive numbers in the model
   condition(np.array([1., 3., 5.]))

.. jupyter-execute::

   # All even numbers in the model
   condition(np.array([2., 4., 6.]))

But if either of them is satisfied, it will return ``False``:

.. jupyter-execute::

   condition(np.array([-1., 4., 6.]))

The ``^`` (XOR) operator
~~~~~~~~~~~~~~~~~~~~~~~~

We can also combine them using the `exclusive or <https://en.wikipedia.org/wiki/Exclusive_or>`__ operator (``^``):

.. jupyter-execute::

   condition = is_positive ^ is_even

Similarly to the ``|`` operator, if both conditions are ``False``, then ``condition`` will also be ``False``:

.. jupyter-execute::

   condition(np.array([-1., 4., 6.]))

And if one of them is ``True`` and the other one is ``False``, then it will be ``True``:

.. jupyter-execute::

   # All positive numbers in the model (but not all even)
   condition(np.array([1., 2., 3.]))

.. jupyter-execute::

   # All even numbers in the model (but not all positive)
   condition(np.array([0., 2., 4.]))

But if both are ``True``, then ``condition`` will be ``False``:

.. jupyter-execute::

   # All even numbers in the model (but not all positive)
   condition(np.array([2., 4., 6.]))

Combining multiple conditions
-----------------------------

We can also combine multiple conditions with these operators.
For example, let's consider a third condition class:

.. jupyter-execute::

   class IsCloseToZero(ii.base.Condition):
      """Simple condition that checks if every value in the model is close to zero."""

      def __call__(self, model) -> bool:
         return bool(np.all(np.abs(model) <= 1e-13))


   is_close_to_zero = IsCloseToZero()

And combine the three of them:

.. jupyter-execute::

   condition = (is_positive & is_even) | is_close_to_zero

.. hint::

   We can use parenthesis to group together logical operations.

This new condition will be ``True`` if all values in the model are positive **and** even, **or** if they are all close to zero.

.. jupyter-execute::

   model = np.array([2., 4., 6.])
   condition(model)

.. jupyter-execute::

   model = np.array([1e-14, -2e-15, 0.])
   condition(model)

But it will fail otherwise:

.. jupyter-execute::

   model = np.array([1., 2., 0.])
   condition(model)

Showing information
-------------------

We can see more detailed information about a composed conditions to see which one of the internal conditions is being satisfied and which one is not.
For that we can use the :meth:`inversion_ideas.base.Condition.info` method:

.. jupyter-execute::

   model = np.array([2., 4., 6.])
   condition.info(model)


Real-world use cases
--------------------

Some common use case for combining conditions is to force the inversion to stop if the :math:`\chi` factor of the data misfit term reaches the target **or** if the whole objective function hasn't changed significantly since the previous iteration.
We can then combine a :class:`inversion_ideas.conditions.ChiTarget` and a :class:`inversion_ideas.conditions.ObjectiveChanged`:

.. jupyter-execute::

   # Define the objective function
   smallness = ii.SimpleSmallness(n_params)
   phi = data_misfit + 0.2 * smallness

   # Define the two conditions
   chi_target = ii.conditions.ChiTarget(data_misfit)
   objective_changed = ii.conditions.ObjectiveChanged(phi, rtol=1e-2)

   # Combine them into a single stopping criterion
   stopping_criterion = chi_target | objective_changed
