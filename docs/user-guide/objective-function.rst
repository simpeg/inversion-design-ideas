.. _objective-function:

Objective function
==================

Defining objective functions
----------------------------

Any objective function is defined as a vector function :math:`\phi: \mathbb{R}^M \rightarrow \mathbb{R}`, i.e. it takes a model vector :math:`\mathbf{m}` with :math:`M` elements and returns a single value.

In the new SimPEG's inversion framework, objective functions are represented by a child of the :class:`inversion_ideas.base.Objective`, like the :class:`inversion_ideas.DataMisfit` for example.

.. jupyter-execute::

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
   data_misfit

Once defined, the objective functions can only be evaluated on model vectors with ``n_params`` elements. We can check that property on every objective function:

.. jupyter-execute::

   print("Expected size of model vector:", data_misfit.n_params)

Objective functions can be evaluated by calling them:

.. jupyter-execute::

   # Define some model
   model = np.array([-2., -1., 0., 1., 2.])

   # Evaluate the data_misfit on that model
   data_misfit(model)

All objective functions have two methods to compute its derivatives:

* :meth:`~inversion_ideas.base.Objective.gradient` computes the gradient of the objective function: :math:`\nabla \phi(\mathbf{m})`.
* :meth:`~inversion_ideas.base.Objective.hessian` computes the Hessian of the objective function, i.e. the matrix of the second derivatives: :math:`\nabla^2 \phi(\mathbf{m})`.

.. jupyter-execute::

   # Get gradient
   gradient = data_misfit.gradient(model)
   gradient

.. jupyter-execute::

   # Get Hessian
   hessian = data_misfit.hessian(model)
   hessian


.. important::

   By default, the :class:`~inversion_ideas.DataMisfit` and many other
   objective functions in the framework won't build the whole Hessian matrix, but
   return a :class:`~scipy.sparse.linalg.LinearOperator` instead.
   Hessian matrices are ``n_params`` :math:`\times` ``n_params`` square matrices
   that can grow very large when ``n_params`` is big.
   A :class:`~scipy.sparse.linalg.LinearOperator` allows us to operate with such
   matrix as if it were an array, but without storing it entirely in memory.

   For example, we can compute the dot product of the ``hessian`` with the
   ``model`` vector:

   .. jupyter-execute::

      hessian @ model


Renaming a function
-------------------

An objective function can be renamed using the :meth:`~inversion_ideas.base.Objective.set_name` method. For example:

.. jupyter-execute::

   data_misfit.set_name("data_misfit")
   data_misfit

Let's rename it to its previous name:

.. jupyter-execute::

   data_misfit.set_name("d")
   data_misfit

.. tip::

   The :meth:`~inversion_ideas.base.Objective.set_name` method returns the same objective function, so we can use it directly when defining a new one. For example:

   .. code:: python

      data_misfit = ii.DataMisfit(data, uncertainties, simulation).set_name("data_misfit")


Scaling a function
------------------

We can easily multiply an :class:`~inversion_ideas.base.Objective` function by a scalar, obtaining a :class:`inversion_ideas.base.Scaled` object.
The  :class:`inversion_ideas.base.Scaled` is another type of objective functions that represents an objective function (encapsulated in the :attr:`~inversion_ideas.base.Scaled.function` attribute) multiplied by a scalar (encapsulated in the :attr:`~inversion_ideas.base.Scaled.multiplier` attribute).

For example, we can build a new ``scaled`` function by multiplying the ``data_misfit`` by a ``float``:

.. jupyter-execute::

   scaled = 5.0 * data_misfit
   scaled

We can still access the underlying objects:

.. jupyter-execute::

   scaled.multiplier

.. jupyter-execute::

   scaled.function


.. important::

   The  :attr:`~inversion_ideas.base.Scaled.function` variable stores a reference to the original objective function. The
   :class:`~inversion_ideas.base.Scaled` doesn't perform any copy of it. Therefore ``scaled.function`` **is** the ``data_misfit`` object:

   .. jupyter-execute::

      scaled.function is data_misfit


We can evaluate the ``scaled`` function and get its gradient and Hessian as with any other objective function:

.. jupyter-execute::

   scaled(model)

.. jupyter-execute::

   scaled.gradient(model)

.. jupyter-execute::

   scaled.hessian(model)



Combining functions
-------------------

Add two functions
~~~~~~~~~~~~~~~~~

It's possible to define any linear combination of objective functions.

Consider we have a :class:`~inversion_ideas.TikhonovZero` regularization function:

.. jupyter-execute::

   zeroth = ii.TikhonovZero(n_params)
   zeroth

We can add it to the ``data_misfit`` one by summing them together to obtain a new objective function:

.. jupyter-execute::

   phi = data_misfit + zeroth
   phi

This :class:`inversion_ideas.base.Combo` is an objective function that works as its mathematical counterpart :math:`\phi(\mathbf{m}) = \phi_\text{d}(\mathbf{m}) + \phi_\text{0}(\mathbf{m})` does: we can evaluate it on a given model, get its gradient and Hessian:

.. jupyter-execute::

   phi(model)

.. jupyter-execute::

   phi.gradient(model)

.. jupyter-execute::

   phi.hessian(model)

.. important::

   The :class:`inversion_ideas.base.Combo` class is not meant to be instantiated.
   Add two or more objective functions together to obtain one.

The :class:`inversion_ideas.base.Combo` works as a collection of objective functions. This means that we can get the number of objective functions it contains through the :func:`len` function, and we can also access each one of its terms by index:

.. jupyter-execute::

   len(phi)

.. jupyter-execute::

   print(phi[0])
   print(phi[1])

.. jupyter-execute::

   for i, term in enumerate(phi):
       print(f"Term {i}: {term}")

.. jupyter-execute::

   print(data_misfit in phi)
   print(zeroth in phi)

Add multiple functions
~~~~~~~~~~~~~~~~~~~~~~

We can also add more than two objective functions together. For example, consider that we want to add the ``data_misfit`` and the ``zeroth`` regularization with a :class:`~inversion_ideas.TikhonovFirst` regularization:

.. jupyter-execute::

   # TODO: Use TikhonovFirst here!
   first = ii.TikhonovZero(n_params).set_name("1")
   first

.. jupyter-execute::

   phi = data_misfit + zeroth + first
   phi

By default, when adding more than two objective functions together, we'll get a *nested* :class:`inversion_ideas.base.Combo` object: the first element will be a :class:`inversion_ideas.base.Combo` containing the ``data_misfit`` and the ``zeroth`` regularization, and the second element will be the ``first`` regularization.

.. jupyter-execute::

   print(len(phi))
   print(phi[0])
   print(phi[1])

And since the first element is a :class:`~inversion_ideas.base.Combo`, we can also treat it as a collection:

.. jupyter-execute::

   print(len(phi[0]))
   print(phi[0][0])
   print(phi[0][1])

If we don't want to have this kind of structure,  we can
*flatten* any :class:`~inversion_ideas.base.Combo` through the
:meth:`~inversion_ideas.base.Combo.flatten` method:

.. jupyter-execute::

   flat_phi = phi.flatten()
   flat_phi

.. jupyter-execute::

   len(flat_phi)

.. jupyter-execute::

   print(flat_phi[0])
   print(flat_phi[1])
   print(flat_phi[2])

.. jupyter-execute::

   data_misfit in flat_phi

.. jupyter-execute::

   zeroth in flat_phi

.. jupyter-execute::

   first in flat_phi

Is in or contains?
^^^^^^^^^^^^^^^^^^

When using the ``in`` statement to check whether an objective function is part of a :class:`~inversion_ideas.base.Combo`, we are only checking if that function is one of its elements, but not a recursive search through its nested :class:`~inversion_ideas.base.Combo`.

For example, in we know that ``first`` is the second element of ``phi``:

.. jupyter-execute::

   phi


.. jupyter-execute::

   phi[1]

So, we know that ``first in phi`` should be ``True``:

.. jupyter-execute::

   first in phi

But, the first element of ``phi`` is a :class:`~inversion_ideas.base.Combo` containing ``data_misfit`` and ``zeroth``, so ``data_misfit in phi`` should be ``False``:

.. jupyter-execute::

   data_misfit in phi

And the same goes for ``zeroth``:

.. jupyter-execute::

   zeroth in phi

Alternatively, we can use the :meth:`~inversion_ideas.base.Combo.contains` method to check recursively for a particular term:

.. jupyter-execute::

   phi.contains(data_misfit)

.. jupyter-execute::

   phi.contains(zeroth)

.. jupyter-execute::

   phi.contains(first)


Linear combination of functions
-------------------------------

Since it's possible to scale objective functions and also to add them up, we can define any linear combination of objective functions.
For example, we could build an classic objective function like:

.. math::

   \phi(\mathbf{m}) = \phi_\text{d}(\mathbf{m}) + \beta \left[ \phi_\text{0}(\mathbf{m}) + \alpha \phi_\text{1}(\mathbf{m}) \right]


as follows:

.. jupyter-execute::

   beta = 1e2
   alpha = 3.0
   phi = data_misfit + beta * (zeroth + alpha * first)
   phi

We can evaluate the ``phi`` function and get its gradient and Hessian as with any other objective function:

.. jupyter-execute::

   phi(model)

.. jupyter-execute::

   phi.gradient(model)

.. jupyter-execute::

   phi.hessian(model)


Internal structure
~~~~~~~~~~~~~~~~~~

The ``phi`` object is a :class:`~inversion_ideas.base.Combo` with a nested structure. It contains two elements: the ``data_misfit`` and a :class:`~inversion_ideas.base.Scaled` object that represents the :math:`\beta \left[ phi_\text{0}(\mathbf{m}) + \alpha \phi_\text{1}(\mathbf{m}) \right]`:

.. jupyter-execute::

   len(phi)


.. jupyter-execute::

   phi[0]

.. jupyter-execute::

   phi[1]


Because the ``phi[1]`` is a  :class:`~inversion_ideas.base.Scaled`, it has a ``multiplier`` (the value of :math:`\beta`) and a ``function``:

.. jupyter-execute::

   phi[1].multiplier

.. jupyter-execute::

   phi[1].function

The ``phi[1].function`` is another :class:`~inversion_ideas.base.Combo` that contains the ``zeroth`` and a scaled version of ``first``.

.. jupyter-execute::

   phi[1].function[0]

.. jupyter-execute::

   phi[1].function[1]

We can use the  :meth:`~inversion_ideas.base.Combo.contains` to perform a recursive check for a given objective function within the ``phi`` function. This recursive check also explores
:class:`~inversion_ideas.base.Scaled` objects:

.. jupyter-execute::

   phi.contains(data_misfit)

.. jupyter-execute::

   phi.contains(zeroth)

.. jupyter-execute::

   phi.contains(first)
