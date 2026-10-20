.. _hyperparameters:

Hyperparameters
===============

In the :ref:`beta-cooling` page we made use of the :class:`hyperparams.CooledMultiplier <inversion_ideas.hyperparams.CooledMultiplier>` to easily implement the beta cooling schedule through an object that knows how to *cool down** itself.

The framework offers classes for such objects, that we call **hyperparameters** or **dynamic hyperparameters**. They work as their underlying values (a ``float`` or a Numpy array), but they also have some mechanisms to easily update those values when needed. Usually the updating process happens when their ``update`` method gets called (see :meth:`CooledMultiplier.update <inversion_ideas.hyperparams.CooledMultiplier.update>` as an example).

Scalar hyperparameters
----------------------

We've seen how the
:class:`hyperparams.CooledMultiplier <inversion_ideas.hyperparams.CooledMultiplier>` represents a ``float`` that can be updated throughout an inversion.
That class inherits from the :class:`inversion_ideas.base.Multiplier`, which simply wraps a ``float`` value and implements all the required interface to behave like one.

For example, we can instantiate it with a given value:

.. jupyter-execute::

   import inversion_ideas as ii

   multiplier = ii.base.Multiplier(3.14)
   print(multiplier)

This ``multiplier`` holds the 3.14 float in its ``value`` attribute:

.. jupyter-execute::

   multiplier.value

We can use that ``multiplier`` object as any other ``float``. For example, to perform any aritmetic operation with another ``float``:

.. jupyter-execute::

   print(multiplier + 3.0)
   print(multiplier - 1.0)
   print(multiplier * 2.0)
   print(multiplier / 2.0)
   print(multiplier**3)

.. important::

   Note that when we operate a :class:`~inversion_ideas.base.Multiplier` with a ``float`` we obtain a new ``float``, not another :class:`~inversion_ideas.base.Multiplier` object.

We can also use it to operate with Numpy arrays:

.. jupyter-execute::

   import numpy as np

   array = np.array([1., 2., 3., 4.])
   multiplier * array

.. important::

   Note that when we operate a :class:`~inversion_ideas.base.Multiplier` with a Numpy array we obtain a new array with the resulting values of the operation.
   Modifying the :class:`~inversion_ideas.base.Multiplier` after the operation will not change the value of the resulting array.

The only exception is when we multiply it by an :class:`~inversion_ideas.base.Objective` function. In such case, the resulting :class:`~inversion_ideas.base.Scaled` function will hold the whole :class:`~inversion_ideas.base.Multiplier` as its
:class:`Scaled.multiplier <inversion_ideas.base.Scaled.multiplier>`, instead of only its value as a ``float``.

.. jupyter-execute::

   smallness = ii.SimpleSmallness(n_params=10)

   scaled = multiplier * smallness
   print(scaled)

.. jupyter-execute::

   # Check that the multiplier in the scaled objective function is
   # our `multiplier` object:
   scaled.multiplier is multiplier


Array hyperparameters
---------------------

Analougsly, we can wrap a Numpy array inside a :class:`base.WrappedArray <inversion_ideas.base.WrappedArray>`:

.. jupyter-execute::

   wrapped = ii.base.WrappedArray(np.array([1., 2., 3.]))
   wrapped

The wrapped array lives in its ``array`` attribute:

.. jupyter-execute::

   wrapped.array

Similarly, we can operate with this object as if it were a regular Numpy array:

.. jupyter-execute::

   print(wrapped * 2.0)
   print(wrapped / 2.0)

.. jupyter-execute::

   wrapped * np.array([10., 20., 30.])

.. jupyter-execute::

   matrix = np.arange(9).reshape(3, 3)
   print(matrix)

   matrix @ wrapped

.. important::

   Note that when we operate with a :class:`~inversion_ideas.base.WrappedArray` we obtain a new array with the resulting values of the operation.
   Modifying the :class:`~inversion_ideas.base.WrappedArray` after the operation will not change the value of the resulting array.

We can also pass it to Numpy functions:

.. jupyter-execute::

   np.mean(wrapped)

And even index it:

.. jupyter-execute::

   wrapped[0]

.. jupyter-execute::

   wrapped[:2]
