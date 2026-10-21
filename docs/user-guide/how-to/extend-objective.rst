.. _extend-objective:

Write new objective functions
=============================

One of the goals of the new inversion framework is to be more modular and allow users to write their own objective functions and easily plug them into the inversion framework.
To do so, we need to create our own class for our new objective function and make it a child of :class:`base.Objective <inversion_ideas.base.Objective>`.
Then, we'll need to write implementations for its ``__call__``, ``gradient``, and ``hessian`` methods with the same signatures as the ones in
:class:`base.Objective <inversion_ideas.base.Objective>`, and its ``n_params`` property.
Optionally, we can also add an ``__init__`` method to our new class to be able to pass the required arguments for it to function.

For example, let's consider the following objective function that simply evaluates the norm of the model values:

.. math::

   \phi_\text{m}(\mathbf{m})
   = \sum\limits_{i=1}^M |m_i|^2
   = \lVert \mathbf{m} \rVert^2
   = \mathbf{m}^\text{T} \mathbf{m}

Its gradient will be given by:

.. math::

   \nabla \phi_\text{m}(\mathbf{m}) = 2 \mathbf{m}

And its Hessian it's the identity matrix:

.. math::

   \nabla^2 \phi_\text{m}(\mathbf{m}) = 2 \mathbb{I}

We can implement this objective function as follows:

.. jupyter-execute::

   import numpy as np

   import inversion_ideas as ii


   class ModelNorm(ii.base.Objective):
       """
       Simple model norm regularization.

       Parameters
       ----------
       n_params : int
           Number of elements expected in the ``model`` array.
       """

       def __init__(self, n_params):
           self._n_params = n_params
           self._name = "norm"

       @property
       def n_params(self):
           return self._n_params

       def __call__(self, model):
           return model.T @ model

       def gradient(self, model):
           return 2 * model

       def hessian(self, model):
           return 2 * np.eye(self.n_params)


We can instantiate this function and use its methods:

.. jupyter-execute::

   n_params = 5
   model_norm = ModelNorm(n_params)
   print(model_norm)

.. jupyter-execute::

   model = np.arange(n_params, dtype=np.float64)
   model_norm(model)

.. jupyter-execute::

   model_norm.gradient(model)

.. jupyter-execute::

   model_norm.hessian(model)


We can also choose to change its default name through its ``_name`` attribute:

.. jupyter-execute::

   class ModelNorm(ii.base.Objective):
       """
       Simple model norm regularization.

       Parameters
       ----------
       n_params : int
           Number of elements expected in the ``model`` array.
       """

       def __init__(self, n_params):
           self._n_params = n_params
           self._name = "norm"

       @property
       def n_params(self):
           return self._n_params

       def __call__(self, model):
           return model.T @ model

       def gradient(self, model):
           return 2 * model

       def hessian(self, model):
           return 2 * np.eye(self.n_params)

.. jupyter-execute::

   model_norm = ModelNorm(n_params)
   print(model_norm)


Optimizing the Hessian matrix
-----------------------------

In the last implementation of ``ModelNorm`` we choose to return a dense matrix in the ``hessian`` method:

.. jupyter-execute::

   model_norm.hessian(model)


If ``n_params`` is too large, then this matrix can grow too big to fit in memory.
Since it's mostly filled with zeros in the non-diagonal elements, it might not be strictly needed to occupy all that memory.

One alternative would be to return a sparse matrix that represents the identity matrix:

.. jupyter-execute::

   import numpy as np
   from scipy.sparse import eye_array

   import inversion_ideas as ii


   class ModelNorm(ii.base.Objective):
       """
       Simple model norm regularization.

       Parameters
       ----------
       n_params : int
           Number of elements expected in the ``model`` array.
       """

       def __init__(self, n_params):
           self._n_params = n_params
           self._name = "norm"

       @property
       def n_params(self):
           return self._n_params

       def __call__(self, model):
           return model.T @ model

       def gradient(self, model):
           return 2 * model

       def hessian(self, model):
           return 2 * eye_array(self.n_params)

.. jupyter-execute::

   model_norm = ModelNorm(n_params)
   model_norm.hessian(model)

This strategy avoids storing all those non-diagonal zero elements in the matrix, but we are still storing all the twos in the diagonal.
We could reduce the memory requirements even more by making the ``hessian`` method to return a :class:`~scipy.sparse.linalg.LinearOperator` objects that implements dot products in the same way as
the identity matrix scaled by 2:

.. jupyter-execute::

   import numpy as np
   from scipy.sparse.linalg import LinearOperator

   import inversion_ideas as ii


   class ModelNorm(ii.base.Objective):
       """
       Simple model norm regularization.

       Parameters
       ----------
       n_params : int
           Number of elements expected in the ``model`` array.
       """

       def __init__(self, n_params):
           self._n_params = n_params
           self._name = "norm"

       @property
       def n_params(self):
           return self._n_params

       def __call__(self, model):
           return model.T @ model

       def gradient(self, model):
           return 2 * model

       def hessian(self, model):

           def matvec(v):
               "Dot product of 2I with vector."
               return 2 * v

           def rmatvec(v):
               "Dot product of transpose of 2I with vector."
               return 2 * v

           hessian = LinearOperator(
               shape=(self.n_params, self.n_params),
               dtype=np.float64,
               matvec=matvec,
               rmatvec=rmatvec,
           )
           return hessian

.. jupyter-execute::

   model_norm = ModelNorm(n_params)
   hessian = model_norm.hessian(model)
   hessian

We can use this :class:`~scipy.sparse.linalg.LinearOperator` to compute dot products:

.. jupyter-execute::

   vector = np.random.default_rng(seed=52).random(n_params)

   hessian @ vector


Plugging custom objective functions
-----------------------------------

Once we defined our objective function in a correct way, we can plug it into our inversion framework as with any other objective function from it.

For example, consider the same ``data_misfit`` term we defined in :ref:`objective-function`:

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

Let's build a ``ModelNorm`` regularization for this problem:

.. jupyter-execute::

   model_norm = ModelNorm(data_misfit.n_params)

.. We can estimate a tradeoff-parameter between the ``data_misfit`` and our custom ``model_norm``:

.. .. jupyter-execute::

..    initial_model = np.zeros(data_misfit.n_params)
..    beta = ii.utils.estimate_tradeoff_parameter(data_misfit, model_norm, initial_model)
..    beta

We can build a linear combination for the two objective functions and choose a trade-off parameter for the ``model_norm``:

.. jupyter-execute::

   phi = data_misfit + 30.0 * model_norm
   phi

We can minimize this objective function with the :func:`minimize.conjugate_gradient <inversion_ideas.minimize.conjugate_gradient>` (since the problem is linear):


.. jupyter-execute::

   initial_model = np.zeros(data_misfit.n_params)

   inverted_model = ii.minimize.conjugate_gradient(phi, initial_model)
   inverted_model



