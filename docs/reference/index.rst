.. _api:

API Reference
=============

.. currentmodule:: inversion_ideas


Objective functions
-------------------

Misfits
~~~~~~~

.. autosummary::
   :toctree: api/

     DataMisfit

Regularizations
~~~~~~~~~~~~~~~

General purpose:

.. autosummary::
   :toctree: api/

    TikhonovZero

Mesh-based:

.. autosummary::
   :toctree: api/

    Smallness
    Flatness
    SparseSmallness

Minimizers
----------

.. autosummary::
   :toctree: api/

    GaussNewtonConjugateGradient
    conjugate_gradient


Preconditioners
---------------

.. autosummary::
   :toctree: api/

   BFGSPreconditioner
   JacobiPreconditioner
   get_jacobi_preconditioner

Directives
----------

.. autosummary::
   :toctree: api/

   Irls
   MultiplierCooler
   UpdateSensitivityWeights

Conditions
----------

.. autosummary::
   :toctree: api/

   ChiTarget
   CustomCondition
   ObjectiveChanged
   ModelChanged

Simulations
-----------

.. autosummary::
   :toctree: api/

   wrap_simulation

Inversions and Recipes
----------------------

Inversion class:

.. autosummary::
   :toctree: api/

   Inversion

Inversion logs:

.. autosummary::
   :toctree: api/

   InversionLog
   InversionLogRich

Recipes:

.. autosummary::
   :toctree: api/

   create_l2_inversion
   create_sparse_inversion
   create_tikhonov_regularization


Custom linear operators
-----------------------

.. autosummary::
   :toctree: api/

   operators

Utilities
---------

.. autosummary::
   :toctree: api/

   utils

Errors and warnings
-------------------

.. autosummary::
   :toctree: api/

   ConvergenceWarning


Base submodule
--------------

The :mod:`inversion_ideas.base` submodule contains abstract classes and base classes that work as the foundation of the framework. Inherit from these classes when defining your own subclasses.

.. admonition:: important

   These classes are not meant to be instantiated. They are abstract classes only intended to be inherited from, or containers (like :class:`inversion_ideas.base.Scaled` and :class:`inversion_ideas.base.Combo`) that should not be instantiated by end users either.

.. autosummary::
   :toctree: api/

   base.Objective
   base.Scaled
   base.Combo
   base.Simulation
   base.Minimizer
   base.MinimizerResult
   base.Condition
   base.Directive
