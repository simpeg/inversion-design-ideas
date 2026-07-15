.. _api:

API Reference
=============

.. currentmodule:: inversion_ideas


Objective functions
-------------------

Misfits
~~~~~~~

.. autosummary::
	:toctree: generated/

	DataMisfit

Regularizations
~~~~~~~~~~~~~~~

General purpose:

.. autosummary::
	:toctree: generated/

	TikhonovZero

Mesh-based:

.. autosummary::
	:toctree: generated/

	Smallness
	Flatness
	SparseSmallness

Minimizers
----------

.. autosummary::
	:toctree: generated/

    GaussNewtonConjugateGradient
    conjugate_gradient


Preconditioners
---------------

.. autosummary::
	:toctree: generated/

    BFGSPreconditioner
    JacobiPreconditioner
    get_jacobi_preconditioner

Directives
----------

.. autosummary::
	:toctree: generated/

    Irls
    MultiplierCooler
    UpdateSensitivityWeights

Conditions
----------

.. autosummary::
	:toctree: generated/

    ChiTarget
    CustomCondition
    ObjectiveChanged
    ModelChanged

Simulations
-----------

.. autosummary::
	:toctree: generated/

    wrap_simulation

Inversions and Recipes
----------------------

Inversion class:

.. autosummary::
	:toctree: generated/

    Inversion

Inversion logs:

.. autosummary::
	:toctree: generated/

    InversionLog
    InversionLogRich

Recipes:

.. autosummary::
	:toctree: generated/

    create_l2_inversion
    create_sparse_inversion
    create_tikhonov_regularization


Custom linear operators
-----------------------

.. autosummary::
	:toctree: generated/

    operators

Utilities
---------

.. autosummary::
	:toctree: generated/

    utils

Errors and warnings
-------------------

.. autosummary::
	:toctree: generated/

    ConvergenceWarning


Base submodule
--------------

The :mod:`inversion_ideas.base` submodule contains abstract classes and base classes that work as the foundation of the framework. Inherit from these classes when defining your own subclasses.

.. admonition:: important

	These classes are not meant to be instantiated. They are abstract classes only intended to be inherited from, or containers (like :class:`inversion_ideas.base.Scaled` and :class:`inversion_ideas.base.Combo`) that should not be instantiated by end users either.

.. autosummary::
	:toctree: generated/

	base.Objective
	base.Scaled
	base.Combo
	base.Simulation
	base.Minimizer
	base.MinimizerResult
	base.Condition
	base.Directive
