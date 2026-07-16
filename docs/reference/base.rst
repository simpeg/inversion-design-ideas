.. _api.base:

.. currentmodule:: inversion_ideas

Base submodule
==============

The :mod:`inversion_ideas.base` submodule contains abstract classes and base classes that work as the foundation of the framework. Inherit from these classes when defining your own subclasses.

.. admonition:: important

   These classes are not meant to be instantiated. They are abstract classes only intended to be inherited from, or containers (like :class:`inversion_ideas.base.Scaled` and :class:`inversion_ideas.base.Combo`) that should not be instantiated by end users either.

.. autosummary::
   :toctree: api/

   base
   base.Objective
   base.Scaled
   base.Combo
   base.Simulation
   base.Minimizer
   base.MinimizerResult
   base.Condition
   base.Directive
