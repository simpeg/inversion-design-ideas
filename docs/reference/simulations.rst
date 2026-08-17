.. _api.simulations:

.. currentmodule:: inversion_ideas

Simulations
===========

The prototype for the new inversion framework does not implement any simulation
class that models a geophysical problem. It relies on SimPEG simulation classes instead.
But since the simulation classes in SimPEG are not yet compatible with the new
framework, we offer a *wrapper class* that allow us to extend them through a
:class:`inversion_ideas.base.Simulation` object so they become compatible with the framework.

.. important::

   The ultimate goal is to make SimPEG simulations compatible with the new
   framework.
   The :func:`inversion_ideas.wrap_simulation` function and the
   :func:`inversion_ideas.WrappedSimulation` class listed here are only temporary
   solutions to extend SimPEG simulations in a way that become compatible with
   the new framework.

.. autosummary::
   :toctree: api/

   wrap_simulation
   WrappedSimulation
