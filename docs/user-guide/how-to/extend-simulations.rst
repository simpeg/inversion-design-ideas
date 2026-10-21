.. _extend-simulations:

Write new simulations
=====================

We've seen that it's possible to :ref:`write new objective functions <extend-objective>` and plug them into the new inversion framework by inheriting from the :class:`base.Objective <inversion_ideas.base.Objective>` abstract class and implementing its abstract methods.

We can do something similar if we want to write our own simulations.
In that case we need to create a class that inherits from :class:`base.Simulation <inversion_ideas.base.Simulation>` and implements its
:attr:`base.Simulation.n_data <inversion_ideas.base.Simulation.n_data>`
and
:attr:`base.Simulation.n_params <inversion_ideas.base.Simulation.n_params>`
abstract properties,
and its
:attr:`base.Simulation.__call__ <inversion_ideas.base.Simulation.__call__>`
and
:attr:`base.Simulation.jacobian <inversion_ideas.base.Simulation.jacobian>`
asbtract methods.
