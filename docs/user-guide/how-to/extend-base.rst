.. _extend-base:

The :mod:`~inversion_ideas.base` submodule
==========================================

The new inversion framework is built on top of a set of *base* classes that specify the interface of the child ones that will finally implement the particular methods.
These base classes live in the :mod:`inversion_ideas.base` submodule (see the :ref:`API reference for the base submodule <api.base>`).
The majority of the base classes are `abstract classes <https://docs.python.org/3/library/abc.html>`__ that require child classes to implement all the `abstract methods <https://docs.python.org/3/library/abc.html#abc.abstractmethod>`__ for that class to be instatiable.
We can understand these abstract classes as a design layout that the child classes are forced to follow.

For example, the :class:`inversion_ideas.base.Objective` is an abstract class that works as a base class for every objective function.
The :class:`base.Objective <inversion_ideas.base.Objective>` implements a few methods and properties that will be shared by all objective functions, but define some abstract methods that any child should implement:

* the :meth:`Objective.__call__ <inversion_ideas.base.Objective.__call__>` method,
* the :meth:`Objective.gradient <inversion_ideas.base.Objective.gradient>` method,
* the :meth:`Objective.hessian <inversion_ideas.base.Objective.hessian>` method, and
* the :attr:`Objective.n_params <inversion_ideas.base.Objective.n_params>` property.

We cannot directly instantiate this class:

.. jupyter-execute::
   :raises: TypeError

   import inversion_ideas as ii

   ii.base.Objective()

But we can instantiate the :class:`inversion_ideas.SimpleSmallness` a subclass of
:class:`base.Objective <inversion_ideas.base.Objective>` that implements all its abstract methods and properties.

.. jupyter-execute::

   smallness = ii.SimpleSmallness(4)

.. note::

   The use of abstract classes makes it easier to guarantee that
   non-abstract classes implement the same interface as all its sibling
   classes. This make the framework more modular: a minimizer like
   :class:`~inversion_ideas.minimize.GaussNewtonConjugateGradient` can take
   any :class:`base.Objective <inversion_ideas.base.Objective>` object as an
   objective function, regardless how it's implemented and where it comes from,
   as long as it correctly implements its abstract methods.


The :mod:`inversion_ideas.base` submodule also contains container classes like the :class:`base.Combo <inversion_ideas.base.Combo>` and :class:`base.Combo <inversion_ideas.base.Scaled>`. These classes are non-abstract :class:`~inversion_ideas.base.Objective` that implements all of its abstract methods, but they are not intended to be instantiated by end users: they are formed when adding together two :class:`~inversion_ideas.base.Objective` functions, or when multiplying one of them by a scalar.

.. jupyter-execute::

   flatness = ii.SimpleFlatness(4)
   combo = smallness + flatness
   type(combo)

.. jupyter-execute::

   scaled = 2.4 * smalness
   type(scaled)
