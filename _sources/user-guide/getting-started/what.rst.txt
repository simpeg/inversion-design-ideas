.. _what-is:

What is this new framework?
===========================

The `simpeg/inversion-design-ideas <https://github.com/simpeg/inversion-design-ideas>`__
is a repo that holds a prototype for a redesign of SimPEG's inversion framework.
The main goal of the redesign is to create a new public inversion framework in
SimPEG that:

* Is modular, allowing users to plug custom or third-party minimizers,
  simulations, regularizations, etc.
* Can be easily extended (with minimum extra work) to problems outside the
  traditional :math:`\phi(m) = \phi_d(m) + \beta \phi_m(m)` inversion problem.
* Is implemented in such way that is easier to read, study, and understand by
  beginners that are taking their first steps in inversion theory.
* Is implemented in such way that can be easily extended to more complex and/or
  complicated use cases.
* Defines a clear and minimal interface for each one of the classes.
* Makes use of `abstract classes <https://docs.python.org/3/library/abc.html>`__
  to enforce implementation of required methods and properties.
* Simplifies the inheritance tree by lowering the amount of inheritance levels.

.. warning::

   This repository is still in experimental stages.
   Its purpose is to have a place where we can freely try new ideas without having
   to worry about breaking existing code, and supporting backward compatibility.
