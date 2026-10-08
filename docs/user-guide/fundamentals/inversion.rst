.. _inversions:

Inversions
==========

Most of the inversion we run involve multiple minimization steps.
These inversions usually modify the objective function's hyperparameters
in between minimization steps in order to converge to a desired value of
those hyperparameters during the inversion process itself, or to implement a
particular algorithms (e.g. Iterative-Reweighted Least Squares).

In the following pages we'll see examples of different ways of running
inversions with the inversion framework.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   beta-cooling
   inversion-class


