.. _api.hyperparams:

Hyperparameters
===============

Hyperparameters objects can be used as dynamic multipliers, weights, etc.: they have an ``update`` method that can be used to modify its values during an inversion.

For example, we could use the :class:`~inversion_ideas.hyperparams.CooledMultiplier` as a multiplier that after each call of the :meth:`~inversion_ideas.hyperparams.CooledMultiplier.update` method it will get *cooled down* by a constant factor.

The :class:`~inversion_ideas.hyperparams.SensitivityWeights` can be used as weights to a regularization. When the
:meth:`~inversion_ideas.hyperparams.SensitivityWeights.update` method is called, the sensitivity weights will get updated for a given ``model``.

The ``update`` methods of hyperparameter objects can be included in the ``directives`` list of  :class:`inversion_ideas.Inversion` so they get called after each iteration within the inversion.

.. currentmodule:: inversion_ideas

.. autosummary::
   :toctree: api/

     hyperparams.CooledMultiplier
     hyperparams.SensitivityWeights
