"""
Directives to modify the objective function between iterations of an inversion.
"""

import numpy as np

from ._utils import extract_from_combo
from .base import Combo, Objective, Scaled
from .conditions import ObjectiveChanged
from .data_misfit import DataMisfit
from .typing import Model, SparseRegularization
from .utils import get_logger

__all__ = [
    "Irls",
]


class Irls:
    """
    Apply iterative reweighed least squares (IRLS).

    This directive is intended to work with a single inversion that performs the two
    stages.

    .. warning::

        This directive is still in experimental stages and might change in the future.

    .. note::

        This directive can only be applied to sparse (lp norm) regularizations. In
        summary they should:

        1. have a ``irls`` bool attribute,
        2. have a ``update_irls`` and a ``activate_irls`` methods.


    Parameters
    ----------
    *args : Objective
        Sparse regularizations that will get IRLS updated.
        It can be a single regularization object
        (e.g. :class:`inversion_ideas.SmallnessSparse`), a
        :class:`inversion_ideas.base.Combo`, or a :class:`inversion_ideas.base.Scaled`,
        or multiple of them.
        :class:`inversion_ideas.base.Combo` and
        :class:`inversion_ideas.base.Scaled` regularizations will be explored
        recursively to use regularizations terms that have sensitivity weights that can
        be updated.
    data_misfit : DataMisfit
        Data misfit function that will be evaluated to decide whether to update the
        IRLS on ``sparse``, or to cool the multiplier of ``regularization``.
    regularization_with_beta : Scaled or None, optional
        Regularization that will get its multiplier cooled down.
        If a single ``arg`` is passed, it will be used as the regularization that will
        get its multiplier cooled down. Pass a ``regularization_with_beta`` if another
        regularization's multiplier should be cooled down, or if multiple ``args`` are
        passed.
    chi_l2_target : float, optional
        Target for the chi factor used in the first stage (L2 inversion). Once this
        target is reached, the IRLS will be activated.
    beta_cooling_factor : float, optional
        Cooling factor used to cool down the ``regularization``'s multiplier.
    data_misfit_rtol : float, optional
        Relative tolerance for the data misfit.
        Used to compare the current value of the data misfit with its value after the
        stage one is finished.
    cool_beta : bool, optional
        Whether to cool down beta during the IRLS process.
        If False, make sure you handle beta cooling in other way, like through other
        directive.

        .. warning::
            If False, the Irls directive won't cool down beta during the inversions.
            This might prevent from reaching convergence.
            Make sure you handle beta cooling in other way, like through other
            directive.
    """

    def __init__(
        self,
        *args: Objective,
        data_misfit: DataMisfit,
        regularization_with_beta: Scaled | None = None,
        chi_l2_target=1.0,
        beta_cooling_factor=2.0,
        data_misfit_rtol=1e-1,
        cool_beta=True,
    ):
        if len(args) == 0:
            msg = (
                "Missing sparse regularization. "
                "Pass at least one to the IRLS directive."
            )
            raise TypeError(msg)

        if regularization_with_beta is None:
            # Raise error if multiple sparse regs and regularization_with_beta as None
            if len(args) > 1:
                msg = (
                    "Cannot pass multiple sparse regularizations and leave "
                    "'regularization_with_beta' as None. "
                )
                raise TypeError(msg)
            # Assign the sparse regularization passed through args as the regularization
            # with beta
            (_reg,) = args
            if not isinstance(_reg, Scaled):
                msg = (
                    f"Cannot use {regularization_with_beta} as the "
                    "'regularization_with_beta' since it doesn't have a multiplier "
                    "that can be cooled down. "
                    "Pass a value to 'regularization_with_beta' or pass a scaled "
                    "regularization through the 'args'."
                )
                raise TypeError(msg)
            regularization_with_beta = _reg

        self.regularization_with_beta: Scaled = regularization_with_beta
        self.sparse_regs: list[SparseRegularization] = (
            self._extract_sparse_regularizations(args)
        )
        if not self.sparse_regs:
            msg = (
                "Invalid regularizations passed through the `args` argument. "
                "Couldn't locate any sparse regularization term in them."
            )
            raise TypeError(msg)

        self.data_misfit = data_misfit
        if not hasattr(data_misfit, "chi_factor"):
            msg = "Invalid `data_misfit` object without `chi_factor` method."
            raise TypeError(msg)

        self.data_misfit_rtol = data_misfit_rtol
        self.chi_l2_target = chi_l2_target

        # Define beta cooling variables
        self._beta_cooling_factor = beta_cooling_factor
        self._cool_beta = cool_beta

        # Define a condition for the data misfit.
        # Compare it always with the data misfit obtained with the model from l2
        # inversion.
        self._dmisfit_below_threshold = ObjectiveChanged(
            data_misfit, rtol=self.data_misfit_rtol
        )

    @property
    def cool_beta(self) -> bool:
        """
        Whether if beta will be cooled or not.
        """
        return self._cool_beta

    @property
    def beta_cooling_factor(self) -> float | None:
        """
        Current beta cooling factor.
        """
        if not self.cool_beta:
            return None
        return self._beta_cooling_factor

    def _cool_down_beta(self):
        """Cool down the beta multiplier."""
        self.regularization_with_beta.multiplier /= self.beta_cooling_factor

    def __call__(self, model: Model):
        """
        Apply IRLS.

        Cool down beta or update IRLS depending on the values of the data misfit.
        """
        # Cool down beta until IRLS gets activated
        if not all(sparse_reg.irls for sparse_reg in self.sparse_regs):
            self._stage_one(model)
        else:
            self._stage_two(model)

    def _stage_one(self, model: Model):
        """
        Implement first stage of the IRLS inversion.
        """
        if self.data_misfit.chi_factor(model) < self.chi_l2_target:
            # Activate IRLS if chi target has been met
            for sparse_reg in self.sparse_regs:
                sparse_reg.activate_irls(model)
            # Cache some attributes
            self._model_l2 = model
            self._dmisfit_l2 = self.data_misfit(self._model_l2)
            self._dmisfit_below_threshold.previous = self._dmisfit_l2
            return

        # Cool down beta otherwise
        if self.cool_beta:
            self._cool_down_beta()

    def _stage_two(self, model: Model):
        """
        Implement second stage of the IRLS inversion.
        """
        if not self._dmisfit_below_threshold(model):
            # Cool beta if the data misfit is quite different from the l2 one
            phi_d = self.data_misfit(model)
            # Adjust the cooling factor
            # (following current implementation of UpdateIRLS)
            if self.cool_beta:
                if self.beta_cooling_factor != 1:
                    if phi_d > self._dmisfit_l2:
                        self._beta_cooling_factor = float(
                            1 / np.mean([0.75, self._dmisfit_l2 / phi_d])
                        )
                    else:
                        self._beta_cooling_factor = float(
                            1 / np.mean([2.0, self._dmisfit_l2 / phi_d])
                        )
                self._cool_down_beta()
        else:
            # Update the IRLS
            for sparse_reg in self.sparse_regs:
                sparse_reg.update_irls(model)

    def _extract_sparse_regularizations(
        self, args: tuple[Objective, ...]
    ) -> list[SparseRegularization]:
        """
        Select sparse regularizations recursively from the passed args.
        """

        def is_sparse(regularization: Objective) -> bool:
            return (
                hasattr(regularization, "irls")
                and hasattr(regularization, "update_irls")
                and hasattr(regularization, "activate_irls")
            )

        sparse_regs = []
        for objective in args:
            if isinstance(objective, Scaled | Combo):
                extracted_regs = extract_from_combo(objective, is_sparse)
                for reg in extracted_regs:
                    get_logger().debug(
                        f"Sparse regularization {reg} will get IRLS managed "
                        f"by the {self} directive."
                    )
                sparse_regs += extracted_regs
            elif is_sparse(objective):
                get_logger().debug(
                    f"Sparse regularization {objective} will get IRLS managed "
                    f"by the {self} directive."
                )
                sparse_regs.append(objective)

        return sparse_regs
