"""
Directives to modify the objective function between iterations of an inversion.
"""

import numpy as np
import numpy.typing as npt

from .base import Directive, Objective, Scaled
from .conditions import ObjectiveChanged
from .data_misfit import DataMisfit
from .regularization import SparseSmallness


class MultiplierCooler(Directive):
    r"""
    Cool the multiplier of an objective function.

    Parameters
    ----------
    scaled_objective : Scaled
        Scaled objective function whose multiplier will be cooled.
    cooling_factor : float
        Factor by which the multiplier will be cooled.
    cooling_rate : int, optional
        Cool down the multiplier every ``cooling_rate`` call to this directive.

    Notes
    -----
    Given a scaled objective function :math:`\phi(\mathbf{m}) = \alpha
    \varphi(\mathbf{m})`, and a cooling factor :math:`k`, this directive will *cool* the
    multiplier `\alpha` by dividing it by :math:`k` on every ``cooling_rate`` call to
    the directive.
    """

    def __init__(
        self, scaled_objective: Scaled, cooling_factor: float, cooling_rate: int = 1
    ):
        if not hasattr(scaled_objective, "multiplier"):
            msg = "Invalid 'scaled_objective': it must have a `multiplier` attribute."
            raise TypeError(msg)
        self.regularization = scaled_objective
        self.cooling_factor = cooling_factor
        self.cooling_rate = cooling_rate

    def __call__(self, model: npt.NDArray[np.float64], iteration: int):  # noqa: ARG002
        """
        Cool the multiplier.
        """
        if iteration % self.cooling_rate == 0:
            self.regularization.multiplier /= self.cooling_factor


class IRLS(Directive):
    """
    Apply iterative reweighed least squares (IRLS).

    Parameters
    ----------
    sparse : SparseSmallness
        Sparse regularization that will get IRLS updated.
    data_misfit : Objective
        Data misfit function that will be evaluated to decide whether to update the
        IRLS on ``sparse``, or to cool the multiplier of ``regularization``.
    regularization : Scaled
        Regularization that will get its multiplier cooled down.
    model_stage_one : (nparams) array
        Model obtained after the first stage of the sparse inversion is finished.
        This model should be the one obtained after the L2 inverison.
    beta_cooling_factor : float, optional
        Cooling factor used to cool down the ``regularization``'s multiplier.
    data_misfit_rtol : float, optional
        Relative tolerance for the data misfit.
        Used to compare the current value of the data misfit with its value on
        ``model_stage_one``.
    """

    def __init__(
        self,
        sparse: SparseSmallness,
        *,
        data_misfit: Objective,
        regularization: Scaled,
        model_stage_one: npt.NDArray[np.float64],
        beta_cooling_factor=2.0,
        data_misfit_rtol=1e-1,
    ):
        for method in ("update_irls", "activate_irls"):
            if not hasattr(sparse, method):
                msg = f"Invalid `sparse` object without `{method}` method."
                raise AttributeError(msg)
        if not hasattr(sparse, "irls"):
            msg = "Invalid `sparse` object without `irls` attribute."
            raise AttributeError(msg)
        if not sparse.irls:
            msg = "Sparse regularization should have IRLS active."
            raise ValueError(msg)

        self.data_misfit = data_misfit
        self.sparse = sparse
        self.regularization = regularization
        self.data_misfit_rtol = data_misfit_rtol
        self.beta_cooling_factor = beta_cooling_factor
        self.model_stage_one = model_stage_one

        # Define a beta cooler
        self._beta_cooler = MultiplierCooler(
            regularization, cooling_factor=self.beta_cooling_factor
        )

        # Define a condition for the data misfit.
        # Compare it always with the data misfit obtained with the model from l2
        # inversion.
        self._dmisfit_below_threshold = ObjectiveChanged(
            data_misfit, rtol=self.data_misfit_rtol
        )
        self._dmisfit_l2 = self.data_misfit(self.model_stage_one)
        self._dmisfit_below_threshold.previous = self._dmisfit_l2

    def __call__(self, model: npt.NDArray[np.float64], iteration: int):
        """
        Apply IRLS.

        Cool down beta or update IRLS depending on the values of the data misfit.
        """
        if not self._dmisfit_below_threshold(model):
            # Cool beta if the data misfit is quite different from the l2 one
            phi_d = self.data_misfit(model)
            # Adjust the cooling factor
            # (following current implementation of UpdateIRLS)
            if phi_d > self._dmisfit_l2:
                self._beta_cooler.cooling_factor = 1 / np.mean(
                    [0.75, self._dmisfit_l2 / phi_d]
                )
            else:
                self._beta_cooler.cooling_factor = 1 / np.mean(
                    [2.0, self._dmisfit_l2 / phi_d]
                )
            self._beta_cooler(model, iteration)
        else:
            # Update the IRLS
            self.sparse.update_irls(model)


class IrlsFull(Directive):
    """
    Apply iterative reweighed least squares (IRLS).

    This directive is intended to work with a single inversion that performs the two
    stages.

    Parameters
    ----------
    sparse : SparseSmallness
        Sparse regularization that will get IRLS updated.
    data_misfit : DataMisfit
        Data misfit function that will be evaluated to decide whether to update the
        IRLS on ``sparse``, or to cool the multiplier of ``regularization``.
    regularization : Scaled
        Regularization that will get its multiplier cooled down.
    chi_l2_target : float, optional
        Target for the chi factor used in the first stage (L2 inversion). Once this
        target is reached, the IRLS will be activated.
    beta_cooling_factor : float, optional
        Cooling factor used to cool down the ``regularization``'s multiplier.
    data_misfit_rtol : float, optional
        Relative tolerance for the data misfit.
        Used to compare the current value of the data misfit with its value after the
        stage one is finished.
    """

    def __init__(
        self,
        sparse: SparseSmallness,
        *,
        data_misfit: DataMisfit,
        regularization: Scaled,
        chi_l2_target=1.0,
        beta_cooling_factor=2.0,
        data_misfit_rtol=1e-1,
    ):
        for method in ("update_irls", "activate_irls"):
            if not hasattr(sparse, method):
                msg = f"Invalid `sparse` object without `{method}` method."
                raise TypeError(msg)
        if not hasattr(sparse, "irls"):
            msg = "Invalid `sparse` object without `irls` attribute."
            raise TypeError(msg)
        if not hasattr(data_misfit, "chi"):
            msg = "Invalid `data_misfit` object without `chi` method."
            raise TypeError(msg)

        self.data_misfit = data_misfit
        self.sparse = sparse
        self.regularization = regularization
        self.data_misfit_rtol = data_misfit_rtol
        self.beta_cooling_factor = beta_cooling_factor
        self.chi_l2_target = chi_l2_target

        # Define a beta cooler
        self._beta_cooler = MultiplierCooler(
            regularization, cooling_factor=self.beta_cooling_factor
        )

        # Define a condition for the data misfit.
        # Compare it always with the data misfit obtained with the model from l2
        # inversion.
        self._dmisfit_below_threshold = ObjectiveChanged(
            data_misfit, rtol=self.data_misfit_rtol
        )

    def __call__(self, model: npt.NDArray[np.float64], iteration: int):
        """
        Apply IRLS.

        Cool down beta or update IRLS depending on the values of the data misfit.
        """
        # Cool down beta until IRLS gets activated
        if not self.sparse.irls:
            self._stage_one(model, iteration)
        else:
            self._stage_two(model, iteration)

    def _stage_one(self, model, iteration):
        """
        Implement first stage of the IRLS inversion.
        """
        if self.data_misfit.chi(model) < self.chi_l2_target:
            # Activate IRLS if chi target has been met
            self.sparse.activate_irls(model)
            # Cache some attributes
            self._model_l2 = model
            self._dmisfit_l2 = self.data_misfit(self._model_l2)
            self._dmisfit_below_threshold.previous = self._dmisfit_l2
            return

        # Cool down beta otherwise
        self._beta_cooler(model, iteration)

    def _stage_two(self, model, iteration):
        """
        Implement second stage of the IRLS inversion.
        """
        if not self._dmisfit_below_threshold(model):
            # Cool beta if the data misfit is quite different from the l2 one
            phi_d = self.data_misfit(model)
            # Adjust the cooling factor
            # (following current implementation of UpdateIRLS)
            if phi_d > self._dmisfit_l2:
                self._beta_cooler.cooling_factor = 1 / np.mean(
                    [0.75, self._dmisfit_l2 / phi_d]
                )
            else:
                self._beta_cooler.cooling_factor = 1 / np.mean(
                    [2.0, self._dmisfit_l2 / phi_d]
                )
            self._beta_cooler(model, iteration)
        else:
            # Update the IRLS
            self.sparse.update_irls(model)
