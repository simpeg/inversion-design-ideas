"""
Directives to modify the objective function between iterations of an inversion.
"""
import warnings
import numpy as np

from ._utils import extract_from_combo
from .base import Combo, Directive, Objective, Scaled, Simulation
from .conditions import ObjectiveChanged
from .data_misfit import DataMisfit
from .typing import Model, SparseRegularization
from .utils import get_logger, get_sensitivity_weights


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

    def __call__(self, model: Model, iteration: int):  # noqa: ARG002
        """
        Cool the multiplier.
        """
        if iteration % self.cooling_rate == 0:
            self.regularization.multiplier /= self.cooling_factor


class Irls(Directive):
    """
    Apply iterative reweighed least squares (IRLS).

    This directive is intended to work with a single inversion that performs the two
    stages.

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
        self.sparse_regs: list[
            SparseRegularization
        ] = self._extract_sparse_regularizations(args)
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

        # Define a beta cooler
        self._beta_cooler = (
            MultiplierCooler(
                self.regularization_with_beta, cooling_factor=beta_cooling_factor
            )
            if cool_beta
            else None
        )

        # Define a condition for the data misfit.
        # Compare it always with the data misfit obtained with the model from l2
        # inversion.
        self._dmisfit_below_threshold = ObjectiveChanged(
            data_misfit, rtol=self.data_misfit_rtol
        )

    @property
    def beta_cooling_factor(self) -> float | None:
        """
        Current beta cooling factor.
        """
        if self._beta_cooler is None:
            return None
        return self._beta_cooler.cooling_factor

    def __call__(self, model: Model, iteration: int):
        """
        Apply IRLS.

        Cool down beta or update IRLS depending on the values of the data misfit.
        """
        # Cool down beta until IRLS gets activated
        if not all(sparse_reg.irls for sparse_reg in self.sparse_regs):
            self._stage_one(model, iteration)
        else:
            self._stage_two(model, iteration)

    def _stage_one(self, model: Model, iteration: int):
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
        if self._beta_cooler is not None:
            self._beta_cooler(model, iteration)

    def _stage_two(self, model: Model, iteration: int):
        """
        Implement second stage of the IRLS inversion.
        """
        if not self._dmisfit_below_threshold(model):
            # Cool beta if the data misfit is quite different from the l2 one
            phi_d = self.data_misfit(model)
            # Adjust the cooling factor
            # (following current implementation of UpdateIRLS)
            if self._beta_cooler is not None:
                if self._beta_cooler.cooling_factor != 1:
                    if phi_d > self._dmisfit_l2:
                        self._beta_cooler.cooling_factor = float(
                            1 / np.mean([0.75, self._dmisfit_l2 / phi_d])
                        )
                    else:
                        self._beta_cooler.cooling_factor = float(
                            1 / np.mean([2.0, self._dmisfit_l2 / phi_d])
                        )
                self._beta_cooler(model, iteration)
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


class UpdateSensitivityWeights(Directive):
    """
    Update sensitivity weights on regularizations.

    .. note::

        This directive can only be applied to regularizations that:
        1. have a ``cell_weights`` attribute,
        2. the ``cell_weights`` attribute is a dictionary,
        3. the ``cell_weights`` attribute contains weights under the key specified
           through the ``weights_key`` argument ("sensitivity" by default).

    Parameters
    ----------
    *args : Objective
        Regularizations to which the sensitivity weights will be updated.
        If a :class:`inversion_ideas.base.Combo` or
        a :class:`inversion_ideas.base.Scaled` are passed, they will be explored
        recursively to use regularizations that have sensitivity weights that can be
        updated.
    simulation : Simulation
        Simulation used to get the jacobian matrix that will be used while updating the
        sensitivity weights.
    weights_key : str, optional
        Key used to store the sensitivity weights on the regularization's
        ``cell_weights`` dictionary. Only the weights under this key will be updated.
    **kwargs
        Extra arguments passed to the
        :func:`inversion_ideas.utils.get_sensitivity_weights` function.

    See Also
    --------
    inversion_ideas.utils.get_sensitivity_weights
    """

    def __init__(
        self,
        *args: Objective,
        simulation: Simulation,
        weights_key: str = "sensitivity",
        **kwargs,
    ):
        if not args:
            msg = "Missing regularization. Pass at least one."
            raise TypeError(msg)

        self.weights_key = weights_key
        self.simulation = simulation
        self.kwargs = kwargs
        self.regularizations: list[Objective] = self._extract_regularizations(args)

        if not self.regularizations:
            msg = (
                "Invalid regularizations passed through the `args` argument. "
                "Couldn't locate any regularization term to update "
                "their sensitivity weights."
            )
            raise TypeError(msg)

    def __call__(self, model: Model, iteration: int):  # noqa: ARG002
        """
        Update sensitivity weights.
        """
        # Compute the jacobian and the new sensitivity weights
        jacobian = self.simulation.jacobian(model)
        self._check_jacobian_type(jacobian)
        new_sensitivity_weights = get_sensitivity_weights(jacobian, **self.kwargs)

        # Update sensitivity weights on regularizations
        for regularization in self.regularizations:
            self._check_cell_weights(regularization)
            regularization.cell_weights[self.weights_key] = new_sensitivity_weights

    def _extract_regularizations(self, args: tuple[Objective, ...]) -> list[Objective]:
        """
        Select regularizations to update their sensitivity weights.

        Extract a selection of the regularizations passed as arguments to build the
        ``self.regularizations`` attribute. Follow this criteria:

        - Any objective function that is not a ``Combo`` or a ``Scaled`` will be added
          as is. We'll check if the regularization has sensitivity weights (see below).
        - Any ``Combo`` or ``Scaled`` will be recursively explored to extract any
          regularization function contained by them that has sensitivity weights.

        A regularization is considered to have sensitivity weights if:

        1. Has a ``cell_weights`` attribute.
        2. Its ``cell_weights`` attribute is a dictionary.
        3. Its ``cell_weights`` attribute has a key equal to ``self.weights_key``.
        """

        def has_sensitivity_weights(regularization: Objective) -> bool:
            return (
                hasattr(regularization, "cell_weights")
                and isinstance(regularization.cell_weights, dict)
                and self.weights_key in regularization.cell_weights
            )

        regularizations = []
        for objective in args:
            if isinstance(objective, Scaled | Combo):
                extracted_regs = extract_from_combo(objective, has_sensitivity_weights)
                for reg in extracted_regs:
                    get_logger().debug(
                        f"Sensitivity weights of {reg} will be updated "
                        f"by the {self} directive."
                    )
                regularizations += extracted_regs
            else:
                self._check_cell_weights(objective)
                regularizations.append(objective)

        return regularizations

    def _check_jacobian_type(self, jacobian):
        """Check if jacobian is a dense array."""
        if not isinstance(jacobian, np.ndarray):
            msg = (
                "Cannot compute sensitivity weights for simulation "
                f"{self.simulation} : its jacobian is a {type(jacobian)}. "
                "It must be a dense array."
            )
            raise TypeError(msg)

    def _check_cell_weights(self, regularization: Objective):
        """Sanity checks for cell_weights in regularization."""
        # Check if regularization have cell_weights attribute
        if not hasattr(regularization, "cell_weights"):
            msg = (
                "Missing `cell-weights` attribute in regularization "
                f"'{regularization}'."
            )
            raise AttributeError(msg)

        if not isinstance(regularization.cell_weights, dict):
            msg = (
                f"Invalid `cell_weights` attribute of type '{type(regularization)}' "
                f"for the '{regularization}'. It must be a dictionary."
            )
            raise TypeError(msg)
        if self.weights_key not in regularization.cell_weights:
            msg = (
                f"Missing '{self.weights_key}' weights in "
                f"{regularization}.cell_weights. "
            )
            raise KeyError(msg)
