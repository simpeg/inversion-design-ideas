"""
Ideas for inversion framework.
"""

from . import base, typing, utils
from ._version import __version__
from .conditions import ChiTarget, CustomCondition, ModelChanged, ObjectiveChanged
from .data_misfit import DataMisfit
from .directives import (
    Irls,
    MultiplierCooler,
    UpdateSensitivityWeights,
)
from .errors import ConvergenceWarning
from .inversion import Inversion
from .inversion_log import InversionLog, InversionLogRich
from .minimize import GaussNewtonConjugateGradient, conjugate_gradient
from .preconditioners import JacobiPreconditioner, get_jacobi_preconditioner
from .recipes import create_l2_inversion, create_sparse_inversion
from .regularization import Flatness, Smallness, SparseSmallness, TikhonovZero
from .simulations import wrap_simulation

__all__ = [
    "ChiTarget",
    "ConvergenceWarning",
    "CustomCondition",
    "DataMisfit",
    "Flatness",
    "GaussNewtonConjugateGradient",
    "Inversion",
    "InversionLog",
    "InversionLogRich",
    "Irls",
    "JacobiPreconditioner",
    "ModelChanged",
    "MultiplierCooler",
    "ObjectiveChanged",
    "Smallness",
    "SparseSmallness",
    "TikhonovZero",
    "UpdateSensitivityWeights",
    "__version__",
    "base",
    "conjugate_gradient",
    "create_l2_inversion",
    "create_sparse_inversion",
    "get_jacobi_preconditioner",
    "typing",
    "utils",
    "wrap_simulation",
]
