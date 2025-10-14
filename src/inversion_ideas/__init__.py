"""
Ideas for inversion framework.
"""

from . import base, typing, utils
from ._version import __version__
from .conditions import ChiTarget, CustomCondition, ModelChanged, ObjectiveChanged
from .constructors import create_inversion, create_sparse_inversion
from .data_misfit import DataMisfit
from .directives import (
    MultiplierCooler,
    UpdateSensitivityWeights,
    IRLS,
    IrlsFull,
)
from .errors import ConvergenceWarning
from .inversion import Inversion
from .inversion_log import InversionLog, InversionLogRich
from .minimize import GaussNewtonConjugateGradient, conjugate_gradient
from .preconditioners import JacobiPreconditioner, get_jacobi_preconditioner
from .regularization import Flatness, Smallness, TikhonovZero, SparseSmallness
from .simulations import wrap_simulation

__all__ = [
    "IRLS",
    "ChiTarget",
    "ConvergenceWarning",
    "CustomCondition",
    "DataMisfit",
    "Flatness",
    "GaussNewtonConjugateGradient",
    "Inversion",
    "InversionLog",
    "InversionLogRich",
    "IrlsFull",
    "JacobiPreconditioner",
    "ModelChanged",
    "MultiplierCooler",
    "ObjectiveChanged",
    "SparseSmallness",
    "Smallness",
    "TikhonovZero",
    "UpdateSensitivityWeights",
    "__version__",
    "base",
    "conjugate_gradient",
    "create_inversion",
    "create_sparse_inversion",
    "get_jacobi_preconditioner",
    "typing",
    "utils",
    "wrap_simulation",
]
