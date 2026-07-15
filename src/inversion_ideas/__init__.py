"""
Ideas for inversion framework.
"""

from . import base, errors, operators, typing, utils
from ._version import __version__
from .conditions import ChiTarget, CustomCondition, ModelChanged, ObjectiveChanged
from .data_misfit import DataMisfit
from .directives import (
    Irls,
    MultiplierCooler,
    UpdateSensitivityWeights,
)
from .inversion import Inversion
from .inversion_log import InversionLog, InversionLogRich
from .minimize import GaussNewtonConjugateGradient, conjugate_gradient
from .preconditioners import (
    BFGSPreconditioner,
    JacobiPreconditioner,
    get_jacobi_preconditioner,
)
from .recipes import (
    create_l2_inversion,
    create_sparse_inversion,
    create_tikhonov_regularization,
)
from .regularization import Flatness, Smallness, SparseSmallness, TikhonovZero
from .simulations import wrap_simulation

__all__ = [
    "BFGSPreconditioner",
    "ChiTarget",
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
    "create_tikhonov_regularization",
    "errors",
    "get_jacobi_preconditioner",
    "operators",
    "typing",
    "utils",
    "wrap_simulation",
]
