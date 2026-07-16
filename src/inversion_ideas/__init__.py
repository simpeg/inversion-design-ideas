"""
Ideas for inversion framework.
"""

from . import base, conditions, decorators, directives, errors, operators, typing, utils
from ._version import __version__
from .data_misfit import DataMisfit
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
from .wires import Wires

__all__ = [
    "BFGSPreconditioner",
    "DataMisfit",
    "Flatness",
    "GaussNewtonConjugateGradient",
    "Inversion",
    "InversionLog",
    "InversionLogRich",
    "JacobiPreconditioner",
    "Smallness",
    "SparseSmallness",
    "TikhonovZero",
    "Wires",
    "__version__",
    "base",
    "conjugate_gradient",
    "create_l2_inversion",
    "create_sparse_inversion",
    "create_tikhonov_regularization",
    "decorators",
    "directives",
    "errors",
    "get_jacobi_preconditioner",
    "operators",
    "typing",
    "utils",
    "wrap_simulation",
]
