"""
Ideas for inversion framework.
"""

from . import (
    base,
    conditions,
    decorators,
    directives,
    errors,
    hyperparams,
    minimize,
    operators,
    typing,
    utils,
)
from ._version import __version__
from .data_misfit import DataMisfit
from .inversion import Inversion
from .inversion_log import InversionLog, InversionLogRich
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
from .regularization import (
    Flatness,
    SimpleFlatness,
    SimpleSmallness,
    Smallness,
    SparseSmallness,
)
from .simulations import LinearRegressor, WrappedSimulation, wrap_simulation

__all__ = [
    "BFGSPreconditioner",
    "DataMisfit",
    "Flatness",
    "Inversion",
    "InversionLog",
    "InversionLogRich",
    "JacobiPreconditioner",
    "LinearRegressor",
    "SimpleFlatness",
    "SimpleSmallness",
    "Smallness",
    "SparseSmallness",
    "__version__",
    "base",
    "create_l2_inversion",
    "create_sparse_inversion",
    "create_tikhonov_regularization",
    "decorators",
    "directives",
    "errors",
    "get_jacobi_preconditioner",
    "hyperparams",
    "minimize",
    "operators",
    "typing",
    "utils",
    "wrap_simulation",
]


def __getattr__(name):
    if name in ("GaussNewtonConjugateGradient", "conjugate_gradient"):
        import warnings  # ruff: ignore[PLC0415]

        warnings.warn(
            f"Importing '{name}' from top level of the module is deprecated. "
            "Import it from `inversion_ideas.minimize` instead. ",
            FutureWarning,
            stacklevel=2,
        )
        return getattr(minimize, name)
    raise AttributeError(name)
