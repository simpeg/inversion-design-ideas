"""
Custom errors and warnings.
"""

__all__ = [
    "ConvergenceError",
    "ConvergenceWarning",
    "NotInitializedError",
]


class ConvergenceWarning(Warning):
    """
    Warning raised for issues with convergence.
    """


class ConvergenceError(Exception):
    """
    Error raised for failed convergence.

    Parameters
    ----------
    message : str
        Explanation of the error
    """

    def __init__(self, message=None):
        self.message = message if message is not None else ""
        super().__init__(self.message)


class NotInitializedError(Exception):
    """
    Exception raised when inversion is not yet initialized.

    Parameters
    ----------
    message : str
        Explanation of the error
    """

    def __init__(self, message=None):
        self.message = message if message is not None else ""
        super().__init__(self.message)
