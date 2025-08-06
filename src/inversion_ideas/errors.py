"""
Custom errors and warnings.
"""


class ConvergenceWarning(Warning):
    """
    Warning raised for issues with convergence.
    """


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
