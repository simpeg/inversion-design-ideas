"""
Handler to run an inversion.

The :class:`Inversion` class is intended to simplify the process of running a full
inversion, given an objective function, an optimizer, a set of directives that can
modify the objective function after each iteration and optionally a logger.
"""

from .directives import Directive


class Inversion:
    """
    Inversion runner.
    """

    def __init__(
        self, objective_function, initial_model, optimizer, directives: list[Directive]
    ):
        self.objective_function = objective_function
        self.initial_model = initial_model
        self.optimizer = optimizer
        self.directives = directives
        self.model = initial_model.copy()

    def __next__(self):  # noqa: D105
        # Initialize directives in the first iteration
        if self._counter == 0:
            for directive in self.directives:
                directive.initialize()
            self._initialized = True
        else:
            # Run directives (after the last iteration)
            for directive in self.directives:
                directive()

        # Minimize objective function
        self.model = self.optimizer(self.objective_function, self.model)

        # Increase counter by one
        self._counter += 1

        # Return the model
        return self.model

    def __iter__(self):  # noqa: D105
        # Initialize the counter if it doesn't exist.
        # If iter is called again after the inversion has been initialized already,
        # resume from the last point.
        if not hasattr(self, "_counter"):
            self._counter = 0
        return self

    @property
    def counter(self):
        """
        Iteration counter.
        """
        return self._counter
