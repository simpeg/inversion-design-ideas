"""
Handler to run an inversion.

The :class:`Inversion` class is intended to simplify the process of running a full
inversion, given an objective function, a minimizer, a set of directives that can
modify the objective function after each iteration and optionally a logger.
"""

import typing
from collections.abc import Callable

from .base import Condition, Directive, Minimizer, Objective
from .inversion_log import InversionLog, InversionLogRich
from .typing import Model
from .utils import get_logger


class Inversion:
    """
    Inversion runner.

    Parameters
    ----------
    objective_function : Objective
        Objective function to minimize.
    initial_model : (n_params) array
        Starting model for the inversion.
    minimizer : Minimizer or callable
        Instance of :class:`Minimizer` or callable used to minimize the objective
        function during the inversion. It must take the objective function and a model
        as arguments.
    directives : list of Directive
        List of ``Directive``s used to modify the objective function after each
        iteration.
    stopping_criteria : Condition or callable
        Boolean function that takes the model as argument. If this function returns
        ``True``, then the inversion will stop.
    max_iterations : int, optional
        Max amount of iterations that will be performed. If ``None``, then there will be
        no limit on the total amount of iterations.
    cache_models : bool, optional
        Whether to cache each model after each iteration.
    log : InversionLog or bool, optional
        Instance of :class:`InversionLog` to store information about the inversion.
        If `True`, a default :class:`InversionLog` is going to be used.
        If `False`, no log will be assigned to the inversion, and :attr:`Inversion.log`
        will be ``None``.
    minimizer_kwargs : dict, optional
        Extra arguments that will be passed to the ``minimizer`` when called.
    """

    def __init__(
        self,
        objective_function: Objective,
        initial_model: Model,
        minimizer: Minimizer | Callable[[Objective, Model], Model],
        *,
        directives: typing.Sequence[Directive],
        stopping_criteria: Condition | Callable[[Model], bool],
        max_iterations: int | None = None,
        cache_models=False,
        log: "InversionLog | bool" = True,
        minimizer_kwargs: dict | None = None,
    ):
        self.objective_function = objective_function
        self.initial_model = initial_model
        self.minimizer = minimizer
        self.directives = directives
        self.stopping_criteria = stopping_criteria
        self.max_iterations = max_iterations
        self.cache_models = cache_models
        if minimizer_kwargs is None:
            minimizer_kwargs = {}
        self.minimizer_kwargs = minimizer_kwargs

        # Assign log
        if log is False:
            self.log = None
        elif log is True:
            # TODO: this could fail if the objective function is not
            # phi_d + beta * phi_m. We should try-error here maybe...
            self.log = InversionLogRich.create_from(self.objective_function)
        else:
            self.log = log

        # Assign model as a copy of the initial model
        self.model = initial_model.copy()

        # Initialize the counter
        self._counter = 0

    def __next__(self):
        """
        Run next iteration in the inversion.
        """
        if self.counter == 0:
            # Add initial model to log (only on zeroth iteration)
            if self.log is not None:
                self.log.update(self.counter, self.model)

            # Initialize stopping criteria (if necessary)
            if hasattr(self.stopping_criteria, "initialize"):
                self.stopping_criteria.initialize()

            # Increase counter by one
            self._counter += 1

            # Return the initial model in the zeroth iteration
            return self.model

        # Check for stopping criteria before trying to run the iteration
        if self.stopping_criteria(self.model):
            get_logger().info(
                "🎉 Inversion successfully finished due to stopping criteria."
            )
            raise StopIteration

        # Check if maximum number of iterations have been reached
        if self.max_iterations is not None and self.counter >= self.max_iterations:
            get_logger().info(
                "⚠️ Inversion finished after reaching maximum number of iterations "
                f"({self.max_iterations})."
            )
            raise StopIteration

        # Update stopping criteria (if necessary)
        if hasattr(self.stopping_criteria, "update"):
            self.stopping_criteria.update(self.model)

        # Run directives (only after the zeroth iteration).
        # We update the directives here (and not at the end of this method), so after
        # each iteration the objective function is still the same we passed to the
        # minimizer.
        for directive in self.directives:
            directive(self.model, self.counter)

        # Minimize objective function
        if isinstance(self.minimizer, Minimizer):
            # Keep only the last model of the minimizer iterator
            *_, model = self.minimizer(
                self.objective_function, self.model, **self.minimizer_kwargs
            )
        else:
            model = self.minimizer(
                self.objective_function, self.model, **self.minimizer_kwargs
            )

        # Cache model if required
        if self.cache_models:
            self.models.append(model)

        # Increase counter by one
        self._counter += 1

        # Assign the model to self
        self.model = model

        # Update log
        if self.log is not None:
            self.log.update(self.counter, self.model)

        return self.model

    def __iter__(self):
        return self

    @property
    def counter(self) -> int:
        """
        Iteration counter.
        """
        return self._counter

    @property
    def models(self) -> list:
        """
        Cached inverted models.

        The first model in the list is the initial model, the one that corresponds to
        the zeroth iteration.
        """
        if not self.cache_models:
            msg = "Inversion doesn't have cached models since `cache_model` is `False`."
            raise AttributeError(msg)
        if not hasattr(self, "_models"):
            self._models = [self.initial_model]
        return self._models

    def run(self, show_log=True) -> Model:
        """
        Run the inversion.

        Parameters
        ----------
        show_log : bool, optional
            Whether to show the ``log`` (if it's defined) during the inversion.
        """
        if show_log and self.log is not None:
            if not hasattr(self.log, "live"):
                raise NotImplementedError()
            with self.log.live() as live:
                for _ in self:
                    live.refresh()
        else:
            for _ in self:
                pass
        return self.model
