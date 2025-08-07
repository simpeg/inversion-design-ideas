"""
Handler to run an inversion.

The :class:`Inversion` class is intended to simplify the process of running a full
inversion, given an objective function, an optimizer, a set of directives that can
modify the objective function after each iteration and optionally a logger.
"""

import numbers
import typing
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.live import Live
from rich.table import Table

from .base import Condition, Directive


class Inversion:
    """
    Inversion runner.

    Parameters
    ----------
    objective_function : Objective
        Objective function to minimize.
    initial_model : (n_params) array
        Starting model for the inversion.
    optimizer : Minimizer or callable
        Function or object to use as minimizer. It must take the objective function and
        a model as arguments.
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
    log : InversionLog, optional
        Instance of :class:`InversionLog` to store information about the inversion.
    """

    def __init__(
        self,
        objective_function,
        initial_model,
        optimizer,
        *,
        directives: typing.Sequence[Directive],
        stopping_criteria: Condition | Callable,
        max_iterations: int | None = None,
        cache_models=False,
        log: typing.Optional["InversionLog"] = None,
    ):
        self.objective_function = objective_function
        self.initial_model = initial_model
        self.optimizer = optimizer
        self.directives = directives
        self.stopping_criteria = stopping_criteria
        self.max_iterations = max_iterations
        self.cache_models = cache_models
        self.log = log

        # Assign model as a copy of the initial model
        self.model = initial_model.copy()

        # Initialize the counter
        if not hasattr(self, "_counter"):
            self._counter = 0

        # Add initial model to log (only on first iteration)
        if self.counter == 0 and self.log is not None:
            # TODO:
            # This might trigger evaluation of objective functions, making the
            # initialization of the inversion slow. We should put this somewhere else.
            self.log.update(self.counter, self.model)

    def __next__(self):
        """
        Run next iteration in the inversion.
        """
        # Check for stopping criteria before trying to run the iteration
        if self.stopping_criteria(self.model):
            raise StopIteration

        # Check if maximum number of iterations have been reached
        if self.max_iterations is not None and self.counter > self.max_iterations:
            raise StopIteration

        # Run directives (only after the first iteration)
        if self.counter > 0:
            for directive in self.directives:
                directive(self.model, self.counter)

        # Minimize objective function
        model = self.optimizer(self.objective_function, self.model)

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


class InversionLog:
    """
    Log the outputs of an inversion.

    Parameters
    ----------
    columns : dict
        Dictionary with specification for the columns of the log table.
        The keys are the column titles as strings. The values are callables that will be
        used to generate the value for each row and column. Each callable should take
        two arguments: ``iteration`` (an integer with the number of the iteration) and
        ``model`` (the inverted model as a 1d array).
    """

    def __init__(self, columns: dict[str, Callable]):
        self._columns = columns

    @property
    def columns(self) -> dict[str, Callable]:
        """
        Column specifiers.
        """
        return self._columns

    @property
    def log(self) -> dict[str, list]:
        """
        Inversion log.
        """
        if not hasattr(self, "_log"):
            self._log: dict[str, list] = {col: [] for col in self.columns}
        return self._log

    def update(self, iteration: int, model: npt.NDArray[np.float64]):
        """
        Update the log.
        """
        for title, column_func in self.columns.items():
            self.log[title].append(column_func(iteration, model))


class InversionLogRich(InversionLog):
    """
    Log the outputs of an inversion.

    Parameters
    ----------
    columns : dict
        Dictionary with specification for the columns of the log table.
        The keys are the column titles as strings. The values are callables that will be
        used to generate the value for each row and column. Each callable should take
        two arguments: ``iteration`` (an integer with the number of the iteration) and
        ``model`` (the inverted model as a 1d array).
    """

    def __init__(self, columns: dict[str, Callable], table_kwargs: dict | None = None):
        super().__init__(columns)
        self.table_kwargs = table_kwargs if table_kwargs is not None else {}

    @property
    def table(self) -> Table:
        """
        Table for the inversion log.
        """
        if not hasattr(self, "_table"):
            self._table = Table(**self.table_kwargs)
            for title in self.columns:
                self._table.add_column(title)
        return self._table

    def show(self):
        """
        Show table.
        """
        console = Console()
        console.print(self.table)

    def show_live(self, **kwargs):
        """
        Context manager for live update of the table.
        """
        return Live(self.table, **kwargs)

    def update_table(self):
        """
        Add row to the table given the latest inverted model.

        Parameters
        ----------
        model : (n_params) array
        """
        row = []
        for column in self.columns:
            value = self.log[column][-1]  # last element in the log
            if isinstance(value, numbers.Integral):
                fmt = "d"
            elif isinstance(value, numbers.Real):
                fmt = ".2e"
            else:
                fmt = ""
            row.append(f"{value:{fmt}}")
        self.table.add_row(*row)

    def update(self, iteration: int, model: npt.NDArray[np.float64]):
        """
        Update the log.

        Update the table as well.
        """
        super().update(iteration, model)
        self.update_table()
