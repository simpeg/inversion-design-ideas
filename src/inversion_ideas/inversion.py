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

from .base import Combo, Condition, Directive
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
    log : InversionLog or bool, optional
        Instance of :class:`InversionLog` to store information about the inversion.
        If `True`, a default :class:`InversionLog` is going to be used.
        If `False`, no log will be assigned to the inversion, and :attr:`Inversion.log`
        will be ``None``.
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
        log: "InversionLog | bool" = True,
    ):
        self.objective_function = objective_function
        self.initial_model = initial_model
        self.optimizer = optimizer
        self.directives = directives
        self.stopping_criteria = stopping_criteria
        self.max_iterations = max_iterations
        self.cache_models = cache_models

        # Assign log
        if log is False:
            self.log = None
        elif log is True:
            self.log = InversionLogRich.create_from(self.objective_function)
        else:
            self.log = log

        # Assign model as a copy of the initial model
        self.model = initial_model.copy()

        # Initialize the counter
        if not hasattr(self, "_counter"):
            self._counter = 0

    def __next__(self):
        """
        Run next iteration in the inversion.
        """
        if self.counter == 0 and self.log is not None:
            # Add initial model to log (only on zeroth iteration)
            self.log.update(self.counter, self.model)
            # Initialize stopping criteria (if necessary)
            if hasattr(self.stopping_criteria, "initialize"):
                self.stopping_criteria.initialize()

        # Check for stopping criteria before trying to run the iteration
        if self.stopping_criteria(self.model):
            get_logger().info(
                "🎉 Inversion successfully finished due to stopping critiera."
            )
            raise StopIteration

        # Check if maximum number of iterations have been reached
        if self.max_iterations is not None and self.counter > self.max_iterations:
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
        # optimizer.
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

    def run(self, show_log=True) -> npt.NDArray[np.float64]:
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

    @classmethod
    def create_from(cls, objective_function: Combo) -> typing.Self:
        r"""
        Create the standard log for a classic inversion.

        Parameters
        ----------
        objective_function : Combo
            Combo objective function with two elements: the data misfit and the
            regularization (including a trade-off parameter).

        Returns
        -------
        Self

        Notes
        -----
        The objective function should be of the type:

        .. math::

            \phi(\mathbf{m}) = \phi_d(\mathbf{m}) + \beta \phi_m(\mathbf{m})

        where :math:`\phi_d(m)` is the data misfit term, :math:`\phi_m(\mathbf{m})` is
        the model norm, and :math:`\beta` is the trade-off parameter.
        """
        # TODO: write proper error messages
        assert len(objective_function) == 2
        data_misfit = objective_function[0]
        regularization = objective_function[1]
        assert hasattr(regularization, "multiplier")

        columns = {
            "iter": lambda iteration, _: iteration,
            "beta": lambda _, __: regularization.multiplier,
            "phi_d": lambda _, model: data_misfit(model),
            "phi_m": lambda _, model: regularization.function(model),
            "beta * phi_m": lambda _, model: regularization(model),
            "phi": lambda _, model: objective_function(model),
            "chi": lambda _, model: data_misfit(model) / data_misfit.n_data,
        }
        return cls(columns)


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
    kwargs :
        Pass extra options to :class:`rich.table.Table`.
    """

    def __init__(self, columns: dict[str, Callable], **kwargs):
        super().__init__(columns)
        self.kwargs = kwargs

    @property
    def table(self) -> Table:
        """
        Table for the inversion log.
        """
        if not hasattr(self, "_table"):
            self._table = Table(**self.kwargs)
            for title in self.columns:
                self._table.add_column(title)
        return self._table

    def show(self):
        """
        Show table.
        """
        console = Console()
        console.print(self.table)

    def live(self, **kwargs):
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
