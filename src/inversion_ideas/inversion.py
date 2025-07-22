"""
Handler to run an inversion.

The :class:`Inversion` class is intended to simplify the process of running a full
inversion, given an objective function, an optimizer, a set of directives that can
modify the objective function after each iteration and optionally a logger.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from rich.table import Table

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


@dataclass
class LogColumn:
    """
    Specifier for a column of the ``InversionLog``.
    """

    title: str
    instance: object
    attribute: str
    fmt: str | None


class InversionLog:
    """
    Log the outputs of an inversion.
    """

    def __init__(self, columns: list[LogColumn], table_kwargs: dict | None):
        self._columns = columns
        self.table_kwargs = table_kwargs if table_kwargs is not None else {}

    @property
    def columns(self) -> list[LogColumn]:
        """
        Column specifiers.
        """
        return self._columns

    @property
    def table(self) -> Table:
        """
        Table for the inversion log.
        """
        if not hasattr(self, "_table"):
            self._table = Table(**self.table_kwargs)
            for column in self.columns:
                self._table.add_column(column.title)
        return self._table

    @property
    def log(self) -> dict[str, list]:
        """
        Inversion log.
        """
        if not hasattr(self, "_log"):
            self._log = {col.title: [] for col in self.columns}
        return self._log

    def _update_log(self, model: npt.NDArray[np.float64]):
        """
        Update the log.
        """
        for column in self.columns:
            value = (
                getattr(column.instance, column.attribute)
                if column.attribute != "__call__"
                else column.instance(model)
            )
            self.log[column.title].append(value)

    def update_table(self, model: npt.NDArray[np.float64]):
        """
        Add row to the table given the latest inverted model.

        Parameters
        ----------
        model : (n_params) array
        """
        self._update_log(model)
        row = []
        for column in self.columns:
            value = self.log[column.title][-1]
            fmt = column.fmt
            if fmt is None:
                fmt = "d" if isinstance(value, int) else ".2e"
            row.append(f"{value:{fmt}}")
        self.table.add_row(*row)
