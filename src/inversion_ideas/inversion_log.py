"""
Classes for inversion logs.
"""

import numbers
import typing
import warnings
from collections.abc import Callable, Iterable

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .base import MinimizerResult

try:
    import pandas  # noqa: ICN001
except ImportError:
    pandas = None

from .base import Combo
from .typing import Model


class Column(typing.NamedTuple):
    """
    Column for the ``InversionLog``.
    """

    title: str
    callable: Callable[[int, Model], typing.Any]
    fmt: str | None


class InversionLog:
    """
    Log the outputs of an inversion.

    Parameters
    ----------
    columns : dict
        Dictionary with specification for the columns of the log table.
        The keys are the column titles as strings. The values can be callables that will
        be used to generate the value for each row and column, or ``Column``. Each
        callable should take two arguments: ``iteration`` (an integer with the number
        of the iteration) and ``model`` (the inverted model as a 1d array).
    """

    def __init__(
        self, columns: typing.Mapping[str, Column | Callable[[int, Model], typing.Any]]
    ):
        for name, column in columns.items():
            self.add_column(name, column)

        # Initialize a list of minimizer logs. The first element of it should be None
        # since minimizers are not ran in the first iteration.
        self._minimizer_logs: list[None | MinimizerLog] = [None]

    def update(self, iteration: int, model: Model):
        """
        Update the log.
        """
        for name, column in self.columns.items():
            self.log[name].append(column.callable(iteration, model))

    def get_minimizer_callback(self) -> Callable[[MinimizerResult], None]:
        """
        Return a callable that can be passed to a minimizer.

        This method creates a new :class:`MinimizerLog` that gets stored in a running
        list inside this inversion log, and returns the :func:`MinimizerLog.update`
        method that can be passed as a callback to any minimizer.

        Returns
        -------
        callable
            A callable that can be passed to a minimizer ``callback`` argument.
        """
        minimizer_log = MinimizerLog()
        self._minimizer_logs.append(minimizer_log)
        return minimizer_log.update

    def __len__(self):
        """Return number of entries in the log."""
        if not self.has_records:
            return 0
        n_entries = [len(v) for v in self.log.values()]
        if not all(n_entries[0] == n for n in n_entries):
            msg = "Invalid log with variable entries."
            raise ValueError(msg)
        return n_entries[0]

    @property
    def minimizer_logs(self) -> list["None | MinimizerLog"]:
        """
        List of logs for the minimizer.
        """
        return self._minimizer_logs

    @property
    def has_records(self) -> bool:
        """
        Whether the log has recorded values or not.
        """
        if not hasattr(self, "_log"):
            return False
        has_records = any(bool(c) for c in self.log.values())
        return has_records

    def add_column(
        self, name: str, column: Column | Callable[[int, Model], typing.Any]
    ) -> typing.Self:
        """
        Add column to the log.

        Parameters
        ----------
        name : str
            Name of the column, used in the :attr:`InversionLog.log` dictionary to
            access the recorded values.
        column : Callable | Column
            A callable that takes the ``iteration`` and the ``model`` as arguments, or
            a ``Column``.

        Returns
        -------
        self
        """
        if self.has_records:
            msg = (
                f"{type(self).__name__} has records. "
                "No column can be added after the log has already started "
                "recording values."
            )
            raise TypeError(msg)

        if not hasattr(self, "_columns"):
            self._columns: dict[str, Column] = {}

        if callable(column):
            column = Column(title=name, callable=column, fmt=None)

        self._columns[name] = column
        return self

    @property
    def columns(self) -> dict[str, Column]:
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

    def to_pandas(self, index_col=0):
        """
        Generate a ``pandas.DataFrame`` out of the log.
        """
        if pandas is None:
            msg = "Pandas is missing."
            raise ImportError(msg)
        index = list(self.log.keys())[index_col]
        return pandas.DataFrame(self.log).set_index(index)

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
        assert not hasattr(data_misfit, "multiplier")
        assert not isinstance(data_misfit, Iterable)
        regularization = objective_function[1]
        assert hasattr(regularization, "multiplier")

        columns = {
            "iter": Column(
                title="Iteration", callable=lambda iteration, _: iteration, fmt="d"
            ),
            "beta": Column(
                title="β", callable=lambda _, __: regularization.multiplier, fmt=".2e"
            ),
            "phi_d": Column(
                title="φ_d", callable=lambda _, model: data_misfit(model), fmt=".2e"
            ),
            "phi_m": Column(
                title="φ_m",
                callable=lambda _, model: regularization.function(model),
                fmt=".2e",
            ),
            "beta * phi_m": Column(
                title="β φ_m",
                callable=lambda _, model: regularization(model),
                fmt=".2e",
            ),
            "phi": Column(
                title="φ",
                callable=lambda _, model: objective_function(model),
                fmt=".2e",
            ),
            "chi": Column(
                title="χ",
                callable=lambda _, model: data_misfit(model) / data_misfit.n_data,
                fmt=".2e",
            ),
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

    def __init__(self, columns: dict[str, Callable | Column], **kwargs):
        super().__init__(columns)
        self.kwargs = kwargs

    def __rich__(self) -> Table:
        """
        Return the log as a Rich renderable.
        """
        return self._build_rich_table()

    def _build_rich_table(self):
        """Represent the log as a Rich table."""
        table = Table(*self.log.keys())
        for i in range(len(self)):
            row = []
            for name, column in self.columns.items():
                value = self.log[name][i]
                fmt = column.fmt if column.fmt is not None else self._get_fmt(value)
                row.append(f"{value:{fmt}}")
            table.add_row(*row)
        return table

    @property
    def table(self) -> Table:
        """
        Table for the inversion log.
        """
        warnings.warn("table will be removed", FutureWarning, stacklevel=2)
        if not hasattr(self, "_table"):
            self._table = Table(**self.kwargs)
            for column in self.columns.values():
                self._table.add_column(column.title)
        return self._table

    def show(self):
        """
        Show log through a Rich console.
        """
        console = Console()
        console.print(self)

    def live(self, **kwargs):
        """
        Context manager for live update of the table.
        """
        warnings.warn("live will be removed", FutureWarning, stacklevel=2)
        return Live(self.table, **kwargs)

    def update_table(self):
        """
        Add row to the table given the latest inverted model.

        Parameters
        ----------
        model : (n_params) array
        """
        warnings.warn("update_table will be removed", FutureWarning, stacklevel=2)
        row = []
        for name, column in self.columns.items():
            value = self.log[name][-1]  # last element in the log
            fmt = column.fmt if column.fmt is not None else self._get_fmt(value)
            row.append(f"{value:{fmt}}")
        self.table.add_row(*row)

    def _get_fmt(self, value):
        if isinstance(value, bool):
            fmt = ""
        elif isinstance(value, numbers.Integral):
            fmt = "d"
        elif isinstance(value, numbers.Real):
            fmt = ".2e"
        else:
            fmt = ""
        return fmt


class MinimizerLog:
    """Class to store results of a minimizer in the form of a log."""

    # Columns defined by the fields in MinimizerResult
    columns: typing.ClassVar = {
        "iteration": "d",
        "model": "",
        "conj_grad_iters": "d",
        "line_search_iters": "d",
        "step_norm": ".2e",
    }

    def update(self, minimizer_result: MinimizerResult):
        """
        Use as callback for :class:`inversion_ideas.base.Minimizer`.
        """
        for column in self.columns:
            self.log[column].append(getattr(minimizer_result, column))

    def __len__(self):
        """Return number of entries in the log."""
        if not self.has_records:
            return 0
        n_entries = [len(v) for v in self.log.values()]
        if not all(n_entries[0] == n for n in n_entries):
            msg = "Invalid log with variable entries."
            raise ValueError(msg)
        return n_entries[0]

    @property
    def has_records(self) -> bool:
        """
        Whether the log has recorded values or not.
        """
        if not hasattr(self, "_log"):
            return False
        has_records = any(bool(c) for c in self.log.values())
        return has_records

    @property
    def log(self) -> dict[str, list]:
        """Returns the log."""
        if not hasattr(self, "_log"):
            self._log: dict[str, list] = {col: [] for col in self.columns}
        return self._log

    def __rich__(self) -> Table:
        """
        Return the log as a Rich renderable.
        """
        return self._build_rich_table()

    def _build_rich_table(self):
        """Represent the log as a Rich table."""
        table = Table(*self.log.keys())
        for i in range(len(self)):
            row = []
            for column, fmt in self.columns.items():
                value = self.log[column][i]
                row.append(f"{value:{fmt}}")
            table.add_row(*row)
        return table

    def show(self):
        """
        Show log through a Rich console.
        """
        console = Console()
        console.print(self)
