"""
Multiplier class.
"""

from numbers import Integral, Real

from inversion_ideas.utils import get_logger

from ..base import Multiplier


class CooledMultiplier(Multiplier):
    """
    Multiplier hyperparameter that can be cooled down.

    Wraps a float into an object that can be *cooled down* through the
    :meth:`~inversion_ideas.CooledMultiplier.update` method.

    Parameters
    ----------
    initial_value : float
        Initial value for the multiplier.
    cooling_factor : float
        Factor used to divide (or *cool down*) the multiplier when the
        :meth:`~inversion_ideas.CooledMultiplier.update` method is called.
    cooling_rate : int, optional
        Control how often the multiplier will be cooled down.
        The multiplier will be cooled down every ``cooling_rate`` call of the ``update``
        method.
    mutable : bool, optional
        If False, the wrapped value is immutable, i.e. we cannot change it through
        public properties and methods besides the
        :meth:`~inversion_ideas.CooledMultiplier.update` method.
        If True, the wrapped value is mutable and can be freely modified.

    Examples
    --------
    >>> beta = CooledMultiplier(10.0, cooling_factor=2.0)
    >>> print(beta)
    CooledMultiplier(10.0)

    >>> beta.value
    10.0

    >>> beta.update()
    >>> print(beta)
    CooledMultiplier(5.0)

    >>> beta.value
    5.0
    """

    def __init__(
        self,
        initial_value: float,
        *,
        cooling_factor: float,
        cooling_rate: int = 1,
        mutable=False,
    ):
        self._initial_value = initial_value
        self.cooling_factor = cooling_factor
        self.cooling_rate = cooling_rate
        super().__init__(initial_value, mutable=mutable)

    @property
    def initial_value(self):
        return self._initial_value

    @property
    def cooling_factor(self) -> float:
        """Factor used to cool down the multiplier."""
        return self._cooling_factor

    @cooling_factor.setter
    def cooling_factor(self, value: float):
        """Factor used to cool down the multiplier."""
        if not isinstance(value, Real):
            msg = (
                f"Invalid cooling_factor '{value}' of type '{type(value)}'. "
                "Only floats are accepted."
            )
            raise TypeError(msg)
        self._cooling_factor = value

    @property
    def cooling_rate(self) -> int:
        """How often the multiplier will be cooled down."""
        return self._cooling_rate

    @cooling_rate.setter
    def cooling_rate(self, value: int):
        if not isinstance(value, Integral):
            msg = (
                f"Invalid cooling_rate '{value}' of type '{type(value)}'. "
                "Only positive integers are accepted."
            )
            raise TypeError(msg)
        if value <= 0:
            msg = f"Invalid cooling_rate '{value}'. It must be a positive integer."
            raise ValueError(msg)
        self._cooling_rate = value

    @property
    def counter(self) -> int:
        """Counts calls of the ``update`` method."""
        if not hasattr(self, "_counter"):
            self._counter = 0
        return self._counter

    def update(self, *args):  # ruff: ignore[ARG002]
        """
        Cool down the multiplier.

        The multiplier gets cooled down every :attr:`cooling_factor` calls to this
        method.

        Parameters
        ----------
        *args :
            Any argument will be ignored. They are kept to guarantee compatibility
            with the ``update`` method interface.

        Notes
        -----
        Cool down the multiplier by dividing it by the cooling factor.
        The cooling process happens once every ``cooling_factor`` calls of this method.
        """
        logger = get_logger()

        if (self.counter % self.cooling_rate) == 0:
            # Cool down the multiplier
            self._value /= self.cooling_factor
            # ---  debug ---
            msg = (
                f"Cooled down '{self!r}' on {self.counter}-th call of the "
                "update method. "
                f"Current value of the multiplier: {self}."
            )
            logger.debug(msg)
            # --- end ---
        else:
            # ---  debug ---
            msg = (
                f"Skipped cooling of '{self!r}' on {self.counter}-th call of the "
                "update method. "
                f"Current value of the multiplier: {self}."
            )
            logger.debug(msg)
            # --- end ---

        # Increase counter by one
        self._counter += 1

    def reset(self):
        """
        Reset the multiplier to the initial values.

        The value of the multiplier will be reset to its intial one, and the
        :attr:`counter` will be set to zero.
        """
        self._value = self.initial_value
        self._counter = 0
