"""
Multiplier class.
"""

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
        Factor use to divide (or *cool down*) the multiplier when the
        :meth:`~inversion_ideas.CooledMultiplier.update` method is called.
    mutable : bool, optional
        If False, the wrapped value is immutable, i.e. we cannot change it through
        public properties and methods besides the
        :meth:`~inversion_ideas.CooledMultiplier.update` method.
        If True, the wrapped value is mutable and can be modified.

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

    def __init__(self, initial_value: float, *, cooling_factor: float, mutable=False):
        self._initial_value = initial_value
        self.cooling_factor = cooling_factor
        super().__init__(initial_value, mutable=mutable)

    @property
    def initial_value(self):
        return self._initial_value

    def update(self, *args):  # ruff: ignore[ARG002]
        """
        Cool down the multiplier.

        Notes
        -----
        Cool down the multiplier by dividing it by the cooling factor.
        """
        self._value /= self.cooling_factor
