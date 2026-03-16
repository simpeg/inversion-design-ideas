"""
Test operations for objective functions.
"""

import numpy as np
import pytest

from inversion_ideas.base import Combo, Objective, Scaled


class Dummy(Objective):
    """
    Dummy objective function.
    """

    def __init__(self, n_params):
        self._n_params = n_params

    @property
    def n_params(self):
        return self._n_params

    def __call__(self, model):  # noqa: ARG002
        return 2.0

    def gradient(self, model):  # noqa: ARG002
        return np.ones(self.n_params)

    def hessian(self, model):  # noqa: ARG002
        return np.eye(self.n_params)

    def hessian_diagonal(self, model):  # noqa: ARG002
        return np.ones(self.n_params)


class TestObjectiveOperations:
    """
    Test objective functions operations.

    Test cases:
        - Sum two objective functions, should obtain Combo.
        - Scalar times objective function, should get Scaled.
        - Sum two combos, should generate a Combo (without unpacking).
        - Sum combo and Scaled, should return another Combo (without unpacking).
        - Test iadd and imul:
            - Errors on Objective.
            - Error on imul for Combo.
            - Error on iadd for Scaled.
            - imul works ok for Scaled.
            - idiv works ok for Scaled.
            - iadd works ok for Combo.
    """

    n_params = 5

    def test_add(self):
        a, b = Dummy(self.n_params), Dummy(self.n_params)
        combo = a + b
        assert isinstance(combo, Combo)
        assert len(combo) == 2
        assert a in combo
        assert b in combo
        assert combo[0] is a
        assert combo[1] is b

    def test_add_n(self):
        """
        Test addition of multiple objective functions into nested Combos.

        Since Combos are not unpacked by default, adding together more than 2 objective
        functions create a nested structure of Combos.
        """
        a, b, c, d = tuple(Dummy(self.n_params) for _ in range(4))
        full_combo = a + b + c + d
        assert isinstance(full_combo, Combo)
        assert len(full_combo) == 2  # combo with (a + b + c) and d
        assert a not in full_combo
        assert b not in full_combo
        assert c not in full_combo
        assert full_combo[1] is d

        # First level
        combo = full_combo[0]  # combo with (a + b) and c
        assert isinstance(combo, Combo)
        assert len(combo) == 2
        assert a not in combo
        assert b not in combo
        assert combo[1] is c

        # Second level
        combo = full_combo[0][0]  # combo with a and b
        assert isinstance(combo, Combo)
        assert len(combo) == 2
        assert combo[0] is a
        assert combo[1] is b

    def test_mul(self):
        a = Dummy(self.n_params)
        scalar = 3.14
        scaled = scalar * a
        assert isinstance(scaled, Scaled)
        assert scaled.function is a
        assert scaled.multiplier == scalar

    def test_add_combos(self):
        a, b, c, d = tuple(Dummy(self.n_params) for _ in range(4))
        combo_a = a + b
        combo_b = c + d
        combo = combo_a + combo_b
        assert isinstance(combo, Combo)
        assert len(combo) == 2
        assert combo_a in combo
        assert combo_b in combo
        assert combo[0] is combo_a
        assert combo[1] is combo_b

    def test_add_scaled_and_combo(self):
        a, b, c = tuple(Dummy(self.n_params) for _ in range(3))
        combo = a + b
        scaled = 3.14 * c
        new_combo = combo + scaled
        assert isinstance(new_combo, Combo)
        assert len(new_combo) == 2
        assert combo in new_combo
        assert scaled in new_combo
        assert new_combo[0] is combo
        assert new_combo[1] is scaled

    def test_iadd_combo(self):
        a, b, c = tuple(Dummy(self.n_params) for _ in range(3))
        combo = a + b
        combo_bkp = combo
        combo += c
        assert isinstance(combo, Combo)
        assert combo is combo_bkp  # assert inplace operation
        assert len(combo) == 3
        assert combo[0] is a
        assert combo[1] is b
        assert combo[2] is c

    @pytest.mark.parametrize("function_type", ["objective", "scaled"])
    def test_iadd_error(self, function_type):
        phi, other = Dummy(self.n_params), Dummy(self.n_params)
        if function_type == "scaled":
            phi = 3.5 * phi
        with pytest.raises(TypeError):
            phi += other

    def test_imul_scaled(self):
        a = Dummy(self.n_params)
        scalar = 3.14
        scaled = scalar * a
        scaled_bkp = scaled
        new_scalar = 4.0
        scaled *= new_scalar
        assert isinstance(scaled, Scaled)
        assert scaled is scaled_bkp  # assert inplace operation
        assert scaled.function is a
        assert scaled.multiplier == scalar * new_scalar

    @pytest.mark.parametrize("function_type", ["objective", "combo"])
    def test_imul_error(self, function_type):
        phi = Dummy(self.n_params)
        if function_type == "combo":
            other = Dummy(self.n_params)
            phi = phi + other
        with pytest.raises(TypeError):
            phi *= 2.71

    def test_idiv_scaled(self):
        a = Dummy(self.n_params)
        scalar = 3.14
        scaled = scalar * a
        scaled_bkp = scaled
        new_scalar = 4.0
        scaled /= new_scalar
        assert isinstance(scaled, Scaled)
        assert scaled is scaled_bkp  # assert inplace operation
        assert scaled.function is a
        assert scaled.multiplier == scalar / new_scalar

    @pytest.mark.parametrize("function_type", ["objective", "combo"])
    def test_idiv_error(self, function_type):
        phi = Dummy(self.n_params)
        if function_type == "combo":
            other = Dummy(self.n_params)
            phi = phi + other
        with pytest.raises(TypeError):
            phi /= 2.71


def test_combo_flatten():
    """
    Test flatenning of a Combo.
    """
    a, b, c, d, e = tuple(Dummy(3) for _ in range(5))
    f = 2.5 * c
    g = d + e

    # build combo: (((a + b) + 2.5 * c) + (d + e))
    combo = a + b + f + g
    assert len(combo) == 2

    # Flatten it into: a + b + 2.5 * c + d + e
    flat_combo = combo.flatten()

    # Check the result of the operation
    assert len(flat_combo) == 5
    assert flat_combo[0] is a
    assert flat_combo[1] is b
    assert flat_combo[2] is f
    assert flat_combo[3] is d
    assert flat_combo[4] is e


class TestComboMethods:
    """
    Test ``__call__``, ``gradient`` and ``hessian`` for a ``Combo``.
    """


class TestScaledMethods:
    """
    Test ``__call__``, ``gradient`` and ``hessian`` for a ``Scaled``.
    """
