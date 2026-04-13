"""
Test operations for objective functions.
"""

import itertools
import re

import numpy as np
import pytest
from scipy.sparse import dia_array
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from inversion_ideas.base import Combo, Objective, Scaled

from ..utils import assert_equal_linear_operators


class Dummy(Objective):
    r"""
    Dummy objective function.

    Define a dummy objective function as:

    .. math::

        \phi(\mathbf{m}) = \mathbf{m}^T \mathbf{A}^T \mathbf{A} \mathbf{m},

    where :math:`\mathbf{A}` is a random ``(n, \n_params)`` matrix.

    It's gradient will therefore be:

    .. math::

        \nabla\phi(\mathbf{m}) = \mathbf{A}^T \mathbf{A} \mathbf{m},

    and its Hessian:

    .. math::

        \bar{\bar{\nabla}}\phi(\mathbf{m}) = \mathbf{A}^T \mathbf{A}.

    Parameters
    ----------
    n_params : int
        Number of parameters for the objective function.
    seed : int or numpy.random.Generator or numpy.random.RandomState or None, optional
        Random seed used to define the :math:`\mathbf{A}` matrix.
    hessian_type : {"dense", "sparse", "linop"}, optional
        Type of Hessian matrix: "dense" matrix, "sparse" matrix or "linop" as in
        a ``LinearOperator``.
    """

    def __init__(self, n_params, seed=None, hessian_type="dense"):
        self._n_params = n_params
        rng = np.random.default_rng(seed=seed)
        self.a_matrix = rng.uniform(size=(n_params, n_params))
        if hessian_type not in ("dense", "sparse", "linop"):
            msg = f"Invalid hessian_type '{hessian_type}'."
            raise ValueError(msg)
        self.hessian_type = hessian_type

    @property
    def n_params(self):
        return self._n_params

    def __call__(self, model):
        return float(model.T @ self.a_matrix.T @ self.a_matrix @ model)

    def gradient(self, model):
        return self.a_matrix.T @ self.a_matrix @ model

    def hessian(self, model):  # noqa: ARG002
        match self.hessian_type:
            case "dense":
                hessian = self.a_matrix.T @ self.a_matrix
            case "sparse":
                a_sparse = dia_array(self.a_matrix)
                hessian = a_sparse.T @ a_sparse
            case "linop":
                a_linop = aslinearoperator(self.a_matrix)
                hessian = a_linop.T @ a_linop
            case _:
                msg = f"Invalid hessian_type '{self.hessian_type}'."
                raise ValueError(msg)
        return hessian


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

    def test_add_zero(self):
        """
        Test adding objective functions to the zero integer.

        This feature is useful for adding a collection of objective functions through
        the ``sum()`` built-in function.
        """
        phi = Dummy(self.n_params)
        # Test __add__
        result = phi + 0
        assert result is phi
        # Test __radd__
        result = 0 + phi
        assert result is phi

    def test_add_error_no_zero(self):
        """
        Test error when adding integer different than zero to objective function.
        """
        phi = Dummy(self.n_params)
        # Test add
        msg = re.escape(f"Cannot add objective function '{phi}' with '1'.")
        with pytest.raises(ValueError, match=msg):
            phi + 1
        # Test radd
        with pytest.raises(ValueError, match=msg):
            1 + phi

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

    def test_truediv(self):
        a = Dummy(self.n_params)
        scalar = 3.14
        scaled = a / scalar
        assert isinstance(scaled, Scaled)
        assert scaled.function is a
        assert scaled.multiplier == 1 / scalar

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

    def test_radd(self):
        """
        Test the __radd__ method of objective functions.

        We'll need to add an object of some dummy class that raises NotImplemented when
        calling __add__ to trigger __radd__.
        """

        class DummyNonObjectiveFunction:
            n_params = self.n_params

        a = DummyNonObjectiveFunction()
        b = Dummy(self.n_params)
        combo = a + b
        assert isinstance(combo, Combo)
        assert len(combo) == 2
        assert combo[0] is a
        assert combo[1] is b

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

    def test_floordiv_error(self):
        phi = Dummy(self.n_params)
        with pytest.raises(TypeError, match="Floor division is not implemented"):
            phi // 2.71

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

    def test_add_incompatible_error(self):
        """
        Test error when adding two objective functions with different ``n_params``.
        """
        phi_a, phi_b = Dummy(3), Dummy(4)
        msg = re.escape(
            f"Invalid objective functions {phi_a}, {phi_b} with different n_params: "
            "3, 4, respectively."
        )
        with pytest.raises(ValueError, match=msg):
            phi_a + phi_b

    def test_iadd_incompatible_error(self):
        """
        Test error when adding in-place objective functions with different ``n_params``.
        """
        combo = Dummy(3) + Dummy(3)
        phi_c = Dummy(4)
        msg = re.escape(
            f"Trying to add objective function '{phi_c}' with invalid "
            "n_params (4) different from the one of "
            f"'{combo}' (3)."
        )
        with pytest.raises(ValueError, match=msg):
            combo += phi_c

    def test_sum(self):
        """
        Test adding a collection of objective functions through ``sum()``.
        """
        # Add multiple objective functions
        # --------------------------------
        phi_a, phi_b, phi_c, phi_d = [
            Dummy(self.n_params).set_name(n) for n in ("a", "b", "c", "d")
        ]
        collection = [phi_a, phi_b, phi_c, phi_d]

        combo = sum(collection)
        expected = phi_a + phi_b + phi_c + phi_d
        assert combo == expected

        combo = sum(collection).flatten()
        expected = (phi_a + phi_b + phi_c + phi_d).flatten()
        assert combo == expected

        # Add objective functions including scaled in the collection
        # ----------------------------------------------------------
        scaled_a, scaled_d = 3.0 * phi_a, 5.2 * phi_d
        collection = [scaled_a, phi_b, phi_c, scaled_d]

        combo = sum(collection)
        expected = scaled_a + phi_b + phi_c + scaled_d
        assert combo == expected

        combo = sum(collection).flatten()
        expected = (scaled_a + phi_b + phi_c + scaled_d).flatten()
        assert combo == expected

        # Add objective functions including combos in the collection
        # ----------------------------------------------------------
        combo_a = phi_a + phi_b
        collection = [combo_a, phi_c, phi_d]

        combo = sum(collection)
        expected = combo_a + phi_c + phi_d
        assert combo == expected

        combo = sum(collection).flatten()
        expected = (combo_a + phi_c + phi_d).flatten()
        assert combo == expected

        # Check that order in collection matters
        # --------------------------------------
        collection = [phi_a, phi_b, phi_c, phi_d]
        combo = phi_d + phi_c + phi_a + phi_b
        assert combo != sum(collection)

        collection = [phi_a, phi_b, phi_c, phi_d]
        combo = (phi_d + phi_c + phi_a + phi_b).flatten()
        assert combo != sum(collection).flatten()


class TestObjectiveHessian:
    """
    Test the default implementation of ``hessian_approx`` and ``hessian_diagonal``.
    """

    n_params = 5

    @pytest.fixture
    def model(self):
        rng = np.random.default_rng(seed=42)
        model = rng.uniform(size=self.n_params)
        return model

    @pytest.mark.parametrize("hessian_type", ["dense", "sparse"])
    def test_hessian_approx(self, model, hessian_type):
        """
        Test if the default implementation of ``hessian_approx`` returns the Hessian.
        """
        phi = Dummy(3, seed=42, hessian_type=hessian_type)
        assert_equal_linear_operators(phi.hessian(model), phi.hessian_approx(model))

    def test_hessian_approx_linop_error(self, model):
        """
        Test error on ``hessian_approx`` if hessian is a LinearOperator.
        """
        phi = Dummy(3, seed=42, hessian_type="linop")
        msg = re.escape("Cannot build a 'hessian_approx' for objective function")
        with pytest.raises(TypeError, match=msg):
            phi.hessian_approx(model)

    @pytest.mark.parametrize("hessian_type", ["dense", "sparse"])
    def test_hessian_diagonal(self, model, hessian_type):
        """
        Test default implementation of ``hessian_diagonal``.
        """
        phi = Dummy(3, seed=42, hessian_type=hessian_type)
        np.testing.assert_equal(
            phi.hessian(model).diagonal(), phi.hessian_diagonal(model)
        )

    def test_hessian_diagonal_linop_error(self, model):
        """
        Test error on ``hessian_diagonal`` if hessian is a LinearOperator.
        """
        phi = Dummy(3, seed=42, hessian_type="linop")
        msg = re.escape("Cannot get 'hessian_diagonal' for objective function")
        with pytest.raises(TypeError, match=msg):
            phi.hessian_diagonal(model)


class TestComboExtraMethods:
    """
    Test additional methods of the Combo class.
    """

    def test_combo_flatten(self):
        """
        Test flattening of a Combo.
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

    def test_combo_error(self):
        """
        Test error when `functions` argument is not a sequence.
        """
        msg = re.escape(
            "Invalid 'functions' argument of type 'float'. "
            "It must be a sequence of `Objective` functions."
        )
        with pytest.raises(TypeError, match=msg):
            Combo(functions=3.14)

    def test_empty_combo_error(self):
        """
        Test if an empty combo raises an error.
        """
        # Check error when defining an empty Combo
        msg = re.escape(
            "Invalid empty 'functions' argument. "
            "The list of objective functions must contain at least one function."
        )
        with pytest.raises(ValueError, match=msg):
            Combo(functions=[])

        # Check error when accessing the Combo.functions with no functions
        combo = Combo([Dummy(n_params=3)])
        combo._functions.pop()  # empty the combo
        msg = re.escape("Invalid empty `Combo` without functions.")
        with pytest.raises(ValueError, match=msg):
            combo.functions  # noqa: B018


class TestComboContains:
    """
    Test the contains method of the Combo functions.
    """

    n_params = 5

    def test_shallow(self):
        """
        Test a shallow container.
        """
        # Two elements
        phi_a, phi_b = Dummy(self.n_params), Dummy(self.n_params)
        combo = phi_a + phi_b
        assert combo.contains(phi_a)
        assert combo.contains(phi_b)
        phi_c = Dummy(self.n_params)
        assert not combo.contains(phi_c)

        # Three elements
        combo = (phi_a + phi_b + phi_c).flatten()
        assert combo.contains(phi_a)
        assert combo.contains(phi_b)
        assert combo.contains(phi_c)
        phi_d = Dummy(self.n_params)
        assert not combo.contains(phi_d)

    def test_nested(self):
        """
        Test a nested Combo.
        """
        phi_a, phi_b, phi_c, phi_d = (Dummy(self.n_params) for _ in range(4))
        combo = (phi_a + phi_b) + (phi_c + phi_d)
        assert combo.contains(phi_a)
        assert combo.contains(phi_b)
        assert combo.contains(phi_c)
        assert combo.contains(phi_d)

    def test_scaled_combo(self):
        """
        Test contains against a Combo within a Scaled.
        """
        phi_a, phi_b, phi_c = (Dummy(self.n_params) for _ in range(3))
        inner_combo = phi_b + phi_c
        scaled_combo = 5.24 * inner_combo
        combo = phi_a + scaled_combo
        assert combo.contains(inner_combo)
        assert combo.contains(phi_a)
        assert combo.contains(phi_b)
        assert combo.contains(phi_c)

    def test_contains_scaled(self):
        """
        Test contains against a function in a Scaled class.
        """
        phi_a, phi_b = Dummy(self.n_params), Dummy(self.n_params)
        scaled = 3.0 * phi_a
        combo = scaled + phi_b
        assert combo.contains(scaled)
        assert combo.contains(phi_a)
        other_scaled = 3.0 * phi_a
        assert not combo.contains(other_scaled)

    def test_contains_combo(self):
        """
        Test contains against another Combo.
        """
        phi_a, phi_b, phi_c = (Dummy(self.n_params) for _ in range(3))
        inner_combo = phi_a + phi_b
        combo = inner_combo + phi_c
        assert combo.contains(inner_combo)
        assert combo.contains(phi_a)
        assert combo.contains(phi_b)
        assert combo.contains(phi_c)
        other_combo = phi_a + phi_b
        assert not combo.contains(other_combo)


class TestComboMethods:
    """
    Test ``__call__``, ``gradient`` and ``hessian`` for a ``Combo``.
    """

    n_params = 5

    @pytest.fixture
    def model(self):
        rng = np.random.default_rng(seed=42)
        model = rng.uniform(size=self.n_params)
        return model

    def test_call(self, model):
        """
        Test the call method of Combo objective functions.
        """
        phi_a, phi_b = Dummy(self.n_params, seed=42), Dummy(self.n_params, seed=43)
        combo = phi_a + phi_b
        np.testing.assert_allclose(combo(model), phi_a(model) + phi_b(model))

    def test_gradient(self, model):
        """
        Test the gradient method of Combo objective functions.
        """
        phi_a, phi_b = Dummy(self.n_params, seed=42), Dummy(self.n_params, seed=43)
        combo = phi_a + phi_b
        np.testing.assert_allclose(
            combo.gradient(model), phi_a.gradient(model) + phi_b.gradient(model)
        )

    @pytest.mark.parametrize(
        "hessian_types",
        [
            pytest.param((type_a, type_b), id=f"{type_a}-{type_b}")
            for type_a, type_b in itertools.combinations_with_replacement(
                ("dense", "sparse", "linop"), 2
            )
        ],
    )
    def test_hessian(self, model, hessian_types):
        """
        Test the hessian method of Combo objective functions.
        """
        type_a, type_b = hessian_types
        phi_a = Dummy(self.n_params, seed=42, hessian_type=type_a)
        phi_b = Dummy(self.n_params, seed=43, hessian_type=type_b)
        combo = phi_a + phi_b

        # Expected hessian
        hessian_a = phi_a.hessian(model)
        hessian_b = phi_b.hessian(model)
        if isinstance(hessian_a, LinearOperator) or isinstance(
            hessian_b, LinearOperator
        ):
            hessian_a, hessian_b = (
                aslinearoperator(hessian_a),
                aslinearoperator(hessian_b),
            )
        assert_equal_linear_operators(combo.hessian(model), hessian_a + hessian_b)

    @pytest.mark.parametrize(
        "hessian_types",
        [
            pytest.param((type_a, type_b), id=f"{type_a}-{type_b}")
            for type_a, type_b in itertools.combinations_with_replacement(
                ("dense", "sparse"), 2
            )
        ],
    )
    def test_hessian_approx(self, model, hessian_types):
        """
        Test the hessian_approx method of Combo objective functions.
        """
        type_a, type_b = hessian_types
        rng = np.random.default_rng(seed=42)
        phi_a = Dummy(self.n_params, seed=rng, hessian_type=type_a)
        phi_b = Dummy(self.n_params, seed=rng, hessian_type=type_b)
        combo = phi_a + phi_b
        assert_equal_linear_operators(
            combo.hessian_approx(model),
            phi_a.hessian_approx(model) + phi_b.hessian_approx(model),
            to_dense=True,
        )

    @pytest.mark.parametrize(
        "hessian_types",
        [
            pytest.param((type_a, type_b), id=f"{type_a}-{type_b}")
            for type_a, type_b in itertools.combinations_with_replacement(
                ("dense", "sparse"), 2
            )
        ],
    )
    def test_hessian_diagonal(self, model, hessian_types):
        """
        Test the hessian_diagonal method of Combo objective functions.
        """
        type_a, type_b = hessian_types
        rng = np.random.default_rng(seed=42)
        phi_a = Dummy(self.n_params, seed=rng, hessian_type=type_a)
        phi_b = Dummy(self.n_params, seed=rng, hessian_type=type_b)
        combo = phi_a + phi_b
        np.testing.assert_equal(
            combo.hessian_diagonal(model),
            phi_a.hessian_diagonal(model) + phi_b.hessian_diagonal(model),
        )


class Failed(Objective):
    """
    A failed objective function that raises errors on every method.

    This class is used to test that ``Scaled`` methods don't call the underlying
    objective function methods if the ``multiplier`` is zero.
    """

    def __init__(self, n_params: int):
        self._n_params = n_params

    @property
    def n_params(self):
        return self._n_params

    def __call__(self, model):
        raise NotImplementedError()

    def gradient(self, model):
        raise NotImplementedError()

    def hessian(self, model):
        raise NotImplementedError()

    def hessian_approx(self, model):
        raise NotImplementedError()

    def hessian_diagonal(self, model):
        raise NotImplementedError()


class TestScaledMethods:
    """
    Test ``__call__``, ``gradient`` and ``hessian`` for a ``Scaled``.
    """

    scalar = 3.1416
    n_params = 5

    @pytest.fixture
    def model(self):
        rng = np.random.default_rng(seed=42)
        model = rng.uniform(size=self.n_params)
        return model

    def test_call(self, model):
        """
        Test the call method of Scaled objective functions.
        """
        phi = Dummy(self.n_params)
        scaled = self.scalar * phi
        np.testing.assert_allclose(scaled(model), self.scalar * phi(model))

    def test_gradient(self, model):
        """
        Test the gradient method of Scaled objective functions.
        """
        phi = Dummy(self.n_params)
        scaled = self.scalar * phi
        np.testing.assert_allclose(
            scaled.gradient(model), self.scalar * phi.gradient(model)
        )

    @pytest.mark.parametrize("hessian_type", ["dense", "sparse", "linop"])
    def test_hessian(self, model, hessian_type):
        """
        Test the hessian method of Scaled objective functions.
        """
        phi = Dummy(self.n_params, hessian_type=hessian_type)
        scaled = self.scalar * phi
        assert_equal_linear_operators(
            scaled.hessian(model), self.scalar * phi.hessian(model)
        )

    @pytest.mark.parametrize("hessian_type", ["dense", "sparse"])
    def test_hessian_approx(self, model, hessian_type):
        """
        Test the ``hessian_approx`` method of Scaled objective functions.
        """
        phi = Dummy(self.n_params, seed=42, hessian_type=hessian_type)
        scaled = self.scalar * phi
        assert_equal_linear_operators(
            scaled.hessian_approx(model),
            self.scalar * phi.hessian_approx(model),
            to_dense=True,
        )

    @pytest.mark.parametrize("hessian_type", ["dense", "sparse"])
    def test_hessian_diagonal(self, model, hessian_type):
        """
        Test the ``hessian_diagonal`` method of Scaled objective functions.
        """
        phi = Dummy(self.n_params, seed=42, hessian_type=hessian_type)
        scaled = self.scalar * phi
        np.testing.assert_equal(
            scaled.hessian_diagonal(model),
            self.scalar * phi.hessian_diagonal(model),
        )

    def test_call_null(self, model):
        """Test calling a Scaled with a zero multiplier."""
        phi = Failed(self.n_params)
        with pytest.raises(NotImplementedError):
            phi(model)

        scaled = 0.0 * phi
        result = scaled(model)
        assert np.isscalar(result)
        assert result == 0.0

    def test_gradient_null(self, model):
        """Test gradient of Scaled with a zero multiplier."""
        phi = Failed(self.n_params)
        with pytest.raises(NotImplementedError):
            phi.gradient(model)

        scaled = 0.0 * phi
        gradient = scaled.gradient(model)
        assert gradient.shape == (self.n_params,)
        np.testing.assert_equal(gradient, 0.0)

    def test_hessian_null(self, model):
        """Test hessian of Scaled with a zero multiplier."""
        phi = Failed(self.n_params)
        with pytest.raises(NotImplementedError):
            phi.hessian(model)

        scaled = 0.0 * phi
        hessian = scaled.hessian(model)
        assert hessian.shape == (self.n_params, self.n_params)
        np.testing.assert_equal(hessian.toarray(), 0.0)

    def test_hessian_approx_null(self, model):
        """Test hessian_approx of Scaled with a zero multiplier."""
        phi = Failed(self.n_params)
        with pytest.raises(NotImplementedError):
            phi.hessian_approx(model)

        scaled = 0.0 * phi
        hessian_approx = scaled.hessian_approx(model)
        assert hessian_approx.shape == (self.n_params, self.n_params)
        np.testing.assert_equal(hessian_approx.toarray(), 0.0)

    def test_hessian_diagonal_null(self, model):
        """Test hessian_diagonal of Scaled with a zero multiplier."""
        phi = Failed(self.n_params)
        with pytest.raises(NotImplementedError):
            phi.hessian_diagonal(model)

        scaled = 0.0 * phi
        hessian_diagonal = scaled.hessian_diagonal(model)
        assert hessian_diagonal.shape == (self.n_params,)
        np.testing.assert_equal(hessian_diagonal, 0.0)


class TestObjectiveFunRepresentations:
    """
    Test representations of the objective function.
    """

    def test_name_setter(self):
        # Test the setter
        phi = Dummy(10)
        assert phi.name is None
        dummy_name = "blah"
        phi.set_name(dummy_name)
        assert phi.name == dummy_name

        # Test the set_name method
        phi = Dummy(10)
        returned_phi = phi.set_name(dummy_name)
        assert phi.name == dummy_name
        assert returned_phi is phi

        # Test None
        phi = Dummy(10).set_name(None)
        assert phi.name is None
        phi = Dummy(10)
        phi.name = None
        assert phi.name is None

    def test_invalid_name(self):
        phi = Dummy(3)
        invalid_name = 32
        msg = re.escape(
            f"Invalid name '{invalid_name}' of type 'int'. "
            "Please provide a string or None."
        )
        with pytest.raises(TypeError, match=msg):
            phi.name = invalid_name
        with pytest.raises(TypeError, match=msg):
            phi.set_name(invalid_name)

    def test_repr(self):
        phi = Dummy(3)
        assert repr(phi) == f"{phi._base_str}(m)"
        phi = Dummy(3).set_name("a")
        assert repr(phi) == f"{phi._base_str}a(m)"

    def test_repr_latex(self):
        phi = Dummy(3)
        assert phi._repr_latex_() == f"${phi._base_latex} (m)$"
        phi = Dummy(3).set_name("a")
        assert phi._repr_latex_() == f"${phi._base_latex}_{{a}} (m)$"


class TestScaledRepresentations:
    """
    Test representations of the scaled objective function.
    """

    @pytest.mark.parametrize(
        ("multiplier", "multiplier_str"),
        [
            (3.4, "3.4"),
            (-3.4, "-3.4"),
            (0.0, "0."),
            (1e3, "1000."),
            (1e-3, "0.001"),
            (1e4, "1.e+04"),
            (1e-4, "1.e-04"),
            (3e-5, "3.e-05"),
            (3e5, "3.e+05"),
        ],
    )
    def test_repr(self, multiplier: float, multiplier_str: str):
        phi = Dummy(3).set_name("a")
        scaled = multiplier * phi
        assert repr(scaled) == f"{multiplier_str} {phi}"

    @pytest.mark.parametrize(
        ("multiplier", "multiplier_str"),
        [
            (3.4, "3.4"),
            (-3.4, "-3.4"),
            (0.0, "0."),
            (1e3, "1000."),
            (1e-3, "0.001"),
            (1e4, "1.e+04"),
            (1e-4, "1.e-04"),
            (3e-5, "3.e-05"),
            (3e5, "3.e+05"),
        ],
    )
    def test_repr_with_combo(self, multiplier: float, multiplier_str: str):
        combo = Dummy(3).set_name("a") + Dummy(3).set_name("b")
        scaled = multiplier * combo
        assert repr(scaled) == f"{multiplier_str} [{combo}]"

    def test_repr_latex(self):
        phi = Dummy(3).set_name("a")
        phi_latex = phi._repr_latex_().strip("$")

        multiplier, multiplier_str = 3.4, "3.4"
        scaled = multiplier * phi
        assert scaled._repr_latex_() == f"${multiplier_str} \\, {phi_latex}$"

        multiplier, multiplier_str = 5.8e3, "5.8 \\cdot 10^{3}"
        scaled = multiplier * phi
        assert scaled._repr_latex_() == f"${multiplier_str} \\, {phi_latex}$"

        multiplier, multiplier_str = 3.4, "3.4"
        combo = phi + Dummy(3).set_name("b")
        combo_latex = combo._repr_latex_().strip("$")
        scaled = multiplier * combo
        assert scaled._repr_latex_() == f"${multiplier_str} \\, [{combo_latex}]$"


class TestComboRepresentations:
    """
    Test representations of the combo objective function.
    """

    def test_repr(self):
        phi_a, phi_b = Dummy(3).set_name("a"), Dummy(3).set_name("b")
        combo = phi_a + phi_b
        assert repr(combo) == f"{phi_a} + {phi_b}"

        phi_c = Dummy(3).set_name("c")
        combo = phi_a + phi_b + phi_c
        assert repr(combo) == f"[{phi_a} + {phi_b}] + {phi_c}"

        combo = (phi_a + phi_b + phi_c).flatten()
        assert repr(combo) == f"{phi_a} + {phi_b} + {phi_c}"

        phi_d = Dummy(3).set_name("d")
        combo = (phi_a + phi_b) + (phi_c + phi_d)
        assert repr(combo) == f"[{phi_a} + {phi_b}] + [{phi_c} + {phi_d}]"

    def test_repr_latex(self):
        phi_a, phi_b = Dummy(3).set_name("a"), Dummy(3).set_name("b")
        combo = phi_a + phi_b

        phi_a_latex = phi_a._repr_latex_().strip("$")
        phi_b_latex = phi_b._repr_latex_().strip("$")
        assert combo._repr_latex_() == f"${phi_a_latex} + {phi_b_latex}$"

        phi_c = Dummy(3).set_name("c")
        phi_c_latex = phi_c._repr_latex_().strip("$")
        combo = phi_a + phi_b + phi_c
        assert (
            combo._repr_latex_() == f"$[{phi_a_latex} + {phi_b_latex}] + {phi_c_latex}$"
        )

        combo = (phi_a + phi_b + phi_c).flatten()
        assert (
            combo._repr_latex_() == f"${phi_a_latex} + {phi_b_latex} + {phi_c_latex}$"
        )

        phi_d = Dummy(3).set_name("d")
        phi_d_latex = phi_d._repr_latex_().strip("$")
        combo = (phi_a + phi_b) + (phi_c + phi_d)
        assert (
            combo._repr_latex_()
            == f"$[{phi_a_latex} + {phi_b_latex}] + [{phi_c_latex} + {phi_d_latex}]$"
        )


class TestInfo:
    """
    Test the ``info`` method of objective functions.
    """

    def test_objective(self, capsys):
        """
        Test the ``info`` method of objective functions.
        """
        phi = Dummy(3)
        phi.info()
        captured = capsys.readouterr()
        first_line, *_ = captured.out.splitlines()
        assert first_line == "Dummy"

    def test_scaled(self, capsys):
        """
        Test the ``info`` method of scaled objective functions.
        """
        phi = 3.2 * Dummy(3)
        phi.info()
        captured = capsys.readouterr()
        first_line, *_ = captured.out.splitlines()
        assert first_line == "Scaled"

    def test_combo(self, capsys):
        """
        Test the ``info`` method of scaled objective functions.
        """
        phi = 3.2 * Dummy(3) + 3.4 * Dummy(3)
        phi.info()
        captured = capsys.readouterr()
        first_line, *_ = captured.out.splitlines()
        assert first_line == "Combo"


class TestEquality:
    """Test equality conditions between objective functions."""

    def test_equal_objective(self):
        phi_a = Dummy(3)
        phi_b = phi_a
        assert phi_a == phi_b
        phi_c = Dummy(3)
        assert phi_a != phi_c

    def test_equal_scaled(self):
        phi_a = Dummy(3)
        scaled_a = 3.0 * phi_a
        scaled_b = 3.0 * phi_a
        assert scaled_a == scaled_b

        scaled_a = 3.0 * phi_a
        scaled_b = -3.0 * phi_a
        assert scaled_a != scaled_b

        phi_b = Dummy(3)
        scaled_a = 3.0 * phi_a
        scaled_b = 3.0 * phi_b
        assert scaled_a != scaled_b

        scaled_a = 3.0 * phi_a
        scaled_b = 2.0 * phi_b
        assert scaled_a != scaled_b

        scaled = 5.0 * phi_a
        assert scaled != phi_a
        scaled = 1.0 * phi_a
        assert scaled != phi_a

    def test_equal_combo(self):
        phi_a, phi_b, phi_c = [Dummy(3) for _ in range(3)]

        # Two different combos with same functions are equal
        combo_1 = phi_a + phi_b
        combo_2 = phi_a + phi_b
        assert combo_1 == combo_2
        combo_1 = phi_a + phi_b + phi_c
        combo_2 = phi_a + phi_b + phi_c
        assert combo_1 == combo_2

        # Combos with same functions but different structure are not equal
        combo_1 = (phi_a + phi_b + phi_c).flatten()
        combo_2 = (phi_a + phi_b) + phi_c
        assert combo_1 != combo_2

        # Combos with scaled with same multipliers are equal
        combo_1 = 3.0 * phi_a + 2.1 * phi_b
        combo_2 = 3.0 * phi_a + 2.1 * phi_b
        assert combo_1 == combo_2

        # Combos with scaled with different multipliers are not equal
        combo_1 = 3.0 * phi_a + 2.1 * phi_b
        combo_2 = 6.0 * phi_a + 4.1 * phi_b
        assert combo_1 != combo_2

        # Combos are never equal to non-combos
        combo = phi_a + phi_b
        assert combo != phi_a
        assert combo != 3.0 * phi_a

        # Combos with different lengths are not equal
        combo_1 = phi_a + phi_b + phi_c
        combo_2 = phi_a + phi_b
        assert combo_1 != combo_2

        # Combos with functions in different order are not equal
        combo_1 = phi_a + phi_b
        combo_2 = phi_b + phi_a
        assert combo_1 != combo_2

    def test_equal_nested_combo(self):
        phi_a, phi_b, phi_c, phi_d = [Dummy(3) for _ in range(4)]

        # Nested combos should be the same if they have the same structure
        combo_1 = (phi_a + phi_b) + (phi_c + phi_d)
        combo_2 = (phi_a + phi_b) + (phi_c + phi_d)
        assert combo_1 == combo_2

        # Nested combos with different structures should be different
        combo_1 = (phi_b + phi_a) + (phi_c + phi_d)
        combo_2 = (phi_a + phi_b) + (phi_c + phi_d)
        assert combo_1 != combo_2
        combo_1 = (phi_a + phi_b) + (phi_d + phi_c)
        combo_2 = (phi_a + phi_b) + (phi_c + phi_d)
        assert combo_1 != combo_2
        combo_1 = (phi_a + phi_b + phi_c) + phi_d
        combo_2 = (phi_a + phi_b) + (phi_c + phi_d)
        assert combo_1 != combo_2


class TestHash:
    """
    Test hash operations for objective functions.

    The ``Objective`` and ``Scaled`` objects are hashable, but the ``Combo`` is not,
    since it behaves like a list.
    """

    def test_objective(self):
        phi_a, phi_b = Dummy(3), Dummy(3)
        assert hash(phi_a) == hash(phi_a)
        assert hash(phi_a) != hash(phi_b)

    def test_scaled(self):
        scaled_a, scaled_b = 3.0 * Dummy(3), 3.0 * Dummy(3)
        assert hash(scaled_a) == hash(scaled_a)
        assert hash(scaled_a) != hash(scaled_b)

        phi = Dummy(3)
        scaled_a = 3.0 * phi
        scaled_b = 3.0 * phi
        assert hash(scaled_a) == hash(scaled_b)

    def test_combo(self):
        """
        Test that Combo is not hashable.
        """
        phi_a, phi_b = Dummy(3), Dummy(3)
        combo = phi_a + phi_b
        assert combo.__hash__ is None
        with pytest.raises(TypeError):
            hash(combo)
