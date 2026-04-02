"""
Test operations for objective functions.
"""

import itertools
import re

import numpy as np
import pytest
from scipy.sparse import dia_array, sparray
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from inversion_ideas.base import Combo, Objective, Scaled


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

    def hessian_diagonal(self, model):  # noqa: ARG002
        return (self.a_matrix.T @ self.a_matrix).diagonal()


def assert_equal_linear_operators(a, b, to_dense=False, seed=None, **kwargs):
    """
    Check if two linear operators are the same.

    If ``a`` and ``b`` are ``LinearOperator``s, they will be compared by computing the
    dot product with random arrays. Only the ``matvec`` and ``rmatvec`` will be tested.

    Parameters
    ----------
    a, b : arrays, sparse arrays, or linear operators
        Arrays or linear operators that will be tested.
    to_dense : bool, optional
        If True, sparse arrays will be converted to dense arrays for testing.
        Use False for big matrices that can be too large to fit in memory.
    seed : int or None, optional
        Random seed used to define a random vector to test ``LinearOperator``s.
        This argument will be ignored if ``a`` and ``b`` are not ``LinearOperator``s.
    **kwargs : dict
        Extra keyword arguments that will be passed to
        :func:`numpy.testing.assert_equal`.
    """
    if to_dense:
        if isinstance(a, sparray):
            a = a.toarray()
        if isinstance(b, sparray):
            b = b.toarray()
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        np.testing.assert_equal(a, b, **kwargs)
    else:
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        # matvec
        rng = np.random.default_rng(seed=seed)
        vector = rng.uniform(size=a.shape[1])
        np.testing.assert_equal(a @ vector, b @ vector, **kwargs)
        # rmatvec
        vector = rng.uniform(size=a.shape[0])
        np.testing.assert_equal(a.T @ vector, b.T @ vector, **kwargs)


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

    def test_hessian_diagonal(self, model):
        """
        Test the hessian_diagonal method of Combo objective functions.
        """
        phi_a, phi_b = Dummy(self.n_params, seed=42), Dummy(self.n_params, seed=43)
        combo = phi_a + phi_b
        np.testing.assert_allclose(
            combo.hessian_diagonal(model),
            phi_a.hessian_diagonal(model) + phi_b.hessian_diagonal(model),
        )


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

    def test_hessian_diagonal(self, model):
        """
        Test the hessian_diagonal method of Scaled objective functions.
        """
        phi = Dummy(self.n_params)
        scaled = self.scalar * phi
        np.testing.assert_allclose(
            scaled.hessian_diagonal(model), self.scalar * phi.hessian_diagonal(model)
        )


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
