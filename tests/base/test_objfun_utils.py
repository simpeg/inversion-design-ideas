"""
Test utility functions used by objective functions' code.
"""

import re

import numpy as np
import pytest
from scipy.sparse import diags_array, sparray
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from inversion_ideas.base.objective_function import _float_to_str, _sum


class TestSum:
    """
    Test custom sum for operators.

    Test cases:

        - All arrays, should return array.
        - One sparse array, should return array.
        - All sparse arrays, should return sparse array.
        - One linear operator, should return linear operator.
        - Put the special objects ones in the beginning and in the middle of the
          generator.
    """

    shape = (25, 10)

    @pytest.fixture
    def matrices(self):
        seeds = (40, 41, 42)
        a, b, c = tuple(
            np.random.default_rng(seed=seed).uniform(size=self.shape) for seed in seeds
        )
        return a, b, c

    @pytest.fixture
    def sparse_arrays(self):
        seeds = (40, 41, 42)
        a, b, c = tuple(
            diags_array(
                np.random.default_rng(seed=seed).uniform(size=self.shape[0]),
                shape=self.shape,
            )
            for seed in seeds
        )
        return a, b, c

    @pytest.fixture
    def vector(self):
        return np.random.default_rng(seed=43).uniform(size=self.shape[1])

    def test_all_arrays(self, matrices):
        # Get the sum
        result = _sum(op for op in matrices)

        # We should recover a dense array
        assert isinstance(result, np.ndarray)

        # Check if result is correct
        a, b, c = matrices
        expected = a + b + c
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("index", [0, 1])
    def test_one_sparse_array(self, matrices, index):
        # Put a sparse array in the list of operators
        operators = list(matrices)
        operators[index] = diags_array(
            np.arange(self.shape[1]), shape=self.shape, dtype=float
        )

        # Get the sum
        result = _sum(op for op in operators)

        # We should recover a dense array
        assert isinstance(result, np.ndarray)

        # Check if result is correct
        a, b, c = operators
        expected = a + b + c
        np.testing.assert_allclose(result, expected)

    def test_all_sparse_arrays(self, sparse_arrays):
        result = _sum(op for op in sparse_arrays)

        # We should recover a sparse array
        assert isinstance(result, sparray)

        # Check if result is correct
        a, b, c = sparse_arrays
        expected = a + b + c
        np.testing.assert_allclose(result.toarray(), expected.toarray())

    @pytest.mark.parametrize("index", [0, 1])
    def test_one_linear_operator(self, matrices, vector, index):
        # Put a linear operator in the list of operators
        operators = list(matrices)
        factor = 5.1
        operators[index] = factor * aslinearoperator(operators[index])

        # Get the sum
        result = _sum(op for op in operators)

        # We should recover a linear operator
        assert isinstance(result, LinearOperator)

        # Check if result is correct
        a, b, c = matrices
        expected = factor * a + b + c if index == 0 else a + factor * b + c
        np.testing.assert_allclose(result @ vector, expected @ vector)


class TestFloatToString:
    """Test the ``_float_to_str`` private function."""

    @pytest.mark.parametrize("precision", [0, -1])
    def test_invalid_precision(self, precision):
        msg = re.escape(f"Invalid precision value '{precision}'")
        with pytest.raises(ValueError, match=msg):
            _float_to_str(3.1416, precision)

    @pytest.mark.parametrize(
        ("number", "string"),
        [
            # Zero
            (0, "0."),
            # Positional
            (3.14, "3.14"),
            (3.1416, "3.142"),
            (-3.14, "-3.14"),
            (-3.1416, "-3.142"),
            (0.001, "0.001"),
            (-0.001, "-0.001"),
            (0.123456, "0.123"),
            (1000.0, "1000."),
            (-1000.0, "-1000."),
            (999.123, "999.123"),
            (999.1235, "999.124"),
            (-999.123, "-999.123"),
            (-999.1235, "-999.124"),
            # Scientific
            (3e-5, "3.e-05"),
            (-3e-5, "-3.e-05"),
            (3.1416e-5, "3.142e-05"),
            (-3.1416e-5, "-3.142e-05"),
            (0.0001, "1.e-04"),
            (-0.0001, "-1.e-04"),
            (1000.123, "1.000e+03"),
            (-1000.123, "-1.000e+03"),
        ],
    )
    def test_float_to_str(self, number, string):
        assert _float_to_str(number) == string
