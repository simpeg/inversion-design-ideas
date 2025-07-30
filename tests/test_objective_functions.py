"""
Test operations for objective functions.
"""
import pytest
import numpy as np
from scipy.sparse import diags_array, sparray
from scipy.sparse.linalg import aslinearoperator, LinearOperator

from inversion_ideas.objective_function import _sum


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
        operators[index] = diags_array(np.arange(self.shape[1]), shape=self.shape)

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
