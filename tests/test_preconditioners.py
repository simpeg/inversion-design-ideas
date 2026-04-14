"""
Test preconditioner functions and classes.
"""

import numpy as np
from scipy.sparse import dia_array, diags_array

from inversion_ideas import get_jacobi_preconditioner

from .utils import Dummy


class TestJacobiPreconditionerFunction:
    """
    Test the ``get_jacobi_preconditioner`` function.
    """

    def test_jacobi(self):
        n_params = 10
        phi = Dummy(n_params)

        expected = diags_array(1 / (phi.a_matrix.T @ phi.a_matrix).diagonal())

        rng = np.random.default_rng(seed=32)
        model = rng.uniform(size=n_params)
        preconditioner = get_jacobi_preconditioner(phi, model)
        assert isinstance(preconditioner, dia_array)

        np.testing.assert_allclose(expected.toarray(), preconditioner.toarray())
