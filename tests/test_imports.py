"""
Tests related to imports.
"""

import pytest

import inversion_ideas as ii


class TestDeprecationMinimizers:
    """Test deprecation warnings when importing minimizers from top level."""

    def test_conjugate_gradient(self):
        with pytest.warns(FutureWarning, match="Importing 'conjugate_gradient'"):
            ii.conjugate_gradient  # ruff: ignore[B018]

    def test_gauss_newton_cg(self):
        with pytest.warns(
            FutureWarning, match="Importing 'GaussNewtonConjugateGradient'"
        ):
            ii.GaussNewtonConjugateGradient  # ruff: ignore[B018]
