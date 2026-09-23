"""
Test utilities of the base submodule.
"""

import re

import pytest

from inversion_ideas.base._utils import float_to_str


class TestFloatToString:
    """Test the ``float_to_str`` private function."""

    @pytest.mark.parametrize("precision", [0, -1])
    def test_invalid_precision(self, precision):
        msg = re.escape(f"Invalid precision value '{precision}'")
        with pytest.raises(ValueError, match=msg):
            float_to_str(3.1416, precision)

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
    def testfloat_to_str(self, number, string):
        assert float_to_str(number) == string
