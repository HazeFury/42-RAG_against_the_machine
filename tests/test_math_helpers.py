import pytest

from src.utils.math_helpers import add, divide


def test_add_positive_numbers() -> None:
    """Test the sum of two positives numbers."""
    assert add(2.0, 3.0) == 5.0


def test_add_negative_numbers() -> None:
    """Test the sum of two negatives numbers."""
    assert add(-1.0, 5.0) == 4.0


def test_divide_normal() -> None:
    """Test a classic division ."""
    assert divide(10.0, 2.0) == 5.0


def test_divide_by_zero() -> None:
    """Test if division by zero raise an error."""
    with pytest.raises(ValueError, match="Division by 0 is impossible."):
        divide(10.0, 0.0)
