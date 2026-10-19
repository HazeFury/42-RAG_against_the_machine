def add(a: float, b: float) -> float:
    """Return the sum of a and b."""
    return a + b


def divide(a: float, b: float) -> float:
    """
    Divise a by b.
    Raise a ValueError b equals 0.
    """
    if b == 0:
        raise ValueError("Division by 0 is impossible.")
    return a / b
