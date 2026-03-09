"""Shared utilities for log parsers."""

from typing import Any

import numpy as np


def extract_time_and_field(
    rows: list[dict[str, Any]],
    field: str,
    time_field: str = "TimeMS",
    time_scale: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract time (seconds) and a named field from message rows.

    Args:
        rows: List of message dicts.
        field: Name of the value field to extract.
        time_field: Name of the time field in each row.
        time_scale: Multiplier to convert raw time values to seconds.

    Returns:
        Tuple of (time_seconds, values) as numpy arrays.
    """
    time = np.array([float(r[time_field]) * time_scale for r in rows])
    values = np.array([float(r[field]) for r in rows])
    return time, values
