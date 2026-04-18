"""
Trendline Engine
Fits linear regression trendlines through swing points.
Used by triangle, wedge, channel, and broadening pattern detectors.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import config
from patterns.swing_detector import SwingPoint


@dataclass
class Trendline:
    """A fitted trendline through swing points."""
    slope: float
    intercept: float
    r_squared: float
    touch_count: int
    points: List[SwingPoint]

    def price_at(self, bar_index):
        """Get the projected price at a given bar index."""
        return self.slope * bar_index + self.intercept

    def is_valid(self):
        """Check if trendline meets minimum quality criteria."""
        return (
            self.touch_count >= config.TRENDLINE_MIN_TOUCHES
            and self.r_squared >= config.TRENDLINE_MIN_R_SQUARED
        )


def fit_trendline(points: List[SwingPoint]) -> Optional[Trendline]:
    """Fit a linear regression trendline through swing points.

    Args:
        points: list of SwingPoints (at least 2)

    Returns:
        Trendline object or None if insufficient points
    """
    if len(points) < 2:
        return None

    x = np.array([p.index for p in points], dtype=float)
    y = np.array([p.price for p in points], dtype=float)

    # Linear regression
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]

    # R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return Trendline(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        touch_count=len(points),
        points=points,
    )


def is_flat(trendline: Trendline, avg_price: float = None, threshold: float = 0.0005) -> bool:
    """Check if trendline is approximately horizontal."""
    if avg_price and avg_price > 0:
        return abs(trendline.slope / avg_price) < threshold
    return abs(trendline.slope) < threshold


def is_rising(trendline: Trendline, avg_price: float = None, threshold: float = 0.0003) -> bool:
    """Check if trendline has a positive slope."""
    if avg_price and avg_price > 0:
        return (trendline.slope / avg_price) > threshold
    return trendline.slope > threshold


def is_falling(trendline: Trendline, avg_price: float = None, threshold: float = -0.0003) -> bool:
    """Check if trendline has a negative slope."""
    if avg_price and avg_price > 0:
        return (trendline.slope / avg_price) < threshold
    return trendline.slope < threshold


def find_convergence(line1: Trendline, line2: Trendline) -> Optional[int]:
    """Find the bar index where two trendlines intersect.

    Returns:
        Bar index of intersection, or None if parallel/diverging
    """
    slope_diff = line1.slope - line2.slope
    if abs(slope_diff) < 1e-10:
        return None  # Parallel lines

    x_intersect = (line2.intercept - line1.intercept) / slope_diff
    return int(round(x_intersect))


def are_converging(line1: Trendline, line2: Trendline, current_bar: int) -> bool:
    """Check if two trendlines are converging (triangle/wedge pattern)."""
    conv_bar = find_convergence(line1, line2)
    if conv_bar is None:
        return False
    return conv_bar > current_bar  # Convergence is in the future


def are_diverging(line1: Trendline, line2: Trendline, current_bar: int) -> bool:
    """Check if two trendlines are diverging (broadening pattern)."""
    gap_now = abs(line1.price_at(current_bar) - line2.price_at(current_bar))
    gap_past = abs(line1.price_at(current_bar - 20) - line2.price_at(current_bar - 20))
    return gap_now > gap_past
