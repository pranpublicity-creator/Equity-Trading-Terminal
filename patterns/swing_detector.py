"""
Swing Point (Pivot) Detector
Identifies local highs and lows in OHLCV data for pattern detection.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

import config


@dataclass
class SwingPoint:
    """A local price extreme (pivot point)."""
    index: int          # Bar index in the DataFrame
    price: float        # High or Low price
    type: str           # 'HIGH' or 'LOW'
    strength: float     # Amplitude relative to neighbors (0-1)
    volume: float       # Volume at this bar
    timestamp: int = 0  # Unix timestamp if available


def find_swings(
    df: pd.DataFrame,
    left: int = None,
    right: int = None,
    min_amplitude_pct: float = None,
) -> List[SwingPoint]:
    """Find all swing highs and lows in OHLCV data.

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns
        left: bars to the left for comparison window
        right: bars to the right for comparison window
        min_amplitude_pct: minimum swing amplitude as fraction of price

    Returns:
        Sorted list of alternating SwingPoints (HIGH, LOW, HIGH, ...)
    """
    left = left or config.SWING_LEFT_BARS
    right = right or config.SWING_RIGHT_BARS
    min_amplitude_pct = min_amplitude_pct or config.SWING_MIN_AMPLITUDE_PCT

    highs = df["high"].values
    lows = df["low"].values
    close = df["close"].values
    volume = df["volume"].values
    timestamps = df["timestamp"].values if "timestamp" in df.columns else np.zeros(len(df))

    swing_highs = _find_swing_highs(highs, lows, volume, timestamps, left, right, min_amplitude_pct)
    swing_lows = _find_swing_lows(highs, lows, volume, timestamps, left, right, min_amplitude_pct)

    # Merge and sort by index
    all_swings = swing_highs + swing_lows
    all_swings.sort(key=lambda s: s.index)

    # Remove consecutive same-type swings (keep strongest)
    return _alternate_swings(all_swings)


def _find_swing_highs(highs, lows, volume, timestamps, left, right, min_amp):
    """Find local maxima in the high series."""
    swings = []
    for i in range(left, len(highs) - right):
        window = highs[i - left:i + right + 1]
        if highs[i] == np.max(window):
            # Calculate amplitude: distance from high to nearest low
            nearby_lows = lows[max(0, i - left):min(len(lows), i + right + 1)]
            min_low = np.min(nearby_lows)
            amplitude = (highs[i] - min_low) / highs[i] if highs[i] > 0 else 0

            if amplitude >= min_amp:
                strength = min(amplitude / 0.05, 1.0)  # Normalize: 5% swing = max strength
                swings.append(SwingPoint(
                    index=i,
                    price=float(highs[i]),
                    type="HIGH",
                    strength=float(strength),
                    volume=float(volume[i]),
                    timestamp=int(timestamps[i]) if timestamps[i] else 0,
                ))
    return swings


def _find_swing_lows(highs, lows, volume, timestamps, left, right, min_amp):
    """Find local minima in the low series."""
    swings = []
    for i in range(left, len(lows) - right):
        window = lows[i - left:i + right + 1]
        if lows[i] == np.min(window):
            nearby_highs = highs[max(0, i - left):min(len(highs), i + right + 1)]
            max_high = np.max(nearby_highs)
            amplitude = (max_high - lows[i]) / max_high if max_high > 0 else 0

            if amplitude >= min_amp:
                strength = min(amplitude / 0.05, 1.0)
                swings.append(SwingPoint(
                    index=i,
                    price=float(lows[i]),
                    type="LOW",
                    strength=float(strength),
                    volume=float(volume[i]),
                    timestamp=int(timestamps[i]) if timestamps[i] else 0,
                ))
    return swings


def _alternate_swings(swings):
    """Ensure swings alternate between HIGH and LOW.
    When consecutive same-type swings occur, keep the most extreme.
    """
    if len(swings) < 2:
        return swings

    result = [swings[0]]
    for s in swings[1:]:
        if s.type != result[-1].type:
            result.append(s)
        else:
            # Same type: keep more extreme
            if s.type == "HIGH" and s.price > result[-1].price:
                result[-1] = s
            elif s.type == "LOW" and s.price < result[-1].price:
                result[-1] = s
    return result


def get_recent_swings(swings, df_len, lookback=100):
    """Filter swings to only those within the lookback window from the end."""
    cutoff = df_len - lookback
    return [s for s in swings if s.index >= cutoff]
