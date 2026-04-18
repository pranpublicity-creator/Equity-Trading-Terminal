"""
Breakout Pattern Detection (5 Patterns)
Based on Bulkowski's Encyclopedia of Chart Patterns.

12. Ascending Triangle    13. Descending Triangle    14. Symmetrical Triangle
15. Rectangle             16. Cup & Handle (+ Inverted)
"""
from typing import List

import numpy as np
import pandas as pd

from patterns.swing_detector import SwingPoint, get_recent_swings
from patterns.trendline_engine import (
    fit_trendline, is_flat, is_rising, is_falling,
    find_convergence, are_converging,
)
from patterns.reversal_patterns import PatternResult


def detect_ascending_triangle(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 100) -> List[PatternResult]:
    """Detect Ascending Triangle (bullish breakout).
    Flat upper resistance + rising lower support, converging.
    Bulkowski: upward breakout ~64% of the time.
    """
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    highs = [s for s in recent if s.type == "HIGH"]
    lows = [s for s in recent if s.type == "LOW"]
    close = df["close"].values
    volume = df["volume"].values
    n = len(df)

    if len(highs) < 2 or len(lows) < 2:
        return results

    upper = fit_trendline(highs)
    lower = fit_trendline(lows)
    if upper is None or lower is None:
        return results

    avg_price = float(np.mean(close[-lookback:])) if n >= lookback else float(np.mean(close))

    # Upper must be flat, lower must be rising
    if not is_flat(upper, avg_price):
        return results
    if not is_rising(lower, avg_price):
        return results

    # Lines must converge
    last_idx = recent[-1].index
    if not are_converging(upper, lower, last_idx):
        return results

    # Min touches
    if upper.touch_count < 2 or lower.touch_count < 2:
        return results

    resistance = upper.price_at(last_idx)
    support = lower.price_at(last_idx)
    height = resistance - support

    if height <= 0 or height / avg_price < 0.02:
        return results

    # Breakout check
    breakout = n > 0 and close[-1] > resistance

    # Volume trend: should decrease then spike at breakout
    pattern_vol = volume[max(0, last_idx - 20):last_idx + 1]
    vol_declining = False
    if len(pattern_vol) > 5:
        first_half = float(np.mean(pattern_vol[:len(pattern_vol) // 2]))
        second_half = float(np.mean(pattern_vol[len(pattern_vol) // 2:]))
        vol_declining = second_half < first_half

    confidence = 0.50
    if upper.r_squared > 0.85:
        confidence += 0.05
    if lower.r_squared > 0.85:
        confidence += 0.05
    if vol_declining:
        confidence += 0.05
    if breakout:
        confidence += 0.15

    results.append(PatternResult(
        pattern_name="ascending_triangle",
        direction="bullish",
        confidence=min(confidence, 1.0),
        entry_price=float(resistance),
        target_price=float(resistance + height),
        stop_loss=float(support),
        neckline=float(resistance),
        breakout_confirmed=breakout,
        volume_confirmed=vol_declining,
        start_index=recent[0].index,
        end_index=last_idx,
    ))
    return results


def detect_descending_triangle(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 100) -> List[PatternResult]:
    """Detect Descending Triangle (bearish breakout).
    Flat lower support + falling upper resistance, converging.
    Bulkowski: downward breakout ~64% of the time.
    """
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    highs = [s for s in recent if s.type == "HIGH"]
    lows = [s for s in recent if s.type == "LOW"]
    close = df["close"].values
    volume = df["volume"].values
    n = len(df)

    if len(highs) < 2 or len(lows) < 2:
        return results

    upper = fit_trendline(highs)
    lower = fit_trendline(lows)
    if upper is None or lower is None:
        return results

    avg_price = float(np.mean(close[-lookback:])) if n >= lookback else float(np.mean(close))

    if not is_falling(upper, avg_price):
        return results
    if not is_flat(lower, avg_price):
        return results
    if not are_converging(upper, lower, recent[-1].index):
        return results
    if upper.touch_count < 2 or lower.touch_count < 2:
        return results

    last_idx = recent[-1].index
    resistance = upper.price_at(last_idx)
    support = lower.price_at(last_idx)
    height = resistance - support

    if height <= 0 or height / avg_price < 0.02:
        return results

    breakout = n > 0 and close[-1] < support

    pattern_vol = volume[max(0, last_idx - 20):last_idx + 1]
    vol_declining = False
    if len(pattern_vol) > 5:
        first_half = float(np.mean(pattern_vol[:len(pattern_vol) // 2]))
        second_half = float(np.mean(pattern_vol[len(pattern_vol) // 2:]))
        vol_declining = second_half < first_half

    confidence = 0.50
    if upper.r_squared > 0.85:
        confidence += 0.05
    if lower.r_squared > 0.85:
        confidence += 0.05
    if vol_declining:
        confidence += 0.05
    if breakout:
        confidence += 0.15

    results.append(PatternResult(
        pattern_name="descending_triangle",
        direction="bearish",
        confidence=min(confidence, 1.0),
        entry_price=float(support),
        target_price=float(support - height),
        stop_loss=float(resistance),
        neckline=float(support),
        breakout_confirmed=breakout,
        volume_confirmed=vol_declining,
        start_index=recent[0].index,
        end_index=last_idx,
    ))
    return results


def detect_symmetrical_triangle(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 100) -> List[PatternResult]:
    """Detect Symmetrical Triangle (can break either way).
    Upper falling + lower rising, converging to apex.
    Bulkowski: breaks in direction of prior trend ~54% of the time.
    """
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    highs = [s for s in recent if s.type == "HIGH"]
    lows = [s for s in recent if s.type == "LOW"]
    close = df["close"].values
    n = len(df)

    if len(highs) < 2 or len(lows) < 2:
        return results

    upper = fit_trendline(highs)
    lower = fit_trendline(lows)
    if upper is None or lower is None:
        return results

    avg_price = float(np.mean(close[-lookback:])) if n >= lookback else float(np.mean(close))

    if not is_falling(upper, avg_price):
        return results
    if not is_rising(lower, avg_price):
        return results

    last_idx = recent[-1].index
    if not are_converging(upper, lower, last_idx):
        return results

    resistance = upper.price_at(last_idx)
    support = lower.price_at(last_idx)
    height = resistance - support

    if height <= 0 or height / avg_price < 0.015:
        return results

    # Determine prior trend direction for bias
    pre_pattern_close = close[max(0, recent[0].index - 20):recent[0].index]
    if len(pre_pattern_close) > 5:
        prior_trend = "bullish" if pre_pattern_close[-1] > pre_pattern_close[0] else "bearish"
    else:
        prior_trend = "bullish"

    # Breakout check
    breakout_bull = close[-1] > resistance if n > 0 else False
    breakout_bear = close[-1] < support if n > 0 else False

    if breakout_bull:
        direction = "bullish"
        entry = float(resistance)
        target = float(resistance + height)
        stop = float(support)
        breakout = True
    elif breakout_bear:
        direction = "bearish"
        entry = float(support)
        target = float(support - height)
        stop = float(resistance)
        breakout = True
    else:
        direction = prior_trend
        entry = float(resistance) if prior_trend == "bullish" else float(support)
        target = float(resistance + height) if prior_trend == "bullish" else float(support - height)
        stop = float(support) if prior_trend == "bullish" else float(resistance)
        breakout = False

    confidence = 0.45
    if upper.r_squared > 0.85 and lower.r_squared > 0.85:
        confidence += 0.10
    if breakout:
        confidence += 0.15

    results.append(PatternResult(
        pattern_name="symmetrical_triangle",
        direction=direction,
        confidence=min(confidence, 1.0),
        entry_price=entry,
        target_price=target,
        stop_loss=stop,
        breakout_confirmed=breakout,
        start_index=recent[0].index,
        end_index=last_idx,
    ))
    return results


def detect_rectangle(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 80) -> List[PatternResult]:
    """Detect Rectangle (horizontal channel / trading range).
    Both upper and lower trendlines flat (range-bound).
    Bulkowski: breaks in direction of prior trend slightly more often.
    """
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    highs = [s for s in recent if s.type == "HIGH"]
    lows = [s for s in recent if s.type == "LOW"]
    close = df["close"].values
    volume = df["volume"].values
    n = len(df)

    if len(highs) < 2 or len(lows) < 2:
        return results

    upper = fit_trendline(highs)
    lower = fit_trendline(lows)
    if upper is None or lower is None:
        return results

    avg_price = float(np.mean(close[-lookback:])) if n >= lookback else float(np.mean(close))

    if not is_flat(upper, avg_price):
        return results
    if not is_flat(lower, avg_price):
        return results

    last_idx = recent[-1].index
    resistance = upper.price_at(last_idx)
    support = lower.price_at(last_idx)
    height = resistance - support

    if height <= 0 or height / avg_price < 0.02:
        return results

    # Prior trend for direction bias
    pre_pattern_close = close[max(0, recent[0].index - 20):recent[0].index]
    if len(pre_pattern_close) > 5:
        prior_trend = "bullish" if pre_pattern_close[-1] > pre_pattern_close[0] else "bearish"
    else:
        prior_trend = "bullish"

    breakout_bull = close[-1] > resistance if n > 0 else False
    breakout_bear = close[-1] < support if n > 0 else False

    if breakout_bull:
        direction, entry, target, stop, breakout = "bullish", resistance, resistance + height, support, True
    elif breakout_bear:
        direction, entry, target, stop, breakout = "bearish", support, support - height, resistance, True
    else:
        if prior_trend == "bullish":
            direction, entry, target, stop = "bullish", resistance, resistance + height, support
        else:
            direction, entry, target, stop = "bearish", support, support - height, resistance
        breakout = False

    # Volume: tends to decrease during rectangle
    pattern_vol = volume[max(0, recent[0].index):last_idx + 1]
    vol_declining = False
    if len(pattern_vol) > 10:
        first_half = float(np.mean(pattern_vol[:len(pattern_vol) // 2]))
        second_half = float(np.mean(pattern_vol[len(pattern_vol) // 2:]))
        vol_declining = second_half < first_half

    confidence = 0.45
    if upper.touch_count >= 3:
        confidence += 0.05
    if lower.touch_count >= 3:
        confidence += 0.05
    if vol_declining:
        confidence += 0.05
    if breakout:
        confidence += 0.15

    results.append(PatternResult(
        pattern_name="rectangle",
        direction=direction,
        confidence=min(confidence, 1.0),
        entry_price=float(entry),
        target_price=float(target),
        stop_loss=float(stop),
        neckline=float(resistance if direction == "bullish" else support),
        breakout_confirmed=breakout,
        volume_confirmed=vol_declining,
        start_index=recent[0].index,
        end_index=last_idx,
    ))
    return results


def detect_cup_and_handle(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 200) -> List[PatternResult]:
    """Detect Cup & Handle pattern (bullish) + Inverted Cup & Handle (bearish).
    Cup: rounded bottom (quadratic fit, a > 0).
    Handle: small pullback after right rim (< 50% of cup depth).
    Bulkowski: one of the most reliable continuation patterns.
    """
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    highs = [s for s in recent if s.type == "HIGH"]
    lows = [s for s in recent if s.type == "LOW"]
    close = df["close"].values
    n = len(df)

    # --- Cup & Handle (bullish) ---
    if len(highs) >= 3 and len(lows) >= 2:
        _detect_cup_handle_bullish(results, df, recent, highs, lows, close, n)

    # --- Inverted Cup & Handle (bearish) ---
    if len(highs) >= 2 and len(lows) >= 3:
        _detect_cup_handle_bearish(results, df, recent, highs, lows, close, n)

    return results


def _detect_cup_handle_bullish(results, df, recent, highs, lows, close, n):
    """Standard Cup & Handle (bullish)."""
    # Need at least: left rim (HIGH), cup bottom area (LOWs), right rim (HIGH)
    for i in range(len(highs) - 1):
        left_rim = highs[i]
        right_rim = highs[i + 1]

        # Rims should be approximately equal height (within 5%)
        rim_diff = abs(left_rim.price - right_rim.price) / max(left_rim.price, right_rim.price)
        if rim_diff > 0.05:
            continue

        # Find the lowest low between the rims
        cup_lows = [s for s in lows if left_rim.index < s.index < right_rim.index]
        if not cup_lows:
            continue

        cup_bottom = min(cup_lows, key=lambda s: s.price)
        cup_depth = min(left_rim.price, right_rim.price) - cup_bottom.price

        if cup_depth <= 0:
            continue

        # Cup must be significant (at least 5% of price)
        if cup_depth / left_rim.price < 0.05:
            continue

        # Check for rounded bottom: quadratic fit through lows in cup
        cup_swing_prices = [s.price for s in cup_lows]
        cup_swing_indices = [s.index for s in cup_lows]
        if len(cup_swing_prices) >= 3:
            coeffs = np.polyfit(cup_swing_indices, cup_swing_prices, 2)
            if coeffs[0] <= 0:  # Must be concave UP (a > 0)
                continue
        elif len(cup_swing_prices) < 2:
            continue

        # Look for handle: small pullback after right rim
        rim_level = min(left_rim.price, right_rim.price)
        handle_region = [s for s in recent if s.index > right_rim.index and s.type == "LOW"]

        handle_depth = 0
        handle_end_idx = right_rim.index
        has_handle = False

        if handle_region:
            handle_low = min(handle_region, key=lambda s: s.price)
            handle_depth = right_rim.price - handle_low.price

            # Handle depth must be < 50% of cup depth
            if 0 < handle_depth < cup_depth * 0.50:
                has_handle = True
                handle_end_idx = handle_low.index

        entry = rim_level
        target = rim_level + cup_depth
        stop = cup_bottom.price if not has_handle else max(
            cup_bottom.price, right_rim.price - cup_depth * 0.50
        )

        breakout = close[-1] > rim_level if n > 0 else False

        confidence = 0.50
        if has_handle:
            confidence += 0.10
        if rim_diff < 0.02:
            confidence += 0.05  # Very symmetric rims
        if len(cup_swing_prices) >= 3:
            confidence += 0.05  # Well-defined round bottom
        if breakout:
            confidence += 0.10

        results.append(PatternResult(
            pattern_name="cup_and_handle",
            variant="cup_and_handle" if has_handle else "cup_no_handle",
            direction="bullish",
            confidence=min(confidence, 1.0),
            entry_price=float(entry),
            target_price=float(target),
            stop_loss=float(stop),
            neckline=float(rim_level),
            breakout_confirmed=breakout,
            start_index=left_rim.index,
            end_index=handle_end_idx,
        ))


def _detect_cup_handle_bearish(results, df, recent, highs, lows, close, n):
    """Inverted Cup & Handle (bearish)."""
    for i in range(len(lows) - 1):
        left_rim = lows[i]
        right_rim = lows[i + 1]

        rim_diff = abs(left_rim.price - right_rim.price) / max(left_rim.price, right_rim.price)
        if rim_diff > 0.05:
            continue

        cup_highs = [s for s in highs if left_rim.index < s.index < right_rim.index]
        if not cup_highs:
            continue

        cup_top = max(cup_highs, key=lambda s: s.price)
        cup_depth = cup_top.price - max(left_rim.price, right_rim.price)

        if cup_depth <= 0 or cup_depth / left_rim.price < 0.05:
            continue

        # Quadratic fit: must be concave DOWN (a < 0) for inverted cup
        cup_swing_prices = [s.price for s in cup_highs]
        cup_swing_indices = [s.index for s in cup_highs]
        if len(cup_swing_prices) >= 3:
            coeffs = np.polyfit(cup_swing_indices, cup_swing_prices, 2)
            if coeffs[0] >= 0:
                continue

        rim_level = max(left_rim.price, right_rim.price)
        handle_region = [s for s in recent if s.index > right_rim.index and s.type == "HIGH"]

        has_handle = False
        handle_end_idx = right_rim.index
        if handle_region:
            handle_high = max(handle_region, key=lambda s: s.price)
            handle_depth = handle_high.price - right_rim.price
            if 0 < handle_depth < cup_depth * 0.50:
                has_handle = True
                handle_end_idx = handle_high.index

        entry = rim_level
        target = rim_level - cup_depth
        stop = cup_top.price

        breakout = close[-1] < rim_level if n > 0 else False

        confidence = 0.50
        if has_handle:
            confidence += 0.10
        if rim_diff < 0.02:
            confidence += 0.05
        if breakout:
            confidence += 0.10

        results.append(PatternResult(
            pattern_name="cup_and_handle",
            variant="inverted_cup_and_handle" if has_handle else "inverted_cup_no_handle",
            direction="bearish",
            confidence=min(confidence, 1.0),
            entry_price=float(entry),
            target_price=float(target),
            stop_loss=float(stop),
            neckline=float(rim_level),
            breakout_confirmed=breakout,
            start_index=left_rim.index,
            end_index=handle_end_idx,
        ))
