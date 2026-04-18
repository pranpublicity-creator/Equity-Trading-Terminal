"""
Volatility Pattern Detection (4 Patterns)
Based on Bulkowski's Encyclopedia of Chart Patterns.

17. Broadening Formation (+ Right-Angled variants)
18. Broadening Top / Bottom
19. Diamond
20. Wedge (Rising / Falling)
"""
from typing import List

import numpy as np
import pandas as pd

from patterns.swing_detector import SwingPoint, get_recent_swings
from patterns.trendline_engine import (
    fit_trendline, is_flat, is_rising, is_falling,
    are_converging, are_diverging,
)
from patterns.reversal_patterns import PatternResult


def detect_broadening_formation(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 100) -> List[PatternResult]:
    """Detect Broadening Formation (expanding price range).
    Upper trendline rising, lower trendline falling — lines diverge.
    Variants: standard, right-angled ascending (flat bottom, rising top),
    right-angled descending (flat top, falling bottom).
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

    last_idx = recent[-1].index
    avg_price = float(np.mean(close[-lookback:])) if n >= lookback else float(np.mean(close))

    # Lines must diverge
    if not are_diverging(upper, lower, last_idx):
        return results

    # Classify variant
    upper_rising = is_rising(upper, avg_price)
    upper_flat = is_flat(upper, avg_price)
    lower_falling = is_falling(lower, avg_price)
    lower_flat = is_flat(lower, avg_price)

    if upper_rising and lower_falling:
        variant = "broadening_standard"
    elif upper_rising and lower_flat:
        variant = "right_angled_ascending"
    elif upper_flat and lower_falling:
        variant = "right_angled_descending"
    else:
        return results  # Not a broadening formation

    resistance = upper.price_at(last_idx)
    support = lower.price_at(last_idx)
    height = resistance - support

    if height <= 0 or height / avg_price < 0.03:
        return results

    # Broadening formations are tricky — breakout direction is uncertain
    # Right-angled ascending tends to break down, descending tends to break up (Bulkowski)
    if variant == "right_angled_ascending":
        direction = "bearish"
        entry = float(support)
        target = float(support - height * 0.60)
        stop = float(resistance)
    elif variant == "right_angled_descending":
        direction = "bullish"
        entry = float(resistance)
        target = float(resistance + height * 0.60)
        stop = float(support)
    else:
        # Standard: check which boundary is broken
        if close[-1] > resistance:
            direction = "bullish"
            entry = float(resistance)
            target = float(resistance + height * 0.50)
            stop = float(support)
        elif close[-1] < support:
            direction = "bearish"
            entry = float(support)
            target = float(support - height * 0.50)
            stop = float(resistance)
        else:
            direction = "bearish"  # Bulkowski: slight bearish bias for standard
            entry = float(support)
            target = float(support - height * 0.50)
            stop = float(resistance)

    breakout = (close[-1] > resistance or close[-1] < support) if n > 0 else False

    confidence = 0.40  # Broadening patterns have lower reliability
    if upper.touch_count >= 3:
        confidence += 0.05
    if lower.touch_count >= 3:
        confidence += 0.05
    if breakout:
        confidence += 0.15

    results.append(PatternResult(
        pattern_name="broadening_formation",
        variant=variant,
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


def detect_broadening_top_bottom(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 100) -> List[PatternResult]:
    """Detect Broadening Top / Broadening Bottom.
    Expanding price range with clear higher highs AND lower lows.
    Top: forms after uptrend (bearish reversal).
    Bottom: forms after downtrend (bullish reversal).
    """
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    highs = [s for s in recent if s.type == "HIGH"]
    lows = [s for s in recent if s.type == "LOW"]
    close = df["close"].values
    n = len(df)

    if len(highs) < 3 or len(lows) < 3:
        return results

    # Check for expanding range: higher highs AND lower lows
    hh = all(highs[i].price < highs[i + 1].price for i in range(len(highs) - 1))
    ll = all(lows[i].price > lows[i + 1].price for i in range(len(lows) - 1))

    if not (hh and ll):
        return results

    # Determine if top or bottom by prior trend
    pre_pattern_start = max(0, recent[0].index - 30)
    pre_close = close[pre_pattern_start:recent[0].index]

    if len(pre_close) < 5:
        return results

    prior_up = pre_close[-1] > pre_close[0]
    last_idx = recent[-1].index
    high_level = highs[-1].price
    low_level = lows[-1].price
    height = high_level - low_level

    if prior_up:
        # Broadening Top (bearish reversal)
        direction = "bearish"
        variant = "broadening_top"
        entry = float(low_level)
        target = float(low_level - height * 0.60)
        stop = float(high_level)
    else:
        # Broadening Bottom (bullish reversal)
        direction = "bullish"
        variant = "broadening_bottom"
        entry = float(high_level)
        target = float(high_level + height * 0.60)
        stop = float(low_level)

    breakout = False
    if n > 0:
        if direction == "bearish" and close[-1] < low_level:
            breakout = True
        elif direction == "bullish" and close[-1] > high_level:
            breakout = True

    confidence = 0.40
    if len(highs) >= 4 and len(lows) >= 4:
        confidence += 0.05
    if breakout:
        confidence += 0.15

    results.append(PatternResult(
        pattern_name="broadening_top_bottom",
        variant=variant,
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


def detect_diamond(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 120) -> List[PatternResult]:
    """Detect Diamond pattern.
    First half: broadening (diverging trendlines).
    Second half: converging (symmetrical triangle).
    Together forms a diamond shape. Relatively rare.
    """
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    close = df["close"].values
    n = len(df)

    if len(recent) < 6:
        return results

    # Split swings roughly in half
    mid = len(recent) // 2
    first_half = recent[:mid + 1]
    second_half = recent[mid:]

    first_highs = [s for s in first_half if s.type == "HIGH"]
    first_lows = [s for s in first_half if s.type == "LOW"]
    second_highs = [s for s in second_half if s.type == "HIGH"]
    second_lows = [s for s in second_half if s.type == "LOW"]

    if len(first_highs) < 2 or len(first_lows) < 2:
        return results
    if len(second_highs) < 2 or len(second_lows) < 2:
        return results

    # First half: diverging
    upper1 = fit_trendline(first_highs)
    lower1 = fit_trendline(first_lows)
    if upper1 is None or lower1 is None:
        return results

    mid_idx = recent[mid].index
    if not are_diverging(upper1, lower1, mid_idx):
        return results

    # Second half: converging
    upper2 = fit_trendline(second_highs)
    lower2 = fit_trendline(second_lows)
    if upper2 is None or lower2 is None:
        return results

    last_idx = recent[-1].index
    if not are_converging(upper2, lower2, last_idx):
        return results

    # Diamond height = widest point (at the middle)
    wide_high = max(s.price for s in recent if s.type == "HIGH")
    wide_low = min(s.price for s in recent if s.type == "LOW")
    height = wide_high - wide_low

    avg_price = float(np.mean(close[-lookback:])) if n >= lookback else float(np.mean(close))
    if height / avg_price < 0.03:
        return results

    # Direction: usually reversal of prior trend
    pre_start = max(0, recent[0].index - 30)
    pre_close = close[pre_start:recent[0].index]
    if len(pre_close) > 5 and pre_close[-1] > pre_close[0]:
        direction = "bearish"  # Diamond top (after uptrend)
    else:
        direction = "bullish"  # Diamond bottom (after downtrend)

    # Entry at the boundary of the narrowing second half
    resistance = upper2.price_at(last_idx)
    support = lower2.price_at(last_idx)

    if direction == "bearish":
        entry = float(support)
        target = float(support - height)
        stop = float(resistance)
    else:
        entry = float(resistance)
        target = float(resistance + height)
        stop = float(support)

    breakout = False
    if n > 0:
        if direction == "bearish" and close[-1] < support:
            breakout = True
        elif direction == "bullish" and close[-1] > resistance:
            breakout = True

    confidence = 0.45
    if breakout:
        confidence += 0.15
    if len(recent) >= 8:
        confidence += 0.05  # Well-defined diamond

    results.append(PatternResult(
        pattern_name="diamond",
        variant="diamond_top" if direction == "bearish" else "diamond_bottom",
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


def detect_wedge(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 100) -> List[PatternResult]:
    """Detect Rising Wedge (bearish) and Falling Wedge (bullish).
    Rising wedge: both lines rising, upper rises slower → converging.
    Falling wedge: both lines falling, lower falls faster → converging.
    Bulkowski: falling wedge = bullish 68%, rising wedge = bearish 69%.
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
    last_idx = recent[-1].index

    # Must converge
    if not are_converging(upper, lower, last_idx):
        return results

    upper_rising = is_rising(upper, avg_price)
    upper_falling = is_falling(upper, avg_price)
    lower_rising = is_rising(lower, avg_price)
    lower_falling = is_falling(lower, avg_price)

    # Rising Wedge: both rising, upper slope < lower slope (converging upward)
    if upper_rising and lower_rising and upper.slope < lower.slope:
        variant = "rising_wedge"
        direction = "bearish"
    # Falling Wedge: both falling, lower slope more negative (converging downward)
    elif upper_falling and lower_falling and upper.slope > lower.slope:
        variant = "falling_wedge"
        direction = "bullish"
    else:
        return results

    resistance = upper.price_at(last_idx)
    support = lower.price_at(last_idx)
    height = resistance - support

    if height <= 0 or height / avg_price < 0.015:
        return results

    if direction == "bullish":  # Falling wedge breaks up
        entry = float(resistance)
        target = float(resistance + height)
        stop = float(support)
    else:  # Rising wedge breaks down
        entry = float(support)
        target = float(support - height)
        stop = float(resistance)

    breakout = False
    if n > 0:
        if direction == "bullish" and close[-1] > resistance:
            breakout = True
        elif direction == "bearish" and close[-1] < support:
            breakout = True

    confidence = 0.50
    if upper.r_squared > 0.85 and lower.r_squared > 0.85:
        confidence += 0.10
    if upper.touch_count >= 3 or lower.touch_count >= 3:
        confidence += 0.05
    if breakout:
        confidence += 0.15

    results.append(PatternResult(
        pattern_name="wedge",
        variant=variant,
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
