"""
Continuation Pattern Detection (3 Patterns)
Based on Bulkowski's Encyclopedia of Chart Patterns.

9. Flag (Bull/Bear)    10. Pennant (Bull/Bear)    11. Measured Move (Up/Down)
"""
from typing import List

import numpy as np
import pandas as pd

from patterns.swing_detector import SwingPoint, get_recent_swings
from patterns.trendline_engine import fit_trendline, is_rising, is_falling, are_converging
from patterns.reversal_patterns import PatternResult


def detect_flag(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 50) -> List[PatternResult]:
    """Detect Bull/Bear Flag patterns.
    Bulkowski: sharp impulse move (flagpole) followed by a short,
    counter-trend parallel channel (the flag). Duration < 3 weeks.
    """
    results = []
    close = df["close"].values
    volume = df["volume"].values
    n = len(df)

    if n < 20:
        return results

    # Scan for flagpole + flag consolidation
    scan_start = max(0, n - lookback)

    for pole_start in range(scan_start, n - 15):
        # Try different flagpole lengths (5–15 bars)
        for pole_len in range(5, min(16, n - pole_start - 5)):
            pole_end = pole_start + pole_len
            if pole_end >= n - 5:
                continue

            pole_move = close[pole_end] - close[pole_start]
            pole_pct = abs(pole_move) / close[pole_start] if close[pole_start] > 0 else 0

            # Flagpole must be a strong impulse (>5%)
            if pole_pct < 0.05:
                continue

            is_bull = pole_move > 0
            pole_height = abs(pole_move)

            # Look for consolidation after the pole (5–25 bars)
            max_flag_len = min(25, n - pole_end)
            if max_flag_len < 5:
                continue

            # Get swings in the flag region
            flag_swings_h = [s for s in swings if s.type == "HIGH" and pole_end <= s.index < pole_end + max_flag_len]
            flag_swings_l = [s for s in swings if s.type == "LOW" and pole_end <= s.index < pole_end + max_flag_len]

            if len(flag_swings_h) < 2 or len(flag_swings_l) < 2:
                continue

            upper_line = fit_trendline(flag_swings_h)
            lower_line = fit_trendline(flag_swings_l)
            if upper_line is None or lower_line is None:
                continue

            # Flag channel width must be < 50% of flagpole
            flag_end_idx = max(s.index for s in flag_swings_h + flag_swings_l)
            mid_idx = (pole_end + flag_end_idx) // 2
            channel_width = abs(upper_line.price_at(mid_idx) - lower_line.price_at(mid_idx))
            if channel_width > pole_height * 0.50:
                continue

            # Flag should slope counter to the pole direction
            avg_price = float(np.mean(close[pole_start:flag_end_idx + 1]))
            if is_bull:
                # Bull flag: slight downward or flat slope
                if is_rising(upper_line, avg_price, threshold=0.001):
                    continue
                direction = "bullish"
                entry = upper_line.price_at(flag_end_idx)
                target = entry + pole_height
                stop = lower_line.price_at(flag_end_idx)
            else:
                # Bear flag: slight upward or flat slope
                if is_falling(lower_line, avg_price, threshold=-0.001):
                    continue
                direction = "bearish"
                entry = lower_line.price_at(flag_end_idx)
                target = entry - pole_height
                stop = upper_line.price_at(flag_end_idx)

            # Volume should decrease during flag
            pole_vol = float(np.mean(volume[pole_start:pole_end]))
            flag_vol = float(np.mean(volume[pole_end:flag_end_idx + 1]))
            vol_declining = flag_vol < pole_vol * 0.8

            # Breakout check
            breakout = False
            if flag_end_idx < n - 1:
                if is_bull and close[flag_end_idx + 1] > upper_line.price_at(flag_end_idx + 1):
                    breakout = True
                elif not is_bull and close[flag_end_idx + 1] < lower_line.price_at(flag_end_idx + 1):
                    breakout = True

            confidence = 0.45
            if vol_declining:
                confidence += 0.10
            if upper_line.r_squared > 0.80 and lower_line.r_squared > 0.80:
                confidence += 0.10
            if pole_pct > 0.08:
                confidence += 0.05  # Tall flagpole
            if breakout:
                confidence += 0.10

            results.append(PatternResult(
                pattern_name="flag",
                variant=f"{'bull' if is_bull else 'bear'}_flag",
                direction=direction,
                confidence=min(confidence, 1.0),
                entry_price=float(entry),
                target_price=float(target),
                stop_loss=float(stop),
                breakout_confirmed=breakout,
                volume_confirmed=vol_declining,
                start_index=pole_start,
                end_index=flag_end_idx,
            ))

        # Only keep the best flag starting from this pole
        if results and results[-1].start_index == pole_start:
            break

    # Deduplicate overlapping flags, keep best
    return _deduplicate(results)


def detect_pennant(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 50) -> List[PatternResult]:
    """Detect Bull/Bear Pennant patterns.
    Like flags but consolidation forms a small symmetrical triangle (converging lines).
    Duration typically shorter than flags (5–20 bars).
    """
    results = []
    close = df["close"].values
    volume = df["volume"].values
    n = len(df)

    if n < 20:
        return results

    scan_start = max(0, n - lookback)

    for pole_start in range(scan_start, n - 15):
        for pole_len in range(5, min(16, n - pole_start - 5)):
            pole_end = pole_start + pole_len
            if pole_end >= n - 5:
                continue

            pole_move = close[pole_end] - close[pole_start]
            pole_pct = abs(pole_move) / close[pole_start] if close[pole_start] > 0 else 0

            if pole_pct < 0.05:
                continue

            is_bull = pole_move > 0
            pole_height = abs(pole_move)

            max_pennant_len = min(20, n - pole_end)
            if max_pennant_len < 5:
                continue

            pennant_highs = [s for s in swings if s.type == "HIGH" and pole_end <= s.index < pole_end + max_pennant_len]
            pennant_lows = [s for s in swings if s.type == "LOW" and pole_end <= s.index < pole_end + max_pennant_len]

            if len(pennant_highs) < 2 or len(pennant_lows) < 2:
                continue

            upper_line = fit_trendline(pennant_highs)
            lower_line = fit_trendline(pennant_lows)
            if upper_line is None or lower_line is None:
                continue

            avg_price = float(np.mean(close[pole_end:pole_end + max_pennant_len]))

            # Pennant = converging lines (upper falling, lower rising)
            if not is_falling(upper_line, avg_price):
                continue
            if not is_rising(lower_line, avg_price):
                continue
            if not are_converging(upper_line, lower_line, pole_end + max_pennant_len):
                continue

            pennant_end = max(s.index for s in pennant_highs + pennant_lows)

            if is_bull:
                direction = "bullish"
                entry = upper_line.price_at(pennant_end)
                target = entry + pole_height
                stop = lower_line.price_at(pennant_end)
            else:
                direction = "bearish"
                entry = lower_line.price_at(pennant_end)
                target = entry - pole_height
                stop = upper_line.price_at(pennant_end)

            pole_vol = float(np.mean(volume[pole_start:pole_end]))
            pennant_vol = float(np.mean(volume[pole_end:pennant_end + 1]))
            vol_declining = pennant_vol < pole_vol * 0.8

            breakout = False
            if pennant_end < n - 1:
                if is_bull and close[pennant_end + 1] > upper_line.price_at(pennant_end + 1):
                    breakout = True
                elif not is_bull and close[pennant_end + 1] < lower_line.price_at(pennant_end + 1):
                    breakout = True

            confidence = 0.45
            if vol_declining:
                confidence += 0.10
            if upper_line.r_squared > 0.80 and lower_line.r_squared > 0.80:
                confidence += 0.10
            if pole_pct > 0.08:
                confidence += 0.05
            if breakout:
                confidence += 0.10

            results.append(PatternResult(
                pattern_name="pennant",
                variant=f"{'bull' if is_bull else 'bear'}_pennant",
                direction=direction,
                confidence=min(confidence, 1.0),
                entry_price=float(entry),
                target_price=float(target),
                stop_loss=float(stop),
                breakout_confirmed=breakout,
                volume_confirmed=vol_declining,
                start_index=pole_start,
                end_index=pennant_end,
            ))
            break  # Best pennant for this pole start

    return _deduplicate(results)


def detect_measured_move(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 200) -> List[PatternResult]:
    """Detect Measured Move Up/Down (ABC pattern).
    Three phases: initial move (A→B), corrective phase (B→C), second move (C→D).
    Second move ≈ first move in magnitude (within 20%).
    Corrective phase retraces 30–70% of first move.
    """
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    n = len(df)

    if len(recent) < 4:
        return results

    # Look for A-B-C-D swing sequence
    for i in range(len(recent) - 3):
        a, b, c, d = recent[i], recent[i + 1], recent[i + 2], recent[i + 3]

        # Measured Move Up: LOW-HIGH-LOW-HIGH
        if a.type == "LOW" and b.type == "HIGH" and c.type == "LOW" and d.type == "HIGH":
            move1 = b.price - a.price  # A→B (up)
            retrace = b.price - c.price  # B→C (down)
            move2 = d.price - c.price  # C→D (up)

            if move1 <= 0 or move2 <= 0:
                continue

            # Retrace must be 30-70% of first move
            retrace_pct = retrace / move1 if move1 > 0 else 0
            if not (0.30 <= retrace_pct <= 0.70):
                continue

            # Second move must be similar to first (within 20%)
            move_ratio = move2 / move1 if move1 > 0 else 0
            if not (0.80 <= move_ratio <= 1.20):
                continue

            # Min move size: 3% of price
            if move1 / a.price < 0.03:
                continue

            # If D is not the last swing, the move is complete; otherwise predict target
            predicted_target = c.price + move1  # D should reach this

            confidence = 0.50
            if 0.90 <= move_ratio <= 1.10:
                confidence += 0.10  # Very symmetric
            if 0.40 <= retrace_pct <= 0.60:
                confidence += 0.05  # Clean retrace

            results.append(PatternResult(
                pattern_name="measured_move",
                variant="measured_move_up",
                direction="bullish",
                confidence=min(confidence, 1.0),
                entry_price=float(c.price),
                target_price=float(predicted_target),
                stop_loss=float(a.price),
                breakout_confirmed=d.price >= predicted_target * 0.95,
                start_index=a.index,
                end_index=d.index,
            ))

        # Measured Move Down: HIGH-LOW-HIGH-LOW
        elif a.type == "HIGH" and b.type == "LOW" and c.type == "HIGH" and d.type == "LOW":
            move1 = a.price - b.price  # A→B (down)
            retrace = c.price - b.price  # B→C (up)
            move2 = c.price - d.price  # C→D (down)

            if move1 <= 0 or move2 <= 0:
                continue

            retrace_pct = retrace / move1 if move1 > 0 else 0
            if not (0.30 <= retrace_pct <= 0.70):
                continue

            move_ratio = move2 / move1 if move1 > 0 else 0
            if not (0.80 <= move_ratio <= 1.20):
                continue

            if move1 / a.price < 0.03:
                continue

            predicted_target = c.price - move1

            confidence = 0.50
            if 0.90 <= move_ratio <= 1.10:
                confidence += 0.10
            if 0.40 <= retrace_pct <= 0.60:
                confidence += 0.05

            results.append(PatternResult(
                pattern_name="measured_move",
                variant="measured_move_down",
                direction="bearish",
                confidence=min(confidence, 1.0),
                entry_price=float(c.price),
                target_price=float(predicted_target),
                stop_loss=float(a.price),
                breakout_confirmed=d.price <= predicted_target * 1.05,
                start_index=a.index,
                end_index=d.index,
            ))

    return _deduplicate(results)


def _deduplicate(results: List[PatternResult]) -> List[PatternResult]:
    """Remove overlapping patterns, keep highest confidence."""
    if len(results) <= 1:
        return results

    results.sort(key=lambda r: r.confidence, reverse=True)
    kept = []
    for r in results:
        overlap = False
        for k in kept:
            if r.start_index <= k.end_index and r.end_index >= k.start_index:
                overlap = True
                break
        if not overlap:
            kept.append(r)
    return kept
