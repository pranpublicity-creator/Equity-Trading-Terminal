"""
Reversal Pattern Detection (8 Patterns)
Based on Bulkowski's Encyclopedia of Chart Patterns.

1. Double Top (4 variants)    2. Double Bottom (4 variants)
3. Head & Shoulders Top       4. Head & Shoulders Bottom
5. Triple Top                 6. Triple Bottom
7. Rounding Top               8. Rounding Bottom
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from patterns.swing_detector import SwingPoint, get_recent_swings


@dataclass
class PatternResult:
    """Universal pattern detection result."""
    pattern_name: str = ""
    variant: str = ""
    direction: str = ""             # 'bullish' or 'bearish'
    confidence: float = 0.0         # 0.0 to 1.0
    entry_price: float = 0.0        # Breakout level
    target_price: float = 0.0       # Measure rule target
    stop_loss: float = 0.0          # Pattern invalidation
    neckline: float = 0.0
    breakout_confirmed: bool = False
    volume_confirmed: bool = False
    delivery_confirmed: bool = False
    oi_confirmed: bool = False
    multi_tf_confirmed: bool = False
    timeframe: str = "15"
    start_index: int = 0
    end_index: int = 0


def detect_double_top(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 100) -> List[PatternResult]:
    """Detect Double Top pattern (bearish reversal).
    Bulkowski: close below lowest low between two peaks = breakout.
    4 variants: Adam&Adam, Adam&Eve, Eve&Adam, Eve&Eve.
    """
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    highs = [s for s in recent if s.type == "HIGH"]

    if len(highs) < 2:
        return results

    close = df["close"].values
    volume = df["volume"].values
    low = df["low"].values

    for i in range(len(highs) - 1):
        h1, h2 = highs[i], highs[i + 1]

        # Price proximity: within 3%
        price_diff = abs(h1.price - h2.price) / h1.price
        if price_diff > 0.03:
            continue

        # Minimum separation: at least 10 bars
        if h2.index - h1.index < 10:
            continue

        # Find neckline (lowest low between peaks)
        valley_slice = low[h1.index:h2.index + 1]
        if len(valley_slice) == 0:
            continue
        neckline = float(np.min(valley_slice))
        neckline_idx = h1.index + int(np.argmin(valley_slice))

        # Minimum height: 2%
        avg_peak = (h1.price + h2.price) / 2
        height = (avg_peak - neckline) / neckline
        if height < 0.02:
            continue

        # Classify variant (Adam=narrow, Eve=wide)
        h1_width = _peak_width(df, h1.index, h1.price)
        h2_width = _peak_width(df, h2.index, h2.price)
        h1_type = "Adam" if h1_width < 3 else "Eve"
        h2_type = "Adam" if h2_width < 3 else "Eve"
        variant = f"{h1_type}_{h2_type}".lower()

        # Volume: h2 volume should be lower than h1 (Bulkowski)
        vol_confirmed = h2.volume < h1.volume

        # Check breakout: any close below neckline after h2
        breakout = False
        if h2.index < len(close) - 1:
            post_close = close[h2.index + 1:]
            breakout = bool(np.any(post_close < neckline))

        # Target: neckline - height
        target = neckline - (avg_peak - neckline)

        # Confidence scoring
        confidence = 0.40  # Base
        confidence += 0.10 * min(height / 0.05, 1.0)           # Height bonus
        confidence += 0.10 if vol_confirmed else 0.0            # Volume
        confidence += 0.10 if breakout else 0.0                 # Breakout
        confidence += 0.05 * (1 - price_diff / 0.03)           # Peak similarity
        confidence += 0.05 if (h2.index - h1.index) > 20 else 0  # Time separation
        confidence = min(confidence, 1.0)

        results.append(PatternResult(
            pattern_name="double_top",
            variant=variant,
            direction="bearish",
            confidence=confidence,
            entry_price=neckline,
            target_price=target,
            stop_loss=max(h1.price, h2.price) * 1.01,
            neckline=neckline,
            breakout_confirmed=breakout,
            volume_confirmed=vol_confirmed,
            start_index=h1.index,
            end_index=h2.index,
        ))

    return results


def detect_double_bottom(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 100) -> List[PatternResult]:
    """Detect Double Bottom pattern (bullish reversal).
    Bulkowski: close above highest high between two valleys = breakout.
    """
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    lows = [s for s in recent if s.type == "LOW"]

    if len(lows) < 2:
        return results

    close = df["close"].values
    high = df["high"].values

    for i in range(len(lows) - 1):
        l1, l2 = lows[i], lows[i + 1]

        price_diff = abs(l1.price - l2.price) / l1.price
        if price_diff > 0.03:
            continue
        if l2.index - l1.index < 10:
            continue

        # Neckline: highest high between valleys
        peak_slice = high[l1.index:l2.index + 1]
        if len(peak_slice) == 0:
            continue
        neckline = float(np.max(peak_slice))

        avg_valley = (l1.price + l2.price) / 2
        height = (neckline - avg_valley) / avg_valley
        if height < 0.02:
            continue

        # Variant classification
        l1_width = _trough_width(df, l1.index, l1.price)
        l2_width = _trough_width(df, l2.index, l2.price)
        l1_type = "Adam" if l1_width < 3 else "Eve"
        l2_type = "Adam" if l2_width < 3 else "Eve"
        variant = f"{l1_type}_{l2_type}".lower()

        # Volume: often increases at second bottom (Bulkowski)
        vol_confirmed = l2.volume > l1.volume * 0.8

        # Breakout check
        breakout = False
        if l2.index < len(close) - 1:
            post_close = close[l2.index + 1:]
            breakout = bool(np.any(post_close > neckline))

        target = neckline + (neckline - avg_valley)

        confidence = 0.40
        confidence += 0.10 * min(height / 0.05, 1.0)
        confidence += 0.10 if vol_confirmed else 0.0
        confidence += 0.10 if breakout else 0.0
        confidence += 0.05 * (1 - price_diff / 0.03)
        confidence += 0.05 if (l2.index - l1.index) > 20 else 0
        confidence = min(confidence, 1.0)

        results.append(PatternResult(
            pattern_name="double_bottom",
            variant=variant,
            direction="bullish",
            confidence=confidence,
            entry_price=neckline,
            target_price=target,
            stop_loss=min(l1.price, l2.price) * 0.99,
            neckline=neckline,
            breakout_confirmed=breakout,
            volume_confirmed=vol_confirmed,
            start_index=l1.index,
            end_index=l2.index,
        ))

    return results


def detect_head_shoulders_top(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 150) -> List[PatternResult]:
    """Detect Head & Shoulders Top (bearish reversal).
    Bulkowski: 5 alternating swings H-L-H-L-H, middle H highest.
    """
    results = []
    recent = get_recent_swings(swings, len(df), lookback)

    if len(recent) < 5:
        return results

    close = df["close"].values

    # Scan for 5-point sequences starting with HIGH
    for i in range(len(recent) - 4):
        pts = recent[i:i + 5]

        # Must alternate H-L-H-L-H
        expected = ["HIGH", "LOW", "HIGH", "LOW", "HIGH"]
        if [p.type for p in pts] != expected:
            continue

        ls, lv, head, rv, rs = pts  # left shoulder, left valley, head, right valley, right shoulder

        # Head must be highest
        if head.price <= ls.price or head.price <= rs.price:
            continue

        # Shoulder symmetry: within 15%
        sym = abs(ls.price - rs.price) / ls.price
        if sym > 0.15:
            continue

        # Neckline through two valleys
        neckline_left = lv.price
        neckline_right = rv.price
        neckline_avg = (neckline_left + neckline_right) / 2

        # Neckline slope (can be sloped per Bulkowski)
        neckline_slope = (neckline_right - neckline_left) / max(rv.index - lv.index, 1)

        # Height
        head_height = head.price - neckline_avg
        if head_height / head.price < 0.02:
            continue

        # Volume: typically diminishes LS → Head → RS
        vol_declining = ls.volume > head.volume > rs.volume * 0.7

        # Breakout: close below neckline after right shoulder
        breakout = False
        neckline_at_rs = neckline_left + neckline_slope * (rs.index - lv.index)
        if rs.index < len(close) - 1:
            for j in range(rs.index + 1, min(rs.index + 30, len(close))):
                nl_at_j = neckline_left + neckline_slope * (j - lv.index)
                if close[j] < nl_at_j:
                    breakout = True
                    break

        # Target: neckline - head_height
        target = neckline_avg - head_height

        confidence = 0.45
        confidence += 0.10 * (1 - sym / 0.15)          # Symmetry bonus
        confidence += 0.10 if vol_declining else 0.0
        confidence += 0.10 if breakout else 0.0
        confidence += 0.05 * min(head_height / (head.price * 0.05), 1.0)
        confidence = min(confidence, 1.0)

        # Check for complex variant (extra shoulders)
        variant = "complex" if _has_extra_shoulders(recent, i, pts) else "standard"

        results.append(PatternResult(
            pattern_name="head_shoulders_top",
            variant=variant,
            direction="bearish",
            confidence=confidence,
            entry_price=neckline_avg,
            target_price=target,
            stop_loss=head.price * 1.01,
            neckline=neckline_avg,
            breakout_confirmed=breakout,
            volume_confirmed=vol_declining,
            start_index=ls.index,
            end_index=rs.index,
        ))

    return results


def detect_head_shoulders_bottom(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 150) -> List[PatternResult]:
    """Detect Head & Shoulders Bottom (bullish reversal). Mirror of H&S Top."""
    results = []
    recent = get_recent_swings(swings, len(df), lookback)

    if len(recent) < 5:
        return results

    close = df["close"].values

    for i in range(len(recent) - 4):
        pts = recent[i:i + 5]
        expected = ["LOW", "HIGH", "LOW", "HIGH", "LOW"]
        if [p.type for p in pts] != expected:
            continue

        ls, lp, head, rp, rs = pts

        # Head must be lowest
        if head.price >= ls.price or head.price >= rs.price:
            continue

        sym = abs(ls.price - rs.price) / ls.price
        if sym > 0.15:
            continue

        neckline_avg = (lp.price + rp.price) / 2
        head_depth = neckline_avg - head.price
        if head_depth / neckline_avg < 0.02:
            continue

        neckline_slope = (rp.price - lp.price) / max(rp.index - lp.index, 1)

        # Volume often increases at right shoulder for bottoms
        vol_confirmed = rs.volume > head.volume * 0.7

        breakout = False
        if rs.index < len(close) - 1:
            for j in range(rs.index + 1, min(rs.index + 30, len(close))):
                nl_at_j = lp.price + neckline_slope * (j - lp.index)
                if close[j] > nl_at_j:
                    breakout = True
                    break

        target = neckline_avg + head_depth
        variant = "complex" if _has_extra_shoulders(recent, i, pts) else "standard"

        confidence = 0.45
        confidence += 0.10 * (1 - sym / 0.15)
        confidence += 0.10 if vol_confirmed else 0.0
        confidence += 0.10 if breakout else 0.0
        confidence += 0.05 * min(head_depth / (neckline_avg * 0.05), 1.0)
        confidence = min(confidence, 1.0)

        results.append(PatternResult(
            pattern_name="head_shoulders_bottom",
            variant=variant,
            direction="bullish",
            confidence=confidence,
            entry_price=neckline_avg,
            target_price=target,
            stop_loss=head.price * 0.99,
            neckline=neckline_avg,
            breakout_confirmed=breakout,
            volume_confirmed=vol_confirmed,
            start_index=ls.index,
            end_index=rs.index,
        ))

    return results


def detect_triple_top(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 120) -> List[PatternResult]:
    """Detect Triple Top (bearish reversal). Three highs within 3% proximity."""
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    highs = [s for s in recent if s.type == "HIGH"]

    if len(highs) < 3:
        return results

    close = df["close"].values
    low = df["low"].values

    for i in range(len(highs) - 2):
        h1, h2, h3 = highs[i], highs[i + 1], highs[i + 2]
        avg_price = (h1.price + h2.price + h3.price) / 3

        # All three within 3% of each other
        if max(h1.price, h2.price, h3.price) - min(h1.price, h2.price, h3.price) > avg_price * 0.03:
            continue

        # Minimum separation
        if h3.index - h1.index < 20:
            continue

        # Support level (lowest low in pattern)
        support = float(np.min(low[h1.index:h3.index + 1]))
        height = avg_price - support
        if height / avg_price < 0.02:
            continue

        breakout = False
        if h3.index < len(close) - 1:
            breakout = bool(np.any(close[h3.index + 1:] < support))

        target = support - height

        confidence = 0.45
        confidence += 0.10 if breakout else 0.0
        confidence += 0.10 * min(height / (avg_price * 0.05), 1.0)
        confidence = min(confidence, 0.95)

        results.append(PatternResult(
            pattern_name="triple_top",
            variant="standard",
            direction="bearish",
            confidence=confidence,
            entry_price=support,
            target_price=target,
            stop_loss=max(h1.price, h2.price, h3.price) * 1.01,
            neckline=support,
            breakout_confirmed=breakout,
            start_index=h1.index,
            end_index=h3.index,
        ))

    return results


def detect_triple_bottom(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 120) -> List[PatternResult]:
    """Detect Triple Bottom (bullish reversal). Three lows within 3% proximity."""
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    lows = [s for s in recent if s.type == "LOW"]

    if len(lows) < 3:
        return results

    close = df["close"].values
    high = df["high"].values

    for i in range(len(lows) - 2):
        l1, l2, l3 = lows[i], lows[i + 1], lows[i + 2]
        avg_price = (l1.price + l2.price + l3.price) / 3

        if max(l1.price, l2.price, l3.price) - min(l1.price, l2.price, l3.price) > avg_price * 0.03:
            continue
        if l3.index - l1.index < 20:
            continue

        resistance = float(np.max(high[l1.index:l3.index + 1]))
        height = resistance - avg_price
        if height / resistance < 0.02:
            continue

        breakout = False
        if l3.index < len(close) - 1:
            breakout = bool(np.any(close[l3.index + 1:] > resistance))

        target = resistance + height

        confidence = 0.45
        confidence += 0.10 if breakout else 0.0
        confidence += 0.10 * min(height / (resistance * 0.05), 1.0)
        confidence = min(confidence, 0.95)

        results.append(PatternResult(
            pattern_name="triple_bottom",
            variant="standard",
            direction="bullish",
            confidence=confidence,
            entry_price=resistance,
            target_price=target,
            stop_loss=min(l1.price, l2.price, l3.price) * 0.99,
            neckline=resistance,
            breakout_confirmed=breakout,
            start_index=l1.index,
            end_index=l3.index,
        ))

    return results


def detect_rounding_top(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 200) -> List[PatternResult]:
    """Detect Rounding Top (bearish reversal). Quadratic fit to swing highs."""
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    highs = [s for s in recent if s.type == "HIGH"]

    if len(highs) < 5:
        return results

    close = df["close"].values
    x = np.array([h.index for h in highs], dtype=float)
    y = np.array([h.price for h in highs], dtype=float)

    # Fit quadratic: y = ax^2 + bx + c
    try:
        coeffs = np.polyfit(x, y, 2)
    except np.linalg.LinAlgError:
        return results

    a = coeffs[0]
    if a >= 0:  # Must be concave down for rounding top
        return results

    # R-squared for quadratic fit
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    if r_sq < 0.6:
        return results

    # Right lip level
    right_lip = highs[-1].price
    left_lip = highs[0].price

    # Breakout: close below right lip
    breakout = False
    last_idx = highs[-1].index
    if last_idx < len(close) - 1:
        breakout = bool(np.any(close[last_idx + 1:] < min(right_lip, left_lip)))

    height = max(y) - min(right_lip, left_lip)
    target = min(right_lip, left_lip) - height

    confidence = 0.35
    confidence += 0.15 * r_sq
    confidence += 0.10 if breakout else 0.0
    confidence += 0.05 * min(len(highs) / 8, 1.0)
    confidence = min(confidence, 0.90)

    results.append(PatternResult(
        pattern_name="rounding_top",
        variant="standard",
        direction="bearish",
        confidence=confidence,
        entry_price=min(right_lip, left_lip),
        target_price=target,
        stop_loss=max(y) * 1.01,
        breakout_confirmed=breakout,
        start_index=highs[0].index,
        end_index=highs[-1].index,
    ))

    return results


def detect_rounding_bottom(df: pd.DataFrame, swings: List[SwingPoint], lookback: int = 200) -> List[PatternResult]:
    """Detect Rounding Bottom (bullish reversal). Quadratic fit to swing lows."""
    results = []
    recent = get_recent_swings(swings, len(df), lookback)
    lows = [s for s in recent if s.type == "LOW"]

    if len(lows) < 5:
        return results

    close = df["close"].values
    x = np.array([l.index for l in lows], dtype=float)
    y = np.array([l.price for l in lows], dtype=float)

    try:
        coeffs = np.polyfit(x, y, 2)
    except np.linalg.LinAlgError:
        return results

    a = coeffs[0]
    if a <= 0:  # Must be concave up
        return results

    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    if r_sq < 0.6:
        return results

    right_lip = lows[-1].price
    left_lip = lows[0].price
    lip = max(right_lip, left_lip)

    breakout = False
    last_idx = lows[-1].index
    if last_idx < len(close) - 1:
        breakout = bool(np.any(close[last_idx + 1:] > lip))

    height = lip - min(y)
    target = lip + height

    confidence = 0.35
    confidence += 0.15 * r_sq
    confidence += 0.10 if breakout else 0.0
    confidence += 0.05 * min(len(lows) / 8, 1.0)
    confidence = min(confidence, 0.90)

    results.append(PatternResult(
        pattern_name="rounding_bottom",
        variant="standard",
        direction="bullish",
        confidence=confidence,
        entry_price=lip,
        target_price=target,
        stop_loss=min(y) * 0.99,
        breakout_confirmed=breakout,
        start_index=lows[0].index,
        end_index=lows[-1].index,
    ))

    return results


# ─── Helpers ────────────────────────────────────────────────

def _peak_width(df, idx, peak_price, threshold_pct=0.02):
    """Measure how wide a peak is (Adam=narrow, Eve=wide). Returns bar count."""
    high = df["high"].values
    threshold = peak_price * (1 - threshold_pct)
    width = 0
    for offset in range(1, 10):
        left_ok = (idx - offset >= 0) and (high[idx - offset] >= threshold)
        right_ok = (idx + offset < len(high)) and (high[idx + offset] >= threshold)
        if left_ok or right_ok:
            width += 1
        else:
            break
    return width


def _trough_width(df, idx, trough_price, threshold_pct=0.02):
    """Measure how wide a trough is."""
    low = df["low"].values
    threshold = trough_price * (1 + threshold_pct)
    width = 0
    for offset in range(1, 10):
        left_ok = (idx - offset >= 0) and (low[idx - offset] <= threshold)
        right_ok = (idx + offset < len(low)) and (low[idx + offset] <= threshold)
        if left_ok or right_ok:
            width += 1
        else:
            break
    return width


def _has_extra_shoulders(all_swings, start_idx, pts):
    """Check if there are additional shoulder-like swings (complex H&S)."""
    if start_idx >= 2:
        pre = all_swings[start_idx - 2:start_idx]
        if any(s.type == pts[0].type for s in pre):
            return True
    if start_idx + 5 < len(all_swings) - 1:
        post = all_swings[start_idx + 5:start_idx + 7]
        if any(s.type == pts[-1].type for s in post):
            return True
    return False
