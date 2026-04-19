"""
Intraday Pattern Detectors (5-min timeframe)
=============================================
These patterns are designed to form and resolve WITHIN a single NSE session
(09:15 – 15:30 IST, 75 bars on 5-min).  They complement the existing
swing/multi-day patterns which run on 15-min data.

Pattern 1 — Opening Range Breakout (ORB)
-----------------------------------------
Logic:
  • Mark the high/low of the first ORB_CANDLES × 5-min bars (default = 6,
    i.e. the 09:15–09:45 half-hour opening range).
  • BUY  setup : current bar closes ABOVE ORB high with volume surge.
  • SELL setup : current bar closes BELOW ORB low  with volume surge.

Entry  : ORB high (BUY) or ORB low (SELL) — the breakout level.
SL     : Opposite side of the range (ORB low for BUY, ORB high for SELL).
Target : Breakout level ± range  (1:1 measure rule = ~R:R 1.0).
         A second target at ±2× range is stored as a comment.

Confidence:
  base  0.65  with volume confirmation  /  0.50 without
  +0.10 if the 15-min regime (passed in) is TRENDING same direction
  +0.05 if multi-session ORB range is consistent (stable open)
  cap   0.90

The returned PatternResult uses timeframe="5" so the signal engine
and UI know this came from 5-min data.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

import config
from patterns.reversal_patterns import PatternResult

logger = logging.getLogger(__name__)

# IST timezone constant
_IST = timezone(timedelta(hours=5, minutes=30))

# NSE session open in seconds-from-midnight IST
_NSE_OPEN_SOD = 9 * 3600 + 15 * 60   # 09:15 = 33 300 s


def _ist_sod(unix_ts: int) -> int:
    """Return seconds-from-midnight IST for a unix timestamp."""
    dt = datetime.fromtimestamp(unix_ts, tz=_IST)
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def _today_ist() -> datetime:
    return datetime.now(_IST).replace(hour=0, minute=0, second=0, microsecond=0)


# ── ORB ──────────────────────────────────────────────────────────────────────

def detect_orb(
    df_5min: pd.DataFrame,
    regime_15min: str = "",
    orb_candles: int = None,
) -> List[PatternResult]:
    """Detect Opening Range Breakout on 5-min OHLCV.

    Args:
        df_5min    : 5-min OHLCV DataFrame with a 'timestamp' column (unix int)
                     OR a DatetimeIndex.  At least ORB_CANDLES + 2 bars required.
        regime_15min: current 15-min regime string (e.g. 'TRENDING_UP').
                      Used only for confidence boost — does NOT block the signal.
        orb_candles: override for config.ORB_CANDLES (used in tests).

    Returns:
        List of PatternResult (0, 1 or 2 items — BUY and/or SELL).
        Empty list if ORB period is not yet complete or range is invalid.
    """
    n_orb    = orb_candles or getattr(config, "ORB_CANDLES", 6)
    vol_mult = getattr(config, "ORB_VOLUME_MULT",   1.3)
    min_rng  = getattr(config, "ORB_MIN_RANGE_PCT", 0.003)
    max_rng  = getattr(config, "ORB_MAX_RANGE_PCT", 0.04)

    if df_5min is None or len(df_5min) < n_orb + 2:
        return []

    # ── Resolve timestamp column ─────────────────────────────────────────────
    df = df_5min.copy().reset_index(drop=False)

    if "timestamp" in df.columns:
        ts_col = df["timestamp"].values.astype(int)
    elif isinstance(df_5min.index, pd.DatetimeIndex):
        ts_col = (df_5min.index.astype("int64") // 10 ** 9).values
        df["timestamp"] = ts_col
    else:
        logger.debug("[ORB] No timestamp column — cannot determine session start")
        return []

    # ── Isolate TODAY's session ───────────────────────────────────────────────
    # "Today" in IST; session starts at 09:15.
    today_ist_midnight_ts = int(_today_ist().timestamp())
    session_start_ts      = today_ist_midnight_ts + _NSE_OPEN_SOD

    # Bars from today's session (sorted oldest-first already)
    today_mask    = ts_col >= session_start_ts
    session_idx   = np.where(today_mask)[0]

    if len(session_idx) < n_orb + 1:
        # ORB period not yet complete OR all bars are from prior sessions
        logger.debug(f"[ORB] Only {len(session_idx)} session bars — ORB period incomplete")
        return []

    # First bar of today's session (offset into df)
    first_idx = session_idx[0]
    # Bars that make up the ORB period
    orb_end_idx   = first_idx + n_orb          # exclusive — first bar after ORB
    # Current (latest) bar position
    current_idx   = len(df) - 1

    if current_idx < orb_end_idx:
        # We're still inside the ORB formation window — no signal yet
        return []

    orb_slice   = df.iloc[first_idx : orb_end_idx]
    current_bar = df.iloc[current_idx]

    orb_high = float(orb_slice["high"].max())
    orb_low  = float(orb_slice["low"].min())
    orb_mid  = (orb_high + orb_low) / 2.0
    orb_rng  = orb_high - orb_low

    if orb_mid <= 0:
        return []

    rng_pct = orb_rng / orb_mid

    # ── Range sanity checks ───────────────────────────────────────────────────
    if rng_pct < min_rng:
        logger.debug(f"[ORB] Range too small: {rng_pct:.4f} < {min_rng}")
        return []
    if rng_pct > max_rng:
        logger.debug(f"[ORB] Range too large (gap/chaos): {rng_pct:.4f} > {max_rng}")
        return []

    # ── Volume comparison ─────────────────────────────────────────────────────
    orb_avg_vol  = float(orb_slice["volume"].mean()) or 1.0
    current_vol  = float(current_bar.get("volume", 0) or 0)
    vol_ok       = current_vol >= orb_avg_vol * vol_mult

    current_close = float(current_bar["close"])
    current_open  = float(current_bar["open"])

    # ── Regime confidence bonus ───────────────────────────────────────────────
    reg = (regime_15min or "").upper()
    regime_bonus_buy  = 0.10 if reg == "TRENDING_UP"   else \
                        0.05 if reg == "BREAKOUT"       else \
                       -0.05 if reg == "TRENDING_DOWN"  else 0.0
    regime_bonus_sell = 0.10 if reg == "TRENDING_DOWN"  else \
                        0.05 if reg == "BREAKOUT"        else \
                       -0.05 if reg == "TRENDING_UP"    else 0.0

    results: List[PatternResult] = []

    # ── BUY setup — close above ORB high ─────────────────────────────────────
    if current_close > orb_high:
        base_conf = 0.65 if vol_ok else 0.50
        conf      = min(0.90, base_conf + regime_bonus_buy)

        # Entry = ORB high (already broken), SL = ORB low, Target = 1× range above
        entry  = orb_high
        sl     = orb_low
        target = orb_high + orb_rng   # 1:1 measure rule

        # Drift check: don't signal if LTP has already run 2× the range past entry
        drift = (current_close - orb_high) / orb_rng if orb_rng > 0 else 0
        if drift > 2.0:
            logger.debug(f"[ORB] BUY suppressed — price drifted {drift:.1f}× range past ORB high")
        else:
            results.append(PatternResult(
                pattern_name       = "orb_breakout",
                variant            = "bullish",
                direction          = "bullish",
                confidence         = round(conf, 3),
                entry_price        = round(entry,  2),
                target_price       = round(target, 2),
                stop_loss          = round(sl,     2),
                neckline           = round(orb_high, 2),
                breakout_confirmed = True,
                volume_confirmed   = vol_ok,
                timeframe          = "5",
                start_index        = int(first_idx),
                end_index          = int(current_idx),
            ))
            logger.debug(
                f"[ORB] BUY detected | range={orb_low:.2f}–{orb_high:.2f} "
                f"({rng_pct*100:.2f}%) | vol_ok={vol_ok} | conf={conf:.2f}"
            )

    # ── SELL setup — close below ORB low ─────────────────────────────────────
    elif current_close < orb_low:
        base_conf = 0.65 if vol_ok else 0.50
        conf      = min(0.90, base_conf + regime_bonus_sell)

        entry  = orb_low
        sl     = orb_high
        target = orb_low - orb_rng   # 1:1 measure rule below

        drift = (orb_low - current_close) / orb_rng if orb_rng > 0 else 0
        if drift > 2.0:
            logger.debug(f"[ORB] SELL suppressed — price drifted {drift:.1f}× range past ORB low")
        else:
            results.append(PatternResult(
                pattern_name       = "orb_breakdown",
                variant            = "bearish",
                direction          = "bearish",
                confidence         = round(conf, 3),
                entry_price        = round(entry,  2),
                target_price       = round(target, 2),
                stop_loss          = round(sl,     2),
                neckline           = round(orb_low, 2),
                breakout_confirmed = True,
                volume_confirmed   = vol_ok,
                timeframe          = "5",
                start_index        = int(first_idx),
                end_index          = int(current_idx),
            ))
            logger.debug(
                f"[ORB] SELL detected | range={orb_low:.2f}–{orb_high:.2f} "
                f"({rng_pct*100:.2f}%) | vol_ok={vol_ok} | conf={conf:.2f}"
            )

    return results


# ── VWAP helpers ─────────────────────────────────────────────────────────────

def _compute_session_vwap(df: pd.DataFrame, session_start_ts: int) -> Optional[float]:
    """Compute session VWAP from session-open bar to the latest bar.

    Typical price = (high + low + close) / 3.
    VWAP = Σ(tp × volume) / Σ(volume)

    Args:
        df              : DataFrame that already has a 'timestamp' column (int unix).
        session_start_ts: unix timestamp of today's 09:15 IST open.

    Returns:
        Current VWAP float, or None if insufficient session data or zero volume.
    """
    mask = df["timestamp"].values.astype(int) >= session_start_ts
    session_df = df[mask]
    if len(session_df) < 2:
        return None
    tp  = (session_df["high"] + session_df["low"] + session_df["close"]) / 3.0
    vol = session_df["volume"].astype(float)
    total_vol = float(vol.sum())
    if total_vol <= 0:
        return None
    return float((tp * vol).sum() / total_vol)


def _resolve_df_ts(df_5min: pd.DataFrame):
    """Return (df_with_ts_col, ts_array) or (None, None) on failure."""
    df = df_5min.copy().reset_index(drop=False)
    if "timestamp" in df.columns:
        ts_col = df["timestamp"].values.astype(int)
    elif isinstance(df_5min.index, pd.DatetimeIndex):
        ts_col = (df_5min.index.astype("int64") // 10 ** 9).values
        df["timestamp"] = ts_col
    else:
        return None, None
    return df, ts_col


# ── VWAP Reclaim (Bullish) ────────────────────────────────────────────────────

def detect_vwap_reclaim(
    df_5min: pd.DataFrame,
    regime_15min: str = "",
) -> List[PatternResult]:
    """Detect VWAP Reclaim (Bullish) on 5-min OHLCV.

    Setup:
      • Price was BELOW session VWAP for ≥ 1 bars, then current bar closes
        ABOVE VWAP with volume ≥ session average.

    Entry  : VWAP level
    SL     : Lowest low of the preceding 10 bars
    Target : VWAP + (VWAP − SL)   [1:1 measured move]

    Confidence:
      base  0.55 with volume  /  0.42 without
      +0.10 if regime is TRENDING_UP or BREAKOUT
      −0.05 if regime is TRENDING_DOWN
      cap   0.85
    """
    if df_5min is None or len(df_5min) < 15:
        return []

    df, ts_col = _resolve_df_ts(df_5min)
    if df is None:
        return []

    today_ist_midnight_ts = int(_today_ist().timestamp())
    session_start_ts      = today_ist_midnight_ts + _NSE_OPEN_SOD
    session_mask          = ts_col >= session_start_ts

    if session_mask.sum() < 5:
        return []

    vwap = _compute_session_vwap(df, session_start_ts)
    if vwap is None or vwap <= 0:
        return []

    current_idx   = len(df) - 1
    if current_idx < 2:
        return []

    current_bar   = df.iloc[current_idx]
    current_close = float(current_bar["close"])

    # Current bar must close ABOVE VWAP
    if current_close <= vwap:
        return []

    # Immediately preceding bar must have closed BELOW VWAP (the "reclaim")
    if float(df.iloc[current_idx - 1]["close"]) >= vwap:
        return []

    # SL = lowest low of preceding 10 bars (before current)
    lookback_slice = df.iloc[max(0, current_idx - 10): current_idx]
    sl = float(lookback_slice["low"].min())
    if sl >= vwap:
        return []  # inverted SL

    target = vwap + (vwap - sl)

    # Volume: at or above session average
    avg_vol     = float(df[session_mask]["volume"].astype(float).mean()) or 1.0
    current_vol = float(current_bar.get("volume", 0) or 0)
    vol_ok      = current_vol >= avg_vol

    reg = (regime_15min or "").upper()
    regime_bonus = (0.10 if reg in ("TRENDING_UP", "BREAKOUT") else
                   -0.05 if reg == "TRENDING_DOWN" else 0.0)

    conf = min(0.85, (0.55 if vol_ok else 0.42) + regime_bonus)

    session_start_i = int(np.where(session_mask)[0][0])

    logger.debug(
        f"[VWAP_RECLAIM] VWAP={vwap:.2f} close={current_close:.2f} "
        f"sl={sl:.2f} vol_ok={vol_ok} conf={conf:.2f}"
    )
    return [PatternResult(
        pattern_name       = "vwap_reclaim",
        variant            = "bullish",
        direction          = "bullish",
        confidence         = round(conf, 3),
        entry_price        = round(vwap,   2),
        target_price       = round(target, 2),
        stop_loss          = round(sl,     2),
        neckline           = round(vwap,   2),
        breakout_confirmed = True,
        volume_confirmed   = vol_ok,
        timeframe          = "5",
        start_index        = session_start_i,
        end_index          = current_idx,
    )]


# ── VWAP Rejection (Bearish) ──────────────────────────────────────────────────

def detect_vwap_rejection(
    df_5min: pd.DataFrame,
    regime_15min: str = "",
) -> List[PatternResult]:
    """Detect VWAP Rejection (Bearish) on 5-min OHLCV.

    Setup:
      • Price was ABOVE session VWAP, then current bar closes BELOW VWAP
        (the rejection bar) with volume ≥ session average.

    Entry  : VWAP level
    SL     : Highest high of the preceding 10 bars
    Target : VWAP − (SL − VWAP)   [1:1 measured move]

    Confidence:
      base  0.55 with volume  /  0.42 without
      +0.10 if regime is TRENDING_DOWN
      +0.05 if regime is BREAKOUT
      −0.05 if regime is TRENDING_UP
      cap   0.85
    """
    if df_5min is None or len(df_5min) < 15:
        return []

    df, ts_col = _resolve_df_ts(df_5min)
    if df is None:
        return []

    today_ist_midnight_ts = int(_today_ist().timestamp())
    session_start_ts      = today_ist_midnight_ts + _NSE_OPEN_SOD
    session_mask          = ts_col >= session_start_ts

    if session_mask.sum() < 5:
        return []

    vwap = _compute_session_vwap(df, session_start_ts)
    if vwap is None or vwap <= 0:
        return []

    current_idx   = len(df) - 1
    if current_idx < 2:
        return []

    current_bar   = df.iloc[current_idx]
    current_close = float(current_bar["close"])

    # Current bar must close BELOW VWAP
    if current_close >= vwap:
        return []

    # Preceding bar must have closed ABOVE VWAP (the "rejection")
    if float(df.iloc[current_idx - 1]["close"]) <= vwap:
        return []

    # SL = highest high of preceding 10 bars
    lookback_slice = df.iloc[max(0, current_idx - 10): current_idx]
    sl = float(lookback_slice["high"].max())
    if sl <= vwap:
        return []  # inverted SL

    target = vwap - (sl - vwap)

    avg_vol     = float(df[session_mask]["volume"].astype(float).mean()) or 1.0
    current_vol = float(current_bar.get("volume", 0) or 0)
    vol_ok      = current_vol >= avg_vol

    reg = (regime_15min or "").upper()
    regime_bonus = (0.10 if reg == "TRENDING_DOWN"             else
                    0.05 if reg == "BREAKOUT"                  else
                   -0.05 if reg == "TRENDING_UP"               else 0.0)

    conf = min(0.85, (0.55 if vol_ok else 0.42) + regime_bonus)

    session_start_i = int(np.where(session_mask)[0][0])

    logger.debug(
        f"[VWAP_REJECT] VWAP={vwap:.2f} close={current_close:.2f} "
        f"sl={sl:.2f} vol_ok={vol_ok} conf={conf:.2f}"
    )
    return [PatternResult(
        pattern_name       = "vwap_rejection",
        variant            = "bearish",
        direction          = "bearish",
        confidence         = round(conf, 3),
        entry_price        = round(vwap,   2),
        target_price       = round(target, 2),
        stop_loss          = round(sl,     2),
        neckline           = round(vwap,   2),
        breakout_confirmed = True,
        volume_confirmed   = vol_ok,
        timeframe          = "5",
        start_index        = session_start_i,
        end_index          = current_idx,
    )]


# ── ABCD Harmonic ─────────────────────────────────────────────────────────────

def _find_5min_pivots(df: pd.DataFrame, lookback: int = 2) -> list:
    """Return pivot highs/lows on 5-min data.

    Each entry is (bar_index, price, 'H' | 'L').
    Uses a simple fractal pivot: bar must be strictly ≥ (highs) or ≤ (lows)
    on both sides within *lookback* bars.
    """
    pivots = []
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    n      = len(df)

    for i in range(lookback, n - lookback):
        is_pivot_high = all(highs[i] >= highs[i - j] for j in range(1, lookback + 1)) and \
                        all(highs[i] >= highs[i + j] for j in range(1, lookback + 1))
        is_pivot_low  = all(lows[i]  <= lows[i  - j] for j in range(1, lookback + 1)) and \
                        all(lows[i]  <= lows[i  + j] for j in range(1, lookback + 1))

        if is_pivot_high:
            pivots.append((i, highs[i], "H"))
        elif is_pivot_low:
            pivots.append((i, lows[i],  "L"))

    return pivots


def detect_abcd(
    df_5min: pd.DataFrame,
    regime_15min: str = "",
) -> List[PatternResult]:
    """Detect ABCD Harmonic pattern on 5-min OHLCV.

    Bullish ABCD  —  A(H) → B(L) → C(H) → D(L)
      BC/AB retracement : 0.50 – 0.886
      CD/AB extension   : 0.786 – 1.272
      Entry LONG at D,  SL = D − 0.10×AB,  Target = C (previous swing high)

    Bearish ABCD  —  A(L) → B(H) → C(L) → D(H)
      Same ratios, Entry SHORT at D, SL = D + 0.10×AB, Target = C (previous swing low)

    D pivot must be within the last 5 bars.

    Confidence:
      base  0.58
      +0.10 if regime aligns (TRENDING_UP / MEAN_REVERSION for bullish;
                               TRENDING_DOWN / MEAN_REVERSION for bearish)
      cap   0.85
    """
    if df_5min is None or len(df_5min) < 20:
        return []

    df, _ = _resolve_df_ts(df_5min)
    if df is None:
        df = df_5min.copy().reset_index(drop=False)

    pivots = _find_5min_pivots(df, lookback=2)
    if len(pivots) < 4:
        return []

    current_idx    = len(df) - 1
    recent_pivots  = pivots[-24:]          # cap search to last 24 pivots
    results: List[PatternResult] = []
    reg = (regime_15min or "").upper()

    for i in range(len(recent_pivots) - 3):
        a_idx, a_price, a_type = recent_pivots[i]
        b_idx, b_price, b_type = recent_pivots[i + 1]
        c_idx, c_price, c_type = recent_pivots[i + 2]
        d_idx, d_price, d_type = recent_pivots[i + 3]

        # Must alternate H–L–H–L or L–H–L–H
        types = [a_type, b_type, c_type, d_type]
        if not all(types[j] != types[j + 1] for j in range(3)):
            continue

        # D must be recent (within last 5 bars from current)
        if current_idx - d_idx > 5:
            continue

        # ── Bullish: A(H) B(L) C(H) D(L) ────────────────────────────────────
        if a_type == "H" and b_type == "L" and c_type == "H" and d_type == "L":
            ab = a_price - b_price        # downleg AB
            if ab <= 0 or c_price <= d_price:
                continue
            bc    = c_price - b_price     # retracement up
            cd    = c_price - d_price     # extension down
            bc_ab = bc / ab
            cd_ab = cd / ab

            if not (0.50 <= bc_ab <= 0.886 and 0.786 <= cd_ab <= 1.272):
                continue

            entry  = d_price
            sl     = d_price - 0.10 * ab
            target = c_price              # recover back to previous swing high

            if target <= entry or sl >= entry:
                continue

            bonus = (0.10 if reg in ("TRENDING_UP", "MEAN_REVERSION") else
                    -0.05 if reg == "TRENDING_DOWN" else 0.0)
            conf  = min(0.85, 0.58 + bonus)

            results.append(PatternResult(
                pattern_name       = "abcd_harmonic",
                variant            = "bullish",
                direction          = "bullish",
                confidence         = round(conf, 3),
                entry_price        = round(entry,  2),
                target_price       = round(target, 2),
                stop_loss          = round(sl,     2),
                neckline           = round(c_price, 2),
                breakout_confirmed = False,
                volume_confirmed   = False,
                timeframe          = "5",
                start_index        = int(a_idx),
                end_index          = int(d_idx),
            ))
            logger.debug(
                f"[ABCD] Bullish A={a_price:.2f} B={b_price:.2f} "
                f"C={c_price:.2f} D={d_price:.2f} | "
                f"BC/AB={bc_ab:.3f} CD/AB={cd_ab:.3f} conf={conf:.2f}"
            )

        # ── Bearish: A(L) B(H) C(L) D(H) ────────────────────────────────────
        elif a_type == "L" and b_type == "H" and c_type == "L" and d_type == "H":
            ab = b_price - a_price        # upleg AB
            if ab <= 0 or c_price >= d_price:
                continue
            bc    = b_price - c_price     # retracement down
            cd    = d_price - c_price     # extension up
            bc_ab = bc / ab
            cd_ab = cd / ab

            if not (0.50 <= bc_ab <= 0.886 and 0.786 <= cd_ab <= 1.272):
                continue

            entry  = d_price
            sl     = d_price + 0.10 * ab
            target = c_price              # fall back to previous swing low

            if target >= entry or sl <= entry:
                continue

            bonus = (0.10 if reg in ("TRENDING_DOWN", "MEAN_REVERSION") else
                    -0.05 if reg == "TRENDING_UP" else 0.0)
            conf  = min(0.85, 0.58 + bonus)

            results.append(PatternResult(
                pattern_name       = "abcd_harmonic",
                variant            = "bearish",
                direction          = "bearish",
                confidence         = round(conf, 3),
                entry_price        = round(entry,  2),
                target_price       = round(target, 2),
                stop_loss          = round(sl,     2),
                neckline           = round(c_price, 2),
                breakout_confirmed = False,
                volume_confirmed   = False,
                timeframe          = "5",
                start_index        = int(a_idx),
                end_index          = int(d_idx),
            ))
            logger.debug(
                f"[ABCD] Bearish A={a_price:.2f} B={b_price:.2f} "
                f"C={c_price:.2f} D={d_price:.2f} | "
                f"BC/AB={bc_ab:.3f} CD/AB={cd_ab:.3f} conf={conf:.2f}"
            )

    if not results:
        return []
    # Return only the highest-confidence result (avoid duplicate signals)
    results.sort(key=lambda p: p.confidence, reverse=True)
    return [results[0]]


# ── NR7 Narrow Range ─────────────────────────────────────────────────────────

def detect_nr7(
    df_5min: pd.DataFrame,
    regime_15min: str = "",
    nr_bars: int = 7,
) -> List[PatternResult]:
    """Detect NR7 Narrow Range compression breakout on 5-min OHLCV.

    Setup:
      • Bar[-2] (the "NR7 bar") has the narrowest high−low range of the last
        N bars (default N = 7).
      • Bar[-1] (current) closes ABOVE NR7 high → BUY breakout.
        Bar[-1] (current) closes BELOW NR7 low  → SELL breakdown.

    Entry  : NR7 bar's high (buy) or low (sell)
    SL     : NR7 bar's low (buy) or high (sell)
    Target : Entry ± 2 × NR7 range   [2:1 measure]

    Confidence:
      base  0.58 with volume surge (≥ 1.2× 10-bar avg) / 0.45 without
      +0.08 if regime aligns (TRENDING_UP / BREAKOUT for buy;
                               TRENDING_DOWN / BREAKOUT for sell)
      −0.05 if regime opposes
      cap   0.82
    """
    if df_5min is None or len(df_5min) < nr_bars + 2:
        return []

    df, _ = _resolve_df_ts(df_5min)
    if df is None:
        df = df_5min.copy().reset_index(drop=False)

    current_idx = len(df) - 1
    nr7_idx     = current_idx - 1         # the NR7 candidate bar

    if nr7_idx < nr_bars - 1:
        return []

    # 7-bar window ending at nr7_idx (inclusive)
    window_slice = df.iloc[nr7_idx - nr_bars + 1: nr7_idx + 1]
    all_ranges   = (window_slice["high"] - window_slice["low"]).values.astype(float)
    nr7_range    = float(df.iloc[nr7_idx]["high"] - df.iloc[nr7_idx]["low"])

    if nr7_range <= 0:
        return []

    # The NR7 bar is the last bar in the window; its range must be the minimum
    if nr7_range > float(all_ranges.min()) + 1e-8:
        return []

    nr7_high = float(df.iloc[nr7_idx]["high"])
    nr7_low  = float(df.iloc[nr7_idx]["low"])

    current_bar   = df.iloc[current_idx]
    current_close = float(current_bar["close"])

    # Volume vs 10-bar rolling average (excluding current bar)
    vol_window  = df.iloc[max(0, current_idx - 10): current_idx]
    avg_vol     = float(vol_window["volume"].astype(float).mean()) or 1.0
    current_vol = float(current_bar.get("volume", 0) or 0)
    vol_ok      = current_vol >= avg_vol * 1.2

    reg = (regime_15min or "").upper()
    results: List[PatternResult] = []

    # ── BUY: close above NR7 high ────────────────────────────────────────────
    if current_close > nr7_high:
        bonus = (0.08 if reg in ("TRENDING_UP", "BREAKOUT") else
                -0.05 if reg == "TRENDING_DOWN" else 0.0)
        conf   = min(0.82, (0.58 if vol_ok else 0.45) + bonus)
        target = nr7_high + 2.0 * nr7_range

        results.append(PatternResult(
            pattern_name       = "nr7_breakout",
            variant            = "bullish",
            direction          = "bullish",
            confidence         = round(conf, 3),
            entry_price        = round(nr7_high, 2),
            target_price       = round(target,   2),
            stop_loss          = round(nr7_low,  2),
            neckline           = round(nr7_high, 2),
            breakout_confirmed = True,
            volume_confirmed   = vol_ok,
            timeframe          = "5",
            start_index        = int(nr7_idx),
            end_index          = int(current_idx),
        ))
        logger.debug(
            f"[NR7] BUY nr7={nr7_low:.2f}–{nr7_high:.2f} range={nr7_range:.2f} "
            f"close={current_close:.2f} vol_ok={vol_ok} conf={conf:.2f}"
        )

    # ── SELL: close below NR7 low ────────────────────────────────────────────
    elif current_close < nr7_low:
        bonus = (0.08 if reg in ("TRENDING_DOWN", "BREAKOUT") else
                -0.05 if reg == "TRENDING_UP" else 0.0)
        conf   = min(0.82, (0.58 if vol_ok else 0.45) + bonus)
        target = nr7_low - 2.0 * nr7_range

        results.append(PatternResult(
            pattern_name       = "nr7_breakdown",
            variant            = "bearish",
            direction          = "bearish",
            confidence         = round(conf, 3),
            entry_price        = round(nr7_low,  2),
            target_price       = round(target,   2),
            stop_loss          = round(nr7_high, 2),
            neckline           = round(nr7_low,  2),
            breakout_confirmed = True,
            volume_confirmed   = vol_ok,
            timeframe          = "5",
            start_index        = int(nr7_idx),
            end_index          = int(current_idx),
        ))
        logger.debug(
            f"[NR7] SELL nr7={nr7_low:.2f}–{nr7_high:.2f} range={nr7_range:.2f} "
            f"close={current_close:.2f} vol_ok={vol_ok} conf={conf:.2f}"
        )

    return results


# ── Inside Bar ────────────────────────────────────────────────────────────────

def detect_inside_bar(
    df_5min: pd.DataFrame,
    regime_15min: str = "",
) -> List[PatternResult]:
    """Detect Inside Bar compression breakout on 5-min OHLCV.

    Pattern structure (3-bar sequence):
      Bar[-3]  — ignored (context)
      Bar[-2]  — Mother Bar  (the wide reference bar)
      Bar[-1]  — Inside Bar  : high < mother_high  AND  low > mother_low
      Bar[ 0]  — Current     : closes ABOVE mother_high → BUY
                               closes BELOW mother_low  → SELL

    Entry  : Mother bar high (buy) or low (sell)
    SL     : Mother bar low  (buy) or high (sell)
    Target : Entry ± 1 × mother range

    Confidence:
      base  0.60 with volume surge (≥ 1.2× 10-bar avg) / 0.48 without
      +0.08 if regime aligns
      −0.05 if regime opposes
      cap   0.85
    """
    if df_5min is None or len(df_5min) < 4:
        return []

    df, _ = _resolve_df_ts(df_5min)
    if df is None:
        df = df_5min.copy().reset_index(drop=False)

    current_idx = len(df) - 1
    if current_idx < 2:
        return []

    current_bar = df.iloc[current_idx]
    inside_bar  = df.iloc[current_idx - 1]
    mother_bar  = df.iloc[current_idx - 2]

    mother_high  = float(mother_bar["high"])
    mother_low   = float(mother_bar["low"])
    mother_range = mother_high - mother_low

    if mother_range <= 0:
        return []

    # Confirm inside bar condition: fully contained within mother bar
    if not (float(inside_bar["high"]) < mother_high and
            float(inside_bar["low"])  > mother_low):
        return []

    current_close = float(current_bar["close"])

    # Volume: current bar vs 10-bar average
    vol_window  = df.iloc[max(0, current_idx - 10): current_idx]
    avg_vol     = float(vol_window["volume"].astype(float).mean()) or 1.0
    current_vol = float(current_bar.get("volume", 0) or 0)
    vol_ok      = current_vol >= avg_vol * 1.2

    reg = (regime_15min or "").upper()
    results: List[PatternResult] = []

    # ── BUY: close above mother high ─────────────────────────────────────────
    if current_close > mother_high:
        bonus = (0.08 if reg in ("TRENDING_UP", "BREAKOUT") else
                -0.05 if reg == "TRENDING_DOWN" else 0.0)
        conf   = min(0.85, (0.60 if vol_ok else 0.48) + bonus)
        target = mother_high + mother_range

        results.append(PatternResult(
            pattern_name       = "inside_bar_breakout",
            variant            = "bullish",
            direction          = "bullish",
            confidence         = round(conf, 3),
            entry_price        = round(mother_high,         2),
            target_price       = round(target,              2),
            stop_loss          = round(mother_low,          2),
            neckline           = round(mother_high,         2),
            breakout_confirmed = True,
            volume_confirmed   = vol_ok,
            timeframe          = "5",
            start_index        = int(current_idx - 2),
            end_index          = int(current_idx),
        ))
        logger.debug(
            f"[INSIDE_BAR] BUY mother={mother_low:.2f}–{mother_high:.2f} "
            f"close={current_close:.2f} vol_ok={vol_ok} conf={conf:.2f}"
        )

    # ── SELL: close below mother low ─────────────────────────────────────────
    elif current_close < mother_low:
        bonus = (0.08 if reg in ("TRENDING_DOWN", "BREAKOUT") else
                -0.05 if reg == "TRENDING_UP" else 0.0)
        conf   = min(0.85, (0.60 if vol_ok else 0.48) + bonus)
        target = mother_low - mother_range

        results.append(PatternResult(
            pattern_name       = "inside_bar_breakdown",
            variant            = "bearish",
            direction          = "bearish",
            confidence         = round(conf, 3),
            entry_price        = round(mother_low,          2),
            target_price       = round(target,              2),
            stop_loss          = round(mother_high,         2),
            neckline           = round(mother_low,          2),
            breakout_confirmed = True,
            volume_confirmed   = vol_ok,
            timeframe          = "5",
            start_index        = int(current_idx - 2),
            end_index          = int(current_idx),
        ))
        logger.debug(
            f"[INSIDE_BAR] SELL mother={mother_low:.2f}–{mother_high:.2f} "
            f"close={current_close:.2f} vol_ok={vol_ok} conf={conf:.2f}"
        )

    return results


# ── Public dispatcher ─────────────────────────────────────────────────────────

def detect_all_intraday(
    df_5min: pd.DataFrame,
    regime_15min: str = "",
) -> List[PatternResult]:
    """Run all intraday pattern detectors on 5-min data.

    Detectors (6 patterns, up to 2 signals each):
      1. ORB          — Opening Range Breakout / Breakdown
      2. VWAP Reclaim — bullish VWAP cross after session dip
      3. VWAP Reject  — bearish VWAP cross after session rally
      4. ABCD Harmonic— 4-leg alternating harmonic (bullish & bearish)
      5. NR7          — narrowest range of last 7 bars → breakout
      6. Inside Bar   — compressed bar inside mother bar → breakout

    Args:
        df_5min      : 5-min OHLCV DataFrame (index = datetime or plain int)
        regime_15min : regime string from 15-min detector, used for
                       confidence bonuses (e.g. 'TRENDING_UP')

    Returns:
        List of PatternResult sorted by confidence descending.
        Each PatternResult has timeframe="5".
    """
    if df_5min is None or len(df_5min) < 10:
        return []

    patterns: List[PatternResult] = []

    _detectors = [
        ("ORB",          detect_orb,            {}),
        ("VWAP_RECLAIM", detect_vwap_reclaim,   {}),
        ("VWAP_REJECT",  detect_vwap_rejection, {}),
        ("ABCD",         detect_abcd,           {}),
        ("NR7",          detect_nr7,            {}),
        ("INSIDE_BAR",   detect_inside_bar,     {}),
    ]

    for name, fn, extra_kwargs in _detectors:
        try:
            found = fn(df_5min, regime_15min=regime_15min, **extra_kwargs)
            patterns.extend(found)
        except Exception as exc:
            logger.warning(f"[intraday] {name} detector error: {exc}")

    patterns.sort(key=lambda p: p.confidence, reverse=True)
    return patterns
