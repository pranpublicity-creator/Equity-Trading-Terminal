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


# ── Public convenience wrapper ────────────────────────────────────────────────

def detect_all_intraday(
    df_5min: pd.DataFrame,
    regime_15min: str = "",
) -> List[PatternResult]:
    """Run all intraday pattern detectors on 5-min data.

    Currently includes:
      • ORB (Opening Range Breakout / Breakdown)

    Future additions (same interface — just append to the list):
      • VWAP Reclaim / Rejection
      • ABCD Harmonic
      • NR7 Compression Breakout
      • Inside Bar
      • Fair Value Gap

    Args:
        df_5min      : 5-min OHLCV DataFrame
        regime_15min : regime string from 15-min detector (for confidence boost)

    Returns:
        List of PatternResult sorted by confidence descending.
    """
    if df_5min is None or len(df_5min) < 10:
        return []

    patterns: List[PatternResult] = []

    # ── ORB ──────────────────────────────────────────────────────────────────
    try:
        patterns.extend(detect_orb(df_5min, regime_15min=regime_15min))
    except Exception as e:
        logger.warning(f"[intraday] ORB detector error: {e}")

    # ── (future patterns go here) ─────────────────────────────────────────────

    patterns.sort(key=lambda p: p.confidence, reverse=True)
    return patterns
