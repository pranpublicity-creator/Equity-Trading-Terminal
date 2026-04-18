"""
Market Regime Detector
Identifies 7 equity market regimes from technical indicators.
Used by signal fusion and adaptive bot for regime-aware weight adjustment.
"""
import numpy as np
import pandas as pd


# 7 Equity Regimes
REGIMES = [
    "TRENDING_UP",
    "TRENDING_DOWN",
    "MEAN_REVERTING",
    "VOLATILE",
    "BREAKOUT",
    "CONSOLIDATION",
    "MOMENTUM",
]


def detect_regime(df: pd.DataFrame, indicators: dict = None) -> dict:
    """Detect the current market regime from OHLCV data and indicators.

    Args:
        df: OHLCV DataFrame with indicator columns
        indicators: pre-computed indicator dict (optional, extracted from df if None)

    Returns:
        dict with regime name, scores for each regime, and confidence
    """
    if len(df) < 50:
        return {"regime": "CONSOLIDATION", "confidence": 0.3, "scores": {}}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # Extract indicators from DataFrame columns or compute simple versions
    adx = _get_col(df, "adx", _compute_simple_adx(high, low, close))
    plus_di = _get_col(df, "plus_di", None)
    minus_di = _get_col(df, "minus_di", None)
    rsi = _get_col(df, "rsi", _compute_simple_rsi(close))
    atr = _get_col(df, "atr", _compute_simple_atr(high, low, close))
    ema20 = _get_col(df, "ema_20", _ema(close, 20))
    ema50 = _get_col(df, "ema_50", _ema(close, 50))
    ema200 = _get_col(df, "ema_200", _ema(close, 200))
    bb_width = _get_col(df, "bb_width", None)
    macd_hist = _get_col(df, "macd_hist", None)
    roc = _get_col(df, "roc", None)

    # Current values (latest bar)
    cur_adx = _last(adx, 15.0)
    cur_rsi = _last(rsi, 50.0)
    cur_ema20 = _last(ema20, close[-1])
    cur_ema50 = _last(ema50, close[-1])
    cur_ema200 = _last(ema200, close[-1])
    cur_atr = _last(atr, 0.0)
    cur_plus_di = _last(plus_di, 0.0)
    cur_minus_di = _last(minus_di, 0.0)

    # ATR ratio vs 30-bar average
    atr_30_avg = float(np.mean(atr[-30:])) if atr is not None and len(atr) >= 30 else cur_atr
    atr_ratio = cur_atr / atr_30_avg if atr_30_avg > 0 else 1.0

    # Score each regime (0 to 1)
    scores = {}

    # TRENDING_UP: ADX > 25, +DI > -DI, EMA20 > EMA50 > EMA200
    trend_up_score = 0.0
    if cur_adx > 25:
        trend_up_score += 0.30
    if cur_plus_di > cur_minus_di:
        trend_up_score += 0.20
    if cur_ema20 > cur_ema50:
        trend_up_score += 0.25
    if cur_ema50 > cur_ema200:
        trend_up_score += 0.25
    scores["TRENDING_UP"] = trend_up_score

    # TRENDING_DOWN: ADX > 25, -DI > +DI, EMA20 < EMA50 < EMA200
    trend_down_score = 0.0
    if cur_adx > 25:
        trend_down_score += 0.30
    if cur_minus_di > cur_plus_di:
        trend_down_score += 0.20
    if cur_ema20 < cur_ema50:
        trend_down_score += 0.25
    if cur_ema50 < cur_ema200:
        trend_down_score += 0.25
    scores["TRENDING_DOWN"] = trend_down_score

    # MEAN_REVERTING: ADX < 20, RSI oscillating 40-60, narrow Bollinger
    mr_score = 0.0
    if cur_adx < 20:
        mr_score += 0.35
    if 40 <= cur_rsi <= 60:
        mr_score += 0.30
    if bb_width is not None:
        cur_bb = _last(bb_width, 0.05)
        if cur_bb < 0.04:
            mr_score += 0.35
    else:
        if atr_ratio < 0.8:
            mr_score += 0.35
    scores["MEAN_REVERTING"] = mr_score

    # VOLATILE: ATR > 2x 30-day avg, ADX > 30
    volatile_score = 0.0
    if atr_ratio > 2.0:
        volatile_score += 0.50
    elif atr_ratio > 1.5:
        volatile_score += 0.30
    if cur_adx > 30:
        volatile_score += 0.30
    # Wide daily range
    daily_range = (high[-1] - low[-1]) / close[-1] if close[-1] > 0 else 0
    if daily_range > 0.03:
        volatile_score += 0.20
    scores["VOLATILE"] = min(volatile_score, 1.0)

    # BREAKOUT: price breaking key level, volume spike, ADX rising from < 20
    breakout_score = 0.0
    # ADX was low and is now rising
    if adx is not None and len(adx) >= 10:
        adx_5_ago = float(adx[-5]) if len(adx) >= 5 else cur_adx
        if adx_5_ago < 20 and cur_adx > adx_5_ago + 5:
            breakout_score += 0.40
    # Price breaking above recent high or below recent low
    if len(close) >= 20:
        recent_high = float(np.max(high[-20:-1]))
        recent_low = float(np.min(low[-20:-1]))
        if close[-1] > recent_high:
            breakout_score += 0.30
        elif close[-1] < recent_low:
            breakout_score += 0.30
    # Volume spike
    if "volume" in df.columns:
        vol = df["volume"].values
        vol_avg = float(np.mean(vol[-20:])) if len(vol) >= 20 else float(vol[-1])
        if vol_avg > 0 and vol[-1] > vol_avg * 1.5:
            breakout_score += 0.30
    scores["BREAKOUT"] = min(breakout_score, 1.0)

    # CONSOLIDATION: ADX < 15, narrow range, low volume
    consol_score = 0.0
    if cur_adx < 15:
        consol_score += 0.40
    if atr_ratio < 0.7:
        consol_score += 0.30
    if "volume" in df.columns:
        vol = df["volume"].values
        vol_avg_20 = float(np.mean(vol[-20:])) if len(vol) >= 20 else float(vol[-1])
        vol_avg_5 = float(np.mean(vol[-5:])) if len(vol) >= 5 else float(vol[-1])
        if vol_avg_20 > 0 and vol_avg_5 < vol_avg_20 * 0.8:
            consol_score += 0.30
    scores["CONSOLIDATION"] = min(consol_score, 1.0)

    # MOMENTUM: RSI > 70 or < 30, strong MACD histogram, high ROC
    momentum_score = 0.0
    if cur_rsi > 70 or cur_rsi < 30:
        momentum_score += 0.35
    if macd_hist is not None:
        cur_macd_h = _last(macd_hist, 0.0)
        if abs(cur_macd_h) > 0:
            momentum_score += 0.30
    if roc is not None:
        cur_roc = _last(roc, 0.0)
        if abs(cur_roc) > 3:
            momentum_score += 0.35
    else:
        roc_5 = (close[-1] / close[-5] - 1) * 100 if len(close) >= 5 and close[-5] > 0 else 0
        if abs(roc_5) > 3:
            momentum_score += 0.35
    scores["MOMENTUM"] = min(momentum_score, 1.0)

    # Pick highest scoring regime
    best_regime = max(scores, key=scores.get)
    best_score = scores[best_regime]

    return {
        "regime": best_regime,
        "confidence": round(best_score, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
    }


def _get_col(df, col_name, fallback):
    """Get column values from DataFrame, or use fallback."""
    if col_name in df.columns:
        vals = df[col_name].values
        return vals[~np.isnan(vals)] if len(vals) > 0 else fallback
    return fallback


def _last(arr, default=0.0):
    """Get last non-NaN value from array."""
    if arr is None or len(arr) == 0:
        return default
    val = float(arr[-1])
    return val if not np.isnan(val) else default


def _ema(data, period):
    if len(data) < period:
        return data
    alpha = 2 / (period + 1)
    result = np.zeros(len(data))
    result[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def _compute_simple_rsi(close, period=14):
    if len(close) < period + 1:
        return np.full(len(close), 50.0)
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gains, np.ones(period) / period, mode="valid")
    avg_loss = np.convolve(losses, np.ones(period) / period, mode="valid")
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    pad = np.full(len(close) - len(rsi), 50.0)
    return np.concatenate([pad, rsi])


def _compute_simple_adx(high, low, close, period=14):
    if len(close) < period * 2:
        return np.full(len(close), 15.0)
    # Simplified ADX approximation
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr = np.convolve(tr, np.ones(period) / period, mode="valid")
    # Rough directional movement
    dm_plus = np.maximum(high[1:] - high[:-1], 0)
    dm_minus = np.maximum(low[:-1] - low[1:], 0)
    dm_plus[dm_plus < dm_minus] = 0
    dm_minus[dm_minus < dm_plus] = 0
    avg_dmp = np.convolve(dm_plus, np.ones(period) / period, mode="valid")
    avg_dmn = np.convolve(dm_minus, np.ones(period) / period, mode="valid")
    min_len = min(len(atr), len(avg_dmp), len(avg_dmn))
    di_plus = avg_dmp[:min_len] / np.maximum(atr[:min_len], 1e-10) * 100
    di_minus = avg_dmn[:min_len] / np.maximum(atr[:min_len], 1e-10) * 100
    dx = abs(di_plus - di_minus) / np.maximum(di_plus + di_minus, 1e-10) * 100
    adx = np.convolve(dx, np.ones(period) / period, mode="valid")
    pad = np.full(len(close) - len(adx), 15.0)
    return np.concatenate([pad, adx])


def _compute_simple_atr(high, low, close, period=14):
    if len(close) < period + 1:
        return np.full(len(close), 0.0)
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr = np.convolve(tr, np.ones(period) / period, mode="valid")
    pad = np.full(len(close) - len(atr), float(atr[0]) if len(atr) > 0 else 0.0)
    return np.concatenate([pad, atr])
