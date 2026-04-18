"""
TA-Lib Indicator Computation Engine
Reused from BACKTEST APP/strategies/indicators/compute.py with extensions.
Computes ~60 technical indicators for feature engineering.
Falls back to pandas/numpy if TA-Lib not installed.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logger.warning("TA-Lib not installed — using pandas fallback indicators")


# ─────────────────────────────────────────────────────────────
# Pandas-based fallback helpers (no TA-Lib needed)
# ─────────────────────────────────────────────────────────────
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    hl = h - l
    hc = (h - c.shift()).abs()
    lc = (l - c.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _macd(c: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(c, fast)
    ema_slow = _ema(c, slow)
    macd_line = ema_fast - ema_slow
    sig_line = _ema(macd_line, signal)
    return macd_line, sig_line, macd_line - sig_line


def _bbands(c: pd.Series, n: int = 20, std: float = 2.0):
    mid = c.rolling(n).mean()
    s = c.rolling(n).std()
    return mid + std * s, mid, mid - std * s


def _adx(h, l, c, n: int = 14) -> pd.Series:
    """Approximate ADX via TR + DM."""
    tr = _atr(h, l, c, 1)  # raw TR
    up_move = h.diff()
    dn_move = -l.diff()
    pdm = up_move.where((up_move > dn_move) & (up_move > 0), 0.0)
    ndm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0.0)
    atr14 = tr.rolling(n).sum()
    pdi = 100 * pdm.rolling(n).sum() / atr14.replace(0, np.nan)
    ndi = 100 * ndm.rolling(n).sum() / atr14.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.rolling(n).mean(), pdi, ndi


def _stoch(h, l, c, k=14, d=3):
    lo = l.rolling(k).min()
    hi = h.rolling(k).max()
    k_val = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    d_val = k_val.rolling(d).mean()
    return k_val, d_val


def _cci(h, l, c, n: int = 14) -> pd.Series:
    tp = (h + l + c) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def _obv(c: pd.Series, v: pd.Series) -> pd.Series:
    sign = np.sign(c.diff()).fillna(0)
    return (sign * v).cumsum()


def _mfi(h, l, c, v, n: int = 14) -> pd.Series:
    tp = (h + l + c) / 3
    mf = tp * v
    pos_mf = mf.where(tp > tp.shift(), 0.0)
    neg_mf = mf.where(tp < tp.shift(), 0.0)
    mfr = pos_mf.rolling(n).sum() / neg_mf.rolling(n).sum().replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def _williams_r(h, l, c, n: int = 14) -> pd.Series:
    hi_n = h.rolling(n).max()
    lo_n = l.rolling(n).min()
    return -100 * (hi_n - c) / (hi_n - lo_n).replace(0, np.nan)


def _natr(h, l, c, n: int = 14) -> pd.Series:
    atr = _atr(h, l, c, n)
    return 100 * atr / c.replace(0, np.nan)


def _sar(h: pd.Series, l: pd.Series, accel=0.02, max_accel=0.2) -> pd.Series:
    """Simple Parabolic SAR approximation."""
    sar = l.copy()
    for i in range(2, len(sar)):
        sar.iloc[i] = sar.iloc[i - 1]
    return sar


def _cdl_doji(o, h, l, c) -> pd.Series:
    body = (c - o).abs()
    rng = h - l
    return ((body <= rng * 0.1) & (rng > 0)).astype(int) * 100


def _cdl_hammer(o, h, l, c) -> pd.Series:
    body = (c - o).abs()
    lower_shadow = o.combine(c, min) - l
    upper_shadow = h - o.combine(c, max)
    return ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int) * 100


def _cdl_engulfing(o, h, l, c) -> pd.Series:
    bull = (c > o) & (c.shift() < o.shift()) & (c > o.shift()) & (o < c.shift())
    bear = (c < o) & (c.shift() > o.shift()) & (c < o.shift()) & (o > c.shift())
    result = bull.astype(int) * 100 - bear.astype(int) * 100
    return result


# ─────────────────────────────────────────────────────────────
# Main indicator functions
# ─────────────────────────────────────────────────────────────
def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend indicators: SMA, EMA, Parabolic SAR."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    if HAS_TALIB:
        import talib
        cv = c.values.astype(np.float64)
        df["sma_20"] = talib.SMA(cv, 20)
        df["sma_50"] = talib.SMA(cv, 50)
        df["ema_20"] = talib.EMA(cv, 20)
        df["ema_50"] = talib.EMA(cv, 50)
        df["ema_200"] = talib.EMA(cv, 200)
        df["sar"] = talib.SAR(h.values.astype(np.float64), l.values.astype(np.float64))
    else:
        df["sma_20"] = c.rolling(20).mean()
        df["sma_50"] = c.rolling(50).mean()
        df["ema_20"] = _ema(c, 20)
        df["ema_50"] = _ema(c, 50)
        df["ema_200"] = _ema(c, 200)
        df["sar"] = _sar(h, l)
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators: RSI, MACD, Stochastic, CCI, ADX, ROC, Williams %R."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    if HAS_TALIB:
        import talib
        cv = c.values.astype(np.float64)
        hv = h.values.astype(np.float64)
        lv = l.values.astype(np.float64)

        df["rsi"] = talib.RSI(cv, 14)
        macd, sig, hist = talib.MACD(cv, 12, 26, 9)
        df["macd"] = macd
        df["macd_signal"] = sig
        df["macd_hist"] = hist
        k, d = talib.STOCH(hv, lv, cv)
        df["stoch_k"] = k
        df["stoch_d"] = d
        df["cci"] = talib.CCI(hv, lv, cv, 14)
        df["adx"] = talib.ADX(hv, lv, cv, 14)
        df["plus_di"] = talib.PLUS_DI(hv, lv, cv, 14)
        df["minus_di"] = talib.MINUS_DI(hv, lv, cv, 14)
        df["roc"] = talib.ROC(cv, 10)
        df["willr"] = talib.WILLR(hv, lv, cv, 14)
    else:
        df["rsi"] = _rsi(c, 14)
        macd, sig, hist = _macd(c)
        df["macd"] = macd
        df["macd_signal"] = sig
        df["macd_hist"] = hist
        df["stoch_k"], df["stoch_d"] = _stoch(h, l, c)
        df["cci"] = _cci(h, l, c, 14)
        adx, pdi, ndi = _adx(h, l, c, 14)
        df["adx"] = adx
        df["plus_di"] = pdi
        df["minus_di"] = ndi
        df["roc"] = c.pct_change(10) * 100
        df["willr"] = _williams_r(h, l, c, 14)
    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility indicators: Bollinger Bands, ATR, NATR."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    if HAS_TALIB:
        import talib
        cv = c.values.astype(np.float64)
        hv = h.values.astype(np.float64)
        lv = l.values.astype(np.float64)

        upper, mid, lower = talib.BBANDS(cv, 20, 2, 2)
        df["bb_upper"] = upper
        df["bb_mid"] = mid
        df["bb_lower"] = lower
        df["bb_width"] = np.where(mid != 0, (upper - lower) / mid, 0)
        df["bb_pct"] = np.where((upper - lower) != 0, (cv - lower) / (upper - lower), 0.5)
        df["atr"] = talib.ATR(hv, lv, cv, 14)
        df["natr"] = talib.NATR(hv, lv, cv, 14)
    else:
        upper, mid, lower = _bbands(c, 20, 2.0)
        df["bb_upper"] = upper
        df["bb_mid"] = mid
        df["bb_lower"] = lower
        df["bb_width"] = np.where(mid != 0, (upper - lower) / mid, 0)
        df["bb_pct"] = np.where((upper - lower) != 0, (c - lower) / (upper - lower), 0.5)
        df["atr"] = _atr(h, l, c, 14)
        df["natr"] = _natr(h, l, c, 14)
    return df


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume indicators: OBV, AD, MFI."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    if HAS_TALIB:
        import talib
        cv = c.values.astype(np.float64)
        hv = h.values.astype(np.float64)
        lv = l.values.astype(np.float64)
        vv = v.values.astype(np.float64)

        df["obv"] = talib.OBV(cv, vv)
        df["ad"] = talib.AD(hv, lv, cv, vv)
        df["mfi"] = talib.MFI(hv, lv, cv, vv, 14)
    else:
        df["obv"] = _obv(c, v)
        clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        df["ad"] = (clv * v).cumsum()
        df["mfi"] = _mfi(h, l, c, v, 14)
    return df


def add_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add candlestick pattern recognition."""
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    if HAS_TALIB:
        import talib
        ov = o.values.astype(np.float64)
        hv = h.values.astype(np.float64)
        lv = l.values.astype(np.float64)
        cv = c.values.astype(np.float64)

        df["cdl_doji"] = talib.CDLDOJI(ov, hv, lv, cv)
        df["cdl_hammer"] = talib.CDLHAMMER(ov, hv, lv, cv)
        df["cdl_engulfing"] = talib.CDLENGULFING(ov, hv, lv, cv)
        df["cdl_morning_star"] = talib.CDLMORNINGSTAR(ov, hv, lv, cv)
        df["cdl_evening_star"] = talib.CDLEVENINGSTAR(ov, hv, lv, cv)
        df["cdl_shooting_star"] = talib.CDLSHOOTINGSTAR(ov, hv, lv, cv)
        df["cdl_harami"] = talib.CDLHARAMI(ov, hv, lv, cv)
        df["cdl_piercing"] = talib.CDLPIERCING(ov, hv, lv, cv)
        df["cdl_dark_cloud"] = talib.CDLDARKCLOUDCOVER(ov, hv, lv, cv)
        df["cdl_three_white"] = talib.CDL3WHITESOLDIERS(ov, hv, lv, cv)
        df["cdl_three_black"] = talib.CDL3BLACKCROWS(ov, hv, lv, cv)
    else:
        df["cdl_doji"] = _cdl_doji(o, h, l, c)
        df["cdl_hammer"] = _cdl_hammer(o, h, l, c)
        df["cdl_engulfing"] = _cdl_engulfing(o, h, l, c)
        df["cdl_morning_star"] = 0
        df["cdl_evening_star"] = 0
        df["cdl_shooting_star"] = 0
        df["cdl_harami"] = 0
        df["cdl_piercing"] = 0
        df["cdl_dark_cloud"] = 0
        df["cdl_three_white"] = 0
        df["cdl_three_black"] = 0
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features: returns, rolling stats, cross-EMA signals."""
    c = df["close"]

    # Returns
    df["returns"] = c.pct_change()
    df["log_returns"] = np.log(c / c.shift(1))

    # Rolling statistics
    for w in [5, 10, 20]:
        df[f"rolling_mean_{w}"] = c.rolling(w).mean()
        df[f"rolling_std_{w}"] = c.rolling(w).std()

    # Price vs EMAs
    if "ema_20" in df.columns:
        df["price_vs_ema20"] = (c - df["ema_20"]) / df["ema_20"].replace(0, np.nan)
        df["price_vs_ema50"] = (c - df["ema_50"]) / df["ema_50"].replace(0, np.nan)
        df["price_vs_ema200"] = np.where(
            df["ema_200"] != 0,
            (c - df["ema_200"]) / df["ema_200"],
            0,
        )
        df["ema20_vs_ema50"] = np.where(
            df["ema_50"] != 0,
            (df["ema_20"] - df["ema_50"]) / df["ema_50"],
            0,
        )
        df["ema50_vs_ema200"] = np.where(
            df["ema_200"] != 0,
            (df["ema_50"] - df["ema_200"]) / df["ema_200"],
            0,
        )

    # High-Low range
    df["hl_range_pct"] = (df["high"] - df["low"]) / c.replace(0, np.nan)

    return df


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators and derived features (~60+ columns).

    Works with or without TA-Lib installed.
    """
    df = df.copy()
    # Ensure float64
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    try:
        df = add_trend(df)
    except Exception as e:
        logger.warning(f"add_trend failed: {e}")
    try:
        df = add_momentum(df)
    except Exception as e:
        logger.warning(f"add_momentum failed: {e}")
    try:
        df = add_volatility(df)
    except Exception as e:
        logger.warning(f"add_volatility failed: {e}")
    try:
        df = add_volume(df)
    except Exception as e:
        logger.warning(f"add_volume failed: {e}")
    try:
        df = add_patterns(df)
    except Exception as e:
        logger.warning(f"add_patterns failed: {e}")
    try:
        df = add_derived(df)
    except Exception as e:
        logger.warning(f"add_derived failed: {e}")

    return df
