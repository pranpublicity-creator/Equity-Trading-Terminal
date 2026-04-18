"""
Volume Feature Engineering
Volume spikes, delivery %, OBV slope, MFI signals, Bulkowski volume profiles.
"""
import numpy as np
import pandas as pd


def compute_volume_features(df: pd.DataFrame, delivery_pct=0.0) -> pd.DataFrame:
    """Compute volume-based features.

    Args:
        df: DataFrame with 'volume', 'close', 'obv', 'mfi' columns
        delivery_pct: delivery percentage from MarketDataEnricher (0-100)
    """
    df = df.copy()
    vol = df["volume"].astype(float)

    # Volume SMA ratio
    vol_sma_20 = vol.rolling(20).mean()
    df["volume_sma_ratio"] = np.where(vol_sma_20 > 0, vol / vol_sma_20, 1.0)

    # Volume spike (> 2x 20-bar average)
    df["volume_spike"] = (df["volume_sma_ratio"] > 2.0).astype(int)

    # Volume trend slope (last 10 bars, linear regression)
    df["volume_trend_slope"] = (
        vol.rolling(10)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False)
    )

    # OBV slope (if available)
    if "obv" in df.columns:
        obv = df["obv"]
        df["obv_slope"] = (
            obv.rolling(10)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False)
        )
    else:
        df["obv_slope"] = 0

    # MFI signal
    if "mfi" in df.columns:
        df["mfi_overbought"] = (df["mfi"] > 80).astype(int)
        df["mfi_oversold"] = (df["mfi"] < 20).astype(int)
    else:
        df["mfi_overbought"] = 0
        df["mfi_oversold"] = 0

    # Delivery % features (Indian market specific)
    df["delivery_pct"] = delivery_pct
    df["delivery_institutional"] = 1 if delivery_pct > 50 else 0
    df["delivery_speculative"] = 1 if delivery_pct < 30 else 0

    # Volume profile classification (Bulkowski concept)
    # Classify last 20 bars as U-shape, dome, or flat
    df["volume_profile"] = _classify_volume_profile(vol)

    return df


def _classify_volume_profile(volume_series, window=20):
    """Classify volume into U-shape (0), dome (1), or flat/other (2).
    Bulkowski found U-shaped volume patterns tend to perform better.
    """
    result = pd.Series(2, index=volume_series.index)  # Default: other

    for i in range(window, len(volume_series)):
        window_vol = volume_series.iloc[i - window:i].values
        if len(window_vol) < window:
            continue

        mid = window // 2
        first_half_avg = np.mean(window_vol[:mid])
        middle_avg = np.mean(window_vol[mid - 2:mid + 3])  # 5-bar middle
        second_half_avg = np.mean(window_vol[mid:])

        if first_half_avg > middle_avg and second_half_avg > middle_avg:
            result.iloc[i] = 0  # U-shape
        elif first_half_avg < middle_avg and second_half_avg < middle_avg:
            result.iloc[i] = 1  # Dome
        # else: flat/other = 2

    return result
