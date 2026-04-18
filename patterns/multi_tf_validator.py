"""
Multi-Timeframe Pattern Validator
Confirms patterns by checking higher timeframe alignment.
5m → 15m, 15m → 1h, 1h → daily.
"""
import logging

import numpy as np
import pandas as pd

from patterns.reversal_patterns import PatternResult

logger = logging.getLogger(__name__)

# Timeframe hierarchy: pattern TF → check TF
TF_HIERARCHY = {
    "5": "15",
    "15": "60",
    "60": "D",
    "D": None,
}


class MultiTimeframeValidator:
    """Validates patterns against higher timeframe data."""

    def validate(self, pattern: PatternResult, symbol: str, data_engine, pattern_detector=None) -> PatternResult:
        """Check if higher TF supports pattern direction.

        Args:
            pattern: PatternResult to validate
            symbol: stock symbol
            data_engine: DataEngine instance for fetching higher TF data
            pattern_detector: PatternDetector for running detection on higher TF

        Returns:
            PatternResult with updated confidence and multi_tf_confirmed flag
        """
        higher_tf = TF_HIERARCHY.get(pattern.timeframe)
        if higher_tf is None:
            return pattern  # Already on highest TF

        try:
            higher_df = data_engine.get_cached(symbol, higher_tf)
            if higher_df is None or len(higher_df) < 50:
                return pattern

            adj = 0.0

            # 1. Check EMA trend alignment on higher TF
            trend_adj = self._check_trend_alignment(pattern, higher_df)
            adj += trend_adj

            # 2. Check if same pattern exists on higher TF
            if pattern_detector is not None:
                pattern_adj = self._check_higher_tf_pattern(pattern, higher_df, pattern_detector)
                adj += pattern_adj

            # 3. Check support/resistance proximity on higher TF
            sr_adj = self._check_sr_alignment(pattern, higher_df)
            adj += sr_adj

            if adj > 0:
                pattern.multi_tf_confirmed = True

            pattern.confidence = max(0.0, min(1.0, pattern.confidence + adj))

        except Exception as e:
            logger.warning(f"Multi-TF validation failed for {symbol}: {e}")

        return pattern

    def _check_trend_alignment(self, pattern: PatternResult, higher_df: pd.DataFrame) -> float:
        """Check if higher TF EMA slope supports pattern direction."""
        close = higher_df["close"].values
        if len(close) < 50:
            return 0.0

        # EMA 20 slope on higher TF
        ema20 = _ema(close, 20)
        if len(ema20) < 5:
            return 0.0

        slope = (ema20[-1] - ema20[-5]) / 5
        slope_pct = slope / ema20[-1] if ema20[-1] > 0 else 0

        if pattern.direction == "bullish" and slope_pct > 0.001:
            return 0.10  # Higher TF uptrend supports bullish
        elif pattern.direction == "bearish" and slope_pct < -0.001:
            return 0.10  # Higher TF downtrend supports bearish
        elif pattern.direction == "bullish" and slope_pct < -0.002:
            return -0.10  # Higher TF strong downtrend contradicts
        elif pattern.direction == "bearish" and slope_pct > 0.002:
            return -0.10  # Higher TF strong uptrend contradicts

        return 0.0

    def _check_higher_tf_pattern(self, pattern: PatternResult, higher_df: pd.DataFrame, pattern_detector) -> float:
        """Check if same pattern type exists on higher TF."""
        try:
            higher_patterns = pattern_detector.detect_all(higher_df)
            for hp in higher_patterns:
                if hp.direction == pattern.direction:
                    if hp.pattern_name == pattern.pattern_name:
                        return 0.15  # Same pattern on higher TF
                    else:
                        return 0.05  # Different pattern, same direction
            # Check for contradictory patterns
            for hp in higher_patterns:
                if hp.direction != pattern.direction and hp.confidence > 0.5:
                    return -0.15  # Contradictory high-confidence pattern
        except Exception:
            pass
        return 0.0

    def _check_sr_alignment(self, pattern: PatternResult, higher_df: pd.DataFrame) -> float:
        """Check if pattern entry is near a higher TF support/resistance level."""
        close = higher_df["close"].values
        high = higher_df["high"].values
        low = higher_df["low"].values

        if len(close) < 20:
            return 0.0

        # Find recent swing highs and lows on higher TF (simple method)
        recent_highs = []
        recent_lows = []
        for i in range(5, len(high) - 5):
            if high[i] == max(high[i - 5:i + 6]):
                recent_highs.append(high[i])
            if low[i] == min(low[i - 5:i + 6]):
                recent_lows.append(low[i])

        if not recent_highs and not recent_lows:
            return 0.0

        entry = pattern.entry_price
        atr = float(np.mean(np.abs(np.diff(close[-20:])))) * 2  # Rough ATR proxy

        if atr <= 0:
            return 0.0

        # Check if entry is near a higher TF level
        for level in recent_highs + recent_lows:
            if abs(entry - level) / atr < 1.5:
                return 0.05  # Near a higher TF S/R level

        return 0.0


def _ema(data, period):
    """Simple EMA calculation."""
    if len(data) < period:
        return data
    alpha = 2 / (period + 1)
    ema = np.zeros(len(data))
    ema[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema[period - 1:]
