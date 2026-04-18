"""
Pattern Validator
Adjusts pattern confidence using Bulkowski volume rules + Indian market signals.
Volume confirmation, delivery %, OI/PCR, pattern height, breakout gap.
"""
import logging

import numpy as np
import pandas as pd

from patterns.reversal_patterns import PatternResult

logger = logging.getLogger(__name__)


class PatternValidator:
    """Validates and adjusts pattern confidence with market data."""

    def validate(self, pattern: PatternResult, df: pd.DataFrame, enricher_data: dict = None) -> PatternResult:
        """Update pattern confidence based on volume + Indian market signals.

        Args:
            pattern: raw PatternResult from detector
            df: OHLCV DataFrame
            enricher_data: dict from MarketDataEnricher (can be None)

        Returns:
            PatternResult with adjusted confidence and confirmation flags
        """
        adj = 0.0

        # 1. Volume confirmation (Bulkowski)
        vol_adj = self._check_volume(pattern, df)
        adj += vol_adj
        if vol_adj > 0:
            pattern.volume_confirmed = True

        # 2. Delivery % confirmation (Indian-specific)
        if enricher_data:
            del_adj = self._check_delivery(pattern, enricher_data)
            adj += del_adj
            if del_adj > 0:
                pattern.delivery_confirmed = True

            # 3. OI/PCR confirmation (Indian-specific)
            oi_adj = self._check_oi_pcr(pattern, enricher_data)
            adj += oi_adj
            if oi_adj > 0:
                pattern.oi_confirmed = True

        # 4. Pattern height (Bulkowski: tall patterns perform better)
        adj += self._check_height(pattern, df)

        # 5. Breakout gap
        adj += self._check_breakout_gap(pattern, df)

        pattern.confidence = max(0.0, min(1.0, pattern.confidence + adj))
        return pattern

    def _check_volume(self, pattern: PatternResult, df: pd.DataFrame) -> float:
        """Bulkowski volume rules: U-shaped volume, heavy breakout volume, declining trend."""
        adj = 0.0
        volume = df["volume"].values
        start = max(0, pattern.start_index)
        end = min(len(volume), pattern.end_index + 1)

        if end - start < 5:
            return adj

        pattern_vol = volume[start:end]

        # U-shaped volume: first and last quarters higher than middle
        qlen = max(1, len(pattern_vol) // 4)
        first_q = float(np.mean(pattern_vol[:qlen]))
        mid_q = float(np.mean(pattern_vol[qlen:-qlen])) if len(pattern_vol) > 2 * qlen else first_q
        last_q = float(np.mean(pattern_vol[-qlen:]))

        if mid_q > 0 and first_q > mid_q and last_q > mid_q:
            adj += 0.10  # U-shaped volume pattern

        # Heavy breakout volume (>1.5x 20-bar average)
        if end < len(volume):
            avg_vol_20 = float(np.mean(volume[max(0, end - 20):end]))
            if avg_vol_20 > 0 and end < len(volume):
                breakout_vol = float(volume[min(end, len(volume) - 1)])
                if breakout_vol > avg_vol_20 * 1.5:
                    adj += 0.10

        # Volume declining toward end (typical for triangles, flags)
        if len(pattern_vol) > 10:
            first_half = float(np.mean(pattern_vol[:len(pattern_vol) // 2]))
            second_half = float(np.mean(pattern_vol[len(pattern_vol) // 2:]))
            if first_half > 0 and second_half < first_half * 0.85:
                adj += 0.05

        return adj

    def _check_delivery(self, pattern: PatternResult, enricher_data: dict) -> float:
        """Indian market: delivery % signals institutional participation."""
        delivery_pct = enricher_data.get("delivery_pct", 0)
        if delivery_pct > 50:
            return 0.10   # Institutional participation = reliable breakout
        elif delivery_pct < 30:
            return -0.10  # Speculative = unreliable
        return 0.0

    def _check_oi_pcr(self, pattern: PatternResult, enricher_data: dict) -> float:
        """Indian market: OI/PCR supports pattern direction?"""
        pcr = enricher_data.get("pcr", 1.0)

        if pattern.direction == "bullish":
            if pcr > 1.2:
                return 0.08   # PCR supports bullish
            elif pcr < 0.8:
                return -0.08  # PCR contradicts bullish
        elif pattern.direction == "bearish":
            if pcr < 0.8:
                return 0.08   # PCR supports bearish
            elif pcr > 1.2:
                return -0.08  # PCR contradicts bearish

        return 0.0

    def _check_height(self, pattern: PatternResult, df: pd.DataFrame) -> float:
        """Bulkowski: tall patterns outperform short patterns."""
        if pattern.entry_price <= 0:
            return 0.0

        if pattern.direction == "bullish":
            height = abs(pattern.target_price - pattern.entry_price)
        else:
            height = abs(pattern.entry_price - pattern.target_price)

        height_pct = height / pattern.entry_price

        # Median pattern height is roughly 5-8% for most patterns
        if height_pct > 0.08:
            return 0.05   # Tall pattern
        elif height_pct < 0.03:
            return -0.05  # Short pattern
        return 0.0

    def _check_breakout_gap(self, pattern: PatternResult, df: pd.DataFrame) -> float:
        """Breakout gap = stronger conviction."""
        if not pattern.breakout_confirmed:
            return 0.0

        close = df["close"].values
        open_prices = df["open"].values
        idx = min(pattern.end_index + 1, len(close) - 1)

        if idx <= 0 or idx >= len(close):
            return 0.0

        prev_close = close[idx - 1]
        curr_open = open_prices[idx]

        if pattern.direction == "bullish" and curr_open > prev_close * 1.005:
            return 0.05  # Gap up at breakout
        elif pattern.direction == "bearish" and curr_open < prev_close * 0.995:
            return 0.05  # Gap down at breakout

        return 0.0
