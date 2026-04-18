"""
Master Pattern Detector
Dispatches all 20 pattern detectors and returns deduplicated, validated results.
"""
import logging
from typing import List, Optional

import pandas as pd

import config
from patterns.swing_detector import find_swings, SwingPoint
from patterns.reversal_patterns import PatternResult
from patterns import reversal_patterns
from patterns import continuation_patterns
from patterns import breakout_patterns
from patterns import volatility_patterns
from patterns.pattern_validator import PatternValidator
from patterns.multi_tf_validator import MultiTimeframeValidator

logger = logging.getLogger(__name__)


class PatternDetector:
    """Runs all 20 pattern detectors and returns validated results."""

    def __init__(self):
        self._validator = PatternValidator()
        self._multi_tf = MultiTimeframeValidator()

    def detect_all(self, df: pd.DataFrame, swings: List[SwingPoint] = None,
                   lookback: int = None) -> List[PatternResult]:
        """Run all 20 pattern detectors on the given OHLCV data.

        Args:
            df: OHLCV DataFrame
            swings: pre-computed swing points (computed if None)
            lookback: pattern lookback window (uses config default)

        Returns:
            List of PatternResult sorted by confidence descending
        """
        lookback = lookback or config.PATTERN_LOOKBACK_BARS
        if swings is None:
            swings = find_swings(df)

        if len(swings) < 3:
            return []

        all_patterns = []

        # --- Reversal patterns (8) ---
        all_patterns.extend(reversal_patterns.detect_double_top(df, swings, lookback=100))
        all_patterns.extend(reversal_patterns.detect_double_bottom(df, swings, lookback=100))
        all_patterns.extend(reversal_patterns.detect_head_shoulders_top(df, swings, lookback=150))
        all_patterns.extend(reversal_patterns.detect_head_shoulders_bottom(df, swings, lookback=150))
        all_patterns.extend(reversal_patterns.detect_triple_top(df, swings, lookback=120))
        all_patterns.extend(reversal_patterns.detect_triple_bottom(df, swings, lookback=120))
        all_patterns.extend(reversal_patterns.detect_rounding_top(df, swings, lookback=200))
        all_patterns.extend(reversal_patterns.detect_rounding_bottom(df, swings, lookback=200))

        # --- Continuation patterns (3) ---
        all_patterns.extend(continuation_patterns.detect_flag(df, swings, lookback=50))
        all_patterns.extend(continuation_patterns.detect_pennant(df, swings, lookback=50))
        all_patterns.extend(continuation_patterns.detect_measured_move(df, swings, lookback=200))

        # --- Breakout patterns (5) ---
        all_patterns.extend(breakout_patterns.detect_ascending_triangle(df, swings, lookback=100))
        all_patterns.extend(breakout_patterns.detect_descending_triangle(df, swings, lookback=100))
        all_patterns.extend(breakout_patterns.detect_symmetrical_triangle(df, swings, lookback=100))
        all_patterns.extend(breakout_patterns.detect_rectangle(df, swings, lookback=80))
        all_patterns.extend(breakout_patterns.detect_cup_and_handle(df, swings, lookback=200))

        # --- Volatility patterns (4) ---
        all_patterns.extend(volatility_patterns.detect_broadening_formation(df, swings, lookback=100))
        all_patterns.extend(volatility_patterns.detect_broadening_top_bottom(df, swings, lookback=100))
        all_patterns.extend(volatility_patterns.detect_diamond(df, swings, lookback=120))
        all_patterns.extend(volatility_patterns.detect_wedge(df, swings, lookback=100))

        # Deduplicate overlapping patterns
        all_patterns = self._deduplicate(all_patterns)

        # Sort by confidence
        all_patterns.sort(key=lambda p: p.confidence, reverse=True)

        # Return top N
        return all_patterns[:10]

    def detect_for_symbol(self, symbol: str, df: pd.DataFrame,
                          enricher_data: dict = None,
                          data_engine=None) -> List[PatternResult]:
        """Full detection pipeline with validation for a specific symbol.

        Args:
            symbol: stock symbol (e.g., 'NSE:RELIANCE-EQ')
            df: OHLCV DataFrame
            enricher_data: dict from MarketDataEnricher
            data_engine: DataEngine for multi-TF validation

        Returns:
            Validated, sorted list of PatternResult
        """
        swings = find_swings(df)
        patterns = self.detect_all(df, swings)

        if not patterns:
            return []

        # Validate each pattern
        validated = []
        for p in patterns:
            # Volume + delivery + OI validation
            p = self._validator.validate(p, df, enricher_data)

            # Multi-timeframe validation (if data engine available)
            if data_engine is not None:
                p = self._multi_tf.validate(p, symbol, data_engine, self)

            validated.append(p)

        # Re-sort after validation adjustments
        validated.sort(key=lambda p: p.confidence, reverse=True)

        # Filter below minimum confidence
        validated = [p for p in validated if p.confidence >= config.PATTERN_MIN_CONFIDENCE]

        return validated[:5]

    def _deduplicate(self, patterns: List[PatternResult]) -> List[PatternResult]:
        """Remove overlapping patterns, keeping highest confidence."""
        if len(patterns) <= 1:
            return patterns

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        kept = []
        for p in patterns:
            overlap = False
            for k in kept:
                # Overlap if indices overlap significantly (>50%)
                overlap_start = max(p.start_index, k.start_index)
                overlap_end = min(p.end_index, k.end_index)
                if overlap_end > overlap_start:
                    p_len = max(1, p.end_index - p.start_index)
                    overlap_pct = (overlap_end - overlap_start) / p_len
                    if overlap_pct > 0.50:
                        overlap = True
                        break
            if not overlap:
                kept.append(p)
        return kept
