"""
Signal Fusion Engine
Meta-learner based fusion of patterns, ML models, and market context.
Produces final trade confidence with penalty filters.
"""
import logging

import numpy as np

import config
from models.meta_learner import MetaLearner

logger = logging.getLogger(__name__)


class FusionResult:
    """Result of signal fusion computation."""

    def __init__(self, direction, confidence, components, regime=None):
        self.direction = direction        # 'BUY' or 'SELL'
        self.confidence = confidence      # 0-100
        self.components = components      # dict of individual scores
        self.regime = regime
        self.strength = self._classify()

    def _classify(self):
        if self.confidence >= config.SIGNAL_STRONG_THRESHOLD:
            return "STRONG"
        elif self.confidence >= config.SIGNAL_MODERATE_THRESHOLD:
            return "MODERATE"
        elif self.confidence >= config.SIGNAL_WEAK_THRESHOLD:
            return "WEAK"
        return "NO_TRADE"


class SignalFusion:
    """Combines all model outputs into a single trade decision."""

    def __init__(self):
        self._meta_learner = None
        self._active_strategy: str = ""    # set by compute(); used by _apply_penalties()

    def set_meta_learner(self, meta: MetaLearner):
        self._meta_learner = meta

    def compute(self, patterns, ml_predictions, enricher_data=None,
                indicators=None, regime=None, strategy: str = "") -> FusionResult:
        """Compute fused signal from all components.

        Args:
            patterns: list of PatternResult from pattern detector
            ml_predictions: dict from ModelManager.predict()
            enricher_data: dict from MarketDataEnricher
            indicators: DataFrame with indicator columns
            regime: current market regime string

        Returns:
            FusionResult with direction and confidence (0-100)
        """
        self._active_strategy = strategy   # stored for _apply_penalties
        # Step 1: Collect component scores
        components = self._collect_components(patterns, ml_predictions, enricher_data)

        # Step 2: Direction vote
        direction = self._vote_direction(patterns, ml_predictions)

        # Step 3: Compute raw confidence score
        if self._meta_learner and self._meta_learner.is_trained:
            fusion = self._meta_learner.predict(components, regime)
            raw_score = fusion["confidence"]
        elif ml_predictions and "meta_score" in ml_predictions:
            raw_score = ml_predictions["meta_score"]
        else:
            raw_score = self._static_weighted_score(components, regime)

        # Step 4: Apply penalty filters (direction + strategy now influence ADX/RSI logic)
        score = self._apply_penalties(raw_score, enricher_data, indicators,
                                      direction=direction, strategy=self._active_strategy)

        return FusionResult(
            direction=direction,
            confidence=round(score, 2),
            components=components,
            regime=regime,
        )

    def _collect_components(self, patterns, ml_predictions, enricher_data):
        """Normalize all component scores to 0-1 range."""
        components = {}

        # Pattern score
        if patterns:
            best = max(patterns, key=lambda p: p.confidence)
            components["pattern_confidence"] = best.confidence
        else:
            components["pattern_confidence"] = 0.0

        # ML model probabilities
        if ml_predictions:
            components["lgbm_prob"] = ml_predictions.get("lgbm_prob", 0.5)
            components["xgb_prob"] = ml_predictions.get("xgb_prob", 0.5)
            components["lstm_prob"] = ml_predictions.get("lstm_prob", 0.5)
            components["tft_prob"] = ml_predictions.get("tft_prob", 0.5)
            components["arima_prob"] = ml_predictions.get("arima_prob", 0.5)
            components["prophet_prob"] = ml_predictions.get("prophet_prob", 0.5)
        else:
            for key in ["lgbm_prob", "xgb_prob", "lstm_prob", "tft_prob", "arima_prob", "prophet_prob"]:
                components[key] = 0.5

        # Market context signals
        if enricher_data:
            fii_net = enricher_data.get("fii_net", 0)
            components["fii_signal"] = float(np.clip(0.5 + fii_net / 10000.0, 0.0, 1.0))
            pcr = enricher_data.get("pcr", 1.0)
            components["oi_signal"] = float(np.clip(0.5 + (pcr - 1.0) / 1.0, 0.0, 1.0))
        else:
            components["fii_signal"] = 0.5
            components["oi_signal"] = 0.5

        return components

    # Classic reversal patterns have inherent direction — ML cannot override them
    _FORCED_BEARISH_PATTERNS = {
        "head_shoulders_top", "double_top", "triple_top", "rounding_top",
        "bearish_engulfing", "evening_star", "shooting_star", "hanging_man",
        "rsi_bearish_divergence",          # momentum-fade → sell
    }
    _FORCED_BULLISH_PATTERNS = {
        "head_shoulders_bottom", "double_bottom", "triple_bottom", "rounding_bottom",
        "bullish_engulfing", "morning_star", "hammer", "inverted_head_shoulders",
        "rsi_bullish_divergence",          # momentum-build → buy
    }

    def _vote_direction(self, patterns, ml_predictions) -> str:
        """Majority vote for trade direction.
        Classic reversal patterns (H&S, double top/bottom, etc.) force direction
        regardless of ML model votes — their textbook meaning is absolute.
        """
        # Check if top pattern is a forced-direction reversal
        if patterns:
            top_pattern = patterns[0].pattern_name
            if top_pattern in self._FORCED_BEARISH_PATTERNS:
                return "SELL"
            if top_pattern in self._FORCED_BULLISH_PATTERNS:
                return "BUY"

        votes_bull = 0
        votes_bear = 0

        # Pattern votes (3x weight — patterns are the primary trigger)
        if patterns:
            for p in patterns:
                weight = 3.0 * p.confidence
                if p.direction == "bullish":
                    votes_bull += weight
                else:
                    votes_bear += weight

        # ML model votes
        if ml_predictions:
            for key in ["lgbm_prob", "xgb_prob", "lstm_prob", "tft_prob"]:
                prob = ml_predictions.get(key, 0.5)
                if prob > 0.55:
                    votes_bull += prob
                elif prob < 0.45:
                    votes_bear += (1 - prob)

            # ARIMA trend
            trend = ml_predictions.get("arima_trend", "FLAT")
            if trend == "UP":
                votes_bull += 0.5
            elif trend == "DOWN":
                votes_bear += 0.5

        return "BUY" if votes_bull >= votes_bear else "SELL"

    def _static_weighted_score(self, components, regime=None):
        """Fallback weighted scoring when meta-learner not available."""
        weights = dict(config.FUSION_FALLBACK_WEIGHTS)

        # Regime-adaptive modifiers
        from models.meta_learner import REGIME_WEIGHT_MODIFIERS
        if regime and regime in REGIME_WEIGHT_MODIFIERS:
            modifiers = REGIME_WEIGHT_MODIFIERS[regime]
            for key, mod in modifiers.items():
                if key in weights:
                    weights[key] *= mod

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        score = (
            weights.get("pattern", 0) * components.get("pattern_confidence", 0) +
            weights.get("lgbm", 0) * components.get("lgbm_prob", 0.5) +
            weights.get("xgb", 0) * components.get("xgb_prob", 0.5) +
            weights.get("lstm", 0) * components.get("lstm_prob", 0.5) +
            weights.get("tft", 0) * components.get("tft_prob", 0.5) +
            weights.get("arima", 0) * components.get("arima_prob", 0.5) +
            weights.get("prophet", 0) * components.get("prophet_prob", 0.5) +
            weights.get("fii", 0) * components.get("fii_signal", 0.5) +
            weights.get("oi", 0) * components.get("oi_signal", 0.5)
        )

        return score * 100

    def _apply_penalties(self, score, enricher_data, indicators,
                         direction: str = "", strategy: str = ""):
        """Apply soft penalty filters to raw score.

        Three categories:
          A. ADX (strategy-aware)  — strength & direction alignment
          B. RSI extreme levels    — overbought/oversold relative to trade direction
          C. Volume + market context (unchanged)
        """
        if indicators is not None and len(indicators) > 0:
            ind = indicators.iloc[-1]   # latest bar snapshot

            # ── A. ADX penalty — strategy-aware ────────────────────────────
            # TREND_FOLLOW / MOMENTUM want high ADX (weak ADX = weaker signal)
            # MEAN_REVERSION wants LOW ADX  (high ADX = chasing a strong trend)
            adx = float(ind.get("adx", float("nan")))
            if adx == adx:   # not NaN
                mean_rev = strategy in ("MEAN_REVERSION_BB",)
                if mean_rev:
                    # Mean-reversion strategies: penalise strong trends (ADX > 30)
                    if adx > 35:
                        score *= 0.80   # Very strong trend — avoid counter-trend
                    elif adx > 30:
                        score *= 0.90   # Moderate strong trend — mild caution
                    # ADX ≤ 30: ideal choppy environment, no penalty
                else:
                    # Trend/breakout strategies: penalise flat markets
                    if adx < 15:
                        score *= 0.75   # Dead flat market
                    elif adx < 20:
                        score *= 0.88   # Weak trend — mild penalty
                    # ADX ≥ 20: no penalty

            # ── B. RSI extreme-level penalty ───────────────────────────────
            # RSI divergence patterns are explicitly momentum-fade plays, so
            # they are EXCLUDED from this penalty (divergence RELIES on extremes).
            rsi = float(ind.get("rsi", float("nan")))
            if rsi == rsi and direction:
                dir_up = direction.upper() == "BUY"
                # BUY into overbought: penalise late long entries
                if dir_up and rsi > 80:
                    score *= 0.80
                    logger.debug(f"RSI OB penalty (RSI={rsi:.1f} > 80 on BUY): ×0.80")
                elif dir_up and rsi > 75:
                    score *= 0.90
                    logger.debug(f"RSI OB penalty (RSI={rsi:.1f} > 75 on BUY): ×0.90")
                # SELL into oversold: penalise late short entries
                elif not dir_up and rsi < 20:
                    score *= 0.80
                    logger.debug(f"RSI OS penalty (RSI={rsi:.1f} < 20 on SELL): ×0.80")
                elif not dir_up and rsi < 25:
                    score *= 0.90
                    logger.debug(f"RSI OS penalty (RSI={rsi:.1f} < 25 on SELL): ×0.90")

        # ── C. Volume spike penalty ─────────────────────────────────────────
        if indicators is not None and len(indicators) >= 3:
            if "volume_spike" in indicators.columns:
                recent_spikes = indicators["volume_spike"].iloc[-3:].sum()
                if recent_spikes == 0:
                    score *= 0.88
            elif "volume" in indicators.columns:
                vol = indicators["volume"]
                if len(vol) >= 20:
                    vol_avg = float(vol.iloc[-20:-1].mean()) if len(vol) >= 21 else float(vol.mean())
                    cur_vol = float(vol.iloc[-1])
                    if vol_avg > 0 and cur_vol < vol_avg * 0.8:
                        score *= 0.88

        # ── D. Indian market context ────────────────────────────────────────
        if enricher_data:
            circuit_prox = enricher_data.get("circuit_proximity", 1.0)
            if circuit_prox < 0.05:
                score *= 0.50
            delivery = enricher_data.get("delivery_pct", 40)
            if delivery < 20:
                score *= 0.88

        return max(0.0, min(100.0, score))
