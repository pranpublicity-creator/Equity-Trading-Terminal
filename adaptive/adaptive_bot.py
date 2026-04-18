"""
Adaptive Bot
Auto-switches strategy preset based on detected market regime.
Tracks regime transitions and adjusts signal engine parameters.
Adapted from COMMODITY APP — 7 equity regimes (removed CONTANGO/BACKWARDATION).
"""
import json
import logging
import os
import time

import config
from signals.regime_detector import detect_regime, REGIMES
from signals.strategy_presets import PRESETS, get_preset, DEFAULT_STRATEGY

logger = logging.getLogger(__name__)

# Regime → recommended strategy mapping
REGIME_STRATEGY_MAP = {
    "TRENDING_UP":    "TREND_FOLLOW_ADX",
    "TRENDING_DOWN":  "TREND_FOLLOW_ADX",
    "MEAN_REVERTING": "MEAN_REVERSION_BB",
    "VOLATILE":       "ML_ENSEMBLE",
    "BREAKOUT":       "PATTERN_BREAKOUT",
    "CONSOLIDATION":  "MEAN_REVERSION_BB",
    "MOMENTUM":       "MOMENTUM_RSI_MACD",
}


class AdaptiveBot:
    """Auto-adapts strategy based on market regime detection."""

    def __init__(self, signal_engine=None):
        self.signal_engine = signal_engine
        self.current_regime = "CONSOLIDATION"
        self.current_strategy = DEFAULT_STRATEGY
        self.regime_history = []  # [(timestamp, regime, confidence)]
        self.auto_adapt = True
        self._regime_stable_count = 0
        self._min_stable_bars = 3  # Require regime stable for 3 bars before switching
        self._load_state()

    def update_regime(self, df) -> dict:
        """Detect current regime and adapt strategy if needed.

        Args:
            df: OHLCV DataFrame with indicators

        Returns:
            dict with regime info and any strategy change
        """
        result = detect_regime(df)
        new_regime = result["regime"]
        confidence = result["confidence"]

        changed = False

        if new_regime == self.current_regime:
            self._regime_stable_count += 1
        else:
            self._regime_stable_count = 1

        # Only switch after regime is stable for N bars
        if new_regime != self.current_regime and self._regime_stable_count >= self._min_stable_bars:
            old_regime = self.current_regime
            self.current_regime = new_regime
            self._regime_stable_count = 0

            if self.auto_adapt:
                old_strategy = self.current_strategy
                self.current_strategy = REGIME_STRATEGY_MAP.get(new_regime, DEFAULT_STRATEGY)
                changed = True

                logger.info(
                    f"REGIME CHANGE: {old_regime} → {new_regime} "
                    f"(conf={confidence:.2f}) | Strategy: {old_strategy} → {self.current_strategy}"
                )

        # Record history
        self.regime_history.append({
            "timestamp": time.time(),
            "regime": new_regime,
            "confidence": confidence,
            "strategy": self.current_strategy,
        })

        # Keep last 500 entries
        if len(self.regime_history) > 500:
            self.regime_history = self.regime_history[-500:]

        self._save_state()

        return {
            "regime": self.current_regime,
            "confidence": confidence,
            "strategy": self.current_strategy,
            "regime_scores": result["scores"],
            "changed": changed,
            "stable_bars": self._regime_stable_count,
        }

    def get_current_strategy(self) -> str:
        return self.current_strategy

    def set_strategy(self, strategy_name: str):
        """Manually override strategy (disables auto-adapt)."""
        if strategy_name in PRESETS:
            self.current_strategy = strategy_name
            self.auto_adapt = False
            logger.info(f"Strategy manually set to {strategy_name} (auto-adapt disabled)")

    def enable_auto_adapt(self):
        self.auto_adapt = True
        logger.info("Auto-adapt enabled")

    def disable_auto_adapt(self):
        self.auto_adapt = False
        logger.info("Auto-adapt disabled")

    def get_regime_distribution(self) -> dict:
        """Get distribution of regimes over recent history."""
        if not self.regime_history:
            return {}
        counts = {}
        recent = self.regime_history[-100:]
        for entry in recent:
            r = entry["regime"]
            counts[r] = counts.get(r, 0) + 1
        total = len(recent)
        return {k: round(v / total, 3) for k, v in counts.items()}

    def _save_state(self):
        try:
            data = {
                "current_regime": self.current_regime,
                "current_strategy": self.current_strategy,
                "auto_adapt": self.auto_adapt,
                "regime_history": self.regime_history[-100:],
            }
            with open(config.BOT_CONFIG_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save adaptive state: {e}")

    def _load_state(self):
        if not os.path.exists(config.BOT_CONFIG_FILE):
            return
        try:
            with open(config.BOT_CONFIG_FILE, "r") as f:
                data = json.load(f)
            self.current_regime = data.get("current_regime", "CONSOLIDATION")
            self.current_strategy = data.get("current_strategy", DEFAULT_STRATEGY)
            self.auto_adapt = data.get("auto_adapt", True)
            self.regime_history = data.get("regime_history", [])
        except Exception as e:
            logger.error(f"Failed to load adaptive state: {e}")
