"""
Bayesian Strategy Selector
4-level hierarchy: Global → Sector → Regime → Sector-Regime.
Updates strategy priors based on actual trade outcomes.
Adapted from COMMODITY APP for equity markets.
"""
import json
import logging
import os

import numpy as np

import config
from signals.strategy_presets import PRESETS
from signals.regime_detector import REGIMES

logger = logging.getLogger(__name__)

STRATEGIES = list(PRESETS.keys())
SECTORS = [
    "NIFTY_IT", "NIFTY_BANK", "NIFTY_PHARMA", "NIFTY_AUTO",
    "NIFTY_FMCG", "NIFTY_METAL", "NIFTY_REALTY", "NIFTY_ENERGY",
    "NIFTY_INFRA", "NIFTY_MEDIA", "UNKNOWN",
]


class BayesianSelector:
    """Bayesian strategy selection with 4-level hierarchy."""

    def __init__(self):
        # Prior matrices: {key: {strategy: [alpha, beta]}}
        # alpha = wins + 1, beta = losses + 1 (Beta distribution prior)
        self.global_priors = {}
        self.sector_priors = {}
        self.regime_priors = {}
        self.sector_regime_priors = {}
        self._initialize_priors()
        self._load_priors()

    def select_strategy(self, sector: str = "UNKNOWN", regime: str = "CONSOLIDATION") -> str:
        """Select best strategy using Bayesian posterior probabilities.

        Uses 4-level hierarchy with weighted combination:
        - Global: 20% weight
        - Sector: 25% weight
        - Regime: 25% weight
        - Sector-Regime: 30% weight (most specific)

        Returns:
            Best strategy name
        """
        scores = {}

        for strategy in STRATEGIES:
            # Global score
            g_alpha, g_beta = self.global_priors.get(strategy, [1, 1])
            global_score = g_alpha / (g_alpha + g_beta)

            # Sector score
            sector_key = f"{sector}"
            s_priors = self.sector_priors.get(sector_key, {})
            s_alpha, s_beta = s_priors.get(strategy, [1, 1])
            sector_score = s_alpha / (s_alpha + s_beta)

            # Regime score
            r_priors = self.regime_priors.get(regime, {})
            r_alpha, r_beta = r_priors.get(strategy, [1, 1])
            regime_score = r_alpha / (r_alpha + r_beta)

            # Sector-Regime score (most specific)
            sr_key = f"{sector}_{regime}"
            sr_priors = self.sector_regime_priors.get(sr_key, {})
            sr_alpha, sr_beta = sr_priors.get(strategy, [1, 1])
            sr_score = sr_alpha / (sr_alpha + sr_beta)

            # Weighted combination
            scores[strategy] = (
                0.20 * global_score +
                0.25 * sector_score +
                0.25 * regime_score +
                0.30 * sr_score
            )

        best = max(scores, key=scores.get)
        logger.debug(f"Bayesian selected: {best} (sector={sector}, regime={regime})")
        return best

    def update(self, strategy: str, sector: str, regime: str, won: bool):
        """Update priors based on trade outcome.

        Args:
            strategy: strategy name used
            sector: stock sector
            regime: market regime during trade
            won: True if trade was profitable
        """
        # Update all 4 levels
        # Global
        if strategy not in self.global_priors:
            self.global_priors[strategy] = [1, 1]
        if won:
            self.global_priors[strategy][0] += 1
        else:
            self.global_priors[strategy][1] += 1

        # Sector
        if sector not in self.sector_priors:
            self.sector_priors[sector] = {}
        if strategy not in self.sector_priors[sector]:
            self.sector_priors[sector][strategy] = [1, 1]
        if won:
            self.sector_priors[sector][strategy][0] += 1
        else:
            self.sector_priors[sector][strategy][1] += 1

        # Regime
        if regime not in self.regime_priors:
            self.regime_priors[regime] = {}
        if strategy not in self.regime_priors[regime]:
            self.regime_priors[regime][strategy] = [1, 1]
        if won:
            self.regime_priors[regime][strategy][0] += 1
        else:
            self.regime_priors[regime][strategy][1] += 1

        # Sector-Regime
        sr_key = f"{sector}_{regime}"
        if sr_key not in self.sector_regime_priors:
            self.sector_regime_priors[sr_key] = {}
        if strategy not in self.sector_regime_priors[sr_key]:
            self.sector_regime_priors[sr_key][strategy] = [1, 1]
        if won:
            self.sector_regime_priors[sr_key][strategy][0] += 1
        else:
            self.sector_regime_priors[sr_key][strategy][1] += 1

        self._save_priors()

    def get_strategy_stats(self) -> dict:
        """Get win rates for all strategies at global level."""
        stats = {}
        for strategy in STRATEGIES:
            alpha, beta = self.global_priors.get(strategy, [1, 1])
            total = alpha + beta - 2  # Subtract initial priors
            win_rate = alpha / (alpha + beta)
            stats[strategy] = {
                "win_rate": round(win_rate, 3),
                "trades": total,
                "wins": alpha - 1,
                "losses": beta - 1,
            }
        return stats

    def _initialize_priors(self):
        """Set initial uniform priors for all strategies."""
        for strategy in STRATEGIES:
            self.global_priors[strategy] = [1, 1]

        for sector in SECTORS:
            self.sector_priors[sector] = {s: [1, 1] for s in STRATEGIES}

        for regime in REGIMES:
            self.regime_priors[regime] = {s: [1, 1] for s in STRATEGIES}

        # Set informed priors based on expected regime-strategy performance
        informed = {
            "TRENDING_UP": {"TREND_FOLLOW_ADX": [3, 1], "MOMENTUM_RSI_MACD": [2, 1]},
            "TRENDING_DOWN": {"TREND_FOLLOW_ADX": [3, 1], "MOMENTUM_RSI_MACD": [2, 1]},
            "MEAN_REVERTING": {"MEAN_REVERSION_BB": [3, 1], "ML_ENSEMBLE": [2, 1]},
            "VOLATILE": {"ML_ENSEMBLE": [3, 1], "FUSION_FULL": [2, 1]},
            "BREAKOUT": {"PATTERN_BREAKOUT": [3, 1], "TREND_FOLLOW_ADX": [2, 1]},
            "CONSOLIDATION": {"MEAN_REVERSION_BB": [2, 1], "PATTERN_BREAKOUT": [2, 1]},
            "MOMENTUM": {"MOMENTUM_RSI_MACD": [3, 1], "TREND_FOLLOW_ADX": [2, 1]},
        }
        for regime, priors in informed.items():
            if regime in self.regime_priors:
                for strategy, ab in priors.items():
                    self.regime_priors[regime][strategy] = ab

    def _save_priors(self):
        try:
            data = {
                "global": self.global_priors,
                "sector": self.sector_priors,
                "regime": self.regime_priors,
                "sector_regime": self.sector_regime_priors,
            }
            path = config.STRATEGY_PERFORMANCE_FILE
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Bayesian priors: {e}")

    def _load_priors(self):
        path = config.STRATEGY_PERFORMANCE_FILE
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.global_priors.update(data.get("global", {}))
            self.sector_priors.update(data.get("sector", {}))
            self.regime_priors.update(data.get("regime", {}))
            self.sector_regime_priors.update(data.get("sector_regime", {}))
        except Exception as e:
            logger.error(f"Failed to load Bayesian priors: {e}")
