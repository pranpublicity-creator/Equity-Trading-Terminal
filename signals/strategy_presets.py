"""
Strategy Presets
6 pre-configured trading strategies with different component weightings.
Adapted from COMMODITY APP for equity markets.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class StrategyPreset:
    """A named strategy configuration."""
    name: str
    description: str
    weights: Dict[str, float]
    min_confidence: float
    required_conditions: Dict[str, float]  # indicator: min_value


# 6 Strategy Presets
PRESETS = {
    "PATTERN_BREAKOUT": StrategyPreset(
        name="Pattern Breakout",
        description="Weight patterns highest. Best for clear chart pattern setups.",
        weights={
            "pattern": 0.35, "lgbm": 0.15, "xgb": 0.10, "lstm": 0.10,
            "tft": 0.10, "arima": 0.05, "prophet": 0.05, "fii": 0.05, "oi": 0.05,
        },
        min_confidence=50.0,
        required_conditions={"pattern_confidence": 0.5},
    ),

    "TREND_FOLLOW_ADX": StrategyPreset(
        name="Trend Follow (ADX)",
        description="Weight sequence models highest. Best in strong trending markets.",
        weights={
            "pattern": 0.10, "lgbm": 0.10, "xgb": 0.10, "lstm": 0.25,
            "tft": 0.20, "arima": 0.10, "prophet": 0.05, "fii": 0.05, "oi": 0.05,
        },
        min_confidence=55.0,
        required_conditions={"adx": 25.0},
    ),

    "MEAN_REVERSION_BB": StrategyPreset(
        name="Mean Reversion (Bollinger)",
        description="Weight tabular models highest. Best in range-bound markets.",
        weights={
            "pattern": 0.10, "lgbm": 0.30, "xgb": 0.20, "lstm": 0.10,
            "tft": 0.05, "arima": 0.05, "prophet": 0.10, "fii": 0.05, "oi": 0.05,
        },
        min_confidence=55.0,
        required_conditions={},
    ),

    "MOMENTUM_RSI_MACD": StrategyPreset(
        name="Momentum (RSI/MACD)",
        description="Weight XGBoost + LSTM. Best in momentum-driven markets.",
        weights={
            "pattern": 0.10, "lgbm": 0.15, "xgb": 0.20, "lstm": 0.20,
            "tft": 0.10, "arima": 0.05, "prophet": 0.05, "fii": 0.10, "oi": 0.05,
        },
        min_confidence=55.0,
        required_conditions={},
    ),

    "ML_ENSEMBLE": StrategyPreset(
        name="ML Ensemble",
        description="Weight meta-learner. Requires at least 3 models to agree.",
        weights={
            "pattern": 0.10, "lgbm": 0.20, "xgb": 0.15, "lstm": 0.15,
            "tft": 0.15, "arima": 0.10, "prophet": 0.05, "fii": 0.05, "oi": 0.05,
        },
        min_confidence=60.0,
        required_conditions={},
    ),

    "FUSION_FULL": StrategyPreset(
        name="Full Fusion (Default)",
        description="Balanced weights using meta-learner. Default strategy.",
        weights={
            "pattern": 0.20, "lgbm": 0.25, "xgb": 0.15, "lstm": 0.15,
            "tft": 0.10, "arima": 0.05, "prophet": 0.05, "fii": 0.03, "oi": 0.02,
        },
        min_confidence=50.0,
        required_conditions={},
    ),
}

DEFAULT_STRATEGY = "FUSION_FULL"


def get_preset(name: str) -> StrategyPreset:
    """Get a strategy preset by name."""
    return PRESETS.get(name, PRESETS[DEFAULT_STRATEGY])


def list_presets() -> dict:
    """List all available presets."""
    return {name: preset.description for name, preset in PRESETS.items()}


def check_conditions(preset: StrategyPreset, indicators: dict) -> bool:
    """Check if required conditions for a strategy are met."""
    for indicator, min_val in preset.required_conditions.items():
        actual = indicators.get(indicator, 0)
        if actual < min_val:
            return False
    return True
