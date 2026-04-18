"""
Meta-Learner (STACKING ENSEMBLE)
Combines all base model outputs into a final prediction.
Trained on out-of-fold predictions to avoid data leakage.
Falls back to static weights when not yet trained.
"""
import logging
import os

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

import config

logger = logging.getLogger(__name__)

# Fallback static weights — MUST stay in sync with config.FUSION_FALLBACK_WEIGHTS
# Pattern gets primary weight: it's the main trigger; ML models add confirmation.
FALLBACK_WEIGHTS = {
    "pattern":  0.40,   # Primary trigger
    "lgbm":     0.20,   # Best trained model
    "xgb":      0.15,   # Second best
    "lstm":     0.08,   # Sequence model
    "tft":      0.06,   # Transformer
    "arima":    0.04,   # Trend supplement
    "prophet":  0.03,   # Seasonality
    "fii":      0.02,   # FII flow
    "oi":       0.02,   # OI/PCR
}

# Regime-adaptive weight modifiers
REGIME_WEIGHT_MODIFIERS = {
    "TRENDING_UP":    {"lstm": 1.3, "tft": 1.3, "lgbm": 0.8},
    "TRENDING_DOWN":  {"lstm": 1.3, "tft": 1.3, "lgbm": 0.8},
    "MEAN_REVERTING": {"lgbm": 1.3, "xgb": 1.3, "lstm": 0.7},
    "VOLATILE":       {"pattern": 0.7, "lgbm": 1.2, "xgb": 1.2},
    "BREAKOUT":       {"pattern": 1.5, "lgbm": 1.0, "lstm": 0.8},
    "CONSOLIDATION":  {"pattern": 1.3, "arima": 0.5, "lstm": 0.7},
    "MOMENTUM":       {"lstm": 1.2, "tft": 1.2, "pattern": 0.8},
}


class MetaLearner:
    """Stacking meta-learner that combines base model outputs."""

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_order = [
            "lgbm_prob", "xgb_prob", "lstm_prob", "tft_prob",
            "arima_prob", "prophet_prob", "pattern_confidence", "regime_id",
        ]

    def train(self, meta_features: np.ndarray, y: np.ndarray):
        """Train meta-learner on stacked out-of-fold predictions.

        Args:
            meta_features: (n_samples, 8) — base model probabilities + pattern + regime
            y: binary labels (0=down, 1=up)

        Returns:
            dict with metrics
        """
        if len(meta_features) < 50:
            logger.warning("Not enough meta-features for training")
            return {}

        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

        self.model.fit(meta_features, y)
        self.is_trained = True

        # Metrics
        from sklearn.metrics import accuracy_score, roc_auc_score
        preds = self.model.predict(meta_features)
        proba = self.model.predict_proba(meta_features)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y, preds)),
            "auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else 0.0,
            "coefficients": self.model.coef_[0].tolist(),
        }
        logger.info(f"Meta-learner trained — metrics: {metrics}")
        return metrics

    def predict(self, component_scores: dict, regime: str = None) -> dict:
        """Compute final fused prediction.

        Args:
            component_scores: dict with keys matching base model outputs
                {lgbm_prob, xgb_prob, lstm_prob, tft_prob,
                 arima_prob, prophet_prob, pattern_confidence}
            regime: current market regime (for adaptive weights)

        Returns:
            dict with final_probability (0-1), confidence (0-100), components
        """
        # Extract values with defaults
        lgbm = component_scores.get("lgbm_prob", 0.5)
        xgb = component_scores.get("xgb_prob", 0.5)
        lstm = component_scores.get("lstm_prob", 0.5)
        tft = component_scores.get("tft_prob", 0.5)
        arima = component_scores.get("arima_prob", 0.5)
        prophet = component_scores.get("prophet_prob", 0.5)
        pattern = component_scores.get("pattern_confidence", 0.0)
        fii = component_scores.get("fii_signal", 0.5)
        oi = component_scores.get("oi_signal", 0.5)

        # Regime encoding
        regime_map = {
            "TRENDING_UP": 1, "TRENDING_DOWN": 2, "MEAN_REVERTING": 3,
            "VOLATILE": 4, "BREAKOUT": 5, "CONSOLIDATION": 6, "MOMENTUM": 7,
        }
        regime_id = regime_map.get(regime, 0)

        if self.is_trained and self.model is not None:
            # Use trained meta-learner
            features = np.array([[lgbm, xgb, lstm, tft, arima, prophet, pattern, regime_id]])
            probability = float(self.model.predict_proba(features)[0][1])
        else:
            # Fallback: static weighted average with regime adaptation
            weights = dict(FALLBACK_WEIGHTS)

            if regime and regime in REGIME_WEIGHT_MODIFIERS:
                modifiers = REGIME_WEIGHT_MODIFIERS[regime]
                for key, mod in modifiers.items():
                    if key in weights:
                        weights[key] *= mod

            # Normalize weights
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

            probability = (
                weights["pattern"] * pattern +
                weights["lgbm"] * lgbm +
                weights["xgb"] * xgb +
                weights["lstm"] * lstm +
                weights["tft"] * tft +
                weights["arima"] * arima +
                weights["prophet"] * prophet +
                weights["fii"] * fii +
                weights["oi"] * oi
            )

        probability = float(np.clip(probability, 0.0, 1.0))
        confidence = probability * 100

        return {
            "final_probability": probability,
            "confidence": confidence,
            "components": {
                "lgbm": lgbm, "xgb": xgb, "lstm": lstm, "tft": tft,
                "arima": arima, "prophet": prophet, "pattern": pattern,
                "fii": fii, "oi": oi,
            },
            "regime": regime,
            "used_trained_model": self.is_trained,
        }

    def save(self, symbol: str):
        if not self.is_trained or self.model is None:
            return
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker)
        os.makedirs(path, exist_ok=True)
        joblib.dump({"model": self.model}, os.path.join(path, "meta_learner.pkl"))

    def load(self, symbol: str) -> bool:
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker, "meta_learner.pkl")
        if not os.path.exists(path):
            return False
        try:
            data = joblib.load(path)
            self.model = data["model"]
            self.is_trained = True
            return True
        except Exception:
            return False
