"""
LightGBM Model (PRIMARY direction model)
Fastest training (~2 min), best on tabular data, handles missing values natively.
Input: tabular features (1, ~180). Output: P(price_up) in [0,1].
"""
import logging
import os

import numpy as np
import joblib

import config

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.warning("LightGBM not installed — lgbm_model disabled")


class LightGBMModel:
    """LightGBM classifier for price direction prediction."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False

    def build(self, n_features: int = 180):
        """Initialize model with default hyperparameters."""
        if not HAS_LGBM:
            return
        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """Train the model.

        Args:
            X_train: training features (n_samples, n_features)
            y_train: binary labels (0=down, 1=up)
            X_val: validation features
            y_val: validation labels
            feature_names: list of feature column names
        """
        if not HAS_LGBM:
            logger.error("LightGBM not available")
            return {}

        if self.model is None:
            self.build(X_train.shape[1])

        self.feature_names = feature_names

        callbacks = [lgb.log_evaluation(period=100)]
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=50))
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        else:
            self.model.fit(X_train, y_train, callbacks=callbacks)

        self.is_trained = True

        metrics = self._compute_metrics(X_val, y_val) if X_val is not None else {}
        logger.info(f"LightGBM trained — metrics: {metrics}")
        return metrics

    def predict(self, X) -> float:
        """Predict probability of price going up.

        Args:
            X: features array (1, n_features) or (n_features,)

        Returns:
            float probability in [0, 1]
        """
        if not self.is_trained or self.model is None:
            return 0.5

        X = np.atleast_2d(X).astype(np.float32)
        proba = self.model.predict_proba(X)
        return float(proba[0][1])  # P(class=1) = P(up)

    def predict_batch(self, X) -> np.ndarray:
        """Predict probabilities for multiple samples."""
        if not self.is_trained or self.model is None:
            return np.full(len(X), 0.5)
        X = np.atleast_2d(X).astype(np.float32)
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, importance_type="gain"):
        """Get feature importance scores."""
        if not self.is_trained or self.model is None:
            return {}
        importance = self.model.feature_importances_
        if self.feature_names and len(self.feature_names) == len(importance):
            return dict(sorted(
                zip(self.feature_names, importance),
                key=lambda x: x[1], reverse=True,
            ))
        return dict(enumerate(importance))

    def save(self, symbol: str):
        """Save model to disk."""
        if not self.is_trained:
            return
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker)
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
        }, os.path.join(path, "lgbm_model.pkl"))

    def load(self, symbol: str) -> bool:
        """Load model from disk. Returns True if successful."""
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker, "lgbm_model.pkl")
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.is_trained = True
        return True

    def _compute_metrics(self, X_val, y_val):
        """Compute validation metrics."""
        if X_val is None or y_val is None:
            return {}
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        preds = self.model.predict(X_val)
        proba = self.model.predict_proba(X_val)[:, 1]
        acc = float(accuracy_score(y_val, preds))
        return {
            "accuracy": acc,
            "test_accuracy": acc,   # alias used by app.py progress display
            "f1": float(f1_score(y_val, preds, zero_division=0)),
            "auc": float(roc_auc_score(y_val, proba)) if len(set(y_val)) > 1 else 0.0,
        }
