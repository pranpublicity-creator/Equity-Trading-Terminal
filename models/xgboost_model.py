"""
XGBoost Model (CONFIRMATION model)
Different regularization gives uncorrelated errors with LightGBM.
Input: tabular features (1, ~180). Output: P(price_up) in [0,1].
"""
import logging
import os

import numpy as np
import joblib

import config

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not installed — xgboost_model disabled")


class XGBoostModel:
    """XGBoost classifier for price direction confirmation."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False

    def build(self, n_features: int = 180):
        if not HAS_XGB:
            return
        self.model = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1.0,  # Auto-adjusted in train()
            random_state=123,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            use_label_encoder=False,
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        if not HAS_XGB:
            logger.error("XGBoost not available")
            return {}

        if self.model is None:
            self.build(X_train.shape[1])

        self.feature_names = feature_names

        # Auto-adjust scale_pos_weight for class imbalance
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        if n_pos > 0:
            self.model.set_params(scale_pos_weight=float(n_neg / n_pos))

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

        self.is_trained = True
        metrics = self._compute_metrics(X_val, y_val) if X_val is not None else {}
        logger.info(f"XGBoost trained — metrics: {metrics}")
        return metrics

    def predict(self, X) -> float:
        if not self.is_trained or self.model is None:
            return 0.5
        X = np.atleast_2d(X).astype(np.float32)
        proba = self.model.predict_proba(X)
        return float(proba[0][1])

    def predict_batch(self, X) -> np.ndarray:
        if not self.is_trained or self.model is None:
            return np.full(len(X), 0.5)
        X = np.atleast_2d(X).astype(np.float32)
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, importance_type="weight"):
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
        if not self.is_trained:
            return
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker)
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
        }, os.path.join(path, "xgb_model.pkl"))

    def load(self, symbol: str) -> bool:
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker, "xgb_model.pkl")
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.is_trained = True
        return True

    def _compute_metrics(self, X_val, y_val):
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
