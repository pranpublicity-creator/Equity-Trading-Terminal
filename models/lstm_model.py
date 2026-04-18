"""
LSTM Model (SEQUENCE pattern model)
Captures temporal dependencies in price action sequences.
Input: (batch, 60, ~180). Output: P(price_up) in [0,1].
"""
import logging
import os

import numpy as np
import joblib

import config

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logger.warning("TensorFlow not installed — lstm_model disabled")


class LSTMModel:
    """LSTM classifier for sequence-based price direction prediction."""

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.lookback = config.ML_LOOKBACK_TIMESTEPS

    def build(self, n_features: int = 180, lookback: int = None):
        """Build LSTM architecture."""
        if not HAS_TF:
            return

        lookback = lookback or self.lookback

        self.model = keras.Sequential([
            layers.Input(shape=(lookback, n_features)),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train the LSTM model.

        Args:
            X_train: 3D array (samples, lookback, features)
            y_train: binary labels
            X_val, y_val: validation data
            epochs: max training epochs
            batch_size: training batch size

        Returns:
            dict with validation metrics
        """
        if not HAS_TF:
            logger.error("TensorFlow not available")
            return {}

        if self.model is None:
            self.build(n_features=X_train.shape[2], lookback=X_train.shape[1])

        # Class weights for imbalanced data
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        total = len(y_train)
        class_weight = {
            0: total / (2 * max(n_neg, 1)),
            1: total / (2 * max(n_pos, 1)),
        }

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=10,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0,
        )

        self.is_trained = True

        metrics = self._compute_metrics(X_val, y_val) if X_val is not None else {}
        logger.info(f"LSTM trained — metrics: {metrics}")
        return metrics

    def predict(self, X) -> float:
        """Predict probability of price going up.

        Args:
            X: 3D array (1, lookback, features) or (lookback, features)

        Returns:
            float probability in [0, 1]
        """
        if not self.is_trained or self.model is None:
            return 0.5

        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])

        X = X.astype(np.float32)
        pred = self.model.predict(X, verbose=0)
        return float(pred[0][0])

    def predict_batch(self, X) -> np.ndarray:
        if not self.is_trained or self.model is None:
            return np.full(len(X), 0.5)
        X = X.astype(np.float32)
        return self.model.predict(X, verbose=0).flatten()

    def save(self, symbol: str):
        if not self.is_trained or self.model is None:
            return
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker)
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "lstm_model.keras"))

    def load(self, symbol: str) -> bool:
        if not HAS_TF:
            return False
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker, "lstm_model.keras")
        if not os.path.exists(path):
            return False
        self.model = keras.models.load_model(path)
        self.is_trained = True
        return True

    def _compute_metrics(self, X_val, y_val):
        if X_val is None or y_val is None:
            return {}
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        proba = self.model.predict(X_val, verbose=0).flatten()
        preds = (proba > 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y_val, preds)),
            "f1": float(f1_score(y_val, preds, zero_division=0)),
            "auc": float(roc_auc_score(y_val, proba)) if len(set(y_val)) > 1 else 0.0,
        }
