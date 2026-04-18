"""
Model Manager
Handles loading, saving, versioning, and prediction lifecycle for all models.
Single entry point for getting predictions from all available models.
"""
import logging
import os
import time
import json

import numpy as np

import config
from models.lgbm_model import LightGBMModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from models.tft_model import TemporalFusionTransformerModel
from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel
from models.meta_learner import MetaLearner

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages all ML models for prediction and lifecycle."""

    def __init__(self):
        self._models = {}  # {symbol: {model_name: model_instance}}
        self._load_times = {}  # {symbol: timestamp}

    def predict(self, symbol: str, tabular_features, sequence_features=None,
                close_prices=None, df=None, pattern_confidence=0.0,
                regime=None, enricher_data=None) -> dict:
        """Get predictions from all available models for a symbol.

        Args:
            symbol: stock symbol
            tabular_features: 2D array (1, n_features) for LightGBM/XGBoost
            sequence_features: 3D array (1, lookback, n_features) for LSTM/TFT
            close_prices: 1D array for ARIMA
            df: OHLCV DataFrame for Prophet
            pattern_confidence: best pattern confidence (0-1)
            regime: current market regime string
            enricher_data: market context data

        Returns:
            dict with all model probabilities + meta-learner fusion
        """
        models = self._get_models(symbol)
        predictions = {}

        # --- Tabular models ---
        if tabular_features is not None:
            lgbm = models.get("lgbm")
            if lgbm and lgbm.is_trained:
                predictions["lgbm_prob"] = lgbm.predict(tabular_features)
            else:
                predictions["lgbm_prob"] = 0.5

            xgb = models.get("xgb")
            if xgb and xgb.is_trained:
                predictions["xgb_prob"] = xgb.predict(tabular_features)
            else:
                predictions["xgb_prob"] = 0.5

        # --- Sequence models ---
        if sequence_features is not None:
            lstm = models.get("lstm")
            if lstm and lstm.is_trained:
                predictions["lstm_prob"] = lstm.predict(sequence_features)
            else:
                predictions["lstm_prob"] = 0.5

            tft = models.get("tft")
            if tft and tft.is_trained:
                predictions["tft_prob"] = tft.predict(sequence_features)
            else:
                predictions["tft_prob"] = 0.5
        else:
            predictions["lstm_prob"] = 0.5
            predictions["tft_prob"] = 0.5

        # --- Time series models ---
        arima = models.get("arima")
        if arima and arima.is_trained:
            if close_prices is not None and arima.is_stale():
                arima.update(close_prices[-50:])
            arima_result = arima.predict()
            predictions["arima_prob"] = arima_result["probability"]
            predictions["arima_trend"] = arima_result["trend_direction"]
        else:
            predictions["arima_prob"] = 0.5
            predictions["arima_trend"] = "FLAT"

        prophet = models.get("prophet")
        if prophet and prophet.is_trained:
            prophet_result = prophet.predict()
            predictions["prophet_prob"] = prophet_result["probability"]
            predictions["prophet_seasonal"] = prophet_result["seasonal_bias"]
        else:
            predictions["prophet_prob"] = 0.5
            predictions["prophet_seasonal"] = "FLAT"

        # --- Context signals ---
        predictions["pattern_confidence"] = pattern_confidence

        fii_signal = 0.5
        oi_signal = 0.5
        if enricher_data:
            fii_net = enricher_data.get("fii_net", 0)
            fii_signal = 0.5 + np.clip(fii_net / 10000.0, -0.5, 0.5)
            pcr = enricher_data.get("pcr", 1.0)
            oi_signal = 0.5 + np.clip((pcr - 1.0) / 1.0, -0.5, 0.5)
        predictions["fii_signal"] = fii_signal
        predictions["oi_signal"] = oi_signal

        # --- Meta-learner fusion ---
        meta = models.get("meta_learner")
        if meta:
            fusion = meta.predict(predictions, regime)
            predictions["meta_score"] = fusion["confidence"]
            predictions["final_probability"] = fusion["final_probability"]
            predictions["used_trained_meta"] = fusion["used_trained_model"]
        else:
            # Fallback: create meta-learner with static weights
            fallback_meta = MetaLearner()
            fusion = fallback_meta.predict(predictions, regime)
            predictions["meta_score"] = fusion["confidence"]
            predictions["final_probability"] = fusion["final_probability"]
            predictions["used_trained_meta"] = False

        return predictions

    def _get_models(self, symbol: str) -> dict:
        """Load all models for a symbol (cached in memory)."""
        if symbol in self._models:
            return self._models[symbol]

        models = {}

        # Load each model type
        lgbm = LightGBMModel()
        if lgbm.load(symbol):
            models["lgbm"] = lgbm

        xgb = XGBoostModel()
        if xgb.load(symbol):
            models["xgb"] = xgb

        lstm = LSTMModel()
        if lstm.load(symbol):
            models["lstm"] = lstm

        tft = TemporalFusionTransformerModel()
        if tft.load(symbol):
            models["tft"] = tft

        arima = ARIMAModel()
        if arima.load(symbol):
            models["arima"] = arima

        prophet = ProphetModel()
        if prophet.load(symbol):
            models["prophet"] = prophet

        meta = MetaLearner()
        if meta.load(symbol):
            models["meta_learner"] = meta

        self._models[symbol] = models
        self._load_times[symbol] = time.time()

        loaded_names = list(models.keys())
        logger.info(f"Loaded models for {symbol}: {loaded_names}")
        return models

    def invalidate(self, symbol: str):
        """Clear cached models for a symbol (force reload)."""
        self._models.pop(symbol, None)
        self._load_times.pop(symbol, None)

    def is_stale(self, symbol: str, max_age_days: int = 7) -> bool:
        """Check if models for a symbol need retraining."""
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        meta_path = os.path.join(config.MODELS_DIR, ticker, "metadata.json")
        if not os.path.exists(meta_path):
            return True
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            trained_at = meta.get("trained_at", 0)
            age_days = (time.time() - trained_at) / 86400
            return age_days > max_age_days
        except Exception:
            return True

    def save_metadata(self, symbol: str, metrics: dict):
        """Save training metadata for a symbol."""
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker)
        os.makedirs(path, exist_ok=True)
        meta = {
            "symbol": symbol,
            "trained_at": time.time(),
            "metrics": metrics,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def list_models(self) -> dict:
        """List all trained models with their status."""
        result = {}
        models_dir = config.MODELS_DIR
        if not os.path.exists(models_dir):
            return result

        for ticker in os.listdir(models_dir):
            ticker_path = os.path.join(models_dir, ticker)
            if not os.path.isdir(ticker_path):
                continue

            model_files = [f for f in os.listdir(ticker_path) if f.endswith((".pkl", ".keras", ".pt"))]
            meta_path = os.path.join(ticker_path, "metadata.json")
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                except Exception:
                    pass

            result[ticker] = {
                "model_files": model_files,
                "trained_at": meta.get("trained_at"),
                "metrics": meta.get("metrics", {}),
            }

        return result

    def has_models(self, symbol: str) -> bool:
        """Check if any trained models exist for a symbol."""
        models = self._get_models(symbol)
        return len(models) > 0
