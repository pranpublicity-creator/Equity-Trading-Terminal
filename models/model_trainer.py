"""
Model Trainer
Walk-forward training pipeline for all 7 models + meta-learner.
Handles data preparation, time-ordered splits, and out-of-fold prediction generation.
"""
import logging
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import config
from models.lgbm_model import LightGBMModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from models.tft_model import TemporalFusionTransformerModel
from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel
from models.meta_learner import MetaLearner

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains all models for a symbol using walk-forward cross-validation."""

    def __init__(self, feature_pipeline, data_engine=None):
        self.feature_pipeline = feature_pipeline
        self.data_engine = data_engine

    def train_all_models(self, symbol: str, df: pd.DataFrame,
                         pattern_results=None, enricher_data=None) -> dict:
        """Train all 7 models + meta-learner for a symbol.

        Args:
            symbol: stock symbol
            df: OHLCV DataFrame (at least 90 days of data)
            pattern_results: list of PatternResult (for feature encoding)
            enricher_data: dict from MarketDataEnricher

        Returns:
            dict with model metrics and training status
        """
        start_time = time.time()
        results = {}

        # Build full feature matrix
        logger.info(f"Building features for {symbol}...")
        feature_df = self.feature_pipeline.build_features(
            df, pattern_results, enricher_data, symbol
        )

        if len(feature_df) < 200:
            logger.warning(f"Not enough data for training {symbol}: {len(feature_df)} rows")
            return {"error": "insufficient_data", "rows": len(feature_df)}

        # Create labels: 1 if price goes up in next N bars, 0 otherwise
        horizon = config.ML_PREDICTION_HORIZON
        close = df["close"].values[-len(feature_df):]
        labels = self._create_labels(close, horizon)

        # Trim to aligned length
        valid_len = len(labels)
        feature_df = feature_df.iloc[:valid_len]
        labels = labels[:valid_len]

        # Fit scaler
        self.feature_pipeline.fit_scaler(symbol, feature_df)
        scaled_df = self.feature_pipeline.transform(symbol, feature_df)

        feature_names = list(scaled_df.columns)
        X_tabular = scaled_df.values.astype(np.float32)
        y = labels.astype(np.int32)

        # Time-ordered split: 70% train, 15% val, 15% test
        n = len(X_tabular)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train, y_train = X_tabular[:train_end], y[:train_end]
        X_val, y_val = X_tabular[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X_tabular[val_end:], y[val_end:]

        logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        # --- Train tabular models ---
        lgbm = LightGBMModel()
        lgbm.build(X_train.shape[1])
        results["lgbm"] = lgbm.train(X_train, y_train, X_val, y_val, feature_names)
        lgbm.save(symbol)

        xgb_model = XGBoostModel()
        xgb_model.build(X_train.shape[1])
        results["xgb"] = xgb_model.train(X_train, y_train, X_val, y_val, feature_names)
        xgb_model.save(symbol)

        # --- Train sequence models ---
        lookback = config.ML_LOOKBACK_TIMESTEPS
        X_seq_train = self.feature_pipeline.build_sequence(
            scaled_df.iloc[:train_end], lookback
        )
        y_seq_train = y[lookback:train_end]

        X_seq_val = self.feature_pipeline.build_sequence(
            scaled_df.iloc[:val_end], lookback
        )
        # Validation sequences start after training sequences
        y_seq_val = y[train_end + lookback:val_end] if train_end + lookback < val_end else np.array([])
        # Properly slice validation sequences
        if len(X_seq_val) > len(X_seq_train):
            X_seq_val = X_seq_val[len(X_seq_train):]
        else:
            X_seq_val = np.array([])

        if len(X_seq_train) > 0:
            # Align lengths
            min_len = min(len(X_seq_train), len(y_seq_train))
            X_seq_train = X_seq_train[:min_len]
            y_seq_train = y_seq_train[:min_len]

            lstm = LSTMModel()
            if len(X_seq_val) > 0 and len(y_seq_val) > 0:
                min_val = min(len(X_seq_val), len(y_seq_val))
                results["lstm"] = lstm.train(
                    X_seq_train, y_seq_train,
                    X_seq_val[:min_val], y_seq_val[:min_val],
                )
            else:
                results["lstm"] = lstm.train(X_seq_train, y_seq_train)
            lstm.save(symbol)

            tft = TemporalFusionTransformerModel()
            if len(X_seq_val) > 0 and len(y_seq_val) > 0:
                min_val = min(len(X_seq_val), len(y_seq_val))
                results["tft"] = tft.train(
                    X_seq_train, y_seq_train,
                    X_seq_val[:min_val], y_seq_val[:min_val],
                )
            else:
                results["tft"] = tft.train(X_seq_train, y_seq_train)
            tft.save(symbol)
        else:
            results["lstm"] = {"error": "insufficient_sequence_data"}
            results["tft"] = {"error": "insufficient_sequence_data"}

        # --- Train time series models ---
        close_prices = df["close"].values
        arima = ARIMAModel()
        results["arima"] = arima.train(close_prices)
        arima.save(symbol)

        prophet = ProphetModel()
        results["prophet"] = prophet.train(df)
        prophet.save(symbol)

        # --- Train meta-learner on out-of-fold predictions ---
        meta_results = self._train_meta_learner(
            symbol, X_tabular, y, feature_names, close_prices, df
        )
        results["meta_learner"] = meta_results

        elapsed = time.time() - start_time
        results["total_time_sec"] = round(elapsed, 1)
        results["symbol"] = symbol

        logger.info(f"All models trained for {symbol} in {elapsed:.1f}s")
        return results

    def _train_meta_learner(self, symbol, X_tabular, y, feature_names, close_prices, df):
        """Train meta-learner on out-of-fold predictions from base models."""
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            meta_features_list = []
            meta_labels_list = []

            for train_idx, test_idx in tscv.split(X_tabular):
                X_tr, X_te = X_tabular[train_idx], X_tabular[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                # Train base models on fold
                lgbm = LightGBMModel()
                lgbm.build(X_tr.shape[1])
                lgbm.train(X_tr, y_tr, feature_names=feature_names)

                xgb_m = XGBoostModel()
                xgb_m.build(X_tr.shape[1])
                xgb_m.train(X_tr, y_tr, feature_names=feature_names)

                # Get out-of-fold predictions
                lgbm_oof = lgbm.predict_batch(X_te)
                xgb_oof = xgb_m.predict_batch(X_te)

                # For sequence models and time-series, use defaults
                lstm_oof = np.full(len(X_te), 0.5)
                tft_oof = np.full(len(X_te), 0.5)
                arima_oof = np.full(len(X_te), 0.5)
                prophet_oof = np.full(len(X_te), 0.5)
                pattern_oof = np.zeros(len(X_te))
                regime_oof = np.zeros(len(X_te))

                fold_meta = np.column_stack([
                    lgbm_oof, xgb_oof, lstm_oof, tft_oof,
                    arima_oof, prophet_oof, pattern_oof, regime_oof,
                ])
                meta_features_list.append(fold_meta)
                meta_labels_list.append(y_te)

            meta_features = np.vstack(meta_features_list)
            meta_labels = np.concatenate(meta_labels_list)

            meta = MetaLearner()
            result = meta.train(meta_features, meta_labels)
            meta.save(symbol)
            return result

        except Exception as e:
            logger.error(f"Meta-learner training failed: {e}")
            return {"error": str(e)}

    def _create_labels(self, close: np.ndarray, horizon: int) -> np.ndarray:
        """Create binary labels: 1 if price is higher after `horizon` bars."""
        n = len(close)
        labels = np.zeros(n, dtype=np.int32)
        for i in range(n - horizon):
            if close[i + horizon] > close[i]:
                labels[i] = 1
        # Last `horizon` bars get label 0 (unknown future)
        return labels[:n - horizon]
