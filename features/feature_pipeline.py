"""
Feature Pipeline
Assembles all features (~180) into a clean matrix ready for ML models.
Handles scaling, missing values, and sequence/tabular formatting.
"""
import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

import config
from features.indicator_engine import compute_all
from features.volume_features import compute_volume_features
from features.pattern_features import encode_pattern_features
from features.market_context_features import build_context_features

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Builds feature matrices from OHLCV data, patterns, and market context."""

    def __init__(self):
        self._scalers = {}  # {symbol: StandardScaler}

    def build_features(self, df, pattern_results=None, enricher_data=None, symbol=None):
        """Build complete feature DataFrame from all sources.

        Args:
            df: OHLCV DataFrame with datetime index
            pattern_results: list of PatternResult (can be None)
            enricher_data: dict from MarketDataEnricher (can be None)
            symbol: stock symbol for context features

        Returns:
            DataFrame with ~180 features, NaN-handled
        """
        # 1. Technical indicators (~60 features)
        feat_df = compute_all(df)

        # 2. Volume features (~10 features)
        delivery_pct = 0
        if enricher_data:
            delivery_pct = enricher_data.get("delivery_pct", 0)
        feat_df = compute_volume_features(feat_df, delivery_pct=delivery_pct)

        # 3. Pattern features (~65 features)
        pattern_feats = encode_pattern_features(pattern_results or [])
        for key, val in pattern_feats.items():
            feat_df[key] = val

        # 4. Market context features (~15 features)
        if enricher_data and symbol:
            current_price = feat_df["close"].iloc[-1] if len(feat_df) > 0 else 0
            atr_val = feat_df["atr"].iloc[-1] if "atr" in feat_df.columns and len(feat_df) > 0 else 0
            context_feats = build_context_features(
                symbol, enricher_data,
                current_price=current_price,
                atr=atr_val,
            )
            for key, val in context_feats.items():
                feat_df[key] = val

        # 5. Handle missing values
        feat_df = feat_df.ffill()
        feat_df = feat_df.fillna(0)

        # Remove non-feature columns
        drop_cols = ["open", "high", "low", "close", "volume", "timestamp"]
        feat_df = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns], errors="ignore")

        return feat_df

    def build_sequence(self, feature_df, lookback=None):
        """Build 3D sequence array for LSTM/TFT models.

        Args:
            feature_df: DataFrame from build_features()
            lookback: number of timesteps (default: ML_LOOKBACK_TIMESTEPS)

        Returns:
            np.array of shape (samples, lookback, num_features)
        """
        lookback = lookback or config.ML_LOOKBACK_TIMESTEPS
        values = feature_df.values.astype(np.float32)

        if len(values) < lookback:
            logger.warning(f"Not enough data for sequence: {len(values)} < {lookback}")
            return np.array([])

        sequences = []
        for i in range(lookback, len(values)):
            sequences.append(values[i - lookback:i])

        return np.array(sequences)

    def build_tabular(self, feature_df):
        """Build 2D tabular array for LightGBM/XGBoost.

        Returns:
            np.array of shape (samples, num_features)
        """
        return feature_df.values.astype(np.float32)

    def get_latest_features(self, feature_df, lookback=None):
        """Get the most recent feature row (tabular) or sequence (for prediction).

        Returns:
            tuple: (tabular_row, sequence_array)
        """
        lookback = lookback or config.ML_LOOKBACK_TIMESTEPS

        # Latest tabular row
        tabular = feature_df.iloc[-1:].values.astype(np.float32)

        # Latest sequence
        if len(feature_df) >= lookback:
            sequence = feature_df.iloc[-lookback:].values.astype(np.float32)
            sequence = sequence.reshape(1, lookback, -1)
        else:
            sequence = None

        return tabular, sequence

    def fit_scaler(self, symbol, feature_df):
        """Fit StandardScaler on training data for a symbol."""
        scaler = StandardScaler()
        scaler.fit(feature_df.values.astype(np.float32))
        self._scalers[symbol] = scaler
        self._save_scaler(symbol, scaler)
        return scaler

    def transform(self, symbol, feature_df):
        """Apply fitted scaler to features."""
        scaler = self._scalers.get(symbol) or self._load_scaler(symbol)
        if scaler is None:
            return feature_df
        values = scaler.transform(feature_df.values.astype(np.float32))
        return pd.DataFrame(values, columns=feature_df.columns, index=feature_df.index)

    def _save_scaler(self, symbol, scaler):
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker)
        os.makedirs(path, exist_ok=True)
        joblib.dump(scaler, os.path.join(path, "scaler.pkl"))

    def _load_scaler(self, symbol):
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker, "scaler.pkl")
        if os.path.exists(path):
            scaler = joblib.load(path)
            self._scalers[symbol] = scaler
            return scaler
        return None

    def get_feature_names(self, sample_df=None):
        """Return list of feature column names."""
        if sample_df is not None:
            drop_cols = {"open", "high", "low", "close", "volume", "timestamp"}
            return [c for c in sample_df.columns if c not in drop_cols]
        return []
