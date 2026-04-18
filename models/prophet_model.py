"""
Prophet Model (SEASONALITY model)
Captures intraday + weekly seasonality patterns specific to Indian markets.
Opening volatility, lunch lull, expiry Thursday, Monday gaps, etc.
Light model, re-fitted weekly per stock.
"""
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

import config

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    logger.warning("Prophet not installed — prophet_model disabled")


class ProphetModel:
    """Prophet seasonality model for intraday/weekly patterns."""

    def __init__(self):
        self.model = None
        self.is_trained = False
        self._last_fit_time = 0
        self._cache_ttl = 7 * 24 * 3600  # Re-fit weekly

    def train(self, df: pd.DataFrame):
        """Fit Prophet on OHLCV data with datetime index.

        Args:
            df: OHLCV DataFrame with datetime index or 'timestamp' column

        Returns:
            dict with training info
        """
        if not HAS_PROPHET:
            logger.error("Prophet not available")
            return {}

        if len(df) < 100:
            logger.warning(f"Prophet needs >= 100 rows, got {len(df)}")
            return {}

        try:
            # Prepare Prophet format: ds (datetime), y (close price)
            prophet_df = self._prepare_df(df)
            if prophet_df is None or len(prophet_df) < 50:
                return {}

            self.model = Prophet(
                daily_seasonality=True,    # Intraday patterns
                weekly_seasonality=True,   # Day-of-week effects
                yearly_seasonality=False,  # Not enough data usually
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
            )

            # Add Indian market-specific seasonality
            # Expiry week pattern (monthly, ~22 trading days)
            self.model.add_seasonality(
                name="monthly_expiry",
                period=22,
                fourier_order=3,
            )

            # Suppress Prophet's verbose output
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                self.model.fit(prophet_df)
            finally:
                sys.stdout = old_stdout

            self.is_trained = True
            self._last_fit_time = time.time()
            logger.info("Prophet model fitted successfully")
            return {"trained": True, "data_points": len(prophet_df)}

        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            return {}

    def predict(self, periods: int = None) -> dict:
        """Predict seasonal bias for upcoming periods.

        Args:
            periods: number of 15-min bars to forecast

        Returns:
            dict with seasonal_bias, seasonal_strength, probability
        """
        periods = periods or config.ML_PREDICTION_HORIZON

        if not self.is_trained or self.model is None:
            return {
                "seasonal_bias": "FLAT",
                "seasonal_strength": 0.0,
                "probability": 0.5,
            }

        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, freq="15min")
            forecast = self.model.predict(future)

            # Get the forecast for the prediction horizon
            recent = forecast.tail(periods)
            trend_vals = recent["trend"].values
            yhat = recent["yhat"].values

            # Seasonal components
            if len(yhat) >= 2:
                seasonal_change = (yhat[-1] - yhat[0]) / max(abs(yhat[0]), 1e-10)

                if seasonal_change > 0.001:
                    seasonal_bias = "UP"
                    seasonal_strength = min(abs(seasonal_change) / 0.005, 1.0)
                elif seasonal_change < -0.001:
                    seasonal_bias = "DOWN"
                    seasonal_strength = min(abs(seasonal_change) / 0.005, 1.0)
                else:
                    seasonal_bias = "FLAT"
                    seasonal_strength = 0.2
            else:
                seasonal_bias = "FLAT"
                seasonal_strength = 0.0

            # Map to probability
            if seasonal_bias == "UP":
                probability = 0.5 + 0.2 * seasonal_strength
            elif seasonal_bias == "DOWN":
                probability = 0.5 - 0.2 * seasonal_strength
            else:
                probability = 0.5

            return {
                "seasonal_bias": seasonal_bias,
                "seasonal_strength": float(seasonal_strength),
                "probability": float(np.clip(probability, 0.0, 1.0)),
                "forecast": yhat.tolist(),
            }

        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            return {
                "seasonal_bias": "FLAT",
                "seasonal_strength": 0.0,
                "probability": 0.5,
            }

    def get_current_seasonality(self) -> dict:
        """Get the seasonal component for the current time of day / day of week."""
        if not self.is_trained or self.model is None:
            return {"intraday_bias": 0.0, "weekly_bias": 0.0}

        try:
            now = datetime.now()
            single = pd.DataFrame({"ds": [now]})
            forecast = self.model.predict(single)

            result = {"intraday_bias": 0.0, "weekly_bias": 0.0}
            if "daily" in forecast.columns:
                result["intraday_bias"] = float(forecast["daily"].iloc[0])
            if "weekly" in forecast.columns:
                result["weekly_bias"] = float(forecast["weekly"].iloc[0])
            return result

        except Exception:
            return {"intraday_bias": 0.0, "weekly_bias": 0.0}

    def is_stale(self) -> bool:
        if not self.is_trained:
            return True
        return (time.time() - self._last_fit_time) > self._cache_ttl

    def save(self, symbol: str):
        if not self.is_trained or self.model is None:
            return
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker)
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "last_fit_time": self._last_fit_time,
        }, os.path.join(path, "prophet_model.pkl"))

    def load(self, symbol: str) -> bool:
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker, "prophet_model.pkl")
        if not os.path.exists(path):
            return False
        try:
            data = joblib.load(path)
            self.model = data["model"]
            self._last_fit_time = data.get("last_fit_time", 0)
            self.is_trained = True
            return True
        except Exception:
            return False

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert OHLCV DataFrame to Prophet format (ds, y)."""
        try:
            if "timestamp" in df.columns:
                ds = pd.to_datetime(df["timestamp"], unit="s")
            elif isinstance(df.index, pd.DatetimeIndex):
                ds = df.index
            else:
                return None

            prophet_df = pd.DataFrame({
                "ds": ds,
                "y": df["close"].values,
            })
            prophet_df = prophet_df.dropna()
            return prophet_df

        except Exception:
            return None
