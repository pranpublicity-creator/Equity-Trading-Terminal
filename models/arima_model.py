"""
ARIMA Model (TREND validation)
Uses pmdarima auto_arima for automatic order selection.
Provides trend direction and 6-bar ahead forecast with confidence intervals.
Cached hourly per stock — lightweight and fast (~10 sec).
"""
import logging
import os
import time

import numpy as np
import joblib

import config

logger = logging.getLogger(__name__)

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False
    logger.warning("pmdarima not installed — arima_model disabled")


class ARIMAModel:
    """ARIMA trend direction and forecast model."""

    def __init__(self):
        self.model = None
        self.is_trained = False
        self._last_fit_time = 0
        self._cache_ttl = 3600  # Re-fit every 1 hour

    def train(self, close_prices: np.ndarray):
        """Fit auto_arima on close price series.

        Args:
            close_prices: 1D array of closing prices (at least 60 values)

        Returns:
            dict with training info
        """
        if not HAS_PMDARIMA:
            logger.error("pmdarima not available")
            return {}

        if len(close_prices) < 60:
            logger.warning(f"ARIMA needs >= 60 prices, got {len(close_prices)}")
            return {}

        # Use last 50 bars only — smaller window keeps fitting <5s on any series
        prices = close_prices[-50:].copy()

        # Everything (auto_arima + fallback) runs inside one daemon thread.
        # The main training loop waits at most ARIMA_TIMEOUT seconds then
        # moves on; the daemon thread is abandoned (cannot be killed in CPython,
        # but it will eventually finish and GC itself — it won't block the app).
        import threading
        ARIMA_TIMEOUT = 20   # seconds hard cap for the whole block

        _result = [None]
        _exc    = [None]

        def _fit_all():
            try:
                model = pm.auto_arima(
                    prices,
                    seasonal=False,
                    stepwise=True,
                    max_p=2,
                    max_q=2,
                    max_d=1,
                    start_p=0,
                    start_q=0,
                    information_criterion="aic",
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False,
                    n_jobs=1,
                )
                _result[0] = model
            except Exception:
                # auto_arima failed — try bare ARIMA(1,1,0) as fallback
                try:
                    _result[0] = pm.ARIMA(order=(1, 1, 0)).fit(prices)
                except Exception as e2:
                    _exc[0] = e2

        t = threading.Thread(target=_fit_all, daemon=True)
        t.start()
        t.join(timeout=ARIMA_TIMEOUT)

        if t.is_alive():
            # Both auto_arima and fallback timed out — skip ARIMA for this symbol
            logger.warning(
                f"ARIMA fitting exceeded {ARIMA_TIMEOUT}s hard cap — "
                "marking untrained, signal fusion will ignore ARIMA for this symbol"
            )
            self.is_trained = False
            return {"error": "timeout"}

        if _exc[0] is not None:
            logger.error(f"ARIMA training failed: {_exc[0]}")
            return {"error": str(_exc[0])}

        if _result[0] is None:
            logger.warning("ARIMA returned no model")
            return {}

        self.model = _result[0]
        self.is_trained = True
        self._last_fit_time = time.time()
        order = self.model.order
        logger.info(f"ARIMA fitted — order: {order}")
        return {"order": str(order), "aic": float(self.model.aic())}

    def predict(self, n_periods: int = None) -> dict:
        """Forecast future prices and determine trend direction.

        Args:
            n_periods: number of bars to forecast (default: ML_PREDICTION_HORIZON)

        Returns:
            dict with trend_direction, trend_confidence, forecast, conf_int
        """
        n_periods = n_periods or config.ML_PREDICTION_HORIZON

        if not self.is_trained or self.model is None:
            return {
                "trend_direction": "FLAT",
                "trend_confidence": 0.0,
                "forecast": [],
                "probability": 0.5,
            }

        try:
            forecast, conf_int = self.model.predict(
                n_periods=n_periods,
                return_conf_int=True,
                alpha=0.05,
            )

            # Determine trend direction from forecast slope
            if len(forecast) >= 2:
                slope = (forecast[-1] - forecast[0]) / len(forecast)
                last_price = self.model.arima_res_.data.endog[-1]
                slope_pct = slope / last_price if last_price > 0 else 0

                if slope_pct > 0.001:
                    trend_direction = "UP"
                    trend_confidence = min(abs(slope_pct) / 0.005, 1.0)
                elif slope_pct < -0.001:
                    trend_direction = "DOWN"
                    trend_confidence = min(abs(slope_pct) / 0.005, 1.0)
                else:
                    trend_direction = "FLAT"
                    trend_confidence = 0.3
            else:
                trend_direction = "FLAT"
                trend_confidence = 0.0

            # Map to probability: UP=0.8, FLAT=0.5, DOWN=0.2 (adjusted by confidence)
            if trend_direction == "UP":
                probability = 0.5 + 0.3 * trend_confidence
            elif trend_direction == "DOWN":
                probability = 0.5 - 0.3 * trend_confidence
            else:
                probability = 0.5

            return {
                "trend_direction": trend_direction,
                "trend_confidence": float(trend_confidence),
                "forecast": forecast.tolist(),
                "conf_int": conf_int.tolist(),
                "probability": float(np.clip(probability, 0.0, 1.0)),
            }

        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            return {
                "trend_direction": "FLAT",
                "trend_confidence": 0.0,
                "forecast": [],
                "probability": 0.5,
            }

    def is_stale(self) -> bool:
        """Check if model needs re-fitting."""
        if not self.is_trained:
            return True
        return (time.time() - self._last_fit_time) > self._cache_ttl

    def update(self, new_prices: np.ndarray):
        """Update model with new observations (faster than full refit)."""
        if not self.is_trained or self.model is None:
            return self.train(new_prices)

        try:
            self.model.update(new_prices)
            self._last_fit_time = time.time()
            return {"updated": True}
        except Exception:
            return self.train(new_prices)

    def save(self, symbol: str):
        if not self.is_trained or self.model is None:
            return
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker)
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "last_fit_time": self._last_fit_time,
        }, os.path.join(path, "arima_model.pkl"))

    def load(self, symbol: str) -> bool:
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        path = os.path.join(config.MODELS_DIR, ticker, "arima_model.pkl")
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
