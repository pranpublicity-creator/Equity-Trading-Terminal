"""
Webhook Executor
Receives and processes incoming webhook signals (e.g., from TradingView).
Adapted from COMMODITY APP for NSE equity symbols.
"""
import json
import logging
import time

import config

logger = logging.getLogger(__name__)


class WebhookExecutor:
    """Processes incoming webhook trade alerts."""

    def __init__(self, trade_engine=None, signal_engine=None):
        self.trade_engine = trade_engine
        self.signal_engine = signal_engine
        self._last_webhook = 0
        self._cooldown = 5  # seconds between webhooks

    def process_webhook(self, data: dict) -> dict:
        """Process an incoming webhook payload.

        Expected format:
        {
            "symbol": "RELIANCE",
            "action": "BUY" or "SELL",
            "price": 2450.50,
            "sl": 2420.0,
            "target": 2510.0,
            "source": "tradingview"
        }

        Returns:
            dict with status and details
        """
        now = time.time()
        if now - self._last_webhook < self._cooldown:
            return {"status": "rejected", "reason": "cooldown"}
        self._last_webhook = now

        # Validate payload
        required = ["symbol", "action"]
        for key in required:
            if key not in data:
                return {"status": "error", "reason": f"Missing field: {key}"}

        # Normalize symbol
        symbol = self._normalize_symbol(data["symbol"])
        action = data["action"].upper()
        price = float(data.get("price", 0))
        sl = float(data.get("sl", 0))
        target = float(data.get("target", 0))

        if action not in ("BUY", "SELL"):
            return {"status": "error", "reason": f"Invalid action: {action}"}

        logger.info(f"Webhook received: {action} {symbol} @ {price}")

        # Create a minimal signal
        from signals.signal_engine import TradeSignal
        signal = TradeSignal(
            symbol=symbol,
            direction=action,
            confidence=70.0,  # Webhook signals get moderate confidence
            strength="MODERATE",
            entry_price=price,
            stop_loss=sl,
            target_price=target,
            risk_reward=abs(target - price) / abs(price - sl) if abs(price - sl) > 0 else 0,
            timestamp=now,
        )

        # Process through trade engine
        if self.trade_engine:
            position = self.trade_engine.process_signal(signal)
            if position:
                return {
                    "status": "accepted",
                    "position_id": position.id,
                    "symbol": symbol,
                    "direction": action,
                    "quantity": position.quantity,
                }

        return {"status": "queued", "symbol": symbol, "direction": action}

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert various symbol formats to Fyers format."""
        symbol = symbol.strip().upper()

        # Already in Fyers format
        if symbol.startswith("NSE:") and symbol.endswith("-EQ"):
            return symbol

        # Strip common suffixes
        symbol = symbol.replace(".NS", "").replace(".NSE", "")
        symbol = symbol.replace("-EQ", "")

        # Add Fyers prefix/suffix
        if not symbol.startswith("NSE:"):
            symbol = f"NSE:{symbol}-EQ"
        elif not symbol.endswith("-EQ"):
            symbol = f"{symbol}-EQ"

        return symbol
