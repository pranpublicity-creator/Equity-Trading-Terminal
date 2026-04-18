"""
Entry Gate
Hard validation before any trade is executed.
NSE market hours, risk limits, cooldowns, circuit limits.
Adapted from COMMODITY APP for equity markets.
"""
import logging
from datetime import datetime, time as dtime

import config

logger = logging.getLogger(__name__)


class EntryGate:
    """Validates that a signal passes all hard entry requirements."""

    def __init__(self, trade_engine=None, risk_manager=None):
        self.trade_engine = trade_engine
        self.risk_manager = risk_manager
        self._cooldown = {}  # {symbol: cooldown_until_datetime}

    def validate(self, symbol: str, direction: str, confidence: float,
                 entry_price: float, stop_loss: float, target_price: float,
                 enricher_data: dict = None, volume: float = 0) -> dict:
        """Run all entry gate checks.

        Args:
            symbol: stock symbol
            direction: 'BUY' or 'SELL'
            confidence: fusion confidence (0-100)
            entry_price: proposed entry price
            stop_loss: proposed stop loss
            target_price: proposed target
            enricher_data: market context dict
            volume: current bar volume

        Returns:
            dict with passed (bool), reason (str if blocked)
        """
        checks = [
            self._check_market_hours(),
            self._check_holiday(),
            self._check_confidence(confidence),
            self._check_risk_reward(entry_price, stop_loss, target_price, direction),
            self._check_max_positions(),
            self._check_portfolio_risk(entry_price, stop_loss),
            self._check_cooldown(symbol),
            self._check_circuit_limit(enricher_data),
            self._check_volume(volume),
        ]

        for check in checks:
            if not check["passed"]:
                logger.info(f"Entry gate BLOCKED {symbol} {direction}: {check['reason']}")
                return check

        return {"passed": True, "reason": ""}

    def _check_market_hours(self) -> dict:
        """NSE hours: 09:20 – 15:20 (5 min buffer each side)."""
        now = datetime.now().time()
        open_time = dtime(9, 15 + config.NO_SIGNAL_FIRST_MIN)
        close_time = dtime(15, 30 - config.NO_SIGNAL_LAST_MIN)

        if now < open_time or now > close_time:
            return {"passed": False, "reason": f"Outside trading hours ({open_time}-{close_time})"}
        return {"passed": True, "reason": ""}

    def _check_holiday(self) -> dict:
        """Check NSE holidays."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today in config.NSE_HOLIDAYS_2026:
            return {"passed": False, "reason": f"NSE holiday: {today}"}
        # Weekend check
        if datetime.now().weekday() >= 5:
            return {"passed": False, "reason": "Weekend — market closed"}
        return {"passed": True, "reason": ""}

    def _check_confidence(self, confidence: float) -> dict:
        """Minimum confidence threshold."""
        if confidence < config.SIGNAL_WEAK_THRESHOLD:
            return {"passed": False, "reason": f"Confidence {confidence:.1f} < min {config.SIGNAL_WEAK_THRESHOLD}"}
        return {"passed": True, "reason": ""}

    def _check_risk_reward(self, entry, sl, target, direction) -> dict:
        """Minimum R:R ratio check."""
        if direction == "BUY":
            risk = entry - sl
            reward = target - entry
        else:
            risk = sl - entry
            reward = entry - target

        if risk <= 0:
            return {"passed": False, "reason": "Invalid SL (risk <= 0)"}

        rr = reward / risk
        if rr < config.MIN_RISK_REWARD:
            return {"passed": False, "reason": f"R:R {rr:.2f} < min {config.MIN_RISK_REWARD}"}
        return {"passed": True, "reason": ""}

    def _check_max_positions(self) -> dict:
        """Check concurrent position limit."""
        if self.trade_engine is None:
            return {"passed": True, "reason": ""}

        active = self.trade_engine.get_active_position_count()
        if active >= config.MAX_CONCURRENT_POSITIONS:
            return {"passed": False, "reason": f"Max positions reached ({active}/{config.MAX_CONCURRENT_POSITIONS})"}
        return {"passed": True, "reason": ""}

    def _check_portfolio_risk(self, entry, sl) -> dict:
        """Check if new trade would exceed portfolio risk budget."""
        if self.risk_manager is None:
            return {"passed": True, "reason": ""}

        risk_per_share = abs(entry - sl)
        if not self.risk_manager.can_take_trade(risk_per_share, entry):
            return {"passed": False, "reason": "Portfolio risk budget exceeded"}
        return {"passed": True, "reason": ""}

    def _check_cooldown(self, symbol: str) -> dict:
        """Check if symbol is in cooldown after consecutive losses."""
        if symbol in self._cooldown:
            until = self._cooldown[symbol]
            if datetime.now() < until:
                remaining = (until - datetime.now()).seconds // 60
                return {"passed": False, "reason": f"Cooldown active ({remaining} min remaining)"}
            else:
                del self._cooldown[symbol]
        return {"passed": True, "reason": ""}

    def _check_circuit_limit(self, enricher_data) -> dict:
        """Skip if stock is within 5% of circuit limit."""
        if enricher_data is None:
            return {"passed": True, "reason": ""}

        circuit_prox = enricher_data.get("circuit_proximity", 1.0)
        if circuit_prox < 0.05:
            return {"passed": False, "reason": f"Near circuit limit (proximity: {circuit_prox:.2%})"}
        return {"passed": True, "reason": ""}

    def _check_volume(self, volume: float) -> dict:
        """Skip illiquid stocks."""
        # Volume check is soft — only block if zero
        if volume <= 0:
            return {"passed": False, "reason": "Zero volume — illiquid"}
        return {"passed": True, "reason": ""}

    def set_cooldown(self, symbol: str, minutes: int = 60):
        """Put a symbol in cooldown."""
        from datetime import timedelta
        self._cooldown[symbol] = datetime.now() + timedelta(minutes=minutes)

    def clear_cooldown(self, symbol: str):
        """Remove cooldown for a symbol."""
        self._cooldown.pop(symbol, None)
