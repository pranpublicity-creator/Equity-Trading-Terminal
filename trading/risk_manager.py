"""
Risk Manager
Portfolio-level risk management: position sizing, risk budgets, exposure limits.
"""
import logging

import config

logger = logging.getLogger(__name__)


class RiskManager:
    """Portfolio-level risk management."""

    def __init__(self, trade_engine=None):
        self.trade_engine = trade_engine
        self.total_capital = config.TOTAL_TRADING_CAPITAL

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate optimal position size using modified Kelly criterion.

        Risk per trade = MAX_PORTFOLIO_RISK / MAX_CONCURRENT_POSITIONS
        Size = risk_amount / risk_per_share
        Capped at MAX_CAPITAL_PER_STOCK.
        """
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0 or entry_price <= 0:
            return 0

        # Risk budget per position
        risk_amount = self.total_capital * config.MAX_PORTFOLIO_RISK / config.MAX_CONCURRENT_POSITIONS

        # Position size from risk
        qty_from_risk = int(risk_amount / risk_per_share)

        # Cap by max capital allocation
        max_capital = self.total_capital * config.MAX_CAPITAL_PER_STOCK
        qty_from_capital = int(max_capital / entry_price)

        qty = max(1, min(qty_from_risk, qty_from_capital))
        return qty

    def can_take_trade(self, risk_per_share: float, entry_price: float) -> bool:
        """Check if portfolio can absorb another trade's risk."""
        current_risk = self._get_current_portfolio_risk()
        new_risk = risk_per_share * self.calculate_position_size(entry_price, entry_price - risk_per_share)
        max_risk = self.total_capital * config.MAX_PORTFOLIO_RISK

        return (current_risk + new_risk) <= max_risk

    def get_portfolio_summary(self) -> dict:
        """Get current portfolio risk status."""
        active = []
        total_exposure = 0.0
        total_risk = 0.0
        total_unrealized_pnl = 0.0

        if self.trade_engine:
            active = self.trade_engine.get_active_positions()
            for pos in active:
                exposure = pos.current_price * pos.quantity
                total_exposure += exposure
                risk = abs(pos.entry_price - pos.stop_loss) * pos.quantity
                total_risk += risk
                total_unrealized_pnl += pos.unrealized_pnl

        return {
            "total_capital": self.total_capital,
            "active_positions": len(active),
            "max_positions": config.MAX_CONCURRENT_POSITIONS,
            "total_exposure": round(total_exposure, 2),
            "exposure_pct": round(total_exposure / self.total_capital * 100, 2) if self.total_capital > 0 else 0,
            "total_risk": round(total_risk, 2),
            "risk_pct": round(total_risk / self.total_capital * 100, 2) if self.total_capital > 0 else 0,
            "max_risk_pct": config.MAX_PORTFOLIO_RISK * 100,
            "unrealized_pnl": round(total_unrealized_pnl, 2),
            "daily_pnl": round(self.trade_engine.get_daily_pnl(), 2) if self.trade_engine else 0,
            "available_capital": round(self.total_capital - total_exposure, 2),
        }

    def _get_current_portfolio_risk(self) -> float:
        """Sum of risk across all active positions."""
        if not self.trade_engine:
            return 0.0

        total = 0.0
        for pos in self.trade_engine.get_active_positions():
            total += abs(pos.entry_price - pos.stop_loss) * pos.quantity
        return total
