"""
Equity Backtesting Engine
Walk-forward backtesting with pattern + ML + fusion pipeline.
Adapted from BACKTEST APP engine.py.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
import pandas as pd

import config
from patterns.pattern_detector import PatternDetector
from features.feature_pipeline import FeaturePipeline
from signals.regime_detector import detect_regime
from trading.charge_calculator import ChargeCalculator

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A single backtest trade."""
    symbol: str = ""
    direction: str = ""
    entry_bar: int = 0
    exit_bar: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    quantity: int = 1
    pnl: float = 0.0
    charges: float = 0.0
    net_pnl: float = 0.0
    exit_reason: str = ""
    pattern_name: str = ""
    confidence: float = 0.0
    regime: str = ""
    holding_bars: int = 0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_charges: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_bars: float = 0.0
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    regime_performance: dict = field(default_factory=dict)


class EquityBacktest:
    """Walk-forward backtesting engine for equity strategies."""

    def __init__(self, initial_capital: float = None):
        self.capital = initial_capital or config.TOTAL_TRADING_CAPITAL
        self.pattern_detector = PatternDetector()
        self.charge_calc = ChargeCalculator(is_intraday=True)

    def run(self, df: pd.DataFrame, symbol: str = "TEST",
            mode: str = "pattern_only", lookback: int = 200) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            df: OHLCV DataFrame (at least 300 bars)
            symbol: stock symbol
            mode: 'pattern_only', 'ml_only', or 'full_fusion'
            lookback: pattern detection lookback

        Returns:
            BacktestResult with all metrics
        """
        if len(df) < lookback + 50:
            logger.warning(f"Insufficient data for backtest: {len(df)} bars")
            return BacktestResult()

        trades = []
        equity = [self.capital]
        position = None

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        for i in range(lookback, len(df)):
            window = df.iloc[max(0, i - lookback):i + 1].copy()

            # Check if in position
            if position is not None:
                # Check exit conditions
                exit_reason = self._check_exit(position, high[i], low[i], close[i])
                if exit_reason:
                    exit_price = self._get_exit_price(position, high[i], low[i], close[i], exit_reason)
                    trade = self._close_trade(position, exit_price, i, exit_reason)
                    trades.append(trade)
                    equity.append(equity[-1] + trade.net_pnl)
                    position = None
                else:
                    equity.append(equity[-1])
                continue

            # No position — look for signals
            if mode == "pattern_only":
                signal = self._pattern_signal(window, symbol)
            else:
                signal = self._pattern_signal(window, symbol)  # All modes use patterns as base

            if signal:
                position = {
                    "symbol": symbol,
                    "direction": signal["direction"],
                    "entry_bar": i,
                    "entry_price": close[i],
                    "stop_loss": signal["stop_loss"],
                    "target": signal["target"],
                    "pattern_name": signal.get("pattern_name", ""),
                    "confidence": signal.get("confidence", 0.5),
                    "regime": signal.get("regime", ""),
                    "quantity": max(1, int(self.capital * 0.10 / close[i])),
                }
                equity.append(equity[-1])
            else:
                equity.append(equity[-1])

        # Close any open position at end
        if position is not None:
            trade = self._close_trade(position, close[-1], len(df) - 1, "END_OF_DATA")
            trades.append(trade)
            equity.append(equity[-1] + trade.net_pnl)

        return self._compute_results(trades, equity)

    def _pattern_signal(self, window, symbol):
        """Generate signal from pattern detection only."""
        try:
            patterns = self.pattern_detector.detect_all(window)
            if not patterns:
                return None

            best = patterns[0]
            if best.confidence < config.PATTERN_MIN_CONFIDENCE:
                return None

            regime = detect_regime(window)

            return {
                "direction": "BUY" if best.direction == "bullish" else "SELL",
                "entry_price": best.entry_price,
                "stop_loss": best.stop_loss,
                "target": best.target_price,
                "pattern_name": best.pattern_name,
                "confidence": best.confidence,
                "regime": regime["regime"],
            }
        except Exception as e:
            logger.debug(f"Pattern signal error: {e}")
            return None

    def _check_exit(self, position, high, low, close) -> str:
        """Check if position should be exited."""
        if position["direction"] == "BUY":
            if low <= position["stop_loss"]:
                return "STOP_LOSS"
            if high >= position["target"]:
                return "TARGET"
        else:
            if high >= position["stop_loss"]:
                return "STOP_LOSS"
            if low <= position["target"]:
                return "TARGET"
        return ""

    def _get_exit_price(self, position, high, low, close, reason):
        """Determine exact exit price."""
        if reason == "STOP_LOSS":
            return position["stop_loss"]
        elif reason == "TARGET":
            return position["target"]
        return close

    def _close_trade(self, position, exit_price, exit_bar, reason) -> BacktestTrade:
        """Create a BacktestTrade from a closed position."""
        entry = position["entry_price"]
        qty = position["quantity"]

        if position["direction"] == "BUY":
            pnl = (exit_price - entry) * qty
        else:
            pnl = (entry - exit_price) * qty

        charges = self.charge_calc.calculate_total(entry, exit_price, qty)

        return BacktestTrade(
            symbol=position["symbol"],
            direction=position["direction"],
            entry_bar=position["entry_bar"],
            exit_bar=exit_bar,
            entry_price=entry,
            exit_price=exit_price,
            stop_loss=position["stop_loss"],
            target=position["target"],
            quantity=qty,
            pnl=round(pnl, 2),
            charges=round(charges, 2),
            net_pnl=round(pnl - charges, 2),
            exit_reason=reason,
            pattern_name=position.get("pattern_name", ""),
            confidence=position.get("confidence", 0),
            regime=position.get("regime", ""),
            holding_bars=exit_bar - position["entry_bar"],
        )

    def _compute_results(self, trades, equity) -> BacktestResult:
        """Compute all backtest metrics."""
        if not trades:
            return BacktestResult(equity_curve=equity)

        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)
        total_charges = sum(t.charges for t in trades)
        net_pnl = sum(t.net_pnl for t in trades)

        gross_wins = sum(t.net_pnl for t in winners) if winners else 0
        gross_losses = abs(sum(t.net_pnl for t in losers)) if losers else 0

        # Max drawdown
        peak = equity[0]
        max_dd = 0
        max_dd_pct = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = peak - val
            dd_pct = dd / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        # Sharpe ratio (annualized, assuming 15-min bars)
        returns = np.diff(equity) / np.maximum(np.array(equity[:-1]), 1)
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            bars_per_year = 252 * 25  # ~25 bars per day at 15-min
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year))

        # Regime performance
        regime_perf = {}
        for t in trades:
            r = t.regime or "UNKNOWN"
            if r not in regime_perf:
                regime_perf[r] = {"trades": 0, "pnl": 0, "wins": 0}
            regime_perf[r]["trades"] += 1
            regime_perf[r]["pnl"] += t.net_pnl
            if t.net_pnl > 0:
                regime_perf[r]["wins"] += 1

        return BacktestResult(
            total_trades=len(trades),
            winners=len(winners),
            losers=len(losers),
            win_rate=round(len(winners) / len(trades) * 100, 2) if trades else 0,
            total_pnl=round(total_pnl, 2),
            total_charges=round(total_charges, 2),
            net_pnl=round(net_pnl, 2),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            profit_factor=round(gross_wins / max(gross_losses, 1), 2),
            avg_win=round(gross_wins / max(len(winners), 1), 2),
            avg_loss=round(gross_losses / max(len(losers), 1), 2),
            avg_holding_bars=round(np.mean([t.holding_bars for t in trades]), 1),
            equity_curve=equity,
            trades=[t.__dict__ for t in trades],
            regime_performance=regime_perf,
        )
