"""
Trade Engine
Manages position lifecycle: entry, SL/target monitoring, exit, P&L tracking.
Adapted from COMMODITY APP — removed calendar spreads, added portfolio tracking.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

import config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An active or closed trading position."""
    id: str = ""
    symbol: str = ""
    direction: str = ""          # 'BUY' or 'SELL'
    entry_price: float = 0.0
    quantity: int = 0
    stop_loss: float = 0.0
    target: float = 0.0
    trailing_stop: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    status: str = "PENDING"      # PENDING, ACTIVE, CLOSED, CANCELLED
    entry_time: float = 0.0
    exit_time: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    signal_confidence: float = 0.0
    pattern_name: str = ""
    regime: str = ""
    charges: float = 0.0
    # Signal reasoning — stored so UI can show "why this trade"
    lgbm_prob: float = 0.5
    xgb_prob: float = 0.5
    arima_trend: str = "FLAT"
    adx: float = 0.0
    vol_ratio: float = 1.0
    risk_reward: float = 0.0
    signal_strength: str = ""     # STRONG / MODERATE / WEAK
    # ── Trade-quality (6-point pre-trade scorecard) ─────────────
    quality_score:  float = 0.0
    quality_grade:  str   = ""
    quality_passed: bool  = True
    quality_report: dict  = field(default_factory=dict)
    # ── Timeframe that generated this position ──────────────────
    timeframe: str = "15"          # "5" = intraday, "15" = swing
    # ── Excursion tracking (updated every LTP tick) ─────────────
    mfe_pct: float = 0.0           # Max Favourable Excursion % (best unrealised %)
    mae_pct: float = 0.0           # Max Adverse  Excursion % (worst, stored as negative)
    bars_in_trade: int = 0         # candle-bar counter (incremented on each quote update)


class TradeEngine:
    """Manages trade execution and position tracking."""

    def __init__(self, fyers_manager=None, risk_manager=None, charge_calculator=None):
        self.fyers = fyers_manager
        self.risk_manager = risk_manager
        self.charge_calc = charge_calculator
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.auto_execute = True    # ON by default — toggle off to disable live orders
        self._daily_pnl = 0.0
        # ── Post-trade re-entry cooldown (per-symbol) ───────────────
        # symbol -> unix-ts until which new signals for this symbol are blocked.
        # Populated by _close_position().  Checked in process_signal().
        self._symbol_cooldown_until: Dict[str, float] = {}
        # Consecutive loss tracker for loss-streak amplification
        self._consecutive_losses: Dict[str, int] = {}
        self._load_positions()

    def process_signal(self, signal) -> Optional[Position]:
        """Process a trade signal and create a position.

        Args:
            signal: TradeSignal from SignalEngine

        Returns:
            Position if trade opened, None otherwise
        """
        # ── Duplicate symbol guard ──────────────────────────────
        # Never open a second position in the same stock while one is ACTIVE or PENDING.
        symbol_active = any(
            p.symbol == signal.symbol and p.status in ("ACTIVE", "PENDING")
            for p in self.positions.values()
        )
        if symbol_active:
            logger.debug(f"Duplicate blocked: {signal.symbol} already has an open position")
            return None

        # ── Post-trade re-entry cooldown ────────────────────────
        # Prevents the "close → same pattern still firing → re-enter"
        # loop that produced 10 MAXHEALTH / 7 BRITANNIA trades.
        cd_until = self._symbol_cooldown_until.get(signal.symbol, 0)
        now = time.time()
        if now < cd_until:
            remaining = int(cd_until - now)
            logger.info(
                f"Cooldown blocked: {signal.symbol} re-entry in {remaining}s "
                f"(just closed — pattern still active)"
            )
            return None

        # Check daily loss limit
        if self._daily_pnl <= -config.MAX_LOSS_PER_DAY:
            logger.warning(f"Daily loss limit hit: {self._daily_pnl}")
            return None

        # ── Trade-quality hard gate (AUTO mode only) ─────────────
        # Manual mode keeps the signal queued so the user can override,
        # but in AUTO we must not auto-execute a low-grade trade.
        if (getattr(config, "TRADE_QUALITY_ENABLED", False)
                and self.auto_execute
                and not getattr(signal, "quality_passed", True)):
            grade = getattr(signal, "quality_grade", "?")
            score = getattr(signal, "quality_score", 0.0)
            reasons = (signal.quality_report or {}).get("fail_reasons", [])
            logger.info(
                f"QUALITY-GATE blocked {signal.symbol} {signal.direction} "
                f"grade={grade} score={score:.2f} "
                f"min={config.TRADE_QUALITY_MIN_SCORE} "
                f"fails={reasons[:3]}"
            )
            return None

        # Calculate position size
        quantity = self._calculate_quantity(signal)
        if quantity <= 0:
            logger.warning(f"Quantity = 0 for {signal.symbol}")
            return None

        pos_id = f"{signal.symbol}_{int(time.time())}"
        rr = 0.0
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.target_price - signal.entry_price)
        if risk > 0:
            rr = round(reward / risk, 2)
        position = Position(
            id=pos_id,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            target=signal.target_price,
            current_price=signal.entry_price,
            status="PENDING",
            entry_time=time.time(),
            signal_confidence=signal.confidence,
            pattern_name=signal.pattern_name,
            regime=signal.regime,
            # Reasoning fields
            lgbm_prob=getattr(signal, "lgbm_prob", 0.5),
            xgb_prob=getattr(signal, "xgb_prob", 0.5),
            arima_trend=getattr(signal, "arima_trend", "FLAT"),
            risk_reward=rr,
            signal_strength=getattr(signal, "strength", ""),
            quality_score=getattr(signal,  "quality_score",  0.0),
            quality_grade=getattr(signal,  "quality_grade",  ""),
            quality_passed=getattr(signal, "quality_passed", True),
            quality_report=getattr(signal, "quality_report", {}) or {},
            timeframe=getattr(signal,      "timeframe",      "15"),
        )

        if self.auto_execute:
            # AUTO mode: execute every qualifying signal immediately, no confirmation
            success = self._execute_entry(position)
            if success:
                position.status = "ACTIVE"
                self.positions[pos_id] = position
                self._save_positions()
                logger.info(f"AUTO-EXECUTED: {signal.direction} {signal.symbol} x{quantity} @ {signal.entry_price:.2f} conf={signal.confidence:.1f}")
                return position
        else:
            # MANUAL mode: queue ALL signals — user confirms each one
            position.status = "PENDING"
            self.positions[pos_id] = position
            self._save_positions()
            logger.info(f"PENDING (manual): {signal.direction} {signal.symbol} x{quantity} @ {signal.entry_price:.2f} conf={signal.confidence:.1f}")
            return position

        return None

    def confirm_entry(self, pos_id: str) -> bool:
        """Manually confirm a pending position."""
        pos = self.positions.get(pos_id)
        if pos is None or pos.status != "PENDING":
            return False

        success = self._execute_entry(pos)
        if success:
            pos.status = "ACTIVE"
            self._save_positions()
            return True
        return False

    def update_positions(self, quotes: dict):
        """Update all active positions with latest prices.

        Args:
            quotes: {symbol: {ltp: float, ...}}
        """
        for pos_id, pos in list(self.positions.items()):
            if pos.status != "ACTIVE":
                continue

            quote = quotes.get(pos.symbol, {})
            ltp = quote.get("ltp", pos.current_price)
            pos.current_price = ltp

            # Calculate unrealized P&L
            if pos.direction == "BUY":
                pos.unrealized_pnl = (ltp - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - ltp) * pos.quantity

            # ── MAE / MFE tracking ──────────────────────────────
            if pos.entry_price > 0:
                raw_pct = (ltp - pos.entry_price) / pos.entry_price * 100.0
                # For SELL, price falling = favourable; flip sign so direction-normalised
                dir_pct = raw_pct if pos.direction == "BUY" else -raw_pct
                # MFE = highest dir_pct seen (best moment of the trade)
                pos.mfe_pct = round(max(pos.mfe_pct, dir_pct), 3)
                # MAE = lowest dir_pct seen (worst moment, stored negative)
                pos.mae_pct = round(min(pos.mae_pct, dir_pct), 3)
            pos.bars_in_trade += 1     # rough bar count (≈ 3-s poll intervals)

            # Check trailing stop activation
            self._update_trailing_stop(pos)

            # Check exit conditions
            exit_reason = self._check_exit(pos)
            if exit_reason:
                self._close_position(pos, exit_reason, ltp)

        self._save_positions()

    def _check_exit(self, pos: Position) -> str:
        """Check if position should be exited."""
        ltp = pos.current_price

        if pos.direction == "BUY":
            # Stop loss hit
            if ltp <= pos.stop_loss:
                return "STOP_LOSS"
            # Trailing stop hit
            if pos.trailing_stop > 0 and ltp <= pos.trailing_stop:
                return "TRAILING_STOP"
            # Target hit
            if ltp >= pos.target:
                return "TARGET"
        else:  # SELL
            if ltp >= pos.stop_loss:
                return "STOP_LOSS"
            if pos.trailing_stop > 0 and ltp >= pos.trailing_stop:
                return "TRAILING_STOP"
            if ltp <= pos.target:
                return "TARGET"

        return ""

    def _update_trailing_stop(self, pos: Position):
        """Activate and update trailing stop."""
        risk = abs(pos.entry_price - pos.stop_loss)
        activation_gain = risk * config.TRAILING_STOP_ACTIVATION

        if pos.direction == "BUY":
            gain = pos.current_price - pos.entry_price
            if gain >= activation_gain:
                new_trail = pos.current_price - risk * config.TRAILING_STOP_ATR_MULT
                pos.trailing_stop = max(pos.trailing_stop, new_trail)
        else:
            gain = pos.entry_price - pos.current_price
            if gain >= activation_gain:
                new_trail = pos.current_price + risk * config.TRAILING_STOP_ATR_MULT
                if pos.trailing_stop <= 0:
                    pos.trailing_stop = new_trail
                else:
                    pos.trailing_stop = min(pos.trailing_stop, new_trail)

    def _close_position(self, pos: Position, reason: str, exit_price: float):
        """Close a position and record P&L."""
        pos.exit_price = exit_price
        pos.exit_reason = reason
        pos.exit_time = time.time()
        pos.status = "CLOSED"

        if pos.direction == "BUY":
            pos.realized_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pos.realized_pnl = (pos.entry_price - exit_price) * pos.quantity

        # Deduct charges
        if self.charge_calc:
            pos.charges = self.charge_calc.calculate_total(
                pos.entry_price, exit_price, pos.quantity
            )
            pos.realized_pnl -= pos.charges

        self._daily_pnl += pos.realized_pnl

        # Execute exit order
        self._execute_exit(pos)

        # Move to closed
        self.closed_positions.append(pos)
        self.positions.pop(pos.id, None)

        # ── Post-trade cooldown ─────────────────────────────────
        # Base cooldown: config.POST_TRADE_COOLDOWN_SEC (default 30 min).
        # Loss-streak amplification: double the cooldown for every
        # consecutive loss on this symbol (2→60min, 3→120min, capped 4h).
        base_cd = getattr(config, "POST_TRADE_COOLDOWN_SEC", 1800)
        if pos.realized_pnl < 0:
            streak = self._consecutive_losses.get(pos.symbol, 0) + 1
            self._consecutive_losses[pos.symbol] = streak
            cd_seconds = min(base_cd * (2 ** (streak - 1)), 4 * 3600)
        else:
            self._consecutive_losses[pos.symbol] = 0
            cd_seconds = base_cd
        self._symbol_cooldown_until[pos.symbol] = time.time() + cd_seconds

        logger.info(
            f"CLOSED: {pos.symbol} {pos.direction} reason={reason} "
            f"P&L={pos.realized_pnl:.2f} charges={pos.charges:.2f} "
            f"| cooldown {cd_seconds//60}min"
        )

    def _execute_entry(self, pos: Position) -> bool:
        """Place entry order via Fyers API.
        Uses LIMIT order with MPP buffer for slippage protection.
        side: 1 = BUY, -1 = SELL
        """
        if self.fyers is None:
            logger.info(f"Paper trade entry: {pos.direction} {pos.symbol} x{pos.quantity}")
            return True

        try:
            side = 1 if pos.direction == "BUY" else -1
            mpp_pct = config.ORDER_MPP_PCT / 100.0  # e.g. 0.5% → 0.005

            # LIMIT order with MPP buffer: BUY slightly above, SELL slightly below
            if mpp_pct > 0 and pos.entry_price > 0:
                if pos.direction == "BUY":
                    limit_price = round(pos.entry_price * (1 + mpp_pct), 2)
                else:
                    limit_price = round(pos.entry_price * (1 - mpp_pct), 2)
                order_type = "LIMIT"
                logger.info(
                    f"LIMIT entry: {pos.direction} {pos.symbol} x{pos.quantity} "
                    f"@ Rs.{limit_price} (MPP {config.ORDER_MPP_PCT}% from Rs.{pos.entry_price})"
                )
            else:
                limit_price = 0
                order_type = "MARKET"

            result = self.fyers.place_intraday_order(
                symbol=pos.symbol,
                qty=pos.quantity,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
            )
            ok = result is not None and result.get("s") in ("ok", "OK")
            if not ok:
                logger.warning(f"Entry order response: {result}")
            return ok or result is not None
        except Exception as e:
            logger.error(f"Entry order failed: {e}")
            return False

    def _execute_exit(self, pos: Position):
        """Place exit order via Fyers API.
        Uses LIMIT order with MPP buffer for slippage protection.
        """
        if self.fyers is None:
            return

        try:
            side = -1 if pos.direction == "BUY" else 1  # reverse direction to exit
            mpp_pct = config.ORDER_MPP_PCT / 100.0
            exit_price = pos.current_price if pos.current_price > 0 else pos.entry_price

            # Exit with MPP: selling BUY → limit slightly below, covering SELL → limit slightly above
            if mpp_pct > 0 and exit_price > 0:
                if pos.direction == "BUY":
                    limit_price = round(exit_price * (1 - mpp_pct), 2)
                else:
                    limit_price = round(exit_price * (1 + mpp_pct), 2)
                order_type = "LIMIT"
            else:
                limit_price = 0
                order_type = "MARKET"

            self.fyers.place_intraday_order(
                symbol=pos.symbol,
                qty=pos.quantity,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
            )
        except Exception as e:
            logger.error(f"Exit order failed: {e}")

    def _calculate_quantity(self, signal) -> int:
        """Calculate position size based on risk management rules."""
        if self.risk_manager:
            return self.risk_manager.calculate_position_size(
                signal.entry_price, signal.stop_loss
            )

        # Simple fallback: fixed fraction of capital
        risk_per_share = abs(signal.entry_price - signal.stop_loss)
        if risk_per_share <= 0:
            return 0

        max_risk = config.TOTAL_TRADING_CAPITAL * config.MAX_PORTFOLIO_RISK / config.MAX_CONCURRENT_POSITIONS
        qty = int(max_risk / risk_per_share)

        # Cap by max capital per stock
        max_qty = int(config.TOTAL_TRADING_CAPITAL * config.MAX_CAPITAL_PER_STOCK / signal.entry_price)
        return max(1, min(qty, max_qty))

    def cancel_position(self, pos_id: str):
        """Cancel a PENDING position or force-close an ACTIVE one (paper trade)."""
        pos = self.positions.get(pos_id)
        if pos is None:
            return
        if pos.status == "PENDING":
            pos.status = "CANCELLED"
            pos.exit_reason = "MANUAL_CANCEL"
        elif pos.status == "ACTIVE":
            # Force-close at current price (paper trade manual exit)
            exit_price = pos.current_price if pos.current_price > 0 else pos.entry_price
            if pos.direction == "BUY":
                pos.realized_pnl = (exit_price - pos.entry_price) * pos.quantity
            else:
                pos.realized_pnl = (pos.entry_price - exit_price) * pos.quantity
            pos.exit_price  = exit_price
            pos.exit_time   = time.time()
            pos.status      = "CLOSED"
            pos.exit_reason = "MANUAL_CLOSE"
            self._daily_pnl += pos.realized_pnl
            logger.info(
                f"MANUAL CLOSE: {pos.symbol} {pos.direction} x{pos.quantity} "
                f"@ {exit_price:.2f}  P&L={pos.realized_pnl:.2f}"
            )
        else:
            return  # already closed/cancelled
        self.closed_positions.append(pos)
        self.positions.pop(pos_id, None)
        self._save_positions()

    def dedup_positions(self) -> int:
        """Remove duplicate positions for the same symbol, keeping the oldest.
        Returns number of duplicates removed."""
        seen_symbols: set = set()
        to_remove = []
        # Sort by entry_time so we keep the earliest
        ordered = sorted(self.positions.values(), key=lambda p: p.entry_time)
        for pos in ordered:
            if pos.symbol in seen_symbols:
                to_remove.append(pos.id)
            else:
                seen_symbols.add(pos.symbol)
        for pid in to_remove:
            pos = self.positions.pop(pid, None)
            if pos:
                pos.status = "CANCELLED"
                pos.exit_reason = "DEDUP_CLEANUP"
                self.closed_positions.append(pos)
        if to_remove:
            self._save_positions()
            logger.info(f"Dedup: removed {len(to_remove)} duplicate positions: {to_remove}")
        return len(to_remove)

    def get_active_position_count(self) -> int:
        return sum(1 for p in self.positions.values() if p.status == "ACTIVE")

    def get_active_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if p.status == "ACTIVE"]

    def get_daily_pnl(self) -> float:
        return self._daily_pnl

    def get_performance_stats(self) -> dict:
        """Compute advanced performance metrics from closed positions.
        Distinguishes all-time (Net) metrics from TODAY-only metrics.
        """
        from datetime import datetime, timezone, timedelta
        IST = timezone(timedelta(hours=5, minutes=30))
        today_ist = datetime.now(IST).date()

        closed = [p for p in self.closed_positions if p.status == "CLOSED"]

        # Split into today's vs all-time
        def _is_today(p):
            return (p.exit_time and
                    datetime.fromtimestamp(p.exit_time, tz=IST).date() == today_ist)
        closed_today = [p for p in closed if _is_today(p)]

        def _metrics(subset):
            wins   = [p.realized_pnl for p in subset if p.realized_pnl > 0]
            losses = [p.realized_pnl for p in subset if p.realized_pnl <= 0]
            n      = len(subset)
            wr     = round(len(wins) / n * 100, 1) if n > 0 else 0.0
            aw     = round(sum(wins) / len(wins), 2) if wins else 0.0
            al     = round(abs(sum(losses) / len(losses)), 2) if losses else 0.0
            wl     = round(aw / al, 2) if al > 0 else 0.0
            # Max drawdown over subset equity curve
            mdd=0.0; peak=0.0; cum=0.0
            for p in subset:
                cum += p.realized_pnl
                if cum > peak: peak = cum
                dd = peak - cum
                if dd > mdd: mdd = dd
            pnl = round(sum(p.realized_pnl for p in subset), 2)
            return {
                "trades": n, "wins": len(wins), "losses": len(losses),
                "win_rate": wr, "avg_win": aw, "avg_loss": al,
                "wl_ratio": wl, "max_dd": round(mdd, 2), "pnl": pnl,
            }

        net_m   = _metrics(closed)
        today_m = _metrics(closed_today)

        active  = [p for p in self.positions.values() if p.status == "ACTIVE"]
        pending = [p for p in self.positions.values() if p.status == "PENDING"]

        # Capital employed = sum of (entry_price × quantity) for all ACTIVE positions
        capital_used = sum(p.entry_price * p.quantity for p in active)
        total_capital = config.TOTAL_TRADING_CAPITAL
        capital_pct   = round(capital_used / total_capital * 100, 1) if total_capital > 0 else 0.0
        cap_per_stock = round(total_capital * config.MAX_CAPITAL_PER_STOCK)
        cap_total     = total_capital

        return {
            # ── Legacy keys (kept for backwards compat) — map to NET (all-time) ──
            "total_trades":   net_m["trades"],
            "win_count":      net_m["wins"],
            "loss_count":     net_m["losses"],
            "win_rate":       net_m["win_rate"],
            "avg_win":        net_m["avg_win"],
            "avg_loss":       net_m["avg_loss"],
            "wl_ratio":       net_m["wl_ratio"],
            "max_dd":         net_m["max_dd"],
            "total_pnl":      net_m["pnl"],
            # ── Today-scoped metrics (new) ──
            "today_trades":   today_m["trades"],
            "today_win_count":today_m["wins"],
            "today_loss_count":today_m["losses"],
            "today_win_rate": today_m["win_rate"],
            "today_avg_win":  today_m["avg_win"],
            "today_avg_loss": today_m["avg_loss"],
            "today_wl_ratio": today_m["wl_ratio"],
            "today_max_dd":   today_m["max_dd"],
            "today_pnl":      today_m["pnl"],
            # ── Live state ──
            "open_count":     len(active),
            "closed_count":   net_m["trades"],
            "pending_count":  len(pending),
            "daily_pnl":      round(self._daily_pnl, 2),   # live tracker, also today-scoped
            # Capital tracking
            "capital_used":   round(capital_used, 0),
            "capital_pct":    capital_pct,
            "total_capital":  total_capital,
            "cap_per_stock":  cap_per_stock,
            "cap_total":      cap_total,
            "unrealized_pnl": round(sum(p.unrealized_pnl for p in active), 2),
        }

    def reset_daily_pnl(self):
        self._daily_pnl = 0.0

    def _save_positions(self):
        try:
            # ── Merge-on-save: read existing closed trades from file first ──
            # This prevents accidental data loss if closed_positions in memory
            # was reset (e.g., werkzeug reloader, crash recovery).
            existing_closed_by_id = {}
            if os.path.exists(config.TRADES_FILE):
                try:
                    with open(config.TRADES_FILE, "r") as f:
                        old_data = json.load(f)
                    for p_data in old_data.get("closed_today", []):
                        pid = p_data.get("id", "")
                        if pid:
                            existing_closed_by_id[pid] = p_data
                except Exception:
                    pass

            # Build merged closed list: memory takes priority, then file
            memory_closed = {p.id: asdict(p) for p in self.closed_positions}
            merged = dict(existing_closed_by_id)  # start with file data
            merged.update(memory_closed)           # memory overwrites
            closed_list = sorted(merged.values(), key=lambda x: x.get("exit_time", 0))

            data = {
                "active": {k: asdict(v) for k, v in self.positions.items()},
                "closed_today": closed_list[-200:],  # keep up to 200 trades/day
            }
            with open(config.TRADES_FILE, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save positions: {e}")

    def _load_positions(self):
        if not os.path.exists(config.TRADES_FILE):
            return
        try:
            with open(config.TRADES_FILE, "r") as f:
                data = json.load(f)
            for k, v in data.get("active", {}).items():
                self.positions[k] = Position(**v)
            # Restore today's closed positions so trade history survives restarts
            for p_data in data.get("closed_today", []):
                try:
                    self.closed_positions.append(Position(**p_data))
                except Exception:
                    pass  # skip malformed entries
            if self.closed_positions:
                # Recalculate daily P&L from TODAY's closed positions only
                # (closed_positions may contain historical trades from prior sessions)
                from datetime import datetime, timezone, timedelta
                IST = timezone(timedelta(hours=5, minutes=30))
                today_ist = datetime.now(IST).date()
                self._daily_pnl = sum(
                    p.realized_pnl for p in self.closed_positions
                    if p.status == "CLOSED"
                    and p.exit_time
                    and datetime.fromtimestamp(p.exit_time, tz=IST).date() == today_ist
                )
                logger.info(
                    f"Restored {len(self.closed_positions)} closed positions, "
                    f"today's P&L=Rs.{self._daily_pnl:.2f}"
                )
            # Auto-dedup on load — removes duplicate symbols that built up
            # from app restarts (werkzeug reloader) during development
            removed = self.dedup_positions()
            if removed:
                logger.info(f"Startup dedup: removed {removed} duplicate position(s)")
        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
