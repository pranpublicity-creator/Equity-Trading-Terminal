"""
Telegram Bot Command Handler
============================
Polls Telegram's getUpdates API (long-poll) for incoming messages and
responds to slash-commands with live trading data.

All commands are OWNER-ONLY — verified against TELEGRAM_CHAT_ID in config.
A second different chat-id gets a silent "⛔ Unauthorized" reply.

Registered Commands (auto-pushed to Telegram via setMyCommands so the
'/' menu is pre-populated in any Telegram client):

  /help       — list all commands with descriptions
  /status     — scanner state, regime, strategy, daily P&L
  /positions  — all active positions with entry / LTP / P&L
  /pnl        — today's P&L + win-rate breakdown
  /signals    — last 10 signals received this session
  /summary    — full daily summary (trades / winners / losers)
  /start      — start the NSE scanner
  /stop       — stop the NSE scanner
  /close SYM  — close a specific open position (e.g. /close RELIANCE)
  /closeall   — close ALL active positions (requires 2nd confirm in 60 s)
  /config     — show key configuration knobs
  /pause      — same as /stop (alias)

Integration:
  Instantiate TelegramCommander in app.py after TelegramNotifier and call
  commander.start().  Feed signals via commander.push_signal(signal) so
  /signals shows live data.  Expose scanner controls via set_scanner_fns().
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, List, Optional

import requests

import config

logger = logging.getLogger(__name__)

# ── Command menu registered with Telegram ─────────────────────────────────────
_BOT_COMMANDS = [
    ("help",     "List all available commands"),
    ("status",   "Scanner status, regime & daily P&L"),
    ("positions","Show all active positions with live P&L"),
    ("pnl",      "Today's P&L breakdown"),
    ("signals",  "Last 10 signals received this session"),
    ("summary",  "Full daily summary"),
    ("start",    "Start the NSE scanner"),
    ("stop",     "Stop the NSE scanner"),
    ("pause",    "Pause scanner (alias for /stop)"),
    ("close",    "Close a position: /close RELIANCE"),
    ("closeall", "Close ALL active positions (needs confirm)"),
    ("config",   "Show key configuration values"),
]

_LONG_POLL_TIMEOUT = 25   # seconds per getUpdates call
_RETRY_SLEEP       = 5    # seconds to wait after a network error


class TelegramCommander:
    """Receives and handles Telegram slash-commands via long-polling."""

    def __init__(self, notifier, trade_engine=None):
        """
        Args:
            notifier    : TelegramNotifier instance (provides bot_token, chat_id)
            trade_engine: TradeEngine instance for position/P&L queries
        """
        self._notifier      = notifier
        self._trade_engine  = trade_engine
        self._token         = notifier.bot_token or ""
        self._owner_chat    = str(notifier.chat_id or "").strip()

        # Scanner control — injected via set_scanner_fns()
        self._fn_start: Optional[Callable] = None
        self._fn_stop:  Optional[Callable] = None
        self._fn_is_running: Optional[Callable[[], bool]] = None

        # Recent signals buffer (populated via push_signal)
        self._recent_signals: List = []
        self._recent_lock = threading.Lock()

        # Two-step /closeall confirmation: chat_id → expiry timestamp
        self._closeall_pending: dict = {}

        # Polling state
        self._last_uid = 0
        self._running  = False
        self._thread:  Optional[threading.Thread] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_scanner_fns(self,
                        fn_start: Callable,
                        fn_stop:  Callable,
                        fn_is_running: Callable[[], bool]):
        """Inject scanner control callables from app.py."""
        self._fn_start      = fn_start
        self._fn_stop       = fn_stop
        self._fn_is_running = fn_is_running

    def push_signal(self, signal) -> None:
        """Feed a new TradeSignal into the recent-signals buffer (/signals cmd)."""
        with self._recent_lock:
            self._recent_signals.append(signal)
            if len(self._recent_signals) > 50:
                self._recent_signals = self._recent_signals[-50:]

    def start(self) -> None:
        """Start the long-polling loop in a background daemon thread."""
        if not self._token or not self._owner_chat:
            logger.info("[TGCmd] Credentials not set — commander disabled")
            return
        if self._running:
            return
        self._running = True
        self._register_commands()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="TGCommander"
        )
        self._thread.start()
        logger.info("[TGCmd] Started — polling for slash-commands")

    def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False

    # ── Long-polling loop ──────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while self._running:
            try:
                updates = self._get_updates()
                for upd in updates:
                    try:
                        self._dispatch(upd)
                    except Exception as e:
                        logger.debug(f"[TGCmd] dispatch error: {e}")
            except Exception as e:
                logger.debug(f"[TGCmd] poll error: {e}")
                time.sleep(_RETRY_SLEEP)

    def _get_updates(self) -> list:
        url = f"https://api.telegram.org/bot{self._token}/getUpdates"
        params = {
            "offset":           self._last_uid + 1,
            "timeout":          _LONG_POLL_TIMEOUT,
            "allowed_updates":  ["message"],
        }
        try:
            resp = requests.get(url, params=params,
                                timeout=_LONG_POLL_TIMEOUT + 5)
            data = resp.json()
            if data.get("ok") and data.get("result"):
                self._last_uid = data["result"][-1]["update_id"]
                return data["result"]
        except Exception:
            pass
        return []

    def _dispatch(self, update: dict) -> None:
        msg  = update.get("message") or {}
        text = (msg.get("text") or "").strip()
        chat_id = str(msg.get("chat", {}).get("id", ""))

        if not text.startswith("/"):
            return

        # Owner-only gate
        if chat_id != self._owner_chat:
            self._reply(chat_id, "⛔ Unauthorized — this bot is private.")
            return

        # Parse "/command@botname arg1 arg2…"
        parts  = text.split()
        cmd    = parts[0].split("@")[0].lstrip("/").lower()
        args   = parts[1:]

        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler:
            handler(chat_id, args)
        else:
            self._reply(chat_id,
                f"❓ Unknown command: <code>/{cmd}</code>\n"
                f"Send /help for the full list.")

    # ── Command Handlers ───────────────────────────────────────────────────────

    def _cmd_help(self, chat_id: str, args: list) -> None:
        lines = ["<b>📟 Equity Terminal — Bot Commands</b>\n"]
        for cmd, desc in _BOT_COMMANDS:
            lines.append(f"/{cmd}  —  {desc}")
        lines.append("\n<i>All commands are owner-only and real-time.</i>")
        self._reply(chat_id, "\n".join(lines))

    # ── /status ───────────────────────────────────────────────────────────────
    def _cmd_status(self, chat_id: str, args: list) -> None:
        te = self._trade_engine
        running = self._fn_is_running() if self._fn_is_running else False

        daily_pnl = 0.0
        active_n  = 0
        pending_n = 0
        if te:
            try:
                daily_pnl = te.get_daily_pnl()
                active_n  = len(te.get_active_positions())
                pending_n = len(te.get_pending_signals())
            except Exception:
                pass

        icon     = "🟢" if running else "🔴"
        pnl_icon = "💰" if daily_pnl >= 0 else "💸"
        status   = "RUNNING" if running else "STOPPED"

        with self._recent_lock:
            sig_count = len(self._recent_signals)

        self._reply(chat_id, (
            f"{icon} <b>Scanner: {status}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📂 Active positions : {active_n}\n"
            f"⏳ Pending signals  : {pending_n}\n"
            f"📡 Signals this session: {sig_count}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"{pnl_icon} Daily P&L: ₹{daily_pnl:+.2f}\n"
            f"💸 Loss limit: ₹{getattr(config,'MAX_LOSS_PER_DAY',5000):,.0f}"
        ))

    # ── /positions ────────────────────────────────────────────────────────────
    def _cmd_positions(self, chat_id: str, args: list) -> None:
        te = self._trade_engine
        if te is None:
            self._reply(chat_id, "⚠️ Trade engine not connected."); return

        try:
            positions = te.get_active_positions()
        except Exception as e:
            self._reply(chat_id, f"❌ Error reading positions: {e}"); return

        if not positions:
            self._reply(chat_id, "📭 No active positions right now."); return

        lines = [f"<b>📂 Active Positions ({len(positions)})</b>\n"]
        total_pnl = 0.0
        for p in positions:
            ticker  = p.symbol.replace("NSE:", "").replace("-EQ", "")
            pnl     = float(getattr(p, "unrealized_pnl", 0) or 0)
            total_pnl += pnl
            tgt     = float(getattr(p, "target", 0) or getattr(p, "target_price", 0) or 0)
            pattern = (getattr(p, "pattern_name", "") or "ML").replace("_", " ")
            tf      = getattr(p, "timeframe", "15")
            d_icon  = "🟢" if p.direction == "BUY" else "🔴"
            p_icon  = "💚" if pnl >= 0 else "❤️"
            lines.append(
                f"{d_icon} <b>{p.direction} {ticker}</b> [{tf}m]  {p_icon} ₹{pnl:+.0f}\n"
                f"   Entry ₹{p.entry_price:.2f} | SL ₹{p.stop_loss:.2f} | Tgt ₹{tgt:.2f}\n"
                f"   Conf {getattr(p,'signal_confidence',0):.0f}% | {pattern}"
            )
        lines.append(
            f"\n━━━━━━━━━━━━━━━━━━\n"
            f"{'💰' if total_pnl>=0 else '💸'} Total unrealised: ₹{total_pnl:+.0f}"
        )
        self._reply(chat_id, "\n".join(lines))

    # ── /pnl ──────────────────────────────────────────────────────────────────
    def _cmd_pnl(self, chat_id: str, args: list) -> None:
        te = self._trade_engine
        if te is None:
            self._reply(chat_id, "⚠️ Trade engine not connected."); return

        try:
            daily_pnl = te.get_daily_pnl()
            stats = te.get_performance_stats() if hasattr(te, "get_performance_stats") else {}
        except Exception as e:
            self._reply(chat_id, f"❌ Error: {e}"); return

        wins   = stats.get("win_count",   stats.get("winners_today", "—"))
        losses = stats.get("loss_count",  stats.get("losers_today",  "—"))
        trades = stats.get("total_trades",stats.get("trades_today",  "—"))
        wr     = stats.get("win_rate", 0.0)
        avg_w  = stats.get("avg_win",  0.0)
        avg_l  = stats.get("avg_loss", 0.0)

        pnl_icon = "💰" if daily_pnl >= 0 else "💸"
        self._reply(chat_id, (
            f"{pnl_icon} <b>Today's P&L: ₹{daily_pnl:+.2f}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📋 Trades  : {trades}\n"
            f"✅ Winners : {wins}  |  ❌ Losers: {losses}\n"
            f"📈 Win Rate: {wr:.1f}%\n"
            f"💚 Avg Win : ₹{avg_w:+.0f}\n"
            f"❤️ Avg Loss: ₹{avg_l:+.0f}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ Daily loss limit: ₹{getattr(config,'MAX_LOSS_PER_DAY',5000):,.0f}"
        ))

    # ── /signals ──────────────────────────────────────────────────────────────
    def _cmd_signals(self, chat_id: str, args: list) -> None:
        with self._recent_lock:
            recent = list(self._recent_signals)

        if not recent:
            self._reply(chat_id, "📭 No signals recorded this session yet."); return

        n    = min(10, len(recent))
        sigs = list(reversed(recent[-n:]))
        lines = [f"<b>📡 Last {n} Signal(s)</b>\n"]
        for s in sigs:
            ticker  = str(getattr(s, "symbol", "?")).replace("NSE:", "").replace("-EQ", "")
            d       = getattr(s, "direction", "?")
            conf    = getattr(s, "confidence", 0)
            pattern = (getattr(s, "pattern_name", "") or "ML").replace("_", " ")
            tf      = getattr(s, "timeframe", "15")
            entry   = getattr(s, "entry_price", 0)
            regime  = (getattr(s, "regime", "") or "").replace("_", " ")
            icon    = "🟢" if d == "BUY" else "🔴"
            lines.append(
                f"{icon} {d} <b>{ticker}</b> [{tf}m]  {conf:.0f}%\n"
                f"   ₹{entry:.2f} | {pattern} | {regime}"
            )
        self._reply(chat_id, "\n".join(lines))

    # ── /summary ──────────────────────────────────────────────────────────────
    def _cmd_summary(self, chat_id: str, args: list) -> None:
        te = self._trade_engine
        if te is None:
            self._reply(chat_id, "⚠️ Trade engine not connected."); return
        try:
            daily_pnl = te.get_daily_pnl()
            stats     = te.get_performance_stats() if hasattr(te, "get_performance_stats") else {}
            active_n  = len(te.get_active_positions())
        except Exception as e:
            self._reply(chat_id, f"❌ Error: {e}"); return

        with self._recent_lock:
            sig_count = len(self._recent_signals)

        pnl_icon = "💰" if daily_pnl >= 0 else "💸"
        self._reply(chat_id, (
            f"📊 <b>Daily Summary</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"{pnl_icon} P&L      : ₹{daily_pnl:+.2f}\n"
            f"📋 Trades  : {stats.get('total_trades', '—')}\n"
            f"✅ Winners : {stats.get('win_count', '—')}\n"
            f"❌ Losers  : {stats.get('loss_count', '—')}\n"
            f"📈 Win Rate: {stats.get('win_rate', 0.0):.1f}%\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📂 Open positions : {active_n}\n"
            f"📡 Signals today  : {sig_count}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ Loss limit: ₹{getattr(config,'MAX_LOSS_PER_DAY',5000):,.0f}"
        ))

    # ── /start & /stop ────────────────────────────────────────────────────────
    def _cmd_start(self, chat_id: str, args: list) -> None:
        if self._fn_is_running and self._fn_is_running():
            self._reply(chat_id, "ℹ️ Scanner is already running."); return
        if self._fn_start:
            try:
                self._fn_start()
                self._reply(chat_id, "🟢 Scanner started!")
            except Exception as e:
                self._reply(chat_id, f"❌ Could not start: {e}")
        else:
            self._reply(chat_id, "⚠️ Scanner control not wired up.")

    def _cmd_stop(self, chat_id: str, args: list) -> None:
        if self._fn_is_running and not self._fn_is_running():
            self._reply(chat_id, "ℹ️ Scanner is already stopped."); return
        if self._fn_stop:
            try:
                self._fn_stop()
                self._reply(chat_id, "🔴 Scanner stopped.")
            except Exception as e:
                self._reply(chat_id, f"❌ Could not stop: {e}")
        else:
            self._reply(chat_id, "⚠️ Scanner control not wired up.")

    def _cmd_pause(self, chat_id: str, args: list) -> None:
        """Alias for /stop."""
        self._cmd_stop(chat_id, args)

    # ── /close SYMBOL ─────────────────────────────────────────────────────────
    def _cmd_close(self, chat_id: str, args: list) -> None:
        if not args:
            self._reply(chat_id,
                "Usage: <code>/close SYMBOL</code>\n"
                "Example: <code>/close RELIANCE</code>"); return

        te = self._trade_engine
        if te is None:
            self._reply(chat_id, "⚠️ Trade engine not connected."); return

        raw = args[0].upper().replace("NSE:", "").replace("-EQ", "")
        try:
            positions = te.get_active_positions()
        except Exception as e:
            self._reply(chat_id, f"❌ Error reading positions: {e}"); return

        target = None
        for p in positions:
            ticker = p.symbol.replace("NSE:", "").replace("-EQ", "").upper()
            if ticker == raw or p.symbol.upper() == raw:
                target = p
                break

        if target is None:
            tickers = [p.symbol.replace("NSE:", "").replace("-EQ", "") for p in positions]
            self._reply(chat_id,
                f"❌ No active position for <b>{raw}</b>.\n"
                f"Active: {', '.join(tickers) if tickers else 'none'}"); return

        try:
            te.close_position(target.id, reason="telegram_command")
            ticker = target.symbol.replace("NSE:", "").replace("-EQ", "")
            self._reply(chat_id,
                f"✅ <b>{target.direction} {ticker}</b> — close order submitted.")
        except Exception as e:
            self._reply(chat_id, f"❌ Close failed: {e}")

    # ── /closeall (two-step confirm) ──────────────────────────────────────────
    def _cmd_closeall(self, chat_id: str, args: list) -> None:
        te = self._trade_engine
        if te is None:
            self._reply(chat_id, "⚠️ Trade engine not connected."); return

        try:
            positions = te.get_active_positions()
        except Exception as e:
            self._reply(chat_id, f"❌ Error: {e}"); return

        if not positions:
            self._reply(chat_id, "📭 No active positions to close."); return

        now = time.time()
        if self._closeall_pending.get(chat_id, 0) > now:
            # Second call within 60 s → confirmed
            self._closeall_pending.pop(chat_id, None)
            closed, failed = 0, 0
            for p in positions:
                try:
                    te.close_position(p.id, reason="telegram_closeall")
                    closed += 1
                except Exception:
                    failed += 1
            msg = f"✅ Closed {closed} position(s)."
            if failed:
                msg += f"  ⚠️ {failed} failed — check dashboard."
            self._reply(chat_id, msg)
        else:
            # First call → ask for confirmation
            self._closeall_pending[chat_id] = now + 60
            tickers = ", ".join(
                p.symbol.replace("NSE:", "").replace("-EQ", "")
                for p in positions
            )
            self._reply(chat_id,
                f"⚠️ <b>About to close {len(positions)} position(s):</b>\n"
                f"<code>{tickers}</code>\n\n"
                f"Send /closeall again within 60 s to confirm.")

    # ── /config ───────────────────────────────────────────────────────────────
    def _cmd_config(self, chat_id: str, args: list) -> None:
        self._reply(chat_id, (
            f"⚙️ <b>Key Configuration</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 Capital      : ₹{getattr(config,'TOTAL_TRADING_CAPITAL',500000):,.0f}\n"
            f"📊 Universe     : {getattr(config,'DEFAULT_UNIVERSE','NIFTY_200')}\n"
            f"📐 Max positions: {getattr(config,'MAX_CONCURRENT_POSITIONS',15)}\n"
            f"📐 Max/stock    : {getattr(config,'MAX_CAPITAL_PER_STOCK',0.10)*100:.0f}%\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Signal Thresholds\n"
            f"   Strong   ≥ {getattr(config,'SIGNAL_STRONG_THRESHOLD',75)}\n"
            f"   Moderate ≥ {getattr(config,'SIGNAL_MODERATE_THRESHOLD',65)}\n"
            f"   Weak     ≥ {getattr(config,'SIGNAL_WEAK_THRESHOLD',45)}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🛡 Min R:R      : {getattr(config,'MIN_RISK_REWARD',1.0)}\n"
            f"🏅 Quality min  : {getattr(config,'TRADE_QUALITY_MIN_SCORE',0.60):.0%} (grade ≥ C)\n"
            f"⏱ Cooldown     : {getattr(config,'SIGNAL_ENTRY_COOLDOWN',1800)//60} min\n"
            f"💸 Loss limit   : ₹{getattr(config,'MAX_LOSS_PER_DAY',5000):,.0f}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⚡ ORB candles  : {getattr(config,'ORB_CANDLES',6)} × 5 min\n"
            f"📈 Intraday     : {'ON' if getattr(config,'INTRADAY_PATTERNS_ENABLED',True) else 'OFF'}"
        ))

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _reply(self, chat_id: str, text: str) -> None:
        """Send HTML-formatted message to chat_id (non-blocking)."""
        import threading as _t
        token = self._token

        def _do():
            try:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={
                        "chat_id":    chat_id,
                        "text":       text,
                        "parse_mode": "HTML",
                    },
                    timeout=10,
                )
            except Exception as e:
                logger.debug(f"[TGCmd] reply error: {e}")

        _t.Thread(target=_do, daemon=True).start()

    def _register_commands(self) -> None:
        """Push the command list to Telegram so the / menu is auto-populated."""
        url  = f"https://api.telegram.org/bot{self._token}/setMyCommands"
        cmds = [{"command": c, "description": d} for c, d in _BOT_COMMANDS]
        try:
            r = requests.post(url, json={"commands": cmds}, timeout=10)
            if r.ok:
                logger.info(f"[TGCmd] Registered {len(cmds)} commands with Telegram")
            else:
                logger.warning(f"[TGCmd] setMyCommands failed: {r.text[:200]}")
        except Exception as e:
            logger.warning(f"[TGCmd] setMyCommands error: {e}")
