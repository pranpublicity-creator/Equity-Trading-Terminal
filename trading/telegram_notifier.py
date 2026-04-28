"""
Telegram Notifier
Sends trade signals and alerts via Telegram bot.
Reused from COMMODITY APP.
"""
import json
import logging
import os
import threading

import requests

import config

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends notifications via Telegram bot API."""

    def __init__(self):
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self._enabled = bool(self.bot_token and self.chat_id)
        self._load_config()

    def send_signal_alert(self, signal) -> bool:
        """Send a formatted trade signal alert."""
        if not self._enabled:
            return False

        direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
        strength_map = {"STRONG": "⚡", "MODERATE": "💡", "WEAK": "📊"}
        strength_icon = strength_map.get(signal.strength, "📊")

        ticker = signal.symbol.replace("NSE:", "").replace("-EQ", "")

        # Timeframe label
        tf_raw   = str(getattr(signal, "timeframe", "15"))
        tf_label = "⚡ Intraday (5m)" if tf_raw == "5" else "📈 Swing (15m)"

        msg = (
            f"{direction_emoji} <b>{signal.direction} {ticker}</b> {strength_icon}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⏱ Timeframe : {tf_label}\n"
            f"📈 Entry    : ₹{signal.entry_price:.2f}\n"
            f"🛑 SL       : ₹{signal.stop_loss:.2f}\n"
            f"🎯 Target   : ₹{signal.target_price:.2f}\n"
            f"📊 R:R      = {signal.risk_reward:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🤖 Confidence: {signal.confidence:.1f}% ({signal.strength})\n"
            f"📋 Pattern  : {signal.pattern_name or 'ML-driven'}\n"
            f"🏛 Regime   : {signal.regime}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"LGB={signal.lgbm_prob:.2f} XGB={signal.xgb_prob:.2f} "
            f"LSTM={signal.lstm_prob:.2f} TFT={signal.tft_prob:.2f}\n"
            f"ARIMA={signal.arima_trend} PCR={signal.pcr:.2f} "
            f"FII={signal.fii_net:+.0f}Cr"
        )

        return self._send(msg)

    def send_position_closed(self, position) -> bool:
        """Send position closure notification."""
        if not self._enabled:
            return False

        ticker   = position.symbol.replace("NSE:", "").replace("-EQ", "")
        pnl_icon = "💰" if position.realized_pnl >= 0 else "💸"
        tf_raw   = str(getattr(position, "timeframe", "15"))
        tf_label = "⚡ Intraday (5m)" if tf_raw == "5" else "📈 Swing (15m)"

        msg = (
            f"{pnl_icon} <b>CLOSED: {ticker}</b>\n"
            f"⏱ {tf_label} | {position.direction}\n"
            f"Entry: ₹{position.entry_price:.2f} → Exit: ₹{position.exit_price:.2f}\n"
            f"Reason: {position.exit_reason}\n"
            f"P&L: ₹{position.realized_pnl:+.2f} (charges: ₹{position.charges:.2f})"
        )

        return self._send(msg)

    def send_daily_summary(self, summary: dict) -> bool:
        """Send end-of-day summary."""
        if not self._enabled:
            return False

        msg = (
            f"📊 <b>Daily Summary</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Trades: {summary.get('total_trades', 0)}\n"
            f"Winners: {summary.get('winners', 0)} | Losers: {summary.get('losers', 0)}\n"
            f"P&L: ₹{summary.get('daily_pnl', 0):+.2f}\n"
            f"Win Rate: {summary.get('win_rate', 0):.1f}%\n"
            f"Active Positions: {summary.get('active_positions', 0)}"
        )

        return self._send(msg)

    def send_message(self, text: str) -> bool:
        """Send a plain text message."""
        return self._send(text)

    def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send message via Telegram Bot API (non-blocking)."""
        def _do_send():
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                }
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code != 200:
                    logger.error(f"Telegram send failed: {resp.text}")
            except Exception as e:
                logger.error(f"Telegram error: {e}")

        thread = threading.Thread(target=_do_send, daemon=True)
        thread.start()
        return True

    def _load_config(self):
        """Load Telegram config from file if tokens not in config.py."""
        if self._enabled:
            return

        path = config.TELEGRAM_CONFIG_FILE
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                self.bot_token = data.get("bot_token", "")
                self.chat_id = data.get("chat_id", "")
                self._enabled = bool(self.bot_token and self.chat_id)
            except Exception:
                pass

    def save_config(self, bot_token: str, chat_id: str):
        """Save Telegram config to file."""
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)

        os.makedirs(os.path.dirname(config.TELEGRAM_CONFIG_FILE), exist_ok=True)
        with open(config.TELEGRAM_CONFIG_FILE, "w") as f:
            json.dump({"bot_token": bot_token, "chat_id": chat_id}, f)

    def is_enabled(self) -> bool:
        return self._enabled
