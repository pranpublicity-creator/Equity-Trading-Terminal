"""
Watchlist Manager for Equity Trading Terminal
Manages active stock universe with round-robin batch rotation.
Adapted from COMMODITY APP — expanded to handle 50-200 stocks.
"""
import json
import os
import logging
from collections import deque

import config
from core.stock_universe import get_universe, NIFTY_50

logger = logging.getLogger(__name__)


class WatchlistManager:
    """Manages the active watchlist with round-robin batch rotation for rate-limited scanning."""

    _BLACKLIST_FILE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "invalid_symbols.json"
    )

    def __init__(self, universe_name=None):
        self.universe_name = universe_name or config.DEFAULT_UNIVERSE
        self.active_symbols = list(get_universe(self.universe_name))
        self.trading_enabled = {}  # {symbol: bool}
        self._rotation_queue = deque(self.active_symbols)
        self._priority_symbols = set()  # Symbols with active patterns get priority
        self._failed_symbols = {}    # {symbol: (fail_count, last_fail_time)}
        self._FAIL_COOLDOWN = 1800   # Skip symbol for 30 min after 3 failures
        self._FAIL_THRESHOLD = 3     # How many failures before cooldown
        self._persistent_blacklist = set()  # Symbols permanently invalid (survive restarts)
        self._load()
        self._load_blacklist()

    def get_active_symbols(self):
        """Return all active symbols."""
        return list(self.active_symbols)

    def mark_failed(self, symbol, reason=""):
        """Record a fetch failure for a symbol. After threshold, put in cooldown."""
        import time
        count, _ = self._failed_symbols.get(symbol, (0, 0))
        count += 1
        self._failed_symbols[symbol] = (count, time.time())
        # Remove from priority so it stops being hammered
        self._priority_symbols.discard(symbol)
        if count >= self._FAIL_THRESHOLD:
            logger.warning(f"Symbol {symbol} in cooldown after {count} failures: {reason}")
        else:
            logger.debug(f"Symbol {symbol} failure #{count}: {reason}")

    def mark_permanent_failure(self, symbol, reason=""):
        """Immediately blacklist a symbol (e.g., 'Invalid symbol' from Fyers).
        Persists to disk so it survives app restarts.
        """
        import time
        self._failed_symbols[symbol] = (self._FAIL_THRESHOLD, time.time())
        self._priority_symbols.discard(symbol)
        # Persist to disk — survives Flask hot-reload / app restart
        self._persistent_blacklist.add(symbol)
        self._save_blacklist()
        logger.warning(f"Symbol {symbol} permanently blacklisted: {reason}")

    def _load_blacklist(self):
        """Load persisted invalid symbols from disk and pre-populate cooldown."""
        import time
        try:
            if os.path.exists(self._BLACKLIST_FILE):
                with open(self._BLACKLIST_FILE) as f:
                    data = json.load(f)
                self._persistent_blacklist = set(data.get("symbols", []))
                for sym in self._persistent_blacklist:
                    self._failed_symbols[sym] = (self._FAIL_THRESHOLD, time.time())
                if self._persistent_blacklist:
                    logger.info(f"Loaded {len(self._persistent_blacklist)} blacklisted symbols: {sorted(self._persistent_blacklist)}")
        except Exception as e:
            logger.warning(f"Could not load blacklist: {e}")

    def _save_blacklist(self):
        """Persist invalid symbols to disk."""
        try:
            os.makedirs(os.path.dirname(self._BLACKLIST_FILE), exist_ok=True)
            with open(self._BLACKLIST_FILE, "w") as f:
                json.dump({"symbols": sorted(self._persistent_blacklist)}, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save blacklist: {e}")

    def is_in_cooldown(self, symbol):
        """Return True if symbol has too many recent failures."""
        import time
        if symbol not in self._failed_symbols:
            return False
        count, last_fail = self._failed_symbols[symbol]
        if count < self._FAIL_THRESHOLD:
            return False
        # Check if cooldown period has passed
        if time.time() - last_fail > self._FAIL_COOLDOWN:
            del self._failed_symbols[symbol]  # Reset after cooldown
            logger.info(f"Symbol {symbol} cooldown expired — re-enabled")
            return False
        return True

    def get_next_batch(self, batch_size=None):
        """Get next batch of symbols in round-robin rotation.
        Priority symbols (with active patterns) are always included first.
        Symbols in cooldown (repeated failures) are skipped.
        """
        batch_size = batch_size or config.BATCH_SIZE
        batch = []

        # Priority symbols first (skip cooldown ones)
        for sym in list(self._priority_symbols):
            if sym in self.active_symbols and not self.is_in_cooldown(sym) and len(batch) < batch_size:
                batch.append(sym)

        # Fill remaining from rotation queue
        attempts = 0
        while len(batch) < batch_size and attempts < len(self.active_symbols) * 2:
            if not self._rotation_queue:
                self._rotation_queue = deque(self.active_symbols)
            sym = self._rotation_queue.popleft()
            if sym not in batch and not self.is_in_cooldown(sym):
                batch.append(sym)
            attempts += 1

        return batch

    def prioritize(self, symbol):
        """Add symbol to priority set (has active pattern, scanned more often)."""
        if symbol in self.active_symbols and not self.is_in_cooldown(symbol):
            self._priority_symbols.add(symbol)

    def deprioritize(self, symbol):
        """Remove symbol from priority set."""
        self._priority_symbols.discard(symbol)

    def clear_priorities(self):
        """Clear all priority symbols."""
        self._priority_symbols.clear()

    def set_universe(self, universe_name):
        """Switch to a different stock universe."""
        self.universe_name = universe_name
        self.active_symbols = list(get_universe(universe_name))
        self._rotation_queue = deque(self.active_symbols)
        self._priority_symbols.clear()
        self._save()
        return True, f"Universe set to {universe_name} ({len(self.active_symbols)} stocks)"

    def add_symbol(self, symbol):
        """Add a symbol to the active watchlist."""
        if symbol in self.active_symbols:
            return False, f"{symbol} already in watchlist"
        if len(self.active_symbols) >= config.MAX_STOCKS_PER_CYCLE:
            return False, f"Max {config.MAX_STOCKS_PER_CYCLE} stocks"
        self.active_symbols.append(symbol)
        self._rotation_queue.append(symbol)
        self._save()
        return True, f"{symbol} added"

    def remove_symbol(self, symbol):
        """Remove a symbol from the active watchlist."""
        if symbol not in self.active_symbols:
            return False, f"{symbol} not in watchlist"
        self.active_symbols.remove(symbol)
        self._priority_symbols.discard(symbol)
        self._rotation_queue = deque(s for s in self._rotation_queue if s != symbol)
        self._save()
        return True, f"{symbol} removed"

    def set_custom_watchlist(self, symbols):
        """Replace entire watchlist with custom symbols."""
        valid = [s for s in symbols if s.startswith("NSE:")][:config.MAX_STOCKS_PER_CYCLE]
        if not valid:
            return False, "No valid NSE symbols provided"
        self.active_symbols = valid
        self.universe_name = "CUSTOM"
        self._rotation_queue = deque(valid)
        self._priority_symbols.clear()
        self._save()
        return True, f"Custom watchlist set: {len(valid)} stocks"

    def is_trading_enabled(self, symbol):
        """Check if trading is enabled for a symbol."""
        return self.trading_enabled.get(symbol, True)

    def set_trading_enabled(self, symbol, enabled):
        """Enable or disable trading for a symbol."""
        self.trading_enabled[symbol] = bool(enabled)
        self._save()
        return bool(enabled)

    def get_rotation_progress(self):
        """Return how far through the rotation we are."""
        total = len(self.active_symbols)
        remaining = len(self._rotation_queue)
        return {
            "total": total,
            "remaining": remaining,
            "scanned": total - remaining,
            "priority_count": len(self._priority_symbols),
            "universe": self.universe_name,
        }

    def _save(self):
        os.makedirs(os.path.dirname(config.WATCHLIST_FILE), exist_ok=True)
        try:
            with open(config.WATCHLIST_FILE, "w") as f:
                json.dump({
                    "universe_name": self.universe_name,
                    "active_symbols": self.active_symbols,
                    "trading_enabled": self.trading_enabled,
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Watchlist save error: {e}")

    def _load(self):
        if not os.path.exists(config.WATCHLIST_FILE):
            return
        try:
            with open(config.WATCHLIST_FILE) as f:
                data = json.load(f)
            loaded = data.get("active_symbols", [])
            if loaded:
                self.active_symbols = loaded[:config.MAX_STOCKS_PER_CYCLE]
                self._rotation_queue = deque(self.active_symbols)
            self.universe_name = data.get("universe_name", self.universe_name)
            self.trading_enabled = data.get("trading_enabled", {})
            logger.info(f"Loaded watchlist: {len(self.active_symbols)} stocks ({self.universe_name})")
        except Exception as e:
            logger.error(f"Watchlist load error: {e}")
