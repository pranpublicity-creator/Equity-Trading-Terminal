"""
SQLite-Backed Data Engine with Incremental Fetching
Caches OHLCV data locally, fetches only new candles from Fyers API.
Batch processing with rate-limit-aware scheduling.
"""
import sqlite3
import time
import logging
from datetime import datetime, timedelta
from contextlib import contextmanager

import pandas as pd
import numpy as np

import config
from core.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)


class DataEngine:
    """SQLite-cached OHLCV data engine with incremental fetch and batch processing."""

    def __init__(self, fyers_manager):
        self.fyers = fyers_manager
        self.db_path = config.SQLITE_DB
        self._init_db()

    def _init_db(self):
        """Create SQLite tables if they don't exist."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_cache (
                    symbol TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, resolution, timestamp)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_meta (
                    symbol TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    last_update TEXT,
                    row_count INTEGER DEFAULT 0,
                    PRIMARY KEY (symbol, resolution)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_res
                ON ohlcv_cache(symbol, resolution, timestamp)
            """)

    @contextmanager
    def _get_conn(self):
        """Thread-safe SQLite connection context manager."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_cached(self, symbol, resolution="15"):
        """Get cached OHLCV data as DataFrame. Returns None if empty."""
        with self._get_conn() as conn:
            df = pd.read_sql_query(
                "SELECT timestamp, open, high, low, close, volume "
                "FROM ohlcv_cache WHERE symbol=? AND resolution=? "
                "ORDER BY timestamp",
                conn,
                params=(symbol, resolution),
            )
        if df.empty:
            return None
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("datetime", inplace=True)
        return df

    def _get_last_timestamp(self, symbol, resolution):
        """Get the latest cached timestamp for a symbol."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT MAX(timestamp) FROM ohlcv_cache WHERE symbol=? AND resolution=?",
                (symbol, resolution),
            ).fetchone()
        return row[0] if row and row[0] else None

    def _is_cache_fresh(self, symbol, resolution):
        """Check if cache is fresh enough (within CACHE_FRESHNESS_SEC)."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT last_update FROM cache_meta WHERE symbol=? AND resolution=?",
                (symbol, resolution),
            ).fetchone()
        if not row or not row[0]:
            return False
        try:
            last_update = datetime.fromisoformat(row[0])
            age_sec = (datetime.now() - last_update).total_seconds()
            return age_sec < config.CACHE_FRESHNESS_SEC
        except (ValueError, TypeError):
            return False

    def _store_candles(self, symbol, resolution, candles):
        """Store candles to SQLite. Candles format: [[ts, o, h, l, c, v], ...]"""
        if not candles:
            return
        with self._get_conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO ohlcv_cache "
                "(symbol, resolution, timestamp, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(symbol, resolution, c[0], c[1], c[2], c[3], c[4], c[5]) for c in candles],
            )
            count = conn.execute(
                "SELECT COUNT(*) FROM ohlcv_cache WHERE symbol=? AND resolution=?",
                (symbol, resolution),
            ).fetchone()[0]
            conn.execute(
                "INSERT OR REPLACE INTO cache_meta (symbol, resolution, last_update, row_count) "
                "VALUES (?, ?, ?, ?)",
                (symbol, resolution, datetime.now().isoformat(), count),
            )

    # Fyers API max days per single history call by resolution
    _FYERS_MAX_DAYS = {"1": 30, "2": 30, "3": 30, "5": 60, "10": 60,
                       "15": 100, "20": 100, "30": 100, "60": 200, "D": 365}

    def fetch_symbol(self, symbol, resolution="15", days=None):
        """Fetch OHLCV data for a single symbol with incremental update.
        Automatically chunks requests that exceed Fyers API per-call day limits.
        Returns DataFrame or None.
        """
        days = days or config.CANDLE_HISTORY_DAYS

        # Check if cache is fresh (scanner path — skip API call)
        if self._is_cache_fresh(symbol, resolution):
            cached = self.get_cached(symbol, resolution)
            if cached is not None and len(cached) > 50:
                return cached

        # Incremental fetch: from last cached timestamp to now
        last_ts = self._get_last_timestamp(symbol, resolution)
        if last_ts:
            fetch_from = datetime.fromtimestamp(last_ts)
            fetch_days = (datetime.now() - fetch_from).days + 1
            fetch_days = min(fetch_days, 10)   # cap incremental at 10 days
        else:
            fetch_from = datetime.now() - timedelta(days=days)
            fetch_days = days

        try:
            total_fetched = self._chunked_fetch(symbol, resolution, fetch_from, fetch_days)
            if total_fetched == 0:
                self._stamp_fetch_attempt(symbol, resolution)
        except Exception as e:
            from core.fyers_manager import InvalidSymbolError
            if isinstance(e, InvalidSymbolError):
                logger.warning(f"Invalid symbol — blacklisting {symbol}: {e}")
                self._stamp_fetch_attempt(symbol, resolution)
                raise
            logger.error(f"Fetch failed for {symbol}: {e}")
            self._stamp_fetch_attempt(symbol, resolution)

        return self.get_cached(symbol, resolution)

    def _chunked_fetch(self, symbol, resolution, fetch_from, total_days):
        """Split a long fetch into ≤MAX_DAYS chunks to respect Fyers API limits.
        Returns total number of candles stored.
        """
        max_chunk = self._FYERS_MAX_DAYS.get(str(resolution), 100)
        total_stored = 0

        chunk_start = fetch_from
        remaining = total_days

        while remaining > 0:
            chunk_days = min(remaining, max_chunk)
            chunk_end = chunk_start + timedelta(days=chunk_days)
            if chunk_end > datetime.now():
                chunk_end = datetime.now()

            from_str = chunk_start.strftime("%Y-%m-%d")
            to_str   = chunk_end.strftime("%Y-%m-%d")

            candles = self.fyers.get_history(
                symbol, resolution=resolution,
                from_date=from_str, to_date=to_str,
                days=chunk_days,
            )
            if candles:
                self._store_candles(symbol, resolution, candles)
                total_stored += len(candles)
                logger.debug(f"Chunk {from_str}→{to_str}: {len(candles)} candles for {symbol}")

            chunk_start = chunk_end
            remaining  -= chunk_days

        if total_stored:
            logger.info(f"Fetched {total_stored} total candles for {symbol}/{resolution}")
        return total_stored

    def fetch_batch(self, symbols, resolution="15"):
        """Fetch OHLCV data for a batch of symbols.
        Returns: dict[symbol, DataFrame]
        """
        result = {}
        for symbol in symbols:
            df = self.fetch_symbol(symbol, resolution)
            if df is not None:
                result[symbol] = df
        return result

    def fetch_batches(self, all_symbols, resolution="15"):
        """Process all symbols in rate-limited batches.
        Splits into batches of BATCH_SIZE, delays between batches.
        Returns: dict[symbol, DataFrame]
        """
        result = {}
        batch_size = config.BATCH_SIZE

        for i in range(0, len(all_symbols), batch_size):
            batch = all_symbols[i:i + batch_size]
            batch_result = self.fetch_batch(batch, resolution)
            result.update(batch_result)

            # Delay between batches (except after last batch)
            if i + batch_size < len(all_symbols):
                time.sleep(config.BATCH_DELAY)

        return result

    def fetch_multi_timeframe(self, symbol, timeframes=None):
        """Fetch data for multiple timeframes for multi-TF validation.
        Returns: dict[resolution, DataFrame]
        """
        timeframes = timeframes or config.MULTI_TIMEFRAMES
        result = {}
        for tf in timeframes:
            df = self.fetch_symbol(symbol, resolution=tf)
            if df is not None:
                result[tf] = df
        return result

    def _stamp_fetch_attempt(self, symbol, resolution):
        """Mark cache_meta with current time even when no candles were returned.
        This makes _is_cache_fresh() return True for CACHE_FRESHNESS_SEC, so the
        data engine won't hammer the Fyers API for the same (failed) symbol
        within the same freshness window.
        """
        with self._get_conn() as conn:
            # Get existing row_count (don't reset it to 0)
            row = conn.execute(
                "SELECT row_count FROM cache_meta WHERE symbol=? AND resolution=?",
                (symbol, resolution),
            ).fetchone()
            existing_count = row[0] if row else 0
            conn.execute(
                "INSERT OR REPLACE INTO cache_meta (symbol, resolution, last_update, row_count) "
                "VALUES (?, ?, ?, ?)",
                (symbol, resolution, datetime.now().isoformat(), existing_count),
            )

    def invalidate_cache(self, symbol, resolution=None):
        """Remove cached data for a symbol."""
        with self._get_conn() as conn:
            if resolution:
                conn.execute(
                    "DELETE FROM ohlcv_cache WHERE symbol=? AND resolution=?",
                    (symbol, resolution),
                )
                conn.execute(
                    "DELETE FROM cache_meta WHERE symbol=? AND resolution=?",
                    (symbol, resolution),
                )
            else:
                conn.execute("DELETE FROM ohlcv_cache WHERE symbol=?", (symbol,))
                conn.execute("DELETE FROM cache_meta WHERE symbol=?", (symbol,))

    def cleanup_old_data(self, days=90):
        """Remove data older than N days."""
        cutoff = int((datetime.now() - timedelta(days=days)).timestamp())
        with self._get_conn() as conn:
            deleted = conn.execute(
                "DELETE FROM ohlcv_cache WHERE timestamp < ?", (cutoff,)
            ).rowcount
            if deleted:
                logger.info(f"Cleaned up {deleted} old cache rows")

    def get_cache_stats(self):
        """Return cache statistics."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT symbol, resolution, last_update, row_count FROM cache_meta"
            ).fetchall()
        return [
            {"symbol": r[0], "resolution": r[1], "last_update": r[2], "rows": r[3]}
            for r in rows
        ]
