"""
Thread-Safe Dual Token Bucket Rate Limiter
Enforces Fyers API limits: 8 req/sec, 150 req/min with safe buffers.
"""
import time
import threading
import logging
import random
import functools

import config

logger = logging.getLogger(__name__)


class RateLimiter:
    """Dual token bucket rate limiter with per-second and per-minute pools."""

    def __init__(
        self,
        max_per_sec=config.MAX_REQ_PER_SEC,
        max_per_min=config.MAX_REQ_PER_MIN,
    ):
        self.max_per_sec = max_per_sec
        self.max_per_min = max_per_min
        self._tokens_sec = float(max_per_sec)
        self._tokens_min = float(max_per_min)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

        # Stats
        self._total_requests = 0
        self._total_waits = 0
        self._total_wait_time = 0.0

    def _refill(self):
        """Refill both buckets based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now

        # Per-second bucket refills at max_per_sec tokens/sec
        self._tokens_sec = min(
            self.max_per_sec,
            self._tokens_sec + elapsed * self.max_per_sec,
        )
        # Per-minute bucket refills at max_per_min/60 tokens/sec
        self._tokens_min = min(
            self.max_per_min,
            self._tokens_min + elapsed * (self.max_per_min / 60.0),
        )

    def wait_and_acquire(self, count=1):
        """Block until `count` tokens are available from both buckets, then consume them."""
        start = time.monotonic()
        waited = False

        while True:
            with self._lock:
                self._refill()
                if self._tokens_sec >= count and self._tokens_min >= count:
                    self._tokens_sec -= count
                    self._tokens_min -= count
                    self._total_requests += count
                    if waited:
                        wait_time = time.monotonic() - start
                        self._total_waits += 1
                        self._total_wait_time += wait_time
                    return

            waited = True
            time.sleep(0.05)  # 50ms polling interval

    def try_acquire(self, count=1):
        """Non-blocking acquire. Returns True if tokens were consumed, False otherwise."""
        with self._lock:
            self._refill()
            if self._tokens_sec >= count and self._tokens_min >= count:
                self._tokens_sec -= count
                self._tokens_min -= count
                self._total_requests += count
                return True
            return False

    def get_stats(self):
        """Return current rate limiter statistics."""
        with self._lock:
            self._refill()
            return {
                "tokens_sec": round(self._tokens_sec, 2),
                "tokens_min": round(self._tokens_min, 2),
                "max_per_sec": self.max_per_sec,
                "max_per_min": self.max_per_min,
                "total_requests": self._total_requests,
                "total_waits": self._total_waits,
                "total_wait_time": round(self._total_wait_time, 3),
            }


def retry_with_backoff(
    func=None,
    max_attempts=config.RETRY_MAX_ATTEMPTS,
    base_delay=config.RETRY_BASE_DELAY,
    factor=config.RETRY_FACTOR,
    jitter_max=config.RETRY_JITTER_MAX,
    no_retry_exceptions=(),
):
    """Decorator: retry a function with exponential backoff on exception.

    no_retry_exceptions: tuple of exception types that should NOT be retried
    (e.g., InvalidSymbolError — permanent failures that retrying won't fix).
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except no_retry_exceptions:
                    # Permanent failure — re-raise immediately without retry
                    raise
                except Exception as e:
                    last_exc = e
                    if attempt == max_attempts:
                        logger.error(
                            f"{fn.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    delay = base_delay * (factor ** (attempt - 1))
                    delay += random.uniform(0, jitter_max)
                    logger.warning(
                        f"{fn.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
            raise last_exc
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# Global singleton
rate_limiter = RateLimiter()
