"""
Indian Market Data Enricher
Fetches FII/DII flows, delivery %, OI/PCR, sector data, and advance-decline ratio.
All data cached to minimize API calls.
"""
import time
import logging
from datetime import datetime, date
from threading import Lock

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)


class MarketDataEnricher:
    """Fetches and caches Indian market-specific data for signal enhancement."""

    def __init__(self, fyers_manager=None):
        self.fyers = fyers_manager
        self._cache = {}
        self._cache_timestamps = {}
        self._lock = Lock()

        # Cache durations (seconds)
        self.FII_DII_CACHE_SEC = 3600       # 1 hour (daily data)
        self.DELIVERY_CACHE_SEC = 300        # 5 minutes
        self.OPTION_CHAIN_CACHE_SEC = 120    # 2 minutes
        self.SECTOR_CACHE_SEC = 300          # 5 minutes
        self.AD_CACHE_SEC = 300              # 5 minutes

        # NSE session for web scraping
        self._nse_session = None
        self._nse_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        }

    def _is_cached(self, key, max_age_sec):
        """Check if cached data is still fresh."""
        ts = self._cache_timestamps.get(key)
        if ts is None:
            return False
        return (time.time() - ts) < max_age_sec

    def _get_nse_session(self):
        """Get or create NSE session with cookies."""
        if self._nse_session is None:
            self._nse_session = requests.Session()
            self._nse_session.headers.update(self._nse_headers)
            try:
                self._nse_session.get("https://www.nseindia.com/", timeout=10)
            except Exception:
                pass
        return self._nse_session

    # ─────────────────────────────────────────────────────────
    # FII/DII Data
    # ─────────────────────────────────────────────────────────
    def fetch_fii_dii_data(self):
        """Fetch FII/DII daily activity data from NSE.
        Returns: {fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net}
        """
        cache_key = f"fii_dii_{date.today().isoformat()}"
        if self._is_cached(cache_key, self.FII_DII_CACHE_SEC):
            return self._cache[cache_key]

        result = {
            "fii_buy": 0, "fii_sell": 0, "fii_net": 0,
            "dii_buy": 0, "dii_sell": 0, "dii_net": 0,
            "source": "unavailable",
        }

        try:
            session = self._get_nse_session()
            resp = session.get(
                "https://www.nseindia.com/api/fiidiiActivity",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                for entry in data:
                    category = entry.get("category", "")
                    if "FII" in category or "FPI" in category:
                        result["fii_buy"] = float(entry.get("buyValue", 0))
                        result["fii_sell"] = float(entry.get("sellValue", 0))
                        result["fii_net"] = result["fii_buy"] - result["fii_sell"]
                    elif "DII" in category:
                        result["dii_buy"] = float(entry.get("buyValue", 0))
                        result["dii_sell"] = float(entry.get("sellValue", 0))
                        result["dii_net"] = result["dii_buy"] - result["dii_sell"]
                result["source"] = "nse"
        except Exception as e:
            logger.warning(f"FII/DII fetch failed: {e}")

        with self._lock:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
        return result

    # ─────────────────────────────────────────────────────────
    # Delivery Data
    # ─────────────────────────────────────────────────────────
    def fetch_delivery_data(self, symbol):
        """Fetch delivery percentage for a symbol.
        Returns: {delivery_qty, traded_qty, delivery_pct}
        """
        cache_key = f"delivery_{symbol}"
        if self._is_cached(cache_key, self.DELIVERY_CACHE_SEC):
            return self._cache[cache_key]

        result = {"delivery_qty": 0, "traded_qty": 0, "delivery_pct": 0.0}

        try:
            ticker = symbol.replace("NSE:", "").replace("-EQ", "")
            session = self._get_nse_session()
            resp = session.get(
                f"https://www.nseindia.com/api/quote-equity?symbol={ticker}&section=trade_info",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                sec_info = data.get("securityWiseDP", {})
                result["delivery_qty"] = float(sec_info.get("deliveryQuantity", 0))
                result["traded_qty"] = float(sec_info.get("quantityTraded", 0))
                if result["traded_qty"] > 0:
                    result["delivery_pct"] = (
                        result["delivery_qty"] / result["traded_qty"] * 100
                    )
        except Exception as e:
            logger.debug(f"Delivery data fetch failed for {symbol}: {e}")

        with self._lock:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
        return result

    # ─────────────────────────────────────────────────────────
    # Option Chain (OI/PCR)
    # ─────────────────────────────────────────────────────────
    def fetch_option_chain_data(self, symbol):
        """Fetch option chain summary: OI, PCR, max pain, IV rank.
        Returns: {total_ce_oi, total_pe_oi, pcr, max_pain, iv_rank}
        """
        cache_key = f"oc_{symbol}"
        if self._is_cached(cache_key, self.OPTION_CHAIN_CACHE_SEC):
            return self._cache[cache_key]

        result = {
            "total_ce_oi": 0, "total_pe_oi": 0, "pcr": 1.0,
            "max_pain": 0, "iv_rank": 50,
        }

        if self.fyers:
            try:
                oc_data = self.fyers.get_option_chain(symbol)
                if oc_data and "optionsChain" in oc_data:
                    chain = oc_data["optionsChain"]
                    total_ce_oi = sum(
                        item.get("oi", 0) for item in chain if item.get("option_type") == "CE"
                    )
                    total_pe_oi = sum(
                        item.get("oi", 0) for item in chain if item.get("option_type") == "PE"
                    )
                    result["total_ce_oi"] = total_ce_oi
                    result["total_pe_oi"] = total_pe_oi
                    result["pcr"] = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0
            except Exception as e:
                logger.debug(f"Option chain fetch failed for {symbol}: {e}")

        with self._lock:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
        return result

    # ─────────────────────────────────────────────────────────
    # Sector Data
    # ─────────────────────────────────────────────────────────
    def fetch_sector_data(self):
        """Fetch NIFTY sectoral index performance.
        Returns: {sector: {change_pct, relative_strength}}
        """
        cache_key = "sector_data"
        if self._is_cached(cache_key, self.SECTOR_CACHE_SEC):
            return self._cache[cache_key]

        result = {}
        if self.fyers:
            from core.stock_universe import SECTOR_INDICES
            symbols = list(SECTOR_INDICES.values())
            try:
                quotes = self.fyers.get_batch_quotes(symbols)
                nifty_chp = 0
                if quotes:
                    for sym, data in quotes.items():
                        for sector, idx_sym in SECTOR_INDICES.items():
                            if idx_sym == sym:
                                chp = data.get("chp", 0)
                                result[sector] = {"change_pct": chp}
                                if "NIFTY 50" in sym:
                                    nifty_chp = chp
                    for sector in result:
                        result[sector]["relative_strength"] = (
                            result[sector]["change_pct"] - nifty_chp
                        )
            except Exception as e:
                logger.debug(f"Sector data fetch failed: {e}")

        with self._lock:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
        return result

    # ─────────────────────────────────────────────────────────
    # Advance/Decline Ratio
    # ─────────────────────────────────────────────────────────
    def fetch_advance_decline(self):
        """Fetch market breadth data.
        Returns: {advances, declines, unchanged, ad_ratio}
        """
        cache_key = "advance_decline"
        if self._is_cached(cache_key, self.AD_CACHE_SEC):
            return self._cache[cache_key]

        result = {"advances": 0, "declines": 0, "unchanged": 0, "ad_ratio": 1.0}

        try:
            session = self._get_nse_session()
            resp = session.get(
                "https://www.nseindia.com/api/market-data-pre-open?key=NIFTY%2050",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                adv = dec = unch = 0
                for item in data.get("data", []):
                    metadata = item.get("metadata", {})
                    chp = float(metadata.get("pChange", 0))
                    if chp > 0:
                        adv += 1
                    elif chp < 0:
                        dec += 1
                    else:
                        unch += 1
                result["advances"] = adv
                result["declines"] = dec
                result["unchanged"] = unch
                result["ad_ratio"] = adv / dec if dec > 0 else 10.0
        except Exception as e:
            logger.debug(f"Advance/Decline fetch failed: {e}")

        with self._lock:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
        return result

    # ─────────────────────────────────────────────────────────
    # Combined Enriched Context
    # ─────────────────────────────────────────────────────────
    def get_enriched_context(self, symbol):
        """Get all enrichment data for a single symbol.
        Returns combined dict of all market context features.
        """
        fii_dii = self.fetch_fii_dii_data()
        delivery = self.fetch_delivery_data(symbol)
        option_chain = self.fetch_option_chain_data(symbol)
        sector = self.fetch_sector_data()
        ad = self.fetch_advance_decline()

        from core.stock_universe import get_sector
        stock_sector = get_sector(symbol)
        sector_rs = sector.get(stock_sector, {}).get("relative_strength", 0)

        return {
            # FII/DII
            "fii_net": fii_dii.get("fii_net", 0),
            "dii_net": fii_dii.get("dii_net", 0),
            # Delivery
            "delivery_pct": delivery.get("delivery_pct", 0),
            # OI/PCR
            "pcr": option_chain.get("pcr", 1.0),
            "max_pain": option_chain.get("max_pain", 0),
            "iv_rank": option_chain.get("iv_rank", 50),
            # Sector
            "sector": stock_sector,
            "sector_relative_strength": sector_rs,
            # Market Breadth
            "ad_ratio": ad.get("ad_ratio", 1.0),
            "advances": ad.get("advances", 0),
            "declines": ad.get("declines", 0),
        }

    def get_enriched_context_batch(self):
        """Get market-wide context (FII/DII, sector, A/D) — called once per cycle.
        Per-symbol data (delivery, OI) fetched separately.
        """
        return {
            "fii_dii": self.fetch_fii_dii_data(),
            "sector": self.fetch_sector_data(),
            "ad": self.fetch_advance_decline(),
        }
