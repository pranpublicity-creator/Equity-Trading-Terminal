"""
Fyers API Manager for NSE Equity Trading
Handles authentication, quotes, historical data, and order placement.
Adapted from COMMODITY APP/fyers_manager.py — MCX-specific code removed.
"""
import json
import os
import logging
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel

import config
from core.rate_limiter import rate_limiter, retry_with_backoff


class InvalidSymbolError(Exception):
    """Raised when Fyers API rejects a symbol as invalid/unrecognised.
    Permanent failure — do NOT retry, mark symbol for cooldown immediately.
    """
    pass

logger = logging.getLogger(__name__)

_CREDS_FILE = os.path.join(config.DATA_DIR, "api_credentials.json")


class FyersManager:
    def __init__(self):
        self.client_id = config.FYERS_APP_ID
        self.secret_key = config.FYERS_SECRET
        self.redirect_uri = config.FYERS_REDIRECT_URL
        self._load_credential_override()
        self.access_token = None
        self.fyers = None
        self._load_token()

    def _load_credential_override(self):
        """Load App ID / Secret from JSON file if saved via UI."""
        try:
            if os.path.exists(_CREDS_FILE):
                with open(_CREDS_FILE) as f:
                    creds = json.load(f)
                if creds.get("app_id"):
                    self.client_id = creds["app_id"]
                if creds.get("secret"):
                    self.secret_key = creds["secret"]
                logger.info(f"Credentials loaded from file: App ID = {self.client_id}")
        except Exception as e:
            logger.warning(f"Could not load credential override: {e}")

    @staticmethod
    def save_credentials(app_id, secret):
        """Save new credentials to JSON file."""
        try:
            os.makedirs(os.path.dirname(_CREDS_FILE), exist_ok=True)
            with open(_CREDS_FILE, "w") as f:
                json.dump({"app_id": app_id.strip(), "secret": secret.strip()}, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Could not save credentials: {e}")
            return False

    # ----------------------------------------------------------
    # Authentication
    # ----------------------------------------------------------
    def generate_auth_url(self):
        session = fyersModel.SessionModel(
            client_id=self.client_id,
            secret_key=self.secret_key,
            redirect_uri=self.redirect_uri,
            response_type="code",
        )
        return session.generate_authcode()

    def generate_token(self, auth_code):
        try:
            session = fyersModel.SessionModel(
                client_id=self.client_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type="code",
                grant_type="authorization_code",
            )
            session.set_token(auth_code)
            response = session.generate_token()
            if isinstance(response, dict) and (
                response.get("s") == "ok" or "access_token" in response
            ):
                self.access_token = response["access_token"]
                self._init_fyers()
                self._save_token()
                return True, "Authentication successful"
            else:
                msg = response.get("message", str(response)) if isinstance(response, dict) else str(response)
                return False, f"Auth failed: {msg}"
        except Exception as e:
            return False, f"Token error: {str(e)}"

    def is_authenticated(self):
        return self.access_token is not None and self.fyers is not None

    def logout(self):
        self.access_token = None
        self.fyers = None
        if os.path.exists(config.TOKEN_FILE):
            os.remove(config.TOKEN_FILE)

    # ----------------------------------------------------------
    # Market Data — Equity
    # ----------------------------------------------------------
    @retry_with_backoff
    def get_quotes(self, symbols):
        """Get quotes for one or more symbols (comma-separated string).
        Rate-limited: acquires 1 token before each call.
        """
        if not self.is_authenticated():
            return None
        rate_limiter.wait_and_acquire(1)
        resp = self.fyers.quotes(data={"symbols": symbols})
        if resp and resp.get("s") == "ok":
            return resp.get("d", [])
        return None

    def get_equity_quote(self, symbol):
        """Get current quote for a single equity symbol (e.g., NSE:RELIANCE-EQ)."""
        quotes = self.get_quotes(symbol)
        if quotes and len(quotes) > 0:
            v = quotes[0].get("v", {})
            return {
                "ltp": v.get("lp", 0),
                "chp": v.get("chp", 0),
                "ch": v.get("ch", 0),
                "volume": v.get("volume", 0),
                "vwap": v.get("vwap", 0),
                "high": v.get("high_price", 0),
                "low": v.get("low_price", 0),
                "open": v.get("open_price", 0),
                "prev_close": v.get("prev_close_price", 0),
                "symbol": symbol,
            }
        return None

    def get_batch_quotes(self, symbols_list):
        """Batch fetch quotes for multiple equity symbols in a single API call.
        Fyers supports comma-separated symbols.
        Returns: dict[symbol, quote_data]
        """
        if not symbols_list:
            return {}
        symbols_str = ",".join(symbols_list)
        quotes = self.get_quotes(symbols_str)
        result = {}
        if quotes:
            for q in quotes:
                sym = q.get("n", "")
                v = q.get("v", {})
                result[sym] = {
                    "ltp": v.get("lp", 0),
                    "chp": v.get("chp", 0),
                    "ch": v.get("ch", 0),
                    "volume": v.get("volume", 0),
                    "vwap": v.get("vwap", 0),
                    "high": v.get("high_price", 0),
                    "low": v.get("low_price", 0),
                    "open": v.get("open_price", 0),
                    "prev_close": v.get("prev_close_price", 0),
                }
        return result

    # ----------------------------------------------------------
    # Historical Data
    # ----------------------------------------------------------
    @retry_with_backoff(no_retry_exceptions=(InvalidSymbolError,))
    def get_history(self, symbol, resolution="15", days=2, from_date=None, to_date=None):
        """Fetch historical candle data for an equity symbol.
        Rate-limited: acquires 1 token before each call.
        """
        if not self.is_authenticated():
            return []
        rate_limiter.wait_and_acquire(1)
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            resp = self.fyers.history(data={
                "symbol": symbol,
                "resolution": resolution,
                "date_format": "1",
                "range_from": from_date,
                "range_to": to_date,
                "cont_flag": "1",
            })
            if resp and resp.get("s") == "ok":
                return resp.get("candles", [])
            err_msg = resp.get("message", str(resp)) if isinstance(resp, dict) else str(resp)
            # Permanent errors — raise immediately (no retry, instant cooldown)
            if any(kw in err_msg.lower() for kw in ("invalid symbol", "symbol not found", "no data found for given symbol")):
                raise InvalidSymbolError(f"{symbol}: {err_msg}")
            logger.warning(f"History API failed for {symbol} ({resolution}/{days}d): {err_msg}")
            return []
        except Exception as e:
            logger.error(f"History error for {symbol}: {e}")
            raise

    def get_option_chain(self, symbol):
        """Fetch option chain data for OI/PCR analysis.
        symbol: underlying (e.g., 'NSE:RELIANCE-EQ')
        """
        if not self.is_authenticated():
            return None
        try:
            rate_limiter.wait_and_acquire(1)
            # Extract ticker for option chain lookup
            ticker = symbol.replace("NSE:", "").replace("-EQ", "")
            resp = self.fyers.optionchain(data={
                "symbol": f"NSE:{ticker}",
                "strikecount": 10,
                "timestamp": "",
            })
            if resp and resp.get("s") == "ok":
                return resp.get("data", {})
            return None
        except Exception as e:
            logger.error(f"Option chain error for {symbol}: {e}")
            return None

    # ----------------------------------------------------------
    # Order Placement — Equity
    # ----------------------------------------------------------
    def place_order(self, symbol, qty, side, order_type="MARKET", limit_price=0):
        """Place order for NSE equity.
        side: 1=BUY, -1=SELL
        """
        if not self.fyers:
            return {"s": "error", "message": "Fyers not initialized"}

        rate_limiter.wait_and_acquire(1)
        order_data = {
            "symbol": symbol,
            "qty": qty,
            "type": 2 if order_type == "MARKET" else 1,
            "side": side,
            "productType": "CNC",           # Delivery for equity
            "limitPrice": limit_price if order_type == "LIMIT" else 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
        }
        try:
            response = self.fyers.place_order(data=order_data)
            side_str = "BUY" if side == 1 else "SELL"
            logger.info(f"ORDER → {symbol} {side_str} qty={qty} | resp={response}")
            return response
        except Exception as e:
            logger.error(f"ORDER FAILED: {symbol} | {e}")
            return {"s": "error", "message": str(e)}

    def place_intraday_order(self, symbol, qty, side, order_type="MARKET", limit_price=0):
        """Place intraday (MIS) order for equity."""
        if not self.fyers:
            return {"s": "error", "message": "Fyers not initialized"}

        rate_limiter.wait_and_acquire(1)
        order_data = {
            "symbol": symbol,
            "qty": qty,
            "type": 2 if order_type == "MARKET" else 1,
            "side": side,
            "productType": "INTRADAY",
            "limitPrice": limit_price if order_type == "LIMIT" else 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
        }
        try:
            response = self.fyers.place_order(data=order_data)
            side_str = "BUY" if side == 1 else "SELL"
            logger.info(f"INTRADAY ORDER → {symbol} {side_str} qty={qty} | resp={response}")
            return response
        except Exception as e:
            logger.error(f"INTRADAY ORDER FAILED: {symbol} | {e}")
            return {"s": "error", "message": str(e)}

    def get_positions(self):
        if not self.fyers:
            return []
        try:
            rate_limiter.wait_and_acquire(1)
            response = self.fyers.positions()
            if response.get("s") == "ok":
                return response.get("netPositions", [])
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
        return []

    def get_funds(self):
        if not self.fyers:
            return {}
        try:
            rate_limiter.wait_and_acquire(1)
            response = self.fyers.funds()
            if response.get("s") == "ok":
                return response.get("fund_limit", [])
        except Exception as e:
            logger.error(f"Failed to fetch funds: {e}")
        return {}

    # ----------------------------------------------------------
    # Private
    # ----------------------------------------------------------
    def _init_fyers(self):
        log_dir = os.path.join(config.BASE_DIR, "fyers_logs")
        os.makedirs(log_dir, exist_ok=True)
        self.fyers = fyersModel.FyersModel(
            client_id=self.client_id,
            token=self.access_token,
            log_path=log_dir,
        )

    def _save_token(self):
        data = {
            "access_token": self.access_token,
            "timestamp": datetime.now().isoformat(),
            "app_id": self.client_id,
        }
        os.makedirs(os.path.dirname(config.TOKEN_FILE), exist_ok=True)
        with open(config.TOKEN_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def _load_token(self):
        """Load token from own file or shared app token files."""
        for token_path in [
            config.TOKEN_FILE,
            config.PAPER_TRADE_TOKEN,
            config.GEMS_APP_TOKEN,
            config.STRADDLE_APP_TOKEN,
        ]:
            if not os.path.exists(token_path):
                continue
            try:
                with open(token_path, "r") as f:
                    data = json.load(f)
                token_app_id = data.get("app_id")
                if token_app_id and token_app_id != self.client_id:
                    continue
                saved_time = datetime.fromisoformat(data["timestamp"])
                if saved_time.date() == datetime.now().date():
                    self.access_token = data["access_token"]
                    self._init_fyers()
                    logger.info(f"Loaded token from {token_path} (App ID: {self.client_id})")
                    return
            except Exception as e:
                logger.warning(f"Could not load token from {token_path}: {e}")
