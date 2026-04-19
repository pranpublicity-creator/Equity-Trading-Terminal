"""
Configuration for Equity Trading Terminal - NSE/BSE Cash Market
AI-Powered Pattern Detection + ML Signal Fusion + Rate-Limited Execution
"""
import os

# =============================================================
# FYERS API CREDENTIALS
# =============================================================
FYERS_APP_ID = os.environ.get("FYERS_APP_ID", "")
FYERS_SECRET = os.environ.get("FYERS_SECRET", "")
FYERS_REDIRECT_URL = "https://www.google.com"

# =============================================================
# APP SETTINGS
# =============================================================
SECRET_KEY = os.environ.get("SECRET_KEY", "equity-terminal-secret-2026")
HOST = "0.0.0.0"
PORT = 5005
DEBUG = True

# =============================================================
# RATE LIMITS (Safe buffer below Fyers limits: 10/s, 200/min)
# =============================================================
MAX_REQ_PER_SEC = 8
MAX_REQ_PER_MIN = 150
BATCH_SIZE = 15
BATCH_DELAY = 1.5  # seconds between batches
RETRY_BASE_DELAY = 1.0
RETRY_FACTOR = 2.0
RETRY_MAX_ATTEMPTS = 3
RETRY_JITTER_MAX = 0.5

# =============================================================
# STOCK UNIVERSE
# =============================================================
MAX_STOCKS_PER_CYCLE = 100
DEFAULT_UNIVERSE = "NIFTY_200"

# =============================================================
# MARKET HOURS (NSE)
# =============================================================
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
NO_SIGNAL_FIRST_MIN = 5    # No signals in first 5 min after open
NO_SIGNAL_LAST_MIN = 10    # No signals in last 10 min before close
NSE_HOLIDAYS_2026 = [
    "2026-01-26", "2026-03-10", "2026-03-17", "2026-03-30",
    "2026-03-31", "2026-04-02", "2026-04-03", "2026-04-14",
    "2026-05-01", "2026-05-25", "2026-07-07", "2026-08-15",
    "2026-08-19", "2026-10-02", "2026-10-20", "2026-10-21",
    "2026-10-23", "2026-11-04", "2026-11-09", "2026-12-25",
]

# =============================================================
# DATA ENGINE
# =============================================================
CANDLE_RESOLUTION = "15"        # 15-min candles (primary timeframe)
CANDLE_HISTORY_DAYS = 90        # Days of history to cache
CACHE_FRESHNESS_SEC      = 900  # 15 min = 1 candle
CACHE_FRESHNESS_5MIN_SEC = 300  # 5 min  = 1 candle (on-demand intraday fetch)
MULTI_TIMEFRAMES = ["5", "15", "60"]
POLL_INTERVAL = 3               # seconds between poll cycles

# =============================================================
# INTRADAY PATTERN ENGINE (5-min)
# =============================================================
INTRADAY_PATTERNS_ENABLED = True   # master switch for 5-min intraday detectors

# Opening Range Breakout (ORB)
ORB_CANDLES          = 6     # first N × 5-min bars define the range (6 × 5 = 30 min)
ORB_VOLUME_MULT      = 1.3   # breakout bar volume must be ≥ 1.3× ORB-period average
ORB_MIN_RANGE_PCT    = 0.003 # ORB range / midprice must be ≥ 0.3% (filters flat opens)
ORB_MAX_RANGE_PCT    = 0.04  # ORB range / midprice must be ≤ 4.0% (filters gap-up chaos)
ORB_HISTORY_DAYS     = 5     # days of 5-min history to cache per symbol

# =============================================================
# PATTERN ENGINE
# =============================================================
SWING_LEFT_BARS = 5
SWING_RIGHT_BARS = 5
SWING_MIN_AMPLITUDE_PCT = 0.01  # 1% minimum swing
PATTERN_LOOKBACK_BARS = 200
PATTERN_MIN_CONFIDENCE = 0.4    # Min confidence to trigger ML
TRENDLINE_MIN_R_SQUARED = 0.75
TRENDLINE_MIN_TOUCHES = 2

# =============================================================
# ML THRESHOLDS
# =============================================================
ML_LOOKBACK_TIMESTEPS = 60      # Sequence length for LSTM/TFT
ML_PREDICTION_HORIZON = 6       # Predict 6 bars ahead (1.5h on 15m)
ML_TRAIN_DAYS = 90              # Training data window
ML_MIN_ACCURACY = 0.55          # Retrain if accuracy drops below
ML_RETRAIN_DAY = "Saturday"

# =============================================================
# SIGNAL FUSION
# =============================================================
SIGNAL_STRONG_THRESHOLD = 75    # Auto-execute
SIGNAL_MODERATE_THRESHOLD = 65  # Execute with alert
SIGNAL_WEAK_THRESHOLD = 45      # Alert only (lowered from 50 — pattern-only signals are valid)
FUSION_FALLBACK_WEIGHTS = {
    "pattern": 0.40,   # Primary trigger — deserves primary weight
    "lgbm": 0.20,      # Best trained model
    "xgb": 0.15,       # Second best
    "lstm": 0.08,      # Sequence model
    "tft": 0.06,       # Transformer (often untrained)
    "arima": 0.04,     # Trend supplement
    "prophet": 0.03,   # Seasonality (rarely installed)
    "fii": 0.02,       # Market context
    "oi": 0.02,        # Options context
}

# =============================================================
# RISK MANAGEMENT
# =============================================================
MAX_CONCURRENT_POSITIONS = 15      # Paper trade learning phase — capture wide signal sample
MAX_CAPITAL_PER_STOCK = 0.10    # 10% per stock (15 positions × 10% = fully deployed)
MAX_PORTFOLIO_RISK = 0.05       # 5% of total capital
MIN_RISK_REWARD = 1.0        # Chart pattern measure rules give ~1:1 R:R by design
TOTAL_TRADING_CAPITAL = 500000  # Rs 5 Lakh
TRAILING_STOP_ACTIVATION = 1.5  # Activate at 1.5x risk gained
TRAILING_STOP_ATR_MULT = 1.0

# =============================================================
# ORDER EXECUTION
# =============================================================
ORDER_MPP_PCT = 0.5             # Market Protection Price buffer % (configurable via UI)
MAX_PRICE_STALENESS_SECONDS = 30

# =============================================================
# NSE EQUITY CHARGES
# =============================================================
STT_DELIVERY_PCT = 0.1          # 0.1% both sides
STT_INTRADAY_PCT = 0.025        # 0.025% sell side only
EXCHANGE_CHARGES_PCT = 0.00345  # NSE
GST_PCT = 18.0                  # On brokerage + exchange charges
SEBI_CHARGES_PER_CRORE = 10.0   # Rs 10 per crore
STAMP_DUTY_PCT = 0.015          # Buy side
BROKERAGE_PER_ORDER = 20.0      # Rs 20 per order

# =============================================================
# SIGNAL ENGINE (NSE hours)
# =============================================================
SIGNAL_ENTRY_WINDOW_START = "09:20"
SIGNAL_ENTRY_WINDOW_END = "15:20"
SIGNAL_ADX_MIN = 20
SIGNAL_VOLUME_SPIKE_RATIO = 1.5
SIGNAL_ENTRY_COOLDOWN = 1800    # 30 min — hard floor between two signals on the same symbol

# Entry-drift cap for pattern-triggered trades.
# When price drifts this far past the pattern's projected breakout line,
# the breakout is considered "stale / exhausted" and the signal is suppressed
# — prevents phantom fills where entry price = neckline far from current LTP.
MAX_ENTRY_DRIFT_PCT = 0.8        # percent (LTP vs trigger)

# Post-trade per-symbol cooldown (seconds).
# Blocks re-entry into the same stock immediately after a close.
# Doubles on consecutive losses (2 → 60m, 3 → 120m, capped 4h).
# This is what stops the "close → pattern still firing → re-enter"
# loop that produced 10 MAXHEALTH / 7 BRITANNIA trades in one hour.
POST_TRADE_COOLDOWN_SEC = 1800   # 30 min base

# =============================================================
# TRADE QUALITY SCORER  (6-point framework, run before every trade)
# =============================================================
# Every signal is scored on six criteria; total score 0–1 is graded A/B/C/D/F.
# In AUTO mode, signals scoring below TRADE_QUALITY_MIN_SCORE are rejected.
# In MANUAL mode, low-score signals still queue as PENDING but carry a
# ⚠️ badge and the blocking reasons so the user can decide.
TRADE_QUALITY_ENABLED       = True
TRADE_QUALITY_MIN_SCORE     = 0.60   # 0-1; AUTO-mode hard gate (grade ≥ C)
TRADE_QUALITY_MIN_RR        = 1.5    # below this R:R → fail risk/reward check
TRADE_QUALITY_MAX_RR        = 10.0   # above this → almost always SL-too-tight artefact
TRADE_QUALITY_MIN_RISK_PCT  = 0.30   # min stop distance as % of entry (prevents inverted/too-tight SL)
TRADE_QUALITY_MAX_DRIFT_PCT = 1.0    # trigger→entry drift above this fails the "freshness" check

# =============================================================
# REGIME DETECTION
# =============================================================
REGIME_ADX_TREND_MIN = 25
REGIME_ADX_CONSOLIDATION_MAX = 15
REGIME_ADX_MEANREV_MAX = 20
REGIME_ATR_VOLATILE_MULT = 2.0

# =============================================================
# TELEGRAM NOTIFICATIONS
# =============================================================
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# =============================================================
# MAX DAILY LOSS LIMIT
# =============================================================
MAX_LOSS_PER_DAY = 5000         # INR

# =============================================================
# FILE PATHS
# =============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
MODELS_DIR = os.path.join(DATA_DIR, "models")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
TOKEN_FILE = os.path.join(BASE_DIR, "fyers_token.json")
TRADES_FILE = os.path.join(DATA_DIR, "trades.json")
SIGNAL_CONFIG_FILE = os.path.join(DATA_DIR, "signal_config.json")
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlist.json")
BOT_CONFIG_FILE = os.path.join(DATA_DIR, "bot_config.json")
STRATEGY_PERFORMANCE_FILE = os.path.join(DATA_DIR, "strategy_performance.json")
TELEGRAM_CONFIG_FILE = os.path.join(DATA_DIR, "telegram_config.json")
LOG_FILE = os.path.join(BASE_DIR, "app.log")
SQLITE_DB = os.path.join(CACHE_DIR, "ohlcv_cache.db")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Share token with other apps
PAPER_TRADE_TOKEN = os.path.join(BASE_DIR, "..", "PAPER TRADE APP", "fyers_token.json")
GEMS_APP_TOKEN = os.path.join(BASE_DIR, "..", "GEMS Trading", "fyers_token.json")
STRADDLE_APP_TOKEN = os.path.join(BASE_DIR, "..", "STRADDLE APP", "fyers_token.json")
