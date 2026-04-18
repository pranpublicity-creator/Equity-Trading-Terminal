"""
Stock Universe Definitions for NSE Equity
NIFTY 50, NIFTY 100, NIFTY 200, F&O eligible stocks, and custom watchlists.
"""
import json
import os
import logging

import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# NIFTY 50 (as of April 2026)
# ─────────────────────────────────────────────────────────────
NIFTY_50 = [
    "NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ", "NSE:INFY-EQ",
    "NSE:ICICIBANK-EQ", "NSE:HINDUNILVR-EQ", "NSE:ITC-EQ", "NSE:SBIN-EQ",
    "NSE:BHARTIARTL-EQ", "NSE:KOTAKBANK-EQ", "NSE:LT-EQ", "NSE:AXISBANK-EQ",
    "NSE:BAJFINANCE-EQ", "NSE:ASIANPAINT-EQ", "NSE:MARUTI-EQ", "NSE:HCLTECH-EQ",
    "NSE:TITAN-EQ", "NSE:SUNPHARMA-EQ", "NSE:NTPC-EQ",
    "NSE:ULTRACEMCO-EQ", "NSE:WIPRO-EQ", "NSE:POWERGRID-EQ", "NSE:M&M-EQ",
    "NSE:NESTLEIND-EQ", "NSE:TATASTEEL-EQ", "NSE:JSWSTEEL-EQ", "NSE:BAJAJFINSV-EQ",
    "NSE:ADANIENT-EQ", "NSE:ADANIPORTS-EQ", "NSE:TECHM-EQ", "NSE:ONGC-EQ",
    "NSE:COALINDIA-EQ", "NSE:GRASIM-EQ", "NSE:BPCL-EQ", "NSE:DRREDDY-EQ",
    "NSE:CIPLA-EQ", "NSE:APOLLOHOSP-EQ", "NSE:EICHERMOT-EQ", "NSE:DIVISLAB-EQ",
    "NSE:BRITANNIA-EQ", "NSE:HEROMOTOCO-EQ", "NSE:INDUSINDBK-EQ", "NSE:SBILIFE-EQ",
    "NSE:HDFCLIFE-EQ", "NSE:TATACONSUM-EQ", "NSE:BAJAJ-AUTO-EQ", "NSE:HINDALCO-EQ",
    "NSE:SHRIRAMFIN-EQ",
]

# ─────────────────────────────────────────────────────────────
# NIFTY NEXT 50 (NIFTY 100 = NIFTY 50 + NEXT 50)
# ─────────────────────────────────────────────────────────────
NIFTY_NEXT_50 = [
    "NSE:AMBUJACEM-EQ", "NSE:BANKBARODA-EQ", "NSE:BEL-EQ", "NSE:BERGEPAINT-EQ",
    "NSE:BOSCHLTD-EQ", "NSE:CANBK-EQ", "NSE:CHOLAFIN-EQ", "NSE:COLPAL-EQ",
    "NSE:DABUR-EQ", "NSE:DLF-EQ", "NSE:GAIL-EQ", "NSE:GODREJCP-EQ",
    "NSE:HAVELLS-EQ", "NSE:ICICIPRULI-EQ", "NSE:IDEA-EQ", "NSE:INDIGO-EQ",
    "NSE:IOC-EQ", "NSE:IRCTC-EQ", "NSE:JINDALSTEL-EQ", "NSE:LICI-EQ",
    "NSE:LUPIN-EQ", "NSE:MARICO-EQ", "NSE:MCDOWELL-N-EQ", "NSE:MOTHERSON-EQ",
    "NSE:NAUKRI-EQ", "NSE:PFC-EQ", "NSE:PIDILITIND-EQ", "NSE:PNB-EQ",
    "NSE:RECLTD-EQ", "NSE:SAIL-EQ", "NSE:SIEMENS-EQ", "NSE:SRF-EQ",
    "NSE:TATAPOWER-EQ", "NSE:TORNTPHARM-EQ", "NSE:TRENT-EQ", "NSE:UNIONBANK-EQ",
    "NSE:VEDL-EQ", "NSE:ZOMATO-EQ", "NSE:ZYDUSLIFE-EQ", "NSE:ABB-EQ",
    "NSE:ADANIGREEN-EQ", "NSE:ATGL-EQ", "NSE:DMART-EQ", "NSE:HAL-EQ",
    "NSE:ICICIGI-EQ", "NSE:INDUSTOWER-EQ", "NSE:JIOFIN-EQ", "NSE:MAXHEALTH-EQ",
    "NSE:PAYTM-EQ", "NSE:POLYCAB-EQ",
]

# ─────────────────────────────────────────────────────────────
# NIFTY 200 extras (top F&O stocks beyond NIFTY 100)
# ─────────────────────────────────────────────────────────────
NIFTY_200_EXTRAS = [
    "NSE:AARTIIND-EQ", "NSE:ACC-EQ", "NSE:ALKEM-EQ", "NSE:AUROPHARMA-EQ",
    "NSE:BALKRISIND-EQ", "NSE:BANDHANBNK-EQ", "NSE:BATAINDIA-EQ", "NSE:BHEL-EQ",
    "NSE:BIOCON-EQ", "NSE:CANFINHOME-EQ", "NSE:CHAMBLFERT-EQ", "NSE:COFORGE-EQ",
    "NSE:CONCOR-EQ", "NSE:CROMPTON-EQ", "NSE:CUB-EQ", "NSE:CUMMINSIND-EQ",
    "NSE:DEEPAKNTR-EQ", "NSE:DELTACORP-EQ", "NSE:DIXON-EQ", "NSE:ESCORTS-EQ",
    "NSE:EXIDEIND-EQ", "NSE:FEDERALBNK-EQ", "NSE:GMRINFRA-EQ", "NSE:GNFC-EQ",
    "NSE:GODREJPROP-EQ", "NSE:GRANULES-EQ", "NSE:GSPL-EQ", "NSE:GUJGASLTD-EQ",
    "NSE:HINDCOPPER-EQ", "NSE:HINDPETRO-EQ", "NSE:HONAUT-EQ", "NSE:IDFCFIRSTB-EQ",
    "NSE:IEX-EQ", "NSE:INDHOTEL-EQ", "NSE:IRFC-EQ", "NSE:JUBLFOOD-EQ",
    "NSE:LAURUSLABS-EQ", "NSE:LICHSGFIN-EQ", "NSE:LTF-EQ", "NSE:LTTS-EQ",
    "NSE:M&MFIN-EQ", "NSE:MANAPPURAM-EQ", "NSE:METROPOLIS-EQ", "NSE:MFSL-EQ",
    "NSE:MGL-EQ", "NSE:MPHASIS-EQ", "NSE:MUTHOOTFIN-EQ", "NSE:NAM-INDIA-EQ",
    "NSE:NATIONALUM-EQ", "NSE:NAVINFLUOR-EQ", "NSE:NMDC-EQ", "NSE:OBEROIRLTY-EQ",
    "NSE:OFSS-EQ", "NSE:PAGEIND-EQ", "NSE:PERSISTENT-EQ", "NSE:PETRONET-EQ",
    "NSE:PIIND-EQ", "NSE:PVRINOX-EQ", "NSE:RAIN-EQ", "NSE:RAMCOCEM-EQ",
    "NSE:RBLBANK-EQ", "NSE:SBICARD-EQ", "NSE:STAR-EQ", "NSE:SUNTV-EQ",
    "NSE:SYNGENE-EQ", "NSE:TATACHEM-EQ", "NSE:TATACOMM-EQ", "NSE:TATAELXSI-EQ",
    "NSE:TORNTPOWER-EQ", "NSE:TVSMOTOR-EQ", "NSE:UBL-EQ", "NSE:UJJIVANSFB-EQ",
    "NSE:UPL-EQ", "NSE:VOLTAS-EQ", "NSE:WHIRLPOOL-EQ", "NSE:ZEEL-EQ",
    "NSE:ASHOKLEY-EQ", "NSE:AUBANK-EQ", "NSE:BHARATFORG-EQ", "NSE:COROMANDEL-EQ",
    "NSE:CRISIL-EQ", "NSE:DALBHARAT-EQ", "NSE:EMAMILTD-EQ", "NSE:GLENMARK-EQ",
    "NSE:IPCALAB-EQ", "NSE:JKCEMENT-EQ", "NSE:JSL-EQ", "NSE:KAJARIACER-EQ",
    "NSE:KEI-EQ", "NSE:KEC-EQ", "NSE:KPITTECH-EQ", "NSE:L&TFH-EQ",
    "NSE:MCX-EQ", "NSE:NATCOPHARM-EQ", "NSE:NIACL-EQ",
    "NSE:PHOENIXLTD-EQ", "NSE:PRESTIGE-EQ", "NSE:RAJESHEXPO-EQ", "NSE:RELAXO-EQ",
    "NSE:SUNCLAYLTD-EQ", "NSE:SUVENPHAR-EQ",
]

NIFTY_100 = NIFTY_50 + NIFTY_NEXT_50
NIFTY_200 = NIFTY_100 + NIFTY_200_EXTRAS

# ─────────────────────────────────────────────────────────────
# STOCK METADATA (sector mapping for key stocks)
# ─────────────────────────────────────────────────────────────
SECTOR_MAP = {
    "RELIANCE": "ENERGY", "TCS": "IT", "HDFCBANK": "BANK", "INFY": "IT",
    "ICICIBANK": "BANK", "HINDUNILVR": "FMCG", "ITC": "FMCG", "SBIN": "BANK",
    "BHARTIARTL": "TELECOM", "KOTAKBANK": "BANK", "LT": "INFRA", "AXISBANK": "BANK",
    "BAJFINANCE": "FINANCE", "ASIANPAINT": "CONSUMER", "MARUTI": "AUTO",
    "HCLTECH": "IT", "TITAN": "CONSUMER", "SUNPHARMA": "PHARMA",
    "TATAMOTORS": "AUTO", "NTPC": "ENERGY", "ULTRACEMCO": "CEMENT",
    "WIPRO": "IT", "POWERGRID": "ENERGY", "M&M": "AUTO",
    "TATASTEEL": "METAL", "JSWSTEEL": "METAL", "ONGC": "ENERGY",
    "COALINDIA": "ENERGY", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "APOLLOHOSP": "PHARMA", "HINDALCO": "METAL", "TECHM": "IT",
    "DLF": "REALTY", "ADANIENT": "INFRA", "ADANIPORTS": "INFRA",
    "TATAPOWER": "ENERGY", "VEDL": "METAL", "SAIL": "METAL",
    "JINDALSTEL": "METAL", "ZOMATO": "CONSUMER", "HAL": "DEFENCE",
}

# ─────────────────────────────────────────────────────────────
# NIFTY SECTORAL INDICES (for sector rotation tracking)
# ─────────────────────────────────────────────────────────────
SECTOR_INDICES = {
    "IT": "NSE:NIFTY IT-INDEX",
    "BANK": "NSE:NIFTY BANK-INDEX",
    "PHARMA": "NSE:NIFTY PHARMA-INDEX",
    "AUTO": "NSE:NIFTY AUTO-INDEX",
    "FMCG": "NSE:NIFTY FMCG-INDEX",
    "METAL": "NSE:NIFTY METAL-INDEX",
    "REALTY": "NSE:NIFTY REALTY-INDEX",
    "ENERGY": "NSE:NIFTY ENERGY-INDEX",
}


def get_universe(name="NIFTY_200"):
    """Return list of symbols for the named universe."""
    universes = {
        "NIFTY_50": NIFTY_50,
        "NIFTY_100": NIFTY_100,
        "NIFTY_200": NIFTY_200,
        "CUSTOM": load_custom_watchlist(),
    }
    return universes.get(name, NIFTY_200)


def load_custom_watchlist(path=None):
    """Load custom watchlist from JSON file."""
    path = path or os.path.join(config.DATA_DIR, "watchlist.json")
    if not os.path.exists(path):
        return NIFTY_50  # Default fallback
    try:
        with open(path) as f:
            data = json.load(f)
        symbols = data.get("symbols", data) if isinstance(data, dict) else data
        if isinstance(symbols, list) and symbols:
            return symbols
    except Exception as e:
        logger.error(f"Failed to load custom watchlist: {e}")
    return NIFTY_50


def get_sector(symbol):
    """Get sector for a symbol. Symbol can be 'NSE:RELIANCE-EQ' or 'RELIANCE'."""
    ticker = symbol.replace("NSE:", "").replace("-EQ", "")
    return SECTOR_MAP.get(ticker, "OTHER")


def get_ticker(fyers_symbol):
    """Extract ticker from Fyers symbol: 'NSE:RELIANCE-EQ' -> 'RELIANCE'."""
    return fyers_symbol.replace("NSE:", "").replace("-EQ", "")
