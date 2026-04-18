"""
Indian Market Context Features
Converts enricher data into normalized features for ML models.
FII/DII flows, PCR, sector RS, advance-decline, delivery signals.
"""
import numpy as np


def build_context_features(symbol, enricher_data, current_price=0, atr=0):
    """Build market context features from MarketDataEnricher output.

    Args:
        symbol: stock symbol (e.g., 'NSE:RELIANCE-EQ')
        enricher_data: dict from MarketDataEnricher.get_enriched_context()
        current_price: current stock price (for max pain distance)
        atr: current ATR value (for normalization)

    Returns:
        dict with ~15 context features
    """
    features = {}

    # FII/DII features
    fii_net = enricher_data.get("fii_net", 0)
    dii_net = enricher_data.get("dii_net", 0)

    # Normalize FII/DII to a -1 to +1 scale (rough normalization)
    # Typical daily FII activity is in the range of +/- 5000 crore
    features["fii_net_normalized"] = np.clip(fii_net / 5000.0, -1.0, 1.0)
    features["dii_net_normalized"] = np.clip(dii_net / 5000.0, -1.0, 1.0)

    # FII/DII divergence signal
    # FII buying + DII buying = strong bull (2)
    # FII buying + DII selling = FII driven (1)
    # FII selling + DII buying = DII support (-1)
    # FII selling + DII selling = strong bear (-2)
    fii_dir = 1 if fii_net > 0 else -1
    dii_dir = 1 if dii_net > 0 else -1
    features["fii_dii_divergence"] = fii_dir + dii_dir

    # PCR features
    pcr = enricher_data.get("pcr", 1.0)
    features["pcr_raw"] = pcr
    # Map PCR to signal: >1.2 = bullish (+1), <0.8 = bearish (-1), neutral (0)
    if pcr > 1.2:
        features["pcr_signal"] = min((pcr - 1.0) / 0.5, 1.0)
    elif pcr < 0.8:
        features["pcr_signal"] = max((pcr - 1.0) / 0.5, -1.0)
    else:
        features["pcr_signal"] = 0.0

    # Max pain distance
    max_pain = enricher_data.get("max_pain", 0)
    if max_pain > 0 and current_price > 0 and atr > 0:
        features["max_pain_distance"] = (current_price - max_pain) / atr
    else:
        features["max_pain_distance"] = 0.0

    # IV rank (0-100 percentile)
    features["iv_rank"] = enricher_data.get("iv_rank", 50) / 100.0

    # Sector relative strength
    features["sector_relative_strength"] = enricher_data.get("sector_relative_strength", 0)

    # Advance/Decline ratio
    ad_ratio = enricher_data.get("ad_ratio", 1.0)
    features["ad_ratio"] = np.clip(ad_ratio, 0.0, 10.0)
    # Market breadth signal: >2.0 = strong bull, <0.5 = strong bear
    if ad_ratio > 2.0:
        features["market_breadth_signal"] = 1.0
    elif ad_ratio < 0.5:
        features["market_breadth_signal"] = -1.0
    else:
        features["market_breadth_signal"] = (ad_ratio - 1.0) / 1.0

    # Delivery % features
    delivery_pct = enricher_data.get("delivery_pct", 0)
    features["delivery_pct_normalized"] = delivery_pct / 100.0
    features["delivery_institutional_flag"] = 1 if delivery_pct > 50 else 0

    # Circuit proximity (would need current price vs circuit limits)
    features["circuit_proximity"] = 1.0  # Safe default (far from circuit)

    return features
