"""
Portfolio Allocator
Ranks signals by composite score, applies sector diversification,
and allocates capital proportionally across top-5 trades.
"""
import logging
from typing import List, Dict, Optional

import config

logger = logging.getLogger(__name__)

# NSE NIFTY-100 sector mapping
SECTOR_MAP: Dict[str, str] = {
    # Banking & Finance
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "KOTAKBANK": "BANKING",
    "AXISBANK": "BANKING", "SBIN": "BANKING", "INDUSINDBK": "BANKING",
    "BAJFINANCE": "FINANCE", "BAJAJFINSV": "FINANCE", "SBILIFE": "FINANCE",
    "HDFCLIFE": "FINANCE", "SHRIRAMFIN": "FINANCE",
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "TECHM": "IT", "LTIM": "IT",
    # Energy & Oil
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY",
    "COALINDIA": "ENERGY", "NTPC": "ENERGY", "POWERGRID": "ENERGY",
    # Auto
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "M&M": "AUTO",
    "BAJAJ-AUTO": "AUTO", "HEROMOTOCO": "AUTO", "EICHERMOT": "AUTO",
    # Pharma
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "DIVISLAB": "PHARMA", "APOLLOHOSP": "PHARMA",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG",
    # Metal & Materials
    "TATASTEEL": "METAL", "JSWSTEEL": "METAL", "HINDALCO": "METAL",
    "GRASIM": "METAL", "ULTRACEMCO": "CEMENT",
    # Infra & Capital Goods
    "LT": "INFRA", "ADANIPORTS": "INFRA",
    # Consumer & Others
    "TITAN": "CONSUMER", "ASIANPAINT": "CONSUMER",
    "ADANIENT": "CONGLOMERATE",
}


class PortfolioAllocator:
    """Ranks signals and allocates capital with sector diversification."""

    MAX_POSITIONS    = 5
    MAX_EXPOSURE_PCT = 0.30   # 30% total portfolio exposure
    MIN_CONFIDENCE   = 65.0

    def __init__(self, total_capital: float = None):
        self.total_capital = total_capital or config.TOTAL_TRADING_CAPITAL

    def rank_and_allocate(self, signals: list, current_exposure: float = 0.0) -> List[dict]:
        """Rank signals and return top-5 with capital allocations.

        Args:
            signals: list of TradeSignal objects
            current_exposure: already-allocated capital (₹)

        Returns:
            list of dicts: {signal, score, allocated_capital, quantity_hint}
        """
        if not signals:
            return []

        # Score each signal
        scored = []
        for sig in signals:
            if sig.confidence < self.MIN_CONFIDENCE:
                continue
            score = self._composite_score(sig)
            scored.append((score, sig))

        if not scored:
            return []

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Sector diversification — one per sector, keep highest score
        diversified = self._sector_diversify(scored)

        # Take top MAX_POSITIONS
        top = diversified[:self.MAX_POSITIONS]

        # Available budget
        max_exposure = self.total_capital * self.MAX_EXPOSURE_PCT
        available    = max(0.0, max_exposure - current_exposure)
        if available <= 0:
            logger.info("Portfolio allocator: max exposure reached")
            return []

        # Proportional allocation by score
        total_score = sum(s for s, _ in top)
        result = []
        for score, sig in top:
            weight     = score / total_score if total_score > 0 else 1.0 / len(top)
            alloc_cap  = round(available * weight, 2)
            qty_hint   = max(1, int(alloc_cap / sig.entry_price)) if sig.entry_price > 0 else 1
            result.append({
                "symbol":           sig.symbol,
                "direction":        sig.direction,
                "confidence":       sig.confidence,
                "pattern":          sig.pattern_name,
                "score":            round(score, 3),
                "allocated_capital": alloc_cap,
                "quantity_hint":    qty_hint,
                "entry_price":      sig.entry_price,
                "stop_loss":        sig.stop_loss,
                "target_price":     sig.target_price,
                "risk_reward":      sig.risk_reward,
                "sector":           self._get_sector(sig.symbol),
            })

        logger.info(
            f"Portfolio allocator: {len(result)} trades selected from {len(signals)} signals "
            f"(available ₹{available:,.0f})"
        )
        return result

    def _composite_score(self, sig) -> float:
        """Score = 0.40×conf + 0.30×pattern + 0.20×ml_agreement + 0.10×vol_spike."""
        conf_score = sig.confidence / 100.0

        # Pattern score — use pattern_confidence if present
        pat_score  = min(1.0, getattr(sig, "pattern_confidence", sig.confidence / 100.0))

        # ML agreement — fraction of ML models that agree with direction
        is_bull = sig.direction == "BUY"
        ml_votes = [
            sig.lgbm_prob  > 0.5 if is_bull else sig.lgbm_prob  < 0.5,
            sig.xgb_prob   > 0.5 if is_bull else sig.xgb_prob   < 0.5,
            sig.lstm_prob  > 0.5 if is_bull else sig.lstm_prob   < 0.5,
        ]
        ml_agreement = sum(ml_votes) / len(ml_votes)

        # ARIMA directional agreement
        arima = getattr(sig, "arima_trend", "FLAT")
        if (is_bull and arima == "UP") or (not is_bull and arima == "DOWN"):
            ml_agreement = min(1.0, ml_agreement + 0.2)

        # Volume spike proxy — use delivery_pct if available, else 0.5 neutral
        vol_score = min(1.0, max(0.0, getattr(sig, "delivery_pct", 50.0) / 100.0))

        return (0.40 * conf_score +
                0.30 * pat_score  +
                0.20 * ml_agreement +
                0.10 * vol_score)

    def _sector_diversify(self, scored: list) -> list:
        """Keep only the highest-scoring signal per sector."""
        seen_sectors: Dict[str, float] = {}
        result = []
        for score, sig in scored:
            sector = self._get_sector(sig.symbol)
            if sector not in seen_sectors:
                seen_sectors[sector] = score
                result.append((score, sig))
            # else: already have a better signal from this sector
        return result

    def _get_sector(self, symbol: str) -> str:
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        return SECTOR_MAP.get(ticker, "OTHER")
