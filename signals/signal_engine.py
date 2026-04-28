"""
Signal Engine (3-Layer Architecture)
GATE → QUALIFIER → TRIGGER
Adapted from COMMODITY APP for equity markets with 7-model fusion.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

import config
from patterns.pattern_detector import PatternDetector
from features.feature_pipeline import FeaturePipeline
from models.model_manager import ModelManager
from signals.signal_fusion import SignalFusion, FusionResult
from signals.regime_detector import detect_regime
from signals.entry_gate import EntryGate
from signals.trade_quality import score_signal as _score_quality

logger = logging.getLogger(__name__)


# ── Per-symbol journal stats cache ────────────────────────────────────────────
# Keys: clean ticker e.g. "RELIANCE" (no "NSE:" prefix, no "-EQ" suffix).
# Values: dict with win_rate, profit_factor, rank_score, trades, suggested_sl_pct
# Populated by app.py every JOURNAL_CACHE_INTERVAL seconds (default 10 min).
_journal_symbol_cache: dict = {}


def update_journal_cache(stats_dict: dict) -> None:
    """Called by app.py after each JournalEngine compute run.
    stats_dict is keyed by clean ticker, each value is a symbol_stats dict.
    """
    global _journal_symbol_cache
    _journal_symbol_cache = stats_dict or {}
    logger.debug(f"[JOURNAL-CACHE] Updated {len(_journal_symbol_cache)} symbol stats")


@dataclass
class TradeSignal:
    """Complete trade signal with all component data."""
    symbol: str = ""
    timeframe: str = "15"
    direction: str = ""          # 'BUY' or 'SELL'
    confidence: float = 0.0      # 0-100
    strength: str = ""           # 'STRONG', 'MODERATE', 'WEAK'
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    risk_reward: float = 0.0
    pattern_name: str = ""
    pattern_confidence: float = 0.0
    lgbm_prob: float = 0.5
    xgb_prob: float = 0.5
    lstm_prob: float = 0.5
    tft_prob: float = 0.5
    arima_trend: str = "FLAT"
    prophet_seasonal: str = "FLAT"
    regime: str = ""
    delivery_pct: float = 0.0
    pcr: float = 1.0
    fii_net: float = 0.0
    timestamp: float = 0.0
    # ── Chart-overlay metadata (used by Lightweight Charts modal) ──
    # pattern_trigger = the projected breakout line from the pattern
    # (e.g. neckline / resistance). Entry_price is LTP fill, this is the line.
    pattern_trigger: float = 0.0
    neckline: float = 0.0
    # Swing points the pattern was built from, newest last.
    # Each item: {"timestamp": int, "price": float, "type": "HIGH"|"LOW"}
    swings: list = field(default_factory=list)
    # ── Trade-quality scorecard (populated by SignalEngine after scoring) ──
    quality_score:  float = 0.0
    quality_grade:  str   = ""
    quality_passed: bool  = True
    quality_report: dict  = field(default_factory=dict)


def _extract_swings_for_overlay(pattern, df):
    """Return swing points for chart overlay as list of dicts.

    Extracts `pattern.swings` (list of SwingPoint objects) and translates
    each swing's DataFrame index into a unix timestamp so Lightweight Charts
    can place a marker.  Returns [] if pattern has no swing metadata.
    """
    out = []
    try:
        swings = getattr(pattern, "swings", None) or []
        if not len(swings):
            return out
        # df is expected to have 'timestamp' column (int unix seconds)
        ts_col = None
        if "timestamp" in df.columns:
            ts_col = df["timestamp"].values
        for s in swings[-10:]:  # most recent ten swings
            try:
                idx = int(getattr(s, "index", -1))
                price = float(getattr(s, "price", 0) or 0)
                stype = str(getattr(s, "type", "HIGH")).upper()
                if ts_col is not None and 0 <= idx < len(ts_col):
                    t = int(ts_col[idx])
                else:
                    t = 0
                if price > 0 and t > 0:
                    out.append({"timestamp": t, "price": price, "type": stype})
            except Exception:
                continue
    except Exception:
        return out
    return out


class SignalEngine:
    """3-Layer signal processing engine.
    Layer 1 (GATE): Market hours, holidays, circuit limits
    Layer 2 (QUALIFIER): ADX, volume, price stability
    Layer 3 (TRIGGER): Full ML + pattern + fusion pipeline
    """

    def __init__(self, data_engine=None):
        self.pattern_detector = PatternDetector()
        self.feature_pipeline = FeaturePipeline()
        self.model_manager = ModelManager()
        self.signal_fusion = SignalFusion()
        self.entry_gate = EntryGate()
        self.data_engine = data_engine
        self._last_signal_time = {}  # {symbol: timestamp} for cooldown enforcement
        # ── Pattern-instance dedup (per-day) ──────────────────────
        # Tracks (symbol, pattern_name, round(trigger,1), direction) tuples
        # already fired today.  Reset automatically when the IST date rolls.
        self._daily_signature_set: set = set()
        self._daily_signature_date: str = ""

    def evaluate(self, symbol: str, df: pd.DataFrame,
                 enricher_data: dict = None,
                 strategy: str = "FUSION_FULL") -> Optional[TradeSignal]:
        """Full signal evaluation pipeline for a symbol.

        Args:
            symbol: stock symbol (e.g., 'NSE:RELIANCE-EQ')
            df: OHLCV DataFrame with at least 200 bars
            enricher_data: dict from MarketDataEnricher
            strategy: strategy preset name

        Returns:
            TradeSignal if signal generated, None otherwise
        """
        if len(df) < 50:
            return None

        # ===== LAYER 1: GATE =====
        gate_result = self._gate_check(enricher_data)
        if not gate_result["passed"]:
            logger.info(f"[GATE] {symbol} blocked — {gate_result['reason']}")
            return None

        # ===== LAYER 2: QUALIFIER =====
        qual_result = self._qualifier_check(df, enricher_data)
        if not qual_result["passed"]:
            logger.info(f"[QUALIFIER] {symbol} blocked — {qual_result['reason']}")
            return None

        # ===== 5-MIN ON-DEMAND FETCH (Scenario C) =====
        # Only symbols that cleared GATE + QUALIFIER reach this point — keeps
        # the extra API calls to ~10-25 per cycle instead of the full 96.
        df_5min = None
        if getattr(config, "INTRADAY_PATTERNS_ENABLED", True) and self.data_engine is not None:
            try:
                df_5min = self.data_engine.fetch_5min_for_signal(symbol)
            except Exception as _e5:
                logger.debug(f"[5min] fetch skipped for {symbol}: {_e5}")

        # ===== LAYER 3: TRIGGER =====
        return self._trigger(symbol, df, enricher_data, strategy, df_5min=df_5min)

    def _gate_check(self, enricher_data) -> dict:
        """Layer 1: Basic market access checks."""
        # Market hours
        hours_check = self.entry_gate._check_market_hours()
        if not hours_check["passed"]:
            return hours_check

        # Holiday
        holiday_check = self.entry_gate._check_holiday()
        if not holiday_check["passed"]:
            return holiday_check

        # Circuit limit
        circuit_check = self.entry_gate._check_circuit_limit(enricher_data)
        if not circuit_check["passed"]:
            return circuit_check

        return {"passed": True, "reason": ""}

    def _qualifier_check(self, df: pd.DataFrame, enricher_data: dict = None) -> dict:
        """Layer 2: Market quality checks.
        Only HARD blocks here — no spikes/ADX thresholds (those are soft penalties in fusion).
        """
        close = df["close"].values
        volume = df["volume"].values

        # ADX soft floor — only block truly dead markets (ADX < 10 = no movement at all)
        # ADX 10-20 gets a penalty in signal_fusion._apply_penalties, not a hard block.
        if "adx" in df.columns:
            adx_val = df["adx"].iloc[-1]
            if not pd.isna(adx_val):
                adx = float(adx_val)
                if adx < 10:
                    return {"passed": False, "reason": f"ADX extremely flat: {adx:.1f} (dead market)"}

        # Volume floor — only block if truly zero (no liquidity at all)
        # Volume spike ratio is applied as a soft penalty in signal_fusion, not here.
        if len(volume) >= 5:
            recent_avg = float(np.mean(volume[-5:]))
            if recent_avg <= 0:
                return {"passed": False, "reason": "Zero volume — completely illiquid"}

        # Price stability: no >8% gap in last 3 bars (extreme circuit-like moves)
        if len(close) >= 4:
            for i in range(-3, 0):
                gap = abs(close[i] - close[i - 1]) / close[i - 1] if close[i - 1] > 0 else 0
                if gap > 0.08:
                    return {"passed": False, "reason": f"Extreme gap detected: {gap:.1%}"}

        # Advance-decline ratio (market breadth) — only block on extreme breadth collapse
        if enricher_data:
            ad_ratio = enricher_data.get("ad_ratio", 1.0)
            if ad_ratio < 0.15:
                return {"passed": False, "reason": f"Market breadth collapsed: A/D = {ad_ratio:.2f}"}

        return {"passed": True, "reason": ""}

    def _trigger(self, symbol: str, df: pd.DataFrame,
                 enricher_data: dict, strategy: str,
                 df_5min: pd.DataFrame = None) -> Optional[TradeSignal]:
        """Layer 3: Full ML + pattern fusion pipeline."""
        import time

        # 1. Detect patterns (15-min swing patterns)
        patterns = self.pattern_detector.detect_for_symbol(
            symbol, df, enricher_data, self.data_engine
        )

        # 1b. Detect intraday patterns (5-min, on-demand) and PREPEND them
        #     so the fusion step sees them first (higher priority).
        if df_5min is not None:
            regime_hint = detect_regime(df).get("regime", "")
            intraday_pats = self.pattern_detector.detect_intraday(
                df_5min, regime_15min=regime_hint
            )
            if intraday_pats:
                patterns = intraday_pats + list(patterns)

        # 2. Build features
        feature_df = self.feature_pipeline.build_features(
            df, patterns, enricher_data, symbol
        )

        # 3. Detect regime
        regime_result = detect_regime(df)
        regime = regime_result["regime"]

        # 4. Get ML predictions (if any pattern or forced check)
        has_pattern = any(p.confidence >= config.PATTERN_MIN_CONFIDENCE for p in patterns)
        ml_predictions = None

        if has_pattern or self.model_manager.has_models(symbol):
            tabular, sequence = self.feature_pipeline.get_latest_features(feature_df)
            close_prices = df["close"].values

            ml_predictions = self.model_manager.predict(
                symbol=symbol,
                tabular_features=tabular,
                sequence_features=sequence,
                close_prices=close_prices,
                df=df,
                pattern_confidence=patterns[0].confidence if patterns else 0.0,
                regime=regime,
                enricher_data=enricher_data,
            )

        # 5. Fuse signals (strategy passed so penalty layer can be strategy-aware)
        fusion_result = self.signal_fusion.compute(
            patterns=patterns,
            ml_predictions=ml_predictions,
            enricher_data=enricher_data,
            indicators=feature_df if len(feature_df) > 0 else None,
            regime=regime,
            strategy=strategy,
        )

        # Always log scan result so we can see what's happening
        pat_names = [p.pattern_name for p in patterns] if patterns else []
        lgbm_p = round(ml_predictions.get("lgbm_prob", 0.5), 3) if ml_predictions else "no-model"
        xgb_p  = round(ml_predictions.get("xgb_prob",  0.5), 3) if ml_predictions else "no-model"
        logger.info(
            f"[SCAN] {symbol} | regime={regime} | patterns={pat_names} | "
            f"lgbm={lgbm_p} xgb={xgb_p} | "
            f"fusion={fusion_result.confidence:.1f} (threshold={config.SIGNAL_WEAK_THRESHOLD})"
        )

        # ── Journal-based confidence adjustment ──────────────────────────────
        # Use historical per-symbol performance to nudge the fusion confidence.
        # Requires ≥5 closed trades on the symbol to avoid overfitting to noise.
        #   rank_score < 40  →  reduce confidence (up to −15 pts)  poor history
        #   rank_score > 70  →  boost  confidence (up to  +5 pts)  strong history
        #   40 ≤ rank ≤ 70   →  no adjustment (neutral zone)
        _clean_sym = symbol.replace("NSE:", "").replace("-EQ", "")
        _jsym = _journal_symbol_cache.get(_clean_sym)
        if _jsym and _jsym.get("trades", 0) >= 5:
            _rank = float(_jsym.get("rank_score", 50))
            if _rank < 40:
                _adj = round((_rank - 40) / 40.0 * 15.0, 1)   # ≤ 0 (penalty)
            elif _rank > 70:
                _adj = round((_rank - 70) / 30.0 * 5.0,  1)   # ≥ 0 (bonus)
            else:
                _adj = 0.0
            if _adj != 0.0:
                _orig = fusion_result.confidence
                fusion_result.confidence = round(
                    max(0.0, min(100.0, fusion_result.confidence + _adj)), 1
                )
                _sign = "+" if _adj > 0 else ""
                logger.info(
                    f"[JOURNAL-ADJ] {_clean_sym} rank={_rank:.0f}/100 "
                    f"→ conf {_orig:.1f}%→{fusion_result.confidence:.1f}% ({_sign}{_adj})"
                )

        # Check minimum threshold
        if fusion_result.confidence < config.SIGNAL_WEAK_THRESHOLD:
            return None

        # Signal cooldown — suppress repeated signals for same symbol
        import time as _time
        now = _time.time()
        last = self._last_signal_time.get(symbol, 0)
        if now - last < config.SIGNAL_ENTRY_COOLDOWN:
            remaining = int(config.SIGNAL_ENTRY_COOLDOWN - (now - last))
            logger.debug(f"Signal cooldown active for {symbol}: {remaining}s remaining")
            return None

        # 6. Determine entry/SL/target
        entry, sl, target = self._compute_levels(
            fusion_result.direction, patterns, df, feature_df
        )

        if entry <= 0:
            return None

        # 7. Risk-reward check
        if fusion_result.direction == "BUY":
            risk = entry - sl
            reward = target - entry
        else:
            risk = sl - entry
            reward = entry - target

        rr = reward / risk if risk > 0 else 0
        if rr < config.MIN_RISK_REWARD:
            logger.info(
                f"[R:R] {symbol} blocked — R:R {rr:.2f} < min {config.MIN_RISK_REWARD} "
                f"(entry={entry:.2f} sl={sl:.2f} target={target:.2f})"
            )
            return None

        # 8. Entry gate final validation
        gate_check = self.entry_gate.validate(
            symbol=symbol,
            direction=fusion_result.direction,
            confidence=fusion_result.confidence,
            entry_price=entry,
            stop_loss=sl,
            target_price=target,
            enricher_data=enricher_data,
            volume=float(df["volume"].iloc[-1]) if len(df) > 0 else 0,
        )

        if not gate_check["passed"]:
            return None

        # ── Pattern-instance dedup (per-day) ────────────────────
        # The same breakdown / neckline must fire only ONCE per day.
        # Signature: (symbol, pattern_name, round(trigger,1), direction).
        # Date is IST so the set rolls at midnight India time.
        from datetime import datetime, timezone, timedelta
        _ist_today = datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()
        if _ist_today != self._daily_signature_date:
            self._daily_signature_set.clear()
            self._daily_signature_date = _ist_today
        _trigger_ref = float(patterns[0].entry_price) if patterns else float(entry)
        _sig_tuple = (
            symbol,
            patterns[0].pattern_name if patterns else "",
            round(_trigger_ref, 1),
            fusion_result.direction,
        )
        if _sig_tuple in self._daily_signature_set:
            logger.info(f"[DEDUP] {symbol} suppressed — already fired {_sig_tuple} today")
            return None

        # ── Swing-trade SHORT guard ───────────────────────────────────────────
        # 15-minute signals are treated as swing trades that may carry overnight.
        # SEBI/broker rules forbid holding short (naked SELL) equity positions
        # overnight in the cash market.  Drop all SELL signals on the 15m TF.
        _signal_tf = patterns[0].timeframe if patterns else config.CANDLE_RESOLUTION
        if str(_signal_tf) == "15" and fusion_result.direction == "SELL":
            logger.info(
                f"[SWING-SHORT BLOCKED] {symbol} — SELL suppressed on 15m timeframe "
                f"(overnight short not permitted in NSE cash market)"
            )
            return None

        # Build signal
        signal = TradeSignal(
            symbol=symbol,
            timeframe=_signal_tf,
            direction=fusion_result.direction,
            confidence=fusion_result.confidence,
            strength=fusion_result.strength,
            entry_price=entry,
            stop_loss=sl,
            target_price=target,
            risk_reward=round(rr, 2),
            pattern_name=patterns[0].pattern_name if patterns else "",
            pattern_confidence=patterns[0].confidence if patterns else 0.0,
            lgbm_prob=ml_predictions.get("lgbm_prob", 0.5) if ml_predictions else 0.5,
            xgb_prob=ml_predictions.get("xgb_prob", 0.5) if ml_predictions else 0.5,
            lstm_prob=ml_predictions.get("lstm_prob", 0.5) if ml_predictions else 0.5,
            tft_prob=ml_predictions.get("tft_prob", 0.5) if ml_predictions else 0.5,
            arima_trend=ml_predictions.get("arima_trend", "FLAT") if ml_predictions else "FLAT",
            prophet_seasonal=ml_predictions.get("prophet_seasonal", "FLAT") if ml_predictions else "FLAT",
            regime=regime,
            delivery_pct=enricher_data.get("delivery_pct", 0) if enricher_data else 0,
            pcr=enricher_data.get("pcr", 1.0) if enricher_data else 1.0,
            fii_net=enricher_data.get("fii_net", 0) if enricher_data else 0,
            timestamp=time.time(),
            # Chart-overlay metadata
            pattern_trigger=float(patterns[0].entry_price) if patterns else 0.0,
            neckline=float(getattr(patterns[0], "neckline", 0) or 0) if patterns else 0.0,
            swings=_extract_swings_for_overlay(patterns[0], df) if patterns else [],
        )

        # Stamp cooldown timestamp
        self._last_signal_time[symbol] = signal.timestamp

        # Mark this pattern-instance as fired for the day
        self._daily_signature_set.add(_sig_tuple)

        # ── Run 6-point trade-quality scoring & attach to signal ──
        # The same report drives:
        #   • TradeEngine AUTO-mode hard gate
        #   • UI grade badge on top-signal cards / chart modal
        #   • Post-hoc reasoning ("why did we take this trade?")
        try:
            qreport = _score_quality(
                signal,
                pattern=(patterns[0] if patterns else None),
                regime=regime,
                df=df,
            )
            signal.quality_score  = qreport.overall_score
            signal.quality_grade  = qreport.grade
            signal.quality_passed = qreport.passed
            signal.quality_report = qreport.to_dict()
        except Exception as _qe:
            logger.warning(f"Quality scoring failed for {symbol}: {_qe}")

        logger.info(
            f"SIGNAL: {signal.direction} {symbol} @ {entry:.2f} "
            f"SL={sl:.2f} TGT={target:.2f} R:R={rr:.2f} "
            f"Conf={signal.confidence:.1f} ({signal.strength}) "
            f"Pattern={signal.pattern_name} Regime={regime} "
            f"Quality={signal.quality_grade}({signal.quality_score:.2f})"
        )

        return signal

    def _compute_levels(self, direction, patterns, df, feature_df):
        """Compute entry, SL, and target from patterns or ATR.

        CRITICAL FIX (phantom entry bug):
        ---------------------------------
        Pattern's ``entry_price`` is only the *projected breakout / breakdown
        LINE* (e.g. the neckline of a Head & Shoulders, the resistance line
        of an Ascending Triangle).  It is **not** the price at which a trade
        should be booked.  Previous behaviour stamped this projected line as
        the fill, causing "entry = ₹5528.5 when market is ₹5600+" phantom
        fills in paper mode.

        New behaviour:
          1.  Treat ``pattern.entry_price`` as a **trigger threshold**.
              - For BUY  patterns we only proceed if LTP has already closed
                *at or above* the trigger (i.e. the breakout happened).
              - For SELL patterns we only proceed if LTP has already closed
                *at or below* the trigger.
              Otherwise the pattern is still pending confirmation — no
              signal (returns ``(0, 0, 0)`` which the caller treats as
              invalid → signal is suppressed).
          2.  Fill price = **current LTP**, not the pattern line.
          3.  Sanity cap: if LTP has rocketed more than
              ``MAX_ENTRY_DRIFT_PCT`` (default 0.8%) past the trigger, the
              breakout is already exhausted → suppress.
          4.  SL is clamped relative to LTP so risk is measured from the
              real fill, not from the projected line.
          5.  Target is re-projected from LTP using the pattern's height
              (measure rule) so R:R remains valid.  If the resulting R:R
              falls below ``MIN_RISK_REWARD`` we extend the target.
        """
        close = df["close"].values
        current_price = float(close[-1])
        max_drift = getattr(config, "MAX_ENTRY_DRIFT_PCT", 0.8) / 100.0  # 0.8%

        if patterns and patterns[0].entry_price > 0:
            best      = patterns[0]
            trigger   = float(best.entry_price)       # projected breakout line
            pat_sl    = float(best.stop_loss)
            pat_tgt   = float(best.target_price)

            # ── Step 1: has price actually broken through the trigger? ──
            if direction == "BUY"  and current_price < trigger:
                # still below breakout line — pattern not confirmed yet
                logger.debug(
                    f"[LEVELS] BUY suppressed: LTP {current_price:.2f} "
                    f"< trigger {trigger:.2f} (breakout not confirmed)"
                )
                return 0.0, 0.0, 0.0
            if direction == "SELL" and current_price > trigger:
                logger.debug(
                    f"[LEVELS] SELL suppressed: LTP {current_price:.2f} "
                    f"> trigger {trigger:.2f} (breakdown not confirmed)"
                )
                return 0.0, 0.0, 0.0

            # ── Step 2: sanity cap — don't chase a stale breakout ──
            drift = abs(current_price - trigger) / max(trigger, 1e-9)
            if drift > max_drift:
                logger.debug(
                    f"[LEVELS] {direction} suppressed: LTP drifted {drift*100:.2f}% "
                    f"from trigger {trigger:.2f} (>{max_drift*100:.2f}% cap)"
                )
                return 0.0, 0.0, 0.0

            # ── Step 3: fill at LTP, not at the pattern line ──
            entry = current_price

            # ── Step 4: clamp SL using pattern invalidation level, but
            #           enforce a minimum risk measured from LTP ──
            atr = 0.0
            if feature_df is not None and "atr" in feature_df.columns and len(feature_df) > 0:
                atr = float(feature_df["atr"].iloc[-1])
            if atr <= 0:
                atr = entry * 0.015  # fallback

            if direction == "BUY":
                # Stop = pattern invalidation OR 1.5×ATR below LTP (whichever is closer but still below)
                sl = max(pat_sl, entry - 1.5 * atr)
                if sl >= entry:                       # pathological — fall back
                    sl = entry - 1.5 * atr
                # Target = project pattern height from LTP
                pattern_height = abs(pat_tgt - trigger)
                target = entry + max(pattern_height, 2.5 * atr)
            else:  # SELL
                sl = min(pat_sl, entry + 1.5 * atr)
                if sl <= entry:
                    sl = entry + 1.5 * atr
                pattern_height = abs(trigger - pat_tgt)
                target = entry - max(pattern_height, 2.5 * atr)

            # ── Step 5: enforce MIN_RISK_REWARD ──
            risk = (entry - sl) if direction == "BUY" else (sl - entry)
            if risk > 0:
                rr = ((target - entry) if direction == "BUY" else (entry - target)) / risk
                if rr < config.MIN_RISK_REWARD:
                    min_reward = risk * config.MIN_RISK_REWARD
                    target = entry + min_reward if direction == "BUY" else entry - min_reward
            else:
                entry, sl, target = self._atr_levels(direction, current_price, feature_df)
        else:
            entry, sl, target = self._atr_levels(direction, current_price, feature_df)

        return entry, sl, target

    def _atr_levels(self, direction, price, feature_df):
        """ATR-based entry/SL/target."""
        atr = 0
        if feature_df is not None and "atr" in feature_df.columns and len(feature_df) > 0:
            atr = float(feature_df["atr"].iloc[-1])

        if atr <= 0:
            atr = price * 0.015  # Fallback: 1.5% of price

        if direction == "BUY":
            entry = price
            sl = price - 1.5 * atr
            target = price + 2.5 * atr
        else:
            entry = price
            sl = price + 1.5 * atr
            target = price - 2.5 * atr

        return entry, sl, target
