"""
Trade Quality Scorer — 6-point pre-trade validation framework.

Runs on every signal AFTER fusion has produced a candidate but BEFORE the
signal reaches the execution engine.  Each of the six checks returns a
0-1 score with a short human reason.  The overall score is graded A–F
and used by:

  • TradeEngine.process_signal  → hard AUTO-mode gate (score ≥ MIN_SCORE)
  • UI                           → badge on signal cards & chart modal
  • Report                       → post-hoc review ("why did we take this trade?")

The seven checks, ranked by importance:

  1. Breakout freshness    — trigger→entry drift  (stale breakouts kill R:R)
  2. Risk size              — ₹ SL distance as % of entry; hard floor 0.15%, min 0.30%
  3. R:R sanity             — ratio within [MIN_RR, MAX_RR]; catches tight-SL+high-RR artefacts
  4. SL on correct side     — stop beyond pattern-invalidation swing
  5. Regime fit             — pattern type vs current regime
  6. ML alignment           — magnitude-weighted; hard-fails on >70% contra-direction model
  7. DI alignment           — DI+/DI− confirms trade direction

Public API:
  score_signal(signal, pattern, regime, df=None) -> QualityReport
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any
import logging

import config

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────

@dataclass
class QualityCheck:
    """One line-item in the 6-point framework."""
    key:    str          # short id: 'freshness', 'risk_size', 'rr', 'sl_side', 'regime', 'ml'
    name:   str          # human label for UI
    passed: bool
    score:  float        # 0.0–1.0
    value:  str          # e.g. "6.65%", "₹2.42", "12.25×"
    reason: str          # why passed/failed


@dataclass
class QualityReport:
    """Full scorecard for one signal."""
    overall_score: float                          # 0.0–1.0
    grade:         str                            # 'A'..'F'
    passed:        bool                           # ≥ TRADE_QUALITY_MIN_SCORE
    checks:        List[QualityCheck] = field(default_factory=list)

    def fail_reasons(self) -> List[str]:
        return [f"{c.name}: {c.reason}" for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fail_reasons"] = self.fail_reasons()
        return d


# ── Pattern-category taxonomy (used by Regime fit check) ─────
_REVERSAL_PATTERNS = {
    "head_shoulders", "head_shoulders_bottom", "double_top", "double_bottom",
    "triple_top", "triple_bottom", "rounding_top", "rounding_bottom",
    "rising_wedge", "falling_wedge", "diamond_top", "diamond_bottom",
}
_CONTINUATION_PATTERNS = {
    "ascending_triangle", "descending_triangle", "symmetrical_triangle",
    "rectangle", "flag", "pennant", "bull_flag", "bear_flag",
    "cup_handle", "inverse_cup_handle", "channel_up", "channel_down",
}


def _classify(pattern_name: str) -> str:
    n = (pattern_name or "").lower().replace(" ", "_")
    if n in _REVERSAL_PATTERNS: return "REVERSAL"
    if n in _CONTINUATION_PATTERNS: return "CONTINUATION"
    return "UNKNOWN"


# ── Individual checks ────────────────────────────────────────

def _check_freshness(signal, pattern) -> QualityCheck:
    """#1 — Trigger→Entry drift.  Stale breakouts eat most of the R:R."""
    entry   = float(signal.entry_price or 0)
    trigger = float(getattr(signal, "pattern_trigger", 0) or
                    getattr(pattern, "entry_price", 0) or 0)
    if entry <= 0 or trigger <= 0:
        return QualityCheck("freshness", "Breakout freshness", True, 0.7,
                            "n/a", "no trigger reference")
    drift_pct = abs(entry - trigger) / trigger * 100.0
    cap = config.TRADE_QUALITY_MAX_DRIFT_PCT
    if drift_pct <= cap * 0.3:        # <0.3× cap → excellent
        return QualityCheck("freshness", "Breakout freshness", True, 1.0,
                            f"{drift_pct:.2f}%", "price right at the trigger")
    if drift_pct <= cap:              # ≤ cap → acceptable
        return QualityCheck("freshness", "Breakout freshness", True, 0.7,
                            f"{drift_pct:.2f}%", "fresh breakout")
    if drift_pct <= cap * 3:          # stale but not catastrophic
        return QualityCheck("freshness", "Breakout freshness", False, 0.3,
                            f"{drift_pct:.2f}%",
                            f"stale — price drifted {drift_pct:.2f}% past trigger")
    return QualityCheck("freshness", "Breakout freshness", False, 0.0,
                        f"{drift_pct:.2f}%",
                        f"exhausted — {drift_pct:.2f}% past trigger, move is over")


def _check_risk_size(signal) -> QualityCheck:
    """#2 — Risk per share as % of entry.  Guards against artificially
    tight stops that inflate R:R.

    Floors:
      < 0.15% → absolute zero — tick noise will stop this out instantly
      < MIN_RISK_PCT (0.30%) → too tight, likely SL calculation error
    """
    entry = float(signal.entry_price or 0)
    sl    = float(signal.stop_loss or 0)
    if entry <= 0 or sl <= 0:
        return QualityCheck("risk_size", "Risk size", False, 0.0,
                            "n/a", "no SL defined")
    risk_abs = abs(entry - sl)
    risk_pct = (risk_abs / entry) * 100.0
    min_pct  = config.TRADE_QUALITY_MIN_RISK_PCT
    val      = f"₹{risk_abs:.2f} ({risk_pct:.2f}%)"

    # Hard floor: below 0.15% is tick-noise territory — guaranteed stop-out
    if risk_pct < 0.15:
        return QualityCheck("risk_size", "Risk size", False, 0.0, val,
                            f"SL effectively zero — {risk_pct:.2f}% is inside the spread; "
                            f"will stop out on first tick")
    if risk_pct < min_pct:
        return QualityCheck("risk_size", "Risk size", False, 0.2, val,
                            f"SL too tight — {risk_pct:.2f}% < {min_pct:.2f}% min; "
                            f"will stop on normal intraday noise")
    if risk_pct < min_pct * 1.5:
        return QualityCheck("risk_size", "Risk size", True, 0.6, val,
                            "tight but acceptable")
    if risk_pct > 4.0:
        return QualityCheck("risk_size", "Risk size", True, 0.6, val,
                            "wide SL — position size down accordingly")
    return QualityCheck("risk_size", "Risk size", True, 1.0, val,
                        "healthy risk size")


def _check_rr(signal) -> QualityCheck:
    """#3 — R:R within sensible window.

    Two-part check:
    A) Raw ratio vs [MIN_RR, MAX_RR] bounds.
    B) Artefact guard: very high R:R (>5×) with an implausibly tight SL
       (< MIN_RISK_PCT) is almost always a stop-calculation artefact where
       the SL sits inside the noise band, not at a real invalidation level.
       Example: TECHM SELL  entry=1390.50  SL=1394.04 (0.25%)  R:R=5.90×
       → the SL is only ₹3.54 wide; any tick will stop this out.
    """
    rr     = float(getattr(signal, "risk_reward", 0) or 0)
    entry  = float(signal.entry_price or 0)
    sl     = float(signal.stop_loss   or 0)
    min_rr = config.TRADE_QUALITY_MIN_RR
    max_rr = config.TRADE_QUALITY_MAX_RR
    val    = f"{rr:.2f}×"

    if rr <= 0:
        return QualityCheck("rr", "R:R", False, 0.0, val, "R:R not computed")
    if rr < min_rr:
        return QualityCheck("rr", "R:R", False, 0.1, val,
                            f"below {min_rr}× minimum — not worth the risk")
    if rr > max_rr:
        return QualityCheck("rr", "R:R", False, 0.2, val,
                            f"above {max_rr}× — suspiciously high; likely tight-SL artefact")

    # ── Artefact guard: high R:R + tight SL ──────────────────────────────────
    if entry > 0 and sl > 0:
        risk_pct = abs(entry - sl) / entry * 100.0
        if rr > 5.0 and risk_pct < config.TRADE_QUALITY_MIN_RISK_PCT:
            return QualityCheck(
                "rr", "R:R", False, 0.20,
                f"{val}  SL={risk_pct:.2f}%",
                f"R:R {rr:.1f}× with SL only {risk_pct:.2f}% — "
                f"tight-SL artefact; stop is inside tick noise, not a real level"
            )

    # Sweet spot 2–5×
    if 2.0 <= rr <= 5.0:
        return QualityCheck("rr", "R:R", True, 1.0, val, "healthy risk/reward")
    # 5–6× acceptable but verify SL is real
    if 5.0 < rr <= 6.0:
        return QualityCheck("rr", "R:R", True, 0.75, val,
                            "high R:R — double-check SL is at a real level")
    return QualityCheck("rr", "R:R", True, 0.80, val, "acceptable R:R")


def _check_sl_side(signal, pattern) -> QualityCheck:
    """#4 — SL must sit BEYOND the pattern's invalidation swing.
    For BUY: SL should be at-or-below min(swing_lows within pattern).
    For SELL: SL should be at-or-above max(swing_highs within pattern).
    If the SL is INSIDE the pattern range, the stop is inside the move
    itself — any normal pullback stops us out."""
    swings = getattr(pattern, "swings", None) or []
    if not swings:
        return QualityCheck("sl_side", "SL placement", True, 0.7,
                            "n/a", "no swing data (neutral)")
    try:
        lows  = [float(s.price) for s in swings if str(getattr(s, "type", "")).upper() == "LOW"]
        highs = [float(s.price) for s in swings if str(getattr(s, "type", "")).upper() == "HIGH"]
    except Exception:
        return QualityCheck("sl_side", "SL placement", True, 0.7,
                            "n/a", "swing data malformed")

    entry = float(signal.entry_price or 0)
    sl    = float(signal.stop_loss or 0)
    direction = (signal.direction or "").upper()

    if direction == "BUY":
        if not lows:
            return QualityCheck("sl_side", "SL placement", True, 0.7,
                                "no swing lows", "insufficient swings")
        invalid = min(lows)
        if sl <= invalid:
            return QualityCheck("sl_side", "SL placement", True, 1.0,
                                f"SL ₹{sl:.2f} ≤ swing low ₹{invalid:.2f}",
                                "SL correctly below pattern invalidation")
        # SL is ABOVE the pattern's low — inside the move
        intrusion = (invalid - sl) / entry * 100.0 if entry else 0
        return QualityCheck("sl_side", "SL placement", False, 0.2,
                            f"SL ₹{sl:.2f} > swing low ₹{invalid:.2f}",
                            f"SL is INSIDE pattern ({abs(intrusion):.2f}% above invalidation) — "
                            f"will stop on normal pullback")
    elif direction == "SELL":
        if not highs:
            return QualityCheck("sl_side", "SL placement", True, 0.7,
                                "no swing highs", "insufficient swings")
        invalid = max(highs)
        if sl >= invalid:
            return QualityCheck("sl_side", "SL placement", True, 1.0,
                                f"SL ₹{sl:.2f} ≥ swing high ₹{invalid:.2f}",
                                "SL correctly above pattern invalidation")
        intrusion = (sl - invalid) / entry * 100.0 if entry else 0
        return QualityCheck("sl_side", "SL placement", False, 0.2,
                            f"SL ₹{sl:.2f} < swing high ₹{invalid:.2f}",
                            f"SL is INSIDE pattern ({abs(intrusion):.2f}% below invalidation) — "
                            f"will stop on normal pullback")
    return QualityCheck("sl_side", "SL placement", True, 0.5,
                        "no direction", "unknown direction")


def _check_regime_fit(signal, pattern_name, regime) -> QualityCheck:
    """#5 — Pattern type vs regime.
    • Reversal  patterns suit MEAN_REVERTING / opposing-trend.
    • Continuation patterns suit TRENDING / BREAKOUT same-direction.
    • Mismatches score low; UNKNOWN pattern → neutral pass."""
    cat = _classify(pattern_name or "")
    reg = (regime or "").upper()
    direction = (signal.direction or "").upper()

    if cat == "UNKNOWN" or not reg:
        return QualityCheck("regime", "Regime fit", True, 0.7,
                            f"{cat}/{reg}", "unclassified pattern or regime")

    # Direction-aware rules
    if cat == "REVERSAL":
        if reg in ("MEAN_REVERTING", "VOLATILE"):
            return QualityCheck("regime", "Regime fit", True, 1.0,
                                f"{cat}/{reg}", "reversal pattern in mean-reverting regime")
        if reg == "CONSOLIDATION":
            return QualityCheck("regime", "Regime fit", True, 0.8,
                                f"{cat}/{reg}", "reversal in consolidation (okay)")
        # Trending regime same direction as trade → bad for a reversal (swimming upstream)
        if (reg == "TRENDING_UP" and direction == "BUY") or \
           (reg == "TRENDING_DOWN" and direction == "SELL"):
            return QualityCheck("regime", "Regime fit", False, 0.3,
                                f"{cat}/{reg}",
                                "reversal pattern fighting the prevailing trend")
        return QualityCheck("regime", "Regime fit", True, 0.7,
                            f"{cat}/{reg}", "reversal counter-trend (acceptable)")

    if cat == "CONTINUATION":
        if (reg == "TRENDING_UP" and direction == "BUY") or \
           (reg == "TRENDING_DOWN" and direction == "SELL"):
            return QualityCheck("regime", "Regime fit", True, 1.0,
                                f"{cat}/{reg}", "continuation in same-direction trend")
        if reg in ("BREAKOUT", "MOMENTUM"):
            return QualityCheck("regime", "Regime fit", True, 0.9,
                                f"{cat}/{reg}", "continuation in breakout/momentum regime")
        if reg == "CONSOLIDATION":
            return QualityCheck("regime", "Regime fit", True, 0.7,
                                f"{cat}/{reg}", "consolidation — waiting for breakout")
        if reg == "MEAN_REVERTING":
            return QualityCheck("regime", "Regime fit", False, 0.3,
                                f"{cat}/{reg}",
                                "continuation pattern in mean-reverting regime is statistically weak")
        return QualityCheck("regime", "Regime fit", True, 0.6,
                            f"{cat}/{reg}", "mixed signals")

    return QualityCheck("regime", "Regime fit", True, 0.6,
                        f"{cat}/{reg}", "unmapped combination")


def _check_di_alignment(signal, df) -> QualityCheck:
    """#7 — DI+ / DI− directional alignment.

    ADX tells you HOW STRONG a trend is; DI+/DI− tell you WHICH DIRECTION.
    A BUY signal with DI− > DI+ means the trend is actually bearish —
    that's swimming against the current and deserves a penalty.

    Uses pre-computed columns if available, otherwise computes inline.
    """
    direction = (signal.direction or "").upper()
    if not direction or df is None or len(df) < 15:
        return QualityCheck("di_align", "DI Alignment", True, 0.6,
                            "n/a", "insufficient data")

    # Prefer pre-computed columns from indicator_engine
    try:
        if "plus_di" in df.columns and "minus_di" in df.columns:
            pdi = float(df["plus_di"].iloc[-1])
            ndi = float(df["minus_di"].iloc[-1])
        else:
            # Inline DM computation (Wilder's, 14-period)
            h = df["high"].astype(float)
            l = df["low"].astype(float)
            c = df["close"].astype(float)
            up   = h.diff()
            dn   = -l.diff()
            pdm  = up.where((up > dn) & (up > 0), 0.0)
            ndm  = dn.where((dn > up) & (dn > 0), 0.0)
            tr   = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
            atr  = tr.rolling(14).sum().replace(0, float("nan"))
            pdi  = float((100 * pdm.rolling(14).sum() / atr).iloc[-1])
            ndi  = float((100 * ndm.rolling(14).sum() / atr).iloc[-1])

        if pdi != pdi or ndi != ndi:   # NaN guard
            return QualityCheck("di_align", "DI Alignment", True, 0.6,
                                "n/a", "DI not yet computed (warmup)")

        gap   = abs(pdi - ndi)
        bull  = pdi > ndi
        val   = f"DI+={pdi:.1f}  DI−={ndi:.1f}"

        if gap < 5:                    # indeterminate — too close to call
            return QualityCheck("di_align", "DI Alignment", True, 0.6,
                                val, "DI too close to call — neutral")

        if direction == "BUY":
            if bull:
                score = min(1.0, 0.70 + (gap / 60.0) * 0.30)
                return QualityCheck("di_align", "DI Alignment", True, round(score, 2),
                                    val, f"DI+ > DI− by {gap:.1f} pts — bullish bias confirmed")
            else:
                score = max(0.1, 0.40 - (gap / 60.0) * 0.30)
                return QualityCheck("di_align", "DI Alignment", False, round(score, 2),
                                    val, f"DI− > DI+ by {gap:.1f} pts — bearish bias vs BUY signal")
        else:   # SELL
            if not bull:
                score = min(1.0, 0.70 + (gap / 60.0) * 0.30)
                return QualityCheck("di_align", "DI Alignment", True, round(score, 2),
                                    val, f"DI− > DI+ by {gap:.1f} pts — bearish bias confirmed")
            else:
                score = max(0.1, 0.40 - (gap / 60.0) * 0.30)
                return QualityCheck("di_align", "DI Alignment", False, round(score, 2),
                                    val, f"DI+ > DI− by {gap:.1f} pts — bullish bias vs SELL signal")

    except Exception as e:
        return QualityCheck("di_align", "DI Alignment", True, 0.5,
                            "error", f"DI computation failed: {e}")


def _check_ml_alignment(signal) -> QualityCheck:
    """#6 — ML model agreement, magnitude-weighted with contradiction tiers.

    Old approach (binary count) treated XGB=71% bullish on a SELL signal the
    same as XGB=51% — both just "disagree".  That misses the key information:
    a model with 71% conviction AGAINST the direction is a strong red flag.

    New approach — directional score (ds) per model:
      BUY:  ds = (p − 0.5) × 2   →  +1.0 if p=100%,  0 if p=50%, −1.0 if p=0%
      SELL: ds = (0.5 − p) × 2   →  +1.0 if p=0%,    0 if p=50%, −1.0 if p=100%

    Contradiction tiers (worst ds across all models):
      ds < −0.40  (>70% against direction) → HARD FAIL, score ≤ 0.15
      ds < −0.20  (>60% against direction) → MODERATE FAIL, score ≤ 0.30
      ds ≥ −0.20                           → normal scoring from net_ds

    Example — TECHM SELL: XGB=71% bullish → ds = (0.5−0.71)×2 = −0.42
      → Tier 1 hard fail: score ≈ 0.12
    """
    direction = (signal.direction or "").upper()
    lgb  = float(getattr(signal, "lgbm_prob", 0.5) or 0.5)
    xgb  = float(getattr(signal, "xgb_prob",  0.5) or 0.5)
    lstm = float(getattr(signal, "lstm_prob", 0.5) or 0.5)
    tft  = float(getattr(signal, "tft_prob",  0.5) or 0.5)
    arima_trend = (getattr(signal, "arima_trend", "FLAT") or "FLAT").upper()

    active, details = [], []

    for name, p in [("LGB", lgb), ("XGB", xgb), ("LSTM", lstm), ("TFT", tft)]:
        if abs(p - 0.5) < 0.03:          # ±3% from neutral → noise, skip
            details.append(f"{name}=n/a")
            continue
        ds = (p - 0.5) * 2 if direction == "BUY" else (0.5 - p) * 2
        active.append((name, p, ds))
        details.append(f"{name}{'✓' if ds > 0 else '✗'}({p:.0%})")

    # ARIMA vote — binary, fixed ds ±0.60
    if arima_trend in ("UP", "DOWN"):
        arima_ok = (direction == "BUY"  and arima_trend == "UP") or \
                   (direction == "SELL" and arima_trend == "DOWN")
        arima_ds = 0.60 if arima_ok else -0.60
        active.append(("ARIMA", None, arima_ds))
        details.append(f"ARIMA({'✓' if arima_ok else '✗'})")

    if not active:
        return QualityCheck("ml", "ML alignment", True, 0.55, "n/a",
                            "no trained models have strong opinion (neutral)")

    net_ds   = sum(ds for _, _, ds in active) / len(active)
    agreeing = sum(1 for _, _, ds in active if ds > 0)
    total    = len(active)
    worst    = min(active, key=lambda x: x[2])       # most negative = worst contra
    wname, wprob, wds = worst
    wval  = f"{wprob:.0%}" if wprob is not None else "–"

    val        = f"{agreeing}/{total}  net={net_ds:+.2f}"
    detail_str = ", ".join(details)

    # ── Tier 1: HIGH-CONVICTION CONTRADICTION (>70% against direction) ────────
    # ds < −0.40  ⟺  prob > 0.70 against direction
    if wds < -0.40:
        score = round(max(0.0, 0.10 + 0.15 * max(net_ds, 0)), 2)
        return QualityCheck(
            "ml", "ML alignment", False, score, val,
            f"HIGH-CONVICTION CONTRADICTION — {wname} {wval} strongly opposes "
            f"{direction} (ds={wds:+.2f}); models: {detail_str}"
        )

    # ── Tier 2: MODERATE CONTRADICTION (>60% against direction) ──────────────
    # ds < −0.20  ⟺  prob > 0.60 against direction
    if wds < -0.20:
        score = round(max(0.10, 0.28 + 0.22 * max(net_ds, 0)), 2)
        return QualityCheck(
            "ml", "ML alignment", False, score, val,
            f"CONTRADICTION — {wname} {wval} opposes {direction} "
            f"(ds={wds:+.2f}); models: {detail_str}"
        )

    # ── Tier 3: Normal magnitude-weighted scoring ─────────────────────────────
    if net_ds >= 0.30 and agreeing >= 2:
        score = round(min(1.0, 0.65 + 0.35 * net_ds), 2)
        return QualityCheck("ml", "ML alignment", True, score, val,
                            f"Strong ML consensus — {detail_str}")

    if net_ds >= 0.10:
        score = round(min(0.80, 0.50 + 0.30 * net_ds), 2)
        return QualityCheck("ml", "ML alignment", True, score, val,
                            f"ML agrees with direction — {detail_str}")

    if net_ds >= 0.0:
        return QualityCheck("ml", "ML alignment", True, 0.45, val,
                            f"Marginal ML agreement — {detail_str}")

    # Net negative (more models against than for)
    score = round(max(0.10, 0.35 + 0.25 * net_ds), 2)
    return QualityCheck("ml", "ML alignment", False, score, val,
                        f"ML net disagrees with {direction} — {detail_str}")


# ── Aggregator ───────────────────────────────────────────────

def _grade_from_score(score: float) -> str:
    if score >= 0.85: return "A"
    if score >= 0.70: return "B"
    if score >= 0.55: return "C"
    if score >= 0.40: return "D"
    return "F"


def score_signal(signal, pattern=None, regime: str = "", df=None) -> QualityReport:
    """Run the full 7-point framework and return a QualityReport.

    Weights (sum = 1.0):
        freshness  0.18   ← was 0.20
        risk_size  0.18   ← was 0.20
        rr         0.13   ← was 0.15
        sl_side    0.18   ← was 0.20
        regime     0.10   (unchanged)
        ml         0.13   ← was 0.15
        di_align   0.10   ← NEW: DI+/DI− directional alignment
    """
    checks = [
        _check_freshness(signal, pattern),
        _check_risk_size(signal),
        _check_rr(signal),
        _check_sl_side(signal, pattern),
        _check_regime_fit(signal, getattr(signal, "pattern_name", ""), regime),
        _check_ml_alignment(signal),
        _check_di_alignment(signal, df),   # NEW — Check #7
    ]
    weights = {
        "freshness": 0.18,
        "risk_size": 0.18,
        "rr":        0.13,
        "sl_side":   0.18,
        "regime":    0.10,
        "ml":        0.13,
        "di_align":  0.10,   # NEW
    }
    total = sum(c.score * weights.get(c.key, 0) for c in checks)
    grade = _grade_from_score(total)
    passed = total >= config.TRADE_QUALITY_MIN_SCORE
    return QualityReport(
        overall_score=round(total, 3),
        grade=grade,
        passed=passed,
        checks=checks,
    )
