"""
Journal Engine — TradingView-style strategy performance analytics.

Produces the complete set of metrics shown in TradingView's Strategy Report:
  Equity curve, Run-up/Drawdown, Profit Factor, Sharpe, Sortino, CAGR,
  MAE/MFE distributions, per-symbol breakdown, and optimal SL suggestion.

Data flow:
  TradeEngine.closed_positions  ──→  JournalEngine.compute()
                                         ↓
                              /api/journal_stats   (JSON → UI Journal tab)
                              scanner_ranking      (per-symbol quality score)
                              signal_engine        (confidence multiplier)
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

IST = timezone(timedelta(hours=5, minutes=30))
IST_OFF = 19800   # seconds to add for Lightweight Charts IST display


# ── Maths helpers ─────────────────────────────────────────────────────────────

def _sharpe(returns: List[float], rf_daily: float = 0.0) -> float:
    """Annualised Sharpe ratio from a list of daily % returns."""
    n = len(returns)
    if n < 3:
        return 0.0
    mean = sum(returns) / n
    var  = sum((r - mean) ** 2 for r in returns) / max(n - 1, 1)
    std  = math.sqrt(var)
    return round((mean - rf_daily) / std * math.sqrt(252), 2) if std else 0.0


def _sortino(returns: List[float], rf_daily: float = 0.0) -> float:
    """Annualised Sortino ratio — penalises only downside volatility."""
    n = len(returns)
    if n < 3:
        return 0.0
    mean    = sum(returns) / n
    dvar    = sum(min(r, 0) ** 2 for r in returns) / max(n - 1, 1)
    dstd    = math.sqrt(dvar)
    return round((mean - rf_daily) / dstd * math.sqrt(252), 2) if dstd else 0.0


def _cagr(initial: float, total_pnl: float, days: int) -> float:
    """Annualised CAGR %."""
    if initial <= 0 or days <= 0:
        return 0.0
    final  = initial + total_pnl
    years  = days / 365.0
    return round(((max(final, 0.01) / initial) ** (1 / years) - 1) * 100, 2)


def _percentile(vals: List[float], pct: float) -> float:
    """Linear-interpolated percentile."""
    if not vals:
        return 0.0
    sv  = sorted(vals)
    idx = (pct / 100.0) * (len(sv) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
    return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])


def _ist_date(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d") if ts else ""


def _pnl_pct(p) -> float:
    """Trade P&L as % of deployed capital."""
    cap = (p.entry_price or 0) * (p.quantity or 0)
    return round(p.realized_pnl / cap * 100.0, 3) if cap > 0 else 0.0


# ── Main engine ───────────────────────────────────────────────────────────────

class JournalEngine:
    """
    Compute TradingView-style metrics from a list of closed Position objects.

    Required Position fields (all present in trade_engine.Position):
        realized_pnl, entry_price, quantity, direction, entry_time, exit_time,
        symbol, mfe_pct, mae_pct, bars_in_trade
    """

    def __init__(self, total_capital: float = 500_000.0):
        self.total_capital = total_capital

    # ── Public entry point ────────────────────────────────────────────────────

    def compute(self, closed_positions: List) -> dict:
        """Return full analytics dict consumed by /api/journal_stats."""
        closed = [p for p in closed_positions if getattr(p, "status", "") == "CLOSED"]
        if not closed:
            return self._empty()

        cap = self.total_capital
        cs  = sorted(closed, key=lambda p: p.exit_time or 0)

        wins   = [p for p in closed if p.realized_pnl > 0]
        losses = [p for p in closed if p.realized_pnl <= 0]
        longs  = [p for p in closed if p.direction == "BUY"]
        shorts = [p for p in closed if p.direction == "SELL"]

        # ── Equity + drawdown curves ────────────────────────────────────────
        eq_curve, dd_curve = [], []
        cum, peak, max_dd = 0.0, 0.0, 0.0
        for p in cs:
            cum  += p.realized_pnl
            peak  = max(peak, cum)
            dd    = peak - cum
            max_dd = max(max_dd, dd)
            ts    = int(p.exit_time or 0) + IST_OFF
            eq_curve.append({"time": ts, "value": round(cum,   2)})
            dd_curve.append({"time": ts, "value": round(-dd,   2)})

        # ── Daily P&L → Sharpe / Sortino ───────────────────────────────────
        daily: Dict[str, float] = defaultdict(float)
        for p in cs:
            if p.exit_time:
                daily[_ist_date(p.exit_time)] += p.realized_pnl
        daily_pct  = [v / cap * 100.0 for v in daily.values()]
        sharpe     = _sharpe(daily_pct)
        sortino    = _sortino(daily_pct)

        # ── CAGR ────────────────────────────────────────────────────────────
        t0     = cs[0].entry_time or cs[0].exit_time or 0
        t1     = cs[-1].exit_time or 0
        n_days = max(1, int((t1 - t0) / 86400))
        total_pnl = sum(p.realized_pnl for p in closed)
        cagr   = _cagr(cap, total_pnl, n_days)

        # ── Profit factor ────────────────────────────────────────────────────
        gp     = sum(p.realized_pnl for p in wins)
        gl     = abs(sum(p.realized_pnl for p in losses))
        pf     = round(gp / gl, 3) if gl > 0 else 99.0

        # ── Capital efficiency ──────────────────────────────────────────────
        max_deployed = max((p.entry_price * p.quantity for p in closed), default=0)
        acct_size    = max_deployed + max_dd   # capital needed = max position + max DD
        ret_on_acct  = round(total_pnl / acct_size * 100, 2) if acct_size > 0 else 0.0

        # ── Detail breakdown (All / Long / Short) ───────────────────────────
        detail = {
            "all":   self._detail(closed),
            "long":  self._detail(longs),
            "short": self._detail(shorts),
        }

        # ── Per-symbol stats ─────────────────────────────────────────────────
        sym_map: Dict[str, List] = defaultdict(list)
        for p in closed:
            sym = (p.symbol or "").replace("NSE:", "").replace("-EQ", "")
            sym_map[sym].append(p)

        symbol_stats   = {}
        symbol_ranking = []
        for sym, trades in sym_map.items():
            s = self._symbol_stats(sym, trades, cap)
            symbol_stats[sym]  = s
            symbol_ranking.append(s)
        symbol_ranking.sort(key=lambda x: x["rank_score"], reverse=True)

        return {
            "summary": {
                "total_pnl":    round(total_pnl, 2),
                "total_trades": len(closed),
                "open_trades":  0,        # filled by app.py from live positions
                "winning":      len(wins),
                "losing":       len(losses),
                "win_rate":     round(len(wins) / len(closed) * 100, 2),
                "profit_factor": pf,
                "max_dd":       round(max_dd, 2),
                "sharpe":       sharpe,
                "sortino":      sortino,
                "cagr_pct":     cagr,
                "trading_days": n_days,
                "acct_size_required": round(acct_size, 2),
                "ret_on_acct":  ret_on_acct,
                "avg_mae_pct":  round(sum(getattr(p,"mae_pct",0) for p in closed) / len(closed), 3),
                "avg_mfe_pct":  round(sum(getattr(p,"mfe_pct",0) for p in closed) / len(closed), 3),
            },
            "detail":         detail,
            "equity_curve":   eq_curve,
            "dd_curve":       dd_curve,
            "symbol_stats":   symbol_stats,
            "symbol_ranking": symbol_ranking,
        }

    # ── Per-subset detail (mirrors TradingView's Trades Analysis table) ───────

    def _detail(self, subset: List) -> dict:
        if not subset:
            return {k: 0 for k in self._detail_keys()}

        wins   = [p for p in subset if p.realized_pnl > 0]
        losses = [p for p in subset if p.realized_pnl <= 0]
        n      = len(subset)

        gp = sum(p.realized_pnl for p in wins)
        gl = abs(sum(p.realized_pnl for p in losses))
        pf = round(gp / gl, 3) if gl > 0 else 99.0

        def avg_pnl(lst): return round(sum(p.realized_pnl for p in lst) / len(lst), 2) if lst else 0.0
        def avg_pct(lst): return round(sum(_pnl_pct(p) for p in lst) / len(lst), 2)    if lst else 0.0
        def bars(lst):
            b = [p.bars_in_trade for p in lst if getattr(p,"bars_in_trade",0)>0]
            return round(sum(b)/len(b)) if b else 0

        aw_abs = avg_pnl(wins);   aw_pct = avg_pct(wins)
        al_abs = abs(avg_pnl(losses)); al_pct = abs(avg_pct(losses))

        lw = max((p.realized_pnl for p in wins),   default=0.0)
        ll = abs(min((p.realized_pnl for p in losses), default=0.0))

        avg_mae = round(sum(getattr(p,"mae_pct",0) for p in subset)/n, 3)
        avg_mfe = round(sum(getattr(p,"mfe_pct",0) for p in subset)/n, 3)

        # MFE efficiency: how much of the MFE was captured at exit
        mfe_eff_list = []
        for p in subset:
            mfe = getattr(p,"mfe_pct",0)
            pct = _pnl_pct(p)
            if mfe > 0.01:
                mfe_eff_list.append(min(pct/mfe, 1.0))
        mfe_efficiency = round(sum(mfe_eff_list)/len(mfe_eff_list)*100, 1) if mfe_eff_list else 0.0

        return {
            "total":                  n,
            "winning":                len(wins),
            "losing":                 len(losses),
            "pct_profitable":         round(len(wins)/n*100, 2) if n else 0.0,
            "avg_pnl_abs":            avg_pnl(subset),
            "avg_pnl_pct":            avg_pct(subset),
            "avg_win_abs":            aw_abs,
            "avg_win_pct":            aw_pct,
            "avg_loss_abs":           al_abs,
            "avg_loss_pct":           al_pct,
            "ratio_win_loss":         round(aw_abs/al_abs, 3) if al_abs else 0.0,
            "profit_factor":          pf,
            "largest_win_abs":        round(lw, 2),
            "largest_win_pct":        round(max((_pnl_pct(p) for p in wins), default=0.0), 2),
            "largest_win_of_gross":   round(lw/gp*100, 2) if gp else 0.0,
            "largest_loss_abs":       round(ll, 2),
            "largest_loss_pct":       round(abs(min((_pnl_pct(p) for p in losses), default=0.0)), 2),
            "largest_loss_of_gross":  round(ll/gl*100, 2) if gl else 0.0,
            "avg_bars":               bars(subset),
            "avg_bars_winning":       bars(wins),
            "avg_bars_losing":        bars(losses),
            "avg_mae_pct":            avg_mae,
            "avg_mfe_pct":            avg_mfe,
            "mfe_capture_pct":        mfe_efficiency,   # % of available run captured
        }

    def _detail_keys(self):
        return ["total","winning","losing","pct_profitable","avg_pnl_abs","avg_pnl_pct",
                "avg_win_abs","avg_win_pct","avg_loss_abs","avg_loss_pct","ratio_win_loss",
                "profit_factor","largest_win_abs","largest_win_pct","largest_win_of_gross",
                "largest_loss_abs","largest_loss_pct","largest_loss_of_gross",
                "avg_bars","avg_bars_winning","avg_bars_losing",
                "avg_mae_pct","avg_mfe_pct","mfe_capture_pct"]

    # ── Per-symbol stats ──────────────────────────────────────────────────────

    def _symbol_stats(self, sym: str, trades: List, cap: float) -> dict:
        wins   = [p for p in trades if p.realized_pnl > 0]
        losses = [p for p in trades if p.realized_pnl <= 0]
        n      = len(trades)
        wr     = round(len(wins)/n*100, 1) if n else 0.0

        gp = sum(p.realized_pnl for p in wins)
        gl = abs(sum(p.realized_pnl for p in losses))
        pf = round(gp/gl, 2) if gl > 0 else (99.0 if gp > 0 else 0.0)

        # ── Suggested SL: 80th-pct MAE of winning trades ─────────────────
        # "How far against you can the stock go and still win?"
        win_maes = [abs(getattr(p,"mae_pct",0)) for p in wins if getattr(p,"mae_pct",0) < 0]
        suggested_sl = round(_percentile(win_maes, 80), 2) if len(win_maes) >= 3 else None

        # MAE percentiles for distribution insight
        all_maes = [abs(getattr(p,"mae_pct",0)) for p in trades]
        mae_p50  = round(_percentile(all_maes, 50), 2) if all_maes else 0.0
        mae_p80  = round(_percentile(all_maes, 80), 2) if all_maes else 0.0

        avg_mfe  = round(sum(getattr(p,"mfe_pct",0) for p in trades)/n, 2)
        avg_mae  = round(sum(getattr(p,"mae_pct",0) for p in trades)/n, 2)

        # Best/worst single-trade excursion (for Excursion Analysis sheet)
        best_mfe_pct  = round(max((getattr(p,"mfe_pct",0) for p in trades), default=0.0), 3)
        worst_mae_pct = round(max((abs(getattr(p,"mae_pct",0)) for p in trades), default=0.0), 3)

        # MFE capture: what fraction of the best run was captured at exit
        mfe_caps = []
        for p in trades:
            mfe = getattr(p, "mfe_pct", 0) or 0
            if mfe > 0.01:
                ep = getattr(p, "entry_price", 0) or 0
                xp = getattr(p, "exit_price",  0) or 0
                if ep > 0:
                    dir_pct = ((xp - ep)/ep*100) if getattr(p,"direction","") == "BUY" else ((ep - xp)/ep*100)
                    mfe_caps.append(min(dir_pct / mfe, 1.0))
        mfe_capture_pct = round(sum(mfe_caps)/len(mfe_caps)*100, 1) if mfe_caps else 0.0

        # Per-symbol Sharpe (if ≥2 trading days data)
        sym_daily: Dict[str, float] = defaultdict(float)
        for p in trades:
            if p.exit_time:
                sym_daily[_ist_date(p.exit_time)] += p.realized_pnl
        sym_pct  = [v/cap*100.0 for v in sym_daily.values()]
        sym_sh   = _sharpe(sym_pct) if len(sym_pct) >= 2 else 0.0

        # ── Composite rank score 0-100 ────────────────────────────────────
        norm_sh  = max(min(sym_sh,  3.0)/3.0, 0.0)
        norm_wr  = wr / 100.0
        norm_pf  = min(pf, 5.0)  / 5.0
        rank     = round((norm_sh*0.40 + norm_wr*0.30 + norm_pf*0.30) * 100, 1)

        return {
            "symbol":          sym,
            "total_trades":    n,
            "trades":          n,      # alias — scanner JS + report use this shorter key
            "win_rate":        wr,
            "profit_factor":   pf,
            "total_pnl":       round(sum(p.realized_pnl for p in trades), 2),
            "avg_win":         round(gp/len(wins), 2) if wins else 0.0,
            "avg_loss":        round(gl/len(losses), 2) if losses else 0.0,
            "avg_mfe_pct":     avg_mfe,
            "avg_mae_pct":     avg_mae,
            "best_mfe_pct":    best_mfe_pct,
            "worst_mae_pct":   worst_mae_pct,
            "mfe_capture_pct": mfe_capture_pct,
            "mae_p50":         mae_p50,
            "mae_p80":         mae_p80,
            "suggested_sl_pct": suggested_sl,
            "sharpe":          sym_sh,
            "rank_score":      rank,
            # SL quality flag: if current avg SL% < suggested → too tight
            "sl_too_tight":    (suggested_sl is not None and suggested_sl > 0.30),
        }

    # ── Single-symbol drill-down ──────────────────────────────────────────────

    def compute_symbol(self, symbol: str, closed_positions: List) -> dict:
        """Full journal analytics for ONE symbol — powers the popup modal.

        Args:
            symbol : clean ticker e.g. 'KOTAKBANK' (no NSE: / -EQ prefix)
            closed_positions : trade_engine.closed_positions (all symbols)

        Returns dict with keys:
            summary, detail (all/long/short), equity_curve, dd_curve, trade_log
        """
        # Filter to this symbol only
        sym_clean = symbol.replace("NSE:", "").replace("-EQ", "")
        trades = [
            p for p in closed_positions
            if getattr(p, "status", "") == "CLOSED"
            and (p.symbol or "").replace("NSE:", "").replace("-EQ", "") == sym_clean
        ]

        if not trades:
            return self._empty_symbol(sym_clean)

        cap  = self.total_capital
        cs   = sorted(trades, key=lambda p: p.exit_time or 0)
        wins = [p for p in trades if p.realized_pnl > 0]

        # ── Per-symbol equity + drawdown curves ──────────────────────────────
        eq_curve, dd_curve = [], []
        cum, peak, max_dd = 0.0, 0.0, 0.0
        for p in cs:
            cum   += p.realized_pnl
            peak   = max(peak, cum)
            dd     = peak - cum
            max_dd = max(max_dd, dd)
            ts     = int(p.exit_time or 0) + IST_OFF
            eq_curve.append({"time": ts, "value": round(cum,  2)})
            dd_curve.append({"time": ts, "value": round(-dd,  2)})

        # ── Sharpe / Sortino (daily %) ────────────────────────────────────────
        daily: Dict[str, float] = defaultdict(float)
        for p in cs:
            if p.exit_time:
                daily[_ist_date(p.exit_time)] += p.realized_pnl
        daily_pct = [v / cap * 100.0 for v in daily.values()]
        sharpe    = _sharpe(daily_pct)
        sortino   = _sortino(daily_pct)

        # ── CAGR ─────────────────────────────────────────────────────────────
        t0    = cs[0].entry_time or cs[0].exit_time or 0
        t1    = cs[-1].exit_time or 0
        n_days = max(1, int((t1 - t0) / 86400))
        total_pnl = sum(p.realized_pnl for p in trades)
        cagr  = _cagr(cap, total_pnl, n_days)

        # ── Per-symbol stats dict ─────────────────────────────────────────────
        stats = self._symbol_stats(sym_clean, trades, cap)

        # ── Trades Analysis (All / Long / Short) ─────────────────────────────
        longs  = [p for p in trades if p.direction == "BUY"]
        shorts = [p for p in trades if p.direction == "SELL"]
        detail = {
            "all":   self._detail(trades),
            "long":  self._detail(longs),
            "short": self._detail(shorts),
        }

        # ── Trade log (one dict per closed trade) ────────────────────────────
        trade_log = []
        for p in cs:
            ep  = getattr(p, "entry_price",  0) or 0
            xp  = getattr(p, "exit_price",   0) or 0
            qty = getattr(p, "quantity",      0) or 0
            cap_deployed = ep * qty
            pnl_pct = round(p.realized_pnl / cap_deployed * 100.0, 2) if cap_deployed > 0 else 0.0
            tf_raw  = str(getattr(p, "timeframe", "15"))
            tf_lbl  = "Intraday" if tf_raw == "5" else "Swing"
            entry_dt = datetime.fromtimestamp(p.entry_time, tz=IST).strftime("%d %b %y %H:%M") if p.entry_time else "—"
            exit_dt  = datetime.fromtimestamp(p.exit_time,  tz=IST).strftime("%d %b %y %H:%M") if p.exit_time  else "—"
            trade_log.append({
                "entry_date":    entry_dt,
                "exit_date":     exit_dt,
                "timeframe":     tf_lbl,
                "direction":     p.direction,
                "entry_price":   round(ep, 2),
                "exit_price":    round(xp, 2),
                "quantity":      int(qty),
                "pnl":           round(p.realized_pnl, 2),
                "pnl_pct":       pnl_pct,
                "mae_pct":       round(getattr(p, "mae_pct", 0) or 0, 3),
                "mfe_pct":       round(getattr(p, "mfe_pct", 0) or 0, 3),
                "bars":          int(getattr(p, "bars_in_trade", 0) or 0),
                "grade":         getattr(p, "quality_grade", "—") or "—",
                "pattern":       (getattr(p, "pattern_name", "") or "ML").replace("_", " "),
                "exit_reason":   (getattr(p, "exit_reason", "") or "—").replace("_", " "),
                "charges":       round(getattr(p, "charges", 0) or 0, 2),
            })

        # ── Summary card ─────────────────────────────────────────────────────
        summary = {
            "symbol":          sym_clean,
            "total_trades":    len(trades),
            "winning":         len(wins),
            "losing":          len(trades) - len(wins),
            "win_rate":        stats["win_rate"],
            "profit_factor":   stats["profit_factor"],
            "total_pnl":       round(total_pnl, 2),
            "max_dd":          round(max_dd, 2),
            "sharpe":          sharpe,
            "sortino":         sortino,
            "cagr_pct":        cagr,
            "avg_mfe_pct":     stats["avg_mfe_pct"],
            "avg_mae_pct":     stats["avg_mae_pct"],
            "mfe_capture_pct": stats["mfe_capture_pct"],
            "suggested_sl_pct": stats["suggested_sl_pct"],
            "rank_score":      stats["rank_score"],
            "trading_days":    n_days,
        }

        return {
            "symbol":      sym_clean,
            "summary":     summary,
            "detail":      detail,
            "equity_curve": eq_curve,
            "dd_curve":    dd_curve,
            "trade_log":   trade_log,
        }

    def _empty_symbol(self, sym: str) -> dict:
        return {
            "symbol":      sym,
            "summary":     {},
            "detail":      {"all": {}, "long": {}, "short": {}},
            "equity_curve": [],
            "dd_curve":    [],
            "trade_log":   [],
        }

    def _empty(self) -> dict:
        return {
            "summary":        {},
            "detail":         {"all": {}, "long": {}, "short": {}},
            "equity_curve":   [],
            "dd_curve":       [],
            "symbol_stats":   {},
            "symbol_ranking": [],
        }
