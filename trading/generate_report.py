"""
Report Generator — NSE Equity Trading Terminal
Generates XLSX trade analysis reports.
Adapted from COMMODITY APP generate_report.py for equity markets.
"""
import os
import logging
from datetime import datetime

import config

logger = logging.getLogger(__name__)


def _fmt_dt(ts):
    """Unix timestamp → datetime object. Returns None on failure."""
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(float(ts))
    except Exception:
        return None


def _date_str(ts):
    dt = _fmt_dt(ts)
    return dt.strftime("%d/%m/%Y") if dt else ""


def _time_str(ts):
    dt = _fmt_dt(ts)
    return dt.strftime("%H:%M:%S") if dt else ""


def _ddmm_hhmm(ts):
    """DD/MM HH:MM format for the UI trades table."""
    dt = _fmt_dt(ts)
    return dt.strftime("%d/%m %H:%M") if dt else "--"


def generate_trade_report(trade_history, output_path=None):
    """Generate XLSX report from list of closed Position dicts.

    Each item should have fields matching trade_engine.Position dataclass:
    symbol, direction, entry_price, exit_price, quantity, realized_pnl,
    charges, entry_time, exit_time, exit_reason, pattern_name,
    signal_confidence, regime, status.

    Returns output_path on success, None on failure.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    except ImportError:
        logger.error("openpyxl not installed. Run: pip install openpyxl")
        return None

    if not trade_history:
        logger.warning("No trades to report")
        return None

    if output_path is None:
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        output_path = os.path.join(
            config.REPORTS_DIR,
            f"equity_trade_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        )

    wb = Workbook()

    # ── Styles ───────────────────────────────────────────────────
    HDR_FONT   = Font(bold=True, color="FFFFFF", size=11)
    HDR_FILL   = PatternFill(start_color="1A3A5C", end_color="1A3A5C", fill_type="solid")  # dark blue
    PROF_FILL  = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    LOSS_FILL  = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    THIN       = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"),  bottom=Side(style="thin")
    )
    BOLD       = Font(bold=True)
    CENTER     = Alignment(horizontal="center")

    # ── Normalise trade list ──────────────────────────────────────
    closed = [t for t in trade_history if t.get("status", "CLOSED") == "CLOSED"]

    # ── Sheet 1: Trade History ────────────────────────────────────
    ws = wb.active
    ws.title = "Trade History"

    headers = [
        "Trade ID", "Symbol", "Direction", "Pattern", "Confidence %",
        "Entry Date", "Entry Time", "Exit Date", "Exit Time",
        "Entry Price", "Exit Price", "Qty",
        "Gross P&L (₹)", "Charges (₹)", "Net P&L (₹)",
        "Exit Reason", "Regime", "Strategy",
    ]
    col_w = {1:9, 2:14, 3:10, 4:22, 5:12,
             6:12, 7:10, 8:12, 9:10,
             10:13, 11:13, 12:8,
             13:14, 14:13, 15:13,
             16:20, 17:16, 18:20}

    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font      = HDR_FONT
        cell.fill      = HDR_FILL
        cell.alignment = CENTER
        cell.border    = THIN
        ws.column_dimensions[cell.column_letter].width = col_w.get(c, 14)

    for row, t in enumerate(closed, 2):
        net_pnl   = t.get("realized_pnl", 0) or 0
        charges   = t.get("charges", 0) or 0
        gross_pnl = net_pnl + charges

        values = [
            row - 1,
            t.get("symbol", "").replace("NSE:", "").replace("-EQ", ""),
            t.get("direction", ""),
            t.get("pattern_name", ""),
            round(t.get("signal_confidence", 0) or 0, 1),
            _date_str(t.get("entry_time")),
            _time_str(t.get("entry_time")),
            _date_str(t.get("exit_time")),
            _time_str(t.get("exit_time")),
            round(t.get("entry_price", 0) or 0, 2),
            round(t.get("exit_price", 0) or 0, 2),
            t.get("quantity", 0) or 0,
            round(gross_pnl, 2),
            round(charges, 2),
            round(net_pnl, 2),
            t.get("exit_reason", ""),
            t.get("regime", ""),
            t.get("strategy", ""),
        ]
        for c, v in enumerate(values, 1):
            cell = ws.cell(row=row, column=c, value=v)
            cell.border    = THIN
            cell.alignment = CENTER
            if c in (13, 14, 15):
                cell.number_format = "#,##0.00"
                if c == 15:
                    cell.fill = PROF_FILL if net_pnl > 0 else LOSS_FILL

    ws.freeze_panes = "A2"

    # ── Sheet 2: Summary ─────────────────────────────────────────
    ws2 = wb.create_sheet("Summary")
    winners   = [t for t in closed if (t.get("realized_pnl") or 0) > 0]
    losers    = [t for t in closed if (t.get("realized_pnl") or 0) <= 0]
    total_pnl = sum(t.get("realized_pnl", 0) or 0 for t in closed)
    win_rate  = round(len(winners) / len(closed) * 100, 1) if closed else 0
    avg_win   = round(sum(t.get("realized_pnl", 0) for t in winners) / len(winners), 2) if winners else 0
    avg_loss  = round(abs(sum(t.get("realized_pnl", 0) for t in losers) / len(losers)), 2) if losers else 0
    wl_ratio  = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0

    summary_rows = [
        ("Report Date",       datetime.now().strftime("%d/%m/%Y %H:%M")),
        ("Total Trades",      len(closed)),
        ("Winners",           len(winners)),
        ("Losers",            len(losers)),
        ("Win Rate",          f"{win_rate}%"),
        ("Total Net P&L",     f"₹ {total_pnl:,.2f}"),
        ("Avg Win",           f"₹ {avg_win:,.2f}"),
        ("Avg Loss",          f"₹ {avg_loss:,.2f}"),
        ("W/L Ratio",         f"{wl_ratio}×"),
        ("Avg P&L / Trade",   f"₹ {total_pnl/len(closed):,.2f}" if closed else "₹ 0"),
    ]

    ws2.cell(row=1, column=1, value="EQUITY TRADE REPORT — SUMMARY").font = Font(bold=True, size=13)
    for r, (label, value) in enumerate(summary_rows, 2):
        ws2.cell(row=r, column=1, value=label).font = BOLD
        ws2.cell(row=r, column=2, value=value)
    ws2.column_dimensions["A"].width = 24
    ws2.column_dimensions["B"].width = 22

    # ── Sheet 3: Per Strategy ────────────────────────────────────
    ws3 = wb.create_sheet("Per Strategy")
    hdrs3 = ["Strategy", "Trades", "Winners", "Losers", "Win Rate %",
             "Total P&L (₹)", "Avg P&L (₹)", "Status"]
    for c, h in enumerate(hdrs3, 1):
        cell = ws3.cell(row=1, column=c, value=h)
        cell.font = HDR_FONT; cell.fill = HDR_FILL
        cell.alignment = CENTER; cell.border = THIN

    strat_stats: dict = {}
    for t in closed:
        s = t.get("strategy", t.get("regime", "unknown")) or "unknown"
        if s not in strat_stats:
            strat_stats[s] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0}
        pnl = t.get("realized_pnl", 0) or 0
        strat_stats[s]["trades"] += 1
        strat_stats[s]["wins"]   += 1 if pnl > 0 else 0
        strat_stats[s]["losses"] += 1 if pnl <= 0 else 0
        strat_stats[s]["pnl"]    += pnl

    for row, (strat, st) in enumerate(sorted(strat_stats.items(), key=lambda x: x[1]["pnl"], reverse=True), 2):
        wr   = round(st["wins"] / st["trades"] * 100, 1) if st["trades"] > 0 else 0
        avg  = round(st["pnl"] / st["trades"], 2) if st["trades"] > 0 else 0
        vals = [strat, st["trades"], st["wins"], st["losses"], wr,
                round(st["pnl"], 2), avg, "ACTIVE" if st["pnl"] > 0 else "UNDERPERFORMING"]
        for c, v in enumerate(vals, 1):
            cell = ws3.cell(row=row, column=c, value=v)
            cell.border = THIN; cell.alignment = CENTER
            if c == 6:
                cell.number_format = "#,##0.00"
                cell.fill = PROF_FILL if v > 0 else LOSS_FILL
            elif c == 8:
                cell.fill = PROF_FILL if v == "ACTIVE" else LOSS_FILL
    for c in range(1, len(hdrs3) + 1):
        ws3.column_dimensions[ws3.cell(row=1, column=c).column_letter].width = 20

    # ── Sheet 4: Per Pattern ─────────────────────────────────────
    ws4 = wb.create_sheet("Per Pattern")
    hdrs4 = ["Pattern", "Trades", "Winners", "Losers", "Win Rate %",
             "Total P&L (₹)", "Avg P&L (₹)"]
    for c, h in enumerate(hdrs4, 1):
        cell = ws4.cell(row=1, column=c, value=h)
        cell.font = HDR_FONT; cell.fill = HDR_FILL
        cell.alignment = CENTER; cell.border = THIN

    pat_stats: dict = {}
    for t in closed:
        p = t.get("pattern_name", "") or "ML_SIGNAL"
        if not p:
            p = "ML_SIGNAL"
        if p not in pat_stats:
            pat_stats[p] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0}
        pnl = t.get("realized_pnl", 0) or 0
        pat_stats[p]["trades"] += 1
        pat_stats[p]["wins"]   += 1 if pnl > 0 else 0
        pat_stats[p]["losses"] += 1 if pnl <= 0 else 0
        pat_stats[p]["pnl"]    += pnl

    for row, (pat, st) in enumerate(sorted(pat_stats.items(), key=lambda x: x[1]["pnl"], reverse=True), 2):
        wr   = round(st["wins"] / st["trades"] * 100, 1) if st["trades"] > 0 else 0
        avg  = round(st["pnl"] / st["trades"], 2) if st["trades"] > 0 else 0
        vals = [pat, st["trades"], st["wins"], st["losses"], wr,
                round(st["pnl"], 2), avg]
        for c, v in enumerate(vals, 1):
            cell = ws4.cell(row=row, column=c, value=v)
            cell.border = THIN; cell.alignment = CENTER
            if c == 6:
                cell.number_format = "#,##0.00"
                cell.fill = PROF_FILL if v > 0 else LOSS_FILL
    for c in range(1, len(hdrs4) + 1):
        ws4.column_dimensions[ws4.cell(row=1, column=c).column_letter].width = 20

    wb.save(output_path)
    logger.info(f"Equity trade report saved: {output_path}")
    return output_path
