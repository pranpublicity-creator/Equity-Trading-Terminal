"""
EQUITY TRADING TERMINAL
AI-Powered Pattern Detection + ML Signal Fusion + Rate-Limited Execution
Flask + SocketIO on port 5005
"""
import json
import logging
import os
import sys
import time
import threading
from datetime import datetime, timezone, timedelta

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

import config

# Ensure data directories exist
for d in [config.DATA_DIR, config.CACHE_DIR, config.MODELS_DIR, config.LOGS_DIR, config.REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Force UTF-8 on stdout/stderr so Unicode log messages (e.g. "₹")
# survive Windows cp1252 consoles. Python 3.7+ supports reconfigure().
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Logging — file handler uses UTF-8; stream handler inherits from stdout above.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ============================================================
# Initialize all components
# ============================================================
from core.rate_limiter import rate_limiter
from core.fyers_manager import FyersManager, InvalidSymbolError
from core.data_engine import DataEngine
from core.watchlist_manager import WatchlistManager
from core.market_data_enricher import MarketDataEnricher
from core.stock_universe import get_universe

from patterns.pattern_detector import PatternDetector
from features.feature_pipeline import FeaturePipeline
from features.indicator_engine import compute_all
from models.model_manager import ModelManager
from models.model_trainer import ModelTrainer
from signals.signal_engine import SignalEngine
from signals.regime_detector import detect_regime
from adaptive.adaptive_bot import AdaptiveBot
from adaptive.bayesian_selector import BayesianSelector
from trading.trade_engine import TradeEngine
from trading.risk_manager import RiskManager
from trading.charge_calculator import ChargeCalculator
from trading.telegram_notifier import TelegramNotifier
from trading.telegram_commander import TelegramCommander
from trading.webhook_executor import WebhookExecutor
from trading.portfolio_allocator import PortfolioAllocator

# Core
fyers = FyersManager()
data_engine = DataEngine(fyers)
watchlist = WatchlistManager()
enricher = MarketDataEnricher(fyers)

# Pattern + Features
pattern_detector = PatternDetector()
feature_pipeline = FeaturePipeline()
model_manager = ModelManager()

# Signals
signal_engine = SignalEngine(data_engine)

# Trading
charge_calc = ChargeCalculator(is_intraday=True)
trade_engine = TradeEngine(fyers, charge_calculator=charge_calc)
risk_manager = RiskManager(trade_engine)
signal_engine.entry_gate.trade_engine = trade_engine
signal_engine.entry_gate.risk_manager = risk_manager

# Adaptive
adaptive_bot = AdaptiveBot(signal_engine)
bayesian = BayesianSelector()

# Portfolio allocator
portfolio_allocator = PortfolioAllocator()

# Notifications
telegram = TelegramNotifier()
webhook_exec = WebhookExecutor(trade_engine, signal_engine)

# Telegram command handler — polls for /help /status /positions etc.
commander = TelegramCommander(telegram, trade_engine=trade_engine)
commander.set_scanner_fns(
    fn_start      = lambda: _start_poller(),
    fn_stop       = lambda: globals().update(_poller_running=False),
    fn_is_running = lambda: _poller_running,
)
commander.start()

# State
_poller_running = False
_poller_thread = None

# ── IST timezone (no pytz dependency) ───────────────────────────
_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(_IST)

def _ist_hm() -> tuple:
    """Return (hour, minute) in IST."""
    n = _now_ist()
    return n.hour, n.minute

def _ist_time_str() -> str:
    return _now_ist().strftime("%H:%M:%S IST")


# ── Server-side activity log (survives browser refresh) ──────────
_activity_log: list = []           # kept in memory; max 500 entries
_ACTIVITY_LOG_MAX = 500

def _srv_log(msg: str, log_type: str = "info", trade_ts: float = None):
    """Append to server-side activity log and push to all connected clients."""
    ts = trade_ts or time.time()
    ist_str = datetime.fromtimestamp(ts, tz=_IST).strftime("%H:%M:%S")
    entry = {"msg": msg, "type": log_type, "time": ist_str, "ts": ts}
    _activity_log.append(entry)
    if len(_activity_log) > _ACTIVITY_LOG_MAX:
        _activity_log.pop(0)
    try:
        socketio.emit("activity_log_entry", entry)
    except Exception:
        pass


# ── EOD auto-close state (server-side, reset each day) ───────────
_eod_closed_today: bool = False
_eod_last_reset_day: int = -1      # day-of-month when flag was last reset


# ── Training Queue ──────────────────────────────────────────────
import queue as _queue
_train_queue = _queue.Queue()          # items: (symbol, days)
_train_queue_running = False
_train_queue_thread = None
_train_current = None                  # (symbol, days) currently training
_train_completed = []                  # list of {symbol, days, status}


def _start_train_queue_worker():
    """Start the single-threaded training queue worker if not already running."""
    global _train_queue_running, _train_queue_thread
    if _train_queue_running:
        return
    _train_queue_running = True
    _train_queue_thread = threading.Thread(target=_train_queue_worker, daemon=True)
    _train_queue_thread.start()


def _train_queue_worker():
    """Process training requests one at a time from the queue."""
    global _train_queue_running, _train_current, _train_completed
    logger.info("Training queue worker started")
    while True:
        try:
            symbol, days = _train_queue.get(timeout=300)  # wait up to 5 min
        except _queue.Empty:
            _train_queue_running = False
            logger.info("Training queue idle — worker stopped")
            return

        _train_current = (symbol, days)
        ticker = symbol.replace("NSE:", "").replace("-EQ", "")
        socketio.emit("training_queue_update", {
            "current": {"symbol": symbol, "days": days},
            "queue_size": _train_queue.qsize(),
        })
        try:
            _train_model(symbol, days)
            _train_completed.append({"symbol": symbol, "days": days, "status": "ok"})
        except Exception as e:
            logger.error(f"Queue training failed for {ticker}: {e}")
            _train_completed.append({"symbol": symbol, "days": days, "status": "error", "error": str(e)})
        finally:
            _train_current = None
            _train_queue.task_done()
            socketio.emit("training_queue_update", {
                "current": None,
                "queue_size": _train_queue.qsize(),
            })


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    authenticated = fyers.is_authenticated()
    return render_template("index.html", authenticated=authenticated)


@app.route("/auth")
def auth_generate():
    """Generate Fyers auth URL and return it."""
    try:
        url = fyers.generate_auth_url()
        return jsonify({"url": url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/auth/callback", methods=["POST"])
def auth_callback():
    """Process auth code from Fyers redirect."""
    # Accept both form-encoded and JSON
    if request.is_json:
        data = request.get_json()
        auth_code = data.get("auth_code", "")
    else:
        auth_code = request.form.get("auth_code", "")

    auth_code = auth_code.strip()
    if not auth_code:
        return jsonify({"success": False, "message": "No auth code provided"}), 400

    # Extract auth_code from full redirect URL if user pasted the whole thing
    if "auth_code=" in auth_code:
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(auth_code)
            params = parse_qs(parsed.query)
            if "auth_code" in params:
                auth_code = params["auth_code"][0]
        except Exception:
            import re
            match = re.search(r'auth_code=([^&\s#]+)', auth_code)
            if match:
                auth_code = match.group(1)

    success, message = fyers.generate_token(auth_code)
    if success:
        # Reinitialize data engine with authenticated fyers
        data_engine.fyers = fyers
        enricher.fyers = fyers
        logger.info("Fyers authentication successful")
        # Auto-start scanner now that we have a valid token
        if not _poller_running:
            _start_poller()
            logger.info("Scanner auto-started after authentication")
        return jsonify({"success": True, "message": message})
    else:
        logger.warning(f"Fyers auth failed: {message}")
        return jsonify({"success": False, "message": message}), 401


@app.route("/api/indices")
def api_indices():
    """Live NIFTY 50, BANK NIFTY, SENSEX, NIFTY IT quotes."""
    INDICES = {
        "NIFTY": "NSE:NIFTY50-INDEX",
        "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
        "SENSEX": "BSE:SENSEX-INDEX",
        "NIFTYIT": "NSE:NIFTYIT-INDEX",
    }
    result = {}
    if not fyers.is_authenticated():
        return jsonify(result)
    try:
        symbols_str = ",".join(INDICES.values())
        quotes = fyers.get_quotes(symbols_str)
        if quotes:
            sym_map = {v: k for k, v in INDICES.items()}
            for q in quotes:
                sym = q.get("n", "")
                name = sym_map.get(sym, sym)
                v = q.get("v", {})
                result[name] = {
                    "ltp": v.get("lp", 0),
                    "ch": v.get("ch", 0),
                    "chp": v.get("chp", 0),
                }
    except Exception as e:
        logger.error(f"Indices fetch error: {e}")
    return jsonify(result)


@app.route("/api/scanner")
def api_scanner():
    """Return scanner status including last scanned stocks."""
    return jsonify({
        "running": _poller_running,
        "universe": watchlist.universe_name,
        "total_stocks": len(watchlist.active_symbols),
        "progress": watchlist.get_rotation_progress(),
    })


# NOTE: chart data + pattern overlay are served by the pre-existing
#   /api/chart_data?symbol=...&resolution=...
#   /api/pattern_overlay?symbol=...&resolution=...
# routes defined further below.  Frontend (_renderLightweightChart) calls
# both and merges them.  No new endpoint needed here.


@app.route("/api/auth/status")
def auth_status():
    """Check if Fyers is authenticated."""
    return jsonify({
        "authenticated": fyers.is_authenticated(),
        "app_id": fyers.client_id,
    })


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    """Logout from Fyers."""
    fyers.logout()
    return jsonify({"success": True})


@app.route("/api/credentials", methods=["POST"])
def api_save_credentials():
    """Save custom Fyers API credentials."""
    data = request.get_json()
    app_id = data.get("app_id", "").strip()
    secret = data.get("secret", "").strip()
    if not app_id or not secret:
        return jsonify({"success": False, "message": "Both App ID and Secret are required"})
    FyersManager.save_credentials(app_id, secret)
    # Reload credentials
    fyers.client_id = app_id
    fyers.secret_key = secret
    return jsonify({"success": True, "message": f"Credentials saved for {app_id}"})


@app.route("/api/status")
def api_status():
    return jsonify({
        "authenticated": fyers.is_authenticated(),
        "running": _poller_running,
        "universe": watchlist.universe_name,
        "stock_count": len(watchlist.active_symbols),
        "active_positions": trade_engine.get_active_position_count(),
        "daily_pnl": trade_engine.get_daily_pnl(),
        "regime": adaptive_bot.current_regime,
        "strategy": adaptive_bot.current_strategy,
        "rate_limiter": rate_limiter.get_stats(),
        "auto_execute": trade_engine.auto_execute,
        "auto_adapt": adaptive_bot.auto_adapt,
        "mpp_pct": config.ORDER_MPP_PCT,
        "stale_sec": config.MAX_PRICE_STALENESS_SECONDS,
    })


@app.route("/api/positions")
def api_positions():
    from dataclasses import asdict
    active  = [asdict(p) for p in trade_engine.get_active_positions()]
    pending = [asdict(p) for p in trade_engine.positions.values() if p.status == "PENDING"]
    closed  = [asdict(p) for p in trade_engine.closed_positions[-50:]]
    return jsonify({
        "active":      active,
        "pending":     pending,
        "closed":      closed,
        "daily_pnl":   trade_engine.get_daily_pnl(),
        "perf_stats":  trade_engine.get_performance_stats(),
    })


@app.route("/api/portfolio")
def api_portfolio():
    summary = risk_manager.get_portfolio_summary()
    perf    = trade_engine.get_performance_stats()
    summary.update(perf)
    return jsonify(summary)


@app.route("/api/portfolio/stats")
def api_portfolio_stats():
    """Extended portfolio performance stats for the UI panel."""
    return jsonify(trade_engine.get_performance_stats())


@app.route("/api/config/capital/get")
def api_config_capital_get():
    """Return current capital config values for UI population."""
    return jsonify({
        "ok": True,
        "total_capital":  config.TOTAL_TRADING_CAPITAL,
        "stock_pct":      round(config.MAX_CAPITAL_PER_STOCK * 100, 1),
        "max_positions":  config.MAX_CONCURRENT_POSITIONS,
    })


@app.route("/api/config/capital", methods=["POST"])
def api_config_capital():
    """Update capital settings at runtime without restarting the app."""
    from flask import request as freq
    data = freq.get_json(silent=True) or {}
    try:
        total_cap  = float(data.get("total_capital", config.TOTAL_TRADING_CAPITAL))
        stock_pct  = float(data.get("stock_pct",    config.MAX_CAPITAL_PER_STOCK * 100)) / 100.0
        max_pos    = int(data.get("max_positions",   config.MAX_CONCURRENT_POSITIONS))

        # Validate
        if total_cap < 10000:
            return jsonify({"ok": False, "error": "Min capital ₹10,000"})
        if not (1 <= max_pos <= 50):
            return jsonify({"ok": False, "error": "Max positions must be 1–50"})
        if not (0.01 <= stock_pct <= 0.50):
            return jsonify({"ok": False, "error": "Per-stock cap must be 1%–50%"})

        # Apply live — mutate the config module in-place (affects all subsequent calls)
        config.TOTAL_TRADING_CAPITAL   = total_cap
        config.MAX_CAPITAL_PER_STOCK   = stock_pct
        config.MAX_CONCURRENT_POSITIONS = max_pos

        # Also update risk_manager if it holds its own copy
        if risk_manager is not None:
            risk_manager.total_capital = total_cap

        logger.info(
            f"Capital settings updated: total=₹{total_cap:,.0f} "
            f"stock_cap={stock_pct*100:.0f}% max_pos={max_pos}"
        )
        return jsonify({"ok": True,
                        "total_capital": total_cap,
                        "stock_pct": stock_pct * 100,
                        "max_positions": max_pos})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/report/download")
def api_report_download():
    """Generate and stream an XLSX trade report."""
    from flask import send_file
    from dataclasses import asdict
    from trading.generate_report import generate_trade_report

    closed = [asdict(p) for p in trade_engine.closed_positions if p.status == "CLOSED"]
    if not closed:
        return jsonify({"error": "No closed trades to report"}), 404

    try:
        path = generate_trade_report(closed)
        if path is None:
            return jsonify({"error": "Report generation failed — openpyxl may not be installed"}), 500
        return send_file(
            path,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=os.path.basename(path),
        )
    except Exception as e:
        logger.error(f"Report download error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/models")
def api_models():
    """Return full model inventory — all trained stocks with accuracy metrics."""
    raw = model_manager.list_models()
    # Enrich each entry with clean display fields
    result = []
    for ticker, info in sorted(raw.items()):
        metrics = info.get("metrics", {})
        # metrics can be nested under sub-keys from _train_model results dict
        lgbm_m  = metrics.get("lgbm", {}) or {}
        xgb_m   = metrics.get("xgb",  {}) or {}
        arima_m = metrics.get("arima", {}) or {}
        trained_at = info.get("trained_at")
        import datetime as _dt
        if trained_at:
            try:
                trained_str = _dt.datetime.fromtimestamp(trained_at).strftime("%d-%b %H:%M")
            except Exception:
                trained_str = "--"
        else:
            trained_str = "--"

        files = info.get("model_files", [])
        result.append({
            "ticker": ticker,
            "symbol": f"NSE:{ticker}-EQ",
            "trained_at": trained_str,
            "has_lgbm": "lgbm_model.pkl" in files,
            "has_xgb":  "xgb_model.pkl"  in files,
            "has_arima":"arima_model.pkl" in files,
            "has_meta": "meta_learner.pkl" in files,
            "lgbm_acc": round((lgbm_m.get("accuracy") or lgbm_m.get("test_accuracy") or 0) * 100, 1),
            "lgbm_auc": round(lgbm_m.get("auc") or 0, 3),
            "xgb_acc":  round((xgb_m.get("accuracy")  or xgb_m.get("test_accuracy")  or 0) * 100, 1),
            "xgb_auc":  round(xgb_m.get("auc") or 0, 3),
            "arima_order": str(arima_m.get("order", "--")),
            "days": metrics.get("days", "--"),
            "candles": metrics.get("candles", "--"),
        })
    return jsonify({"models": result, "total": len(result)})


@app.route("/api/regime")
def api_regime():
    return jsonify({
        "current": adaptive_bot.current_regime,
        "strategy": adaptive_bot.current_strategy,
        "distribution": adaptive_bot.get_regime_distribution(),
    })


@app.route("/api/strategies")
def api_strategies():
    from signals.strategy_presets import list_presets
    return jsonify({
        "presets": list_presets(),
        "current": adaptive_bot.current_strategy,
        "bayesian_stats": bayesian.get_strategy_stats(),
    })


@app.route("/api/start", methods=["POST"])
def api_start():
    global _poller_running
    if _poller_running:
        return jsonify({"status": "already_running"})
    _start_poller()
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global _poller_running
    _poller_running = False
    return jsonify({"status": "stopped"})


@app.route("/api/set_universe", methods=["POST"])
def api_set_universe():
    data = request.get_json()
    name = data.get("universe", "NIFTY_200")
    watchlist.set_universe(name)
    return jsonify({"status": "ok", "universe": name, "count": len(watchlist.active_symbols)})


@app.route("/api/set_strategy", methods=["POST"])
def api_set_strategy():
    data = request.get_json()
    name = data.get("strategy", "FUSION_FULL")
    adaptive_bot.set_strategy(name)
    return jsonify({"status": "ok", "strategy": name})


@app.route("/api/toggle_auto_execute", methods=["POST"])
def api_toggle_auto():
    trade_engine.auto_execute = not trade_engine.auto_execute
    return jsonify({"auto_execute": trade_engine.auto_execute})


@app.route("/api/toggle_auto_adapt", methods=["POST"])
def api_toggle_adapt():
    if adaptive_bot.auto_adapt:
        adaptive_bot.disable_auto_adapt()
    else:
        adaptive_bot.enable_auto_adapt()
    return jsonify({"auto_adapt": adaptive_bot.auto_adapt})


@app.route("/api/confirm_trade/<pos_id>", methods=["POST"])
def api_confirm_trade(pos_id):
    success = trade_engine.confirm_entry(pos_id)
    return jsonify({"status": "confirmed" if success else "failed"})


@app.route("/api/cancel_trade/<pos_id>", methods=["POST"])
def api_cancel_trade(pos_id):
    """Cancel PENDING or force-close ACTIVE position."""
    trade_engine.cancel_position(pos_id)
    return jsonify({"status": "closed"})


@app.route("/api/positions/cleanup", methods=["POST"])
def api_positions_cleanup():
    """Remove duplicate positions for the same symbol (keep oldest)."""
    removed = trade_engine.dedup_positions()
    return jsonify({"ok": True, "removed": removed,
                    "remaining": len(trade_engine.positions)})


@app.route("/api/eod_force_close", methods=["POST"])
def api_eod_force_close():
    """Force-close ALL active intraday (5-min) positions immediately.
    Used when the normal 15:25 EOD window was missed (e.g. server restart).
    """
    data        = request.get_json(silent=True) or {}
    target_tf   = data.get("timeframe", "5")          # default: intraday only
    close_all   = data.get("close_all", False)         # True = close swing too

    targets = [
        p for p in trade_engine.get_active_positions()
        if close_all or getattr(p, "timeframe", "15") == target_tf
    ]

    if not targets:
        return jsonify({"ok": True, "closed": 0, "message": "No matching open positions"})

    closed = []
    now_ist = _ist_time_str()
    for p in targets:
        try:
            ticker = (p.symbol or "").replace("NSE:", "").replace("-EQ", "")
            trade_engine.cancel_position(p.id)
            _srv_log(
                f"[FORCE CLOSE] {p.direction} {ticker} closed @ ₹{p.current_price or p.entry_price:.2f}"
                f" | P&L ₹{p.realized_pnl:.2f} | {now_ist}",
                log_type="warning",
            )
            closed.append({
                "symbol":  ticker,
                "direction": p.direction,
                "pnl":     round(p.realized_pnl, 2),
                "exit_price": round(p.exit_price or p.entry_price, 2),
            })
            logger.info(f"FORCE CLOSE: {p.direction} {ticker} P&L={p.realized_pnl:.2f}")
        except Exception as e:
            logger.error(f"Force-close failed for {p.symbol}: {e}")

    # Push updated dashboard immediately
    _emit_dashboard_update([], [])

    return jsonify({
        "ok":      True,
        "closed":  len(closed),
        "details": closed,
        "time":    now_ist,
    })


@app.route("/api/journal_stats")
def api_journal_stats():
    """TradingView-style strategy performance report.
    Computes: equity curve, drawdown, Sharpe, Sortino, CAGR, MAE/MFE,
    profit factor, per-symbol breakdown, and suggested SL per symbol.
    """
    try:
        from trading.journal_engine import JournalEngine
        je     = JournalEngine(total_capital=config.TOTAL_TRADING_CAPITAL)
        report = je.compute(trade_engine.closed_positions)
        # Patch in live open count
        if report.get("summary"):
            report["summary"]["open_trades"] = len(trade_engine.get_active_positions())
        return jsonify(report)
    except Exception as e:
        logger.error(f"journal_stats error: {e}", exc_info=True)
        return jsonify({"error": str(e), "summary": {}, "detail": {},
                        "equity_curve": [], "dd_curve": [],
                        "symbol_stats": {}, "symbol_ranking": []})


@app.route("/api/activity_log")
def api_activity_log():
    """Return last 200 server-side activity log entries (survives browser refresh)."""
    return jsonify({"entries": _activity_log[-200:]})


@app.route("/api/universe_symbols")
def api_universe_symbols():
    """Return sorted list of all active symbols in current universe."""
    return jsonify({
        "symbols": sorted(watchlist.active_symbols),
        "universe": watchlist.universe_name,
        "count": len(watchlist.active_symbols),
    })


@app.route("/api/train", methods=["POST"])
def api_train():
    """Queue one or more symbols for sequential background training."""
    data = request.get_json()
    days = int(data.get("days", config.ML_TRAIN_DAYS))
    days = max(30, min(days, 730))

    # Accept either a single symbol or a list
    symbols_raw = data.get("symbols") or ([data.get("symbol")] if data.get("symbol") else [])
    symbols = [s.strip() for s in symbols_raw if s and s.strip()]
    if not symbols:
        return jsonify({"error": "symbol or symbols required"})

    queued = []
    for sym in symbols:
        _train_queue.put((sym, days))
        queued.append(sym)

    _start_train_queue_worker()

    return jsonify({
        "status": "queued",
        "queued": queued,
        "days": days,
        "queue_size": _train_queue.qsize(),
    })


@app.route("/api/train/queue")
def api_train_queue():
    """Return current training queue status."""
    items = list(_train_queue.queue)   # snapshot (not thread-safe but good enough for display)
    return jsonify({
        "current": {"symbol": _train_current[0], "days": _train_current[1]} if _train_current else None,
        "pending": [{"symbol": s, "days": d} for s, d in items],
        "completed": _train_completed[-20:],   # last 20
        "queue_size": _train_queue.qsize(),
        "worker_running": _train_queue_running,
    })


@app.route("/api/train/all", methods=["POST"])
def api_train_all():
    """Queue ALL active watchlist symbols for training (batch mode)."""
    data = request.get_json() or {}
    days = int(data.get("days", config.ML_TRAIN_DAYS))
    days = max(30, min(days, 730))

    symbols = watchlist.get_active_symbols()
    if not symbols:
        return jsonify({"error": "No active symbols in watchlist"})

    already_queued = {s for s, _ in list(_train_queue.queue)}
    if _train_current:
        already_queued.add(_train_current[0])

    added = []
    skipped = []
    for sym in symbols:
        if sym in already_queued:
            skipped.append(sym)
        else:
            _train_queue.put((sym, days))
            added.append(sym)

    _start_train_queue_worker()

    est_min = len(symbols) * (3 if days <= 90 else 5 if days <= 180 else 10)
    return jsonify({
        "status": "queued",
        "queued_count": len(added),
        "skipped_count": len(skipped),
        "total_symbols": len(symbols),
        "days": days,
        "estimated_minutes": est_min,
        "queue_size": _train_queue.qsize(),
    })


@app.route("/api/webhook", methods=["POST"])
def api_webhook():
    data = request.get_json()
    result = webhook_exec.process_webhook(data)
    return jsonify(result)


@app.route("/api/telegram/config", methods=["POST"])
def api_telegram_config():
    data = request.get_json()
    telegram.save_config(data.get("bot_token", ""), data.get("chat_id", ""))
    return jsonify({"status": "saved", "enabled": telegram.is_enabled()})


@app.route("/api/telegram/status", methods=["GET"])
def api_telegram_status():
    return jsonify({
        "enabled": telegram.is_enabled(),
        "chat_id": telegram.chat_id or "",
        "bot_token": telegram.bot_token or ""
    })


@app.route("/api/telegram/test", methods=["POST"])
def api_telegram_test():
    ok = telegram.send_message("✅ Equity Terminal — Telegram test message OK")
    return jsonify({"success": ok})


@app.route("/api/chart_data")
def api_chart_data():
    """Return cached OHLCV candles for the chart modal."""
    symbol = request.args.get("symbol", "")
    resolution = request.args.get("resolution", "15")
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    try:
        df = data_engine.get_cached(symbol, resolution)
        if df is None or df.empty:
            # Try to fetch fresh data (lightweight, last 2 days)
            df = fyers.get_history(symbol, resolution=resolution, days=2)
        if df is None or df.empty:
            return jsonify({"candles": [], "symbol": symbol})
        # Return last 200 candles
        df = df.tail(200).reset_index(drop=True)
        candles = df[["timestamp","open","high","low","close","volume"]].to_dict(orient="records")
        return jsonify({"candles": candles, "symbol": symbol, "resolution": resolution})
    except Exception as e:
        logger.error(f"chart_data error: {e}")
        return jsonify({"candles": [], "error": str(e)})


def _classify_pattern_shape(pattern_name: str) -> str:
    """Map pattern name → shape class used by the chart overlay renderer."""
    n = pattern_name.lower()
    if any(k in n for k in ("triangle", "wedge", "pennant", "diamond", "broadening")):
        return "triangle"
    if "rectangle" in n:
        return "rectangle"
    if any(k in n for k in ("flag", "channel", "cup")):
        return "channel"
    if any(k in n for k in ("head_shoulder", "double_top", "double_bottom",
                             "triple_top", "triple_bottom", "rounding")):
        return "hs"
    return "horizontal"


def _tl_geometry(tl, sample_indices, df):
    """Return [{time, price}] for a trendline at the given bar indices."""
    if tl is None:
        return []
    pts = []
    seen_ts = set()
    for idx in sample_indices:
        if idx < 0 or idx >= len(df):
            continue
        ts = int(df.iloc[idx]["timestamp"])
        if ts in seen_ts:
            continue
        seen_ts.add(ts)
        pts.append({"time": ts, "price": round(float(tl.price_at(idx)), 2)})
    return pts


@app.route("/api/pattern_overlay")
def api_pattern_overlay():
    """Return detected patterns with trendline geometry for chart shape overlay."""
    symbol = request.args.get("symbol", "")
    resolution = request.args.get("resolution", "15")
    if not symbol:
        return jsonify({"patterns": []}), 400
    try:
        df = data_engine.get_cached(symbol, resolution)
        if df is None or df.empty:
            return jsonify({"patterns": [], "symbol": symbol})

        # Run pattern detection
        from patterns.swing_detector import find_swings
        from patterns.trendline_engine import fit_trendline
        swings = find_swings(df)
        patterns = signal_engine.pattern_detector.detect_all(df, swings)

        results = []
        for p in patterns:
            # Map DataFrame indices to timestamps
            si = max(0, min(p.start_index, len(df) - 1))
            ei = max(0, min(p.end_index, len(df) - 1))
            start_ts = int(df.iloc[si]["timestamp"]) if si < len(df) else 0
            end_ts = int(df.iloc[ei]["timestamp"]) if ei < len(df) else 0

            # Gather swing points within the pattern range for visualization
            pattern_swings = [
                {"time": int(df.iloc[min(s.index, len(df)-1)]["timestamp"]),
                 "price": round(s.price, 2),
                 "type": s.type}
                for s in swings
                if si <= s.index <= ei and s.index < len(df)
            ]

            # ── Trendline geometry for shape overlay ─────────────────────
            # Fit upper (highs) and lower (lows) trendlines through the
            # swings inside the pattern range.
            pat_swing_objs = [s for s in swings if si <= s.index <= ei]
            highs_in = [s for s in pat_swing_objs if s.type == "HIGH"]
            lows_in  = [s for s in pat_swing_objs if s.type == "LOW"]
            upper_tl = fit_trendline(highs_in) if len(highs_in) >= 2 else None
            lower_tl = fit_trendline(lows_in)  if len(lows_in)  >= 2 else None

            # 7 evenly-spaced sample indices across [si, ei]
            span = max(1, ei - si)
            sample_indices = [si + round(span * t / 6) for t in range(7)]
            sample_indices = sorted(set(sample_indices + [si, ei]))

            upper_line = _tl_geometry(upper_tl, sample_indices, df)
            lower_line = _tl_geometry(lower_tl, sample_indices, df)
            shape_type = _classify_pattern_shape(p.pattern_name)

            results.append({
                "name": p.pattern_name,
                "variant": p.variant,
                "direction": p.direction,
                "confidence": round(p.confidence, 3),
                "entry_price": round(p.entry_price, 2),
                "stop_loss": round(p.stop_loss, 2),
                "target_price": round(p.target_price, 2),
                "neckline": round(p.neckline, 2),
                "start_time": start_ts,
                "end_time": end_ts,
                "breakout_confirmed": p.breakout_confirmed,
                "swings": pattern_swings,
                "upper_line": upper_line,
                "lower_line": lower_line,
                "shape_type": shape_type,
            })

        return jsonify({"patterns": results, "symbol": symbol})
    except Exception as e:
        logger.error(f"pattern_overlay error: {e}", exc_info=True)
        return jsonify({"patterns": [], "error": str(e)})


@app.route("/api/config/order_settings", methods=["POST"])
def api_config_order_settings():
    data = request.get_json()
    try:
        mpp = float(data.get("mpp_pct", config.ORDER_MPP_PCT))
        stale = int(data.get("stale_sec", config.MAX_PRICE_STALENESS_SECONDS))
        config.ORDER_MPP_PCT = mpp
        config.MAX_PRICE_STALENESS_SECONDS = stale
        return jsonify({"status": "saved", "mpp_pct": mpp, "stale_sec": stale})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 400


# ============================================================
# SocketIO Events
# ============================================================
@socketio.on("connect")
def on_connect():
    logger.info("Client connected")
    socketio.emit("status", {"connected": True, "running": _poller_running})
    # Replay server-side activity log so browser sees full history after refresh
    if _activity_log:
        socketio.emit("activity_log_replay", {"entries": _activity_log[-100:]})


# ============================================================
# Main Poller Loop
# ============================================================
def _start_poller():
    global _poller_running, _poller_thread
    if _poller_running and _poller_thread and _poller_thread.is_alive():
        logger.info("Poller already running — skipping start")
        return
    _poller_running = True
    _poller_thread = threading.Thread(target=_poller_loop, daemon=True)
    _poller_thread.start()
    logger.info("Poller started")


def _watchdog_loop():
    """Auto-restart the poller if it crashes unexpectedly."""
    while True:
        time.sleep(15)
        if _poller_running and (_poller_thread is None or not _poller_thread.is_alive()):
            logger.warning("Poller thread died — auto-restarting...")
            _start_poller()
            socketio.emit("status", {"connected": True, "running": True, "auto_restarted": True})


def _server_eod_close():
    """SERVER-SIDE end-of-day close for all intraday (5-min) positions.

    Runs every poller cycle.  Triggers at 15:25 IST regardless of whether
    the browser is open.  Intraday SELL positions MUST NOT be held overnight
    in NSE cash market — this is the safety net that enforces it.
    """
    global _eod_closed_today, _eod_last_reset_day

    h, m   = _ist_hm()
    today  = _now_ist().day

    # Reset the flag each new calendar day
    if today != _eod_last_reset_day:
        _eod_closed_today   = False
        _eod_last_reset_day = today

    # Only act 15:25–16:30 IST
    in_window = (h == 15 and m >= 25) or (h == 16 and m <= 30)
    if not in_window or _eod_closed_today:
        return

    intraday = [
        p for p in trade_engine.get_active_positions()
        if getattr(p, "timeframe", "15") == "5"
    ]

    _eod_closed_today = True          # set before closing so we don't double-fire

    if not intraday:
        return

    logger.warning(f"[EOD] Server auto-closing {len(intraday)} intraday position(s) at {_ist_time_str()}")
    symbols_closed = []
    for p in intraday:
        try:
            trade_engine.cancel_position(p.id)
            ticker = (p.symbol or "").replace("NSE:", "").replace("-EQ", "")
            logger.info(f"[EOD] Closed {p.direction} {ticker} @ entry ₹{p.entry_price:.2f}")
            symbols_closed.append(ticker)
            _srv_log(
                f"[EOD AUTO-CLOSE] {p.direction} {ticker} closed at market (15:25 IST cutoff)",
                "warning"
            )
        except Exception as e:
            logger.error(f"[EOD] Failed to close {p.symbol}: {e}")

    if symbols_closed:
        socketio.emit("eod_auto_closed", {
            "count":   len(symbols_closed),
            "symbols": symbols_closed,
            "time":    _ist_time_str(),
        })
        try:
            telegram.send_message(
                f"⚠️ EOD AUTO-CLOSE\n"
                f"Closed {len(symbols_closed)} intraday position(s): {', '.join(symbols_closed)}\n"
                f"Time: {_ist_time_str()}"
            )
        except Exception:
            pass


def _poller_loop():
    global _poller_running

    logger.info(f"Poller running — universe: {watchlist.universe_name}, stocks: {len(watchlist.active_symbols)}")

    # Fetch enricher data once per cycle
    enricher_cache = {}
    enricher_refresh_time = 0

    while _poller_running:
        try:
            cycle_start = time.time()

            # ── Server-side EOD close (runs every cycle, idempotent) ─────────
            _server_eod_close()

            # ── Skip new-signal scanning outside market hours ─────────────────
            # Positions are still updated for LTP/PnL; only signal generation
            # is gated here.  Entry-gate in signal_engine does its own check,
            # but this prevents wasting API quota on dead cycles.
            _h, _m = _ist_hm()
            _market_open  = (_h > 9 or (_h == 9 and _m >= 15))
            _market_close = (_h > 15 or (_h == 15 and _m >= 35))
            _scan_active  = _market_open and not _market_close

            # Refresh enricher data every 5 minutes
            if time.time() - enricher_refresh_time > 300:
                try:
                    enricher_cache = enricher.get_enriched_context_batch()
                    enricher_refresh_time = time.time()
                except Exception as e:
                    logger.error(f"Enricher fetch failed: {e}")

            # Throttle scanner when training is active — share API budget
            # Training uses ~3–5 API calls; scanner gets a smaller batch to avoid
            # hitting Fyers' server-side rate limit ("request limit reached")
            # ── After-hours: skip scanning, just update open positions ──────
            if not _scan_active:
                if trade_engine.get_active_position_count() > 0:
                    active_symbols = [p.symbol for p in trade_engine.get_active_positions()]
                    try:
                        quotes = {}
                        for sym in active_symbols:
                            q = fyers.get_equity_quote(sym)
                            if q:
                                quotes[sym] = q
                        trade_engine.update_positions(quotes)
                    except Exception as e:
                        logger.error(f"After-hours position update error: {e}")
                _emit_dashboard_update([], [])
                time.sleep(15)     # poll slower after hours
                continue

            effective_batch = max(3, config.BATCH_SIZE // 3) if _train_queue_running else config.BATCH_SIZE

            # Get next batch
            batch = watchlist.get_next_batch(effective_batch)
            if not batch:
                time.sleep(config.POLL_INTERVAL)
                continue

            # Fetch OHLCV data (one by one so we can track per-symbol failures)
            batch_data = {}
            for sym in batch:
                try:
                    df = data_engine.fetch_symbol(sym, config.CANDLE_RESOLUTION)
                    if df is not None and len(df) > 0:
                        batch_data[sym] = df
                    else:
                        watchlist.mark_failed(sym, "no data returned")
                except InvalidSymbolError as e:
                    # Permanent error — skip symbol for 30 min immediately
                    watchlist.mark_permanent_failure(sym, str(e))
                except Exception as e:
                    logger.error(f"Unexpected fetch error for {sym}: {e}")
                    watchlist.mark_failed(sym, str(e))

            signals_found = []
            scanner_rows = []

            for symbol, df in batch_data.items():
                if df is None or len(df) < 50:
                    continue

                try:
                    # Get enricher data for this symbol
                    sym_enricher = enricher_cache.get(symbol, {})

                    # Compute indicators
                    df_with_indicators = compute_all(df)

                    # Update regime
                    regime_info = adaptive_bot.update_regime(df_with_indicators)
                    regime = regime_info.get("regime", adaptive_bot.current_regime)

                    # Run full signal evaluation
                    signal = signal_engine.evaluate(
                        symbol=symbol,
                        df=df_with_indicators,
                        enricher_data=sym_enricher,
                        strategy=adaptive_bot.current_strategy,
                    )

                    # Build scanner row (always — even without signal)
                    try:
                        ltp = float(df_with_indicators["close"].iloc[-1])
                    except Exception:
                        ltp = 0
                    try:
                        adx_raw = df_with_indicators["adx"].iloc[-1] if "adx" in df_with_indicators.columns else 0
                        adx_val = 0 if (adx_raw != adx_raw) else float(adx_raw)  # NaN check
                    except Exception:
                        adx_val = 0
                    try:
                        # Use [-2] (last COMPLETED candle) not [-1] (current open candle which has
                        # only partial volume). Average also excludes the current open candle.
                        vol_series = df_with_indicators["volume"]
                        completed_vol = float(vol_series.iloc[-2]) if len(vol_series) >= 2 else float(vol_series.iloc[-1])
                        vol_avg = float(vol_series.iloc[-21:-1].mean()) if len(vol_series) >= 21 else float(vol_series.iloc[:-1].mean()) if len(vol_series) >= 2 else 0
                        vol_ratio = completed_vol / vol_avg if vol_avg > 0 else 1.0
                    except Exception:
                        vol_ratio = 1.0

                    row = {
                        "symbol": symbol,
                        "ltp": round(ltp, 2),
                        "regime": regime,
                        "adx": round(adx_val, 1),
                        "vol_ratio": round(vol_ratio, 2),
                        "conf": 0.0,
                        "dir": "--",
                        "pattern": "--",
                        "lgbm": 0.5,
                        "xgb": 0.5,
                        "arima": "--",
                    }

                    if signal:
                        row.update({
                            "conf": round(signal.confidence, 1),
                            "dir": signal.direction,
                            "pattern": signal.pattern_name or "ML",
                            "lgbm": round(signal.lgbm_prob, 2),
                            "xgb": round(signal.xgb_prob, 2),
                            "arima": signal.arima_trend,
                        })
                        signals_found.append(signal)

                        # Process through trade engine
                        trade_engine.process_signal(signal)

                        # ── Server-side activity log entry ────────────────────
                        _ticker = symbol.replace("NSE:", "").replace("-EQ", "")
                        _tf_lbl = "[5m]" if getattr(signal, "timeframe", "15") == "5" else "[15m]"
                        _grade  = getattr(signal, "quality_grade", "")
                        _srv_log(
                            f"{_tf_lbl} {signal.direction} {_ticker} "
                            f"@ ₹{signal.entry_price:.2f} | "
                            f"conf {signal.confidence:.1f}% | "
                            f"{signal.pattern_name or 'ML'} | "
                            f"grade {_grade}",
                            log_type="trade",
                            trade_ts=getattr(signal, "timestamp", None),
                        )

                        # Send Telegram alert
                        telegram.send_signal_alert(signal)

                        # Prioritize this symbol for next scan
                        watchlist.prioritize(symbol)

                    scanner_rows.append(row)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}", exc_info=True)

            # Update active positions with latest quotes
            if trade_engine.get_active_position_count() > 0:
                active_symbols = [p.symbol for p in trade_engine.get_active_positions()]
                try:
                    quotes = {}
                    for sym in active_symbols:
                        q = fyers.get_equity_quote(sym)
                        if q:
                            quotes[sym] = q
                    trade_engine.update_positions(quotes)
                except Exception as e:
                    logger.error(f"Position update error: {e}")

            # Emit dashboard update
            _emit_dashboard_update(signals_found, scanner_rows)

            # Rate-controlled delay
            elapsed = time.time() - cycle_start
            sleep_time = max(0, config.BATCH_DELAY - elapsed)
            time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Poller loop error: {e}", exc_info=True)
            time.sleep(5)


def _emit_dashboard_update(signals, scanner_rows=None):
    """Push dashboard update via SocketIO."""
    try:
        from dataclasses import asdict
        progress = watchlist.get_rotation_progress()

        # Feed new signals into Telegram commander's buffer (/signals command)
        if signals:
            for sig in signals:
                try:
                    commander.push_signal(sig)
                except Exception:
                    pass

        # Top signals: all signals this cycle sorted by confidence desc (capped at 10)
        top_signals = sorted(
            [s.__dict__ for s in signals] if signals else [],
            key=lambda x: x.get("confidence", 0),
            reverse=True
        )[:10]

        # ── Maintain per-symbol signal cache for /api/chart_data overlay ──
        # Keeps the most recent live signal for each symbol (expires after 15 min).
        global _last_signals_by_symbol
        if "_last_signals_by_symbol" not in globals():
            _last_signals_by_symbol = {}
        now_ts = time.time()
        # evict stale
        _last_signals_by_symbol = {
            k: v for k, v in _last_signals_by_symbol.items()
            if now_ts - v.get("timestamp", 0) < 900
        }
        for s in top_signals:
            sym = s.get("symbol")
            if sym:
                _last_signals_by_symbol[sym] = s

        # Portfolio allocator — rank signals for display
        alloc_result = []
        if signals:
            try:
                current_exp = risk_manager.get_portfolio_summary().get("total_exposure", 0)
                alloc_result = portfolio_allocator.rank_and_allocate(signals, current_exp)
            except Exception:
                pass

        data = {
            "timestamp":         time.time(),
            "active_positions":  [asdict(p) for p in trade_engine.get_active_positions()],
            "pending_positions": [asdict(p) for p in trade_engine.positions.values() if p.status == "PENDING"],
            "closed_positions":  [asdict(p) for p in trade_engine.closed_positions[-50:]],
            "daily_pnl":         trade_engine.get_daily_pnl(),
            "regime":            adaptive_bot.current_regime,
            "strategy":          adaptive_bot.current_strategy,
            "signals":           [s.__dict__ for s in signals] if signals else [],
            "top_signals":       top_signals,
            "portfolio_alloc":   alloc_result,
            "perf_stats":        trade_engine.get_performance_stats(),
            "rate_limiter":      rate_limiter.get_stats(),
            "scan_progress":     progress,
            "scanner_rows":      scanner_rows or [],
        }
        socketio.emit("update", data)
    except Exception as e:
        logger.error(f"Dashboard emit error: {e}")


def _train_model(symbol, days=None):
    """Background model training — emits live progress events at every step."""
    import time as _t
    import numpy as _np
    days = days or config.ML_TRAIN_DAYS
    ticker = symbol.replace("NSE:", "").replace("-EQ", "")
    t0 = _t.time()

    def _prog(stage, detail, pct):
        elapsed = round(_t.time() - t0, 1)
        socketio.emit("training_progress", {
            "symbol": symbol, "stage": stage,
            "detail": detail, "pct": pct, "elapsed": elapsed,
        })
        logger.info(f"[TRAIN {ticker}] {pct}% — {detail}")

    try:
        # ── 1. Fetch OHLCV ─────────────────────────────────────
        _prog("fetching_data", f"Fetching {days} days of OHLCV data from Fyers/cache...", 5)
        df = data_engine.fetch_symbol(symbol, config.CANDLE_RESOLUTION, days=days)
        if df is None or len(df) < 100:
            n = len(df) if df is not None else 0
            socketio.emit("training_error", {"symbol": symbol,
                "error": f"Only {n} candles available — need at least 100. Try fewer days or check authentication."})
            return
        candles = len(df)
        _prog("fetching_data", f"Got {candles:,} candles ({days}d). Building feature matrix...", 10)

        # ── 2. Features ────────────────────────────────────────
        _prog("features", "Computing 60+ technical indicators (TA-Lib)...", 14)
        feature_df = feature_pipeline.build_features(df, [], {}, symbol)
        if len(feature_df) < 100:
            socketio.emit("training_error", {"symbol": symbol, "error": "Feature build produced too few rows"})
            return

        horizon = config.ML_PREDICTION_HORIZON
        close = df["close"].values[-len(feature_df):]
        valid_len = len(close) - horizon
        if valid_len < 80:
            socketio.emit("training_error", {"symbol": symbol, "error": "Not enough data after label creation"})
            return

        labels = _np.array([close[i + horizon] > close[i] for i in range(valid_len)], dtype=_np.int32)
        feature_df = feature_df.iloc[:valid_len]
        feature_pipeline.fit_scaler(symbol, feature_df)
        scaled_df = feature_pipeline.transform(symbol, feature_df)
        X = scaled_df.values.astype(_np.float32)
        y = labels
        n = len(X)
        tr = int(n * 0.70); va = int(n * 0.85)
        X_tr, y_tr = X[:tr], y[:tr]
        X_va, y_va = X[tr:va], y[tr:va]
        _prog("features", f"Feature matrix: {n:,} rows x {X.shape[1]} features. Train/Val/Test split done.", 20)

        results = {}

        # ── 3. LightGBM ────────────────────────────────────────
        _prog("lgbm", f"Training LightGBM on {len(X_tr):,} rows...", 23)
        t1 = _t.time()
        from models.lgbm_model import LightGBMModel
        lgbm = LightGBMModel(); lgbm.build(X_tr.shape[1])
        results["lgbm"] = lgbm.train(X_tr, y_tr, X_va, y_va, list(scaled_df.columns))
        lgbm.save(symbol)
        r_lgbm = results["lgbm"]
        acc_lgbm  = r_lgbm.get("accuracy", r_lgbm.get("test_accuracy", 0))
        auc_lgbm  = r_lgbm.get("auc", 0)
        f1_lgbm   = r_lgbm.get("f1", 0)
        _prog("lgbm", f"LightGBM: {_t.time()-t1:.1f}s | Acc {acc_lgbm*100:.1f}%  AUC {auc_lgbm:.3f}  F1 {f1_lgbm:.3f}", 36)

        # ── 4. XGBoost ─────────────────────────────────────────
        _prog("xgb", f"Training XGBoost on {len(X_tr):,} rows...", 38)
        t1 = _t.time()
        from models.xgboost_model import XGBoostModel
        xgb_m = XGBoostModel(); xgb_m.build(X_tr.shape[1])
        results["xgb"] = xgb_m.train(X_tr, y_tr, X_va, y_va, list(scaled_df.columns))
        xgb_m.save(symbol)
        r_xgb   = results["xgb"]
        acc_xgb = r_xgb.get("accuracy", r_xgb.get("test_accuracy", 0))
        auc_xgb = r_xgb.get("auc", 0)
        f1_xgb  = r_xgb.get("f1", 0)
        _prog("xgb", f"XGBoost: {_t.time()-t1:.1f}s | Acc {acc_xgb*100:.1f}%  AUC {auc_xgb:.3f}  F1 {f1_xgb:.3f}", 52)

        # ── 5. TFT (PyTorch) ───────────────────────────────────
        # NVS315 / CPU-only: TFT with 60-step LSTM sequences is too slow on CPU
        # (3–5 min per epoch × 25 epochs = 1+ hour). Skip TFT on CPU-only systems.
        # TFT only runs when a CUDA GPU with compute capability ≥ 3.0 is available.
        try:
            import torch as _torch
            _has_cuda = _torch.cuda.is_available()
        except ImportError:
            _has_cuda = False

        if _has_cuda:
            lookback = config.ML_LOOKBACK_TIMESTEPS
            X_seq_tr = feature_pipeline.build_sequence(scaled_df.iloc[:tr], lookback)
            y_seq_tr = y[lookback:tr]
            X_seq_va = feature_pipeline.build_sequence(scaled_df.iloc[:va], lookback)
            if len(X_seq_va) > len(X_seq_tr):
                X_seq_va = X_seq_va[len(X_seq_tr):]
            else:
                X_seq_va = _np.array([])
            y_seq_va = y[tr + lookback:va] if tr + lookback < va else _np.array([])

            if len(X_seq_tr) > 10:
                min_tr = min(len(X_seq_tr), len(y_seq_tr))
                X_seq_tr2, y_seq_tr2 = X_seq_tr[:min_tr], y_seq_tr[:min_tr]
                _prog("tft", f"Training TFT on GPU — {min_tr} seq, 25 epochs...", 54)
                t1 = _t.time()
                from models.tft_model import TemporalFusionTransformerModel
                tft = TemporalFusionTransformerModel()

                def _tft_epoch_cb(epoch, total, train_loss, val_loss, stopped_early=False):
                    pct = 54 + int((epoch / total) * 16)
                    vl  = f"  val={val_loss:.4f}" if val_loss is not None else ""
                    _prog("tft", f"TFT epoch {epoch}/{total} loss={train_loss:.4f}{vl} [{_t.time()-t1:.0f}s]", pct)

                tft_kw = dict(on_epoch=_tft_epoch_cb)
                if len(X_seq_va) > 0 and len(y_seq_va) > 0:
                    mv = min(len(X_seq_va), len(y_seq_va))
                    results["tft"] = tft.train(X_seq_tr2, y_seq_tr2, X_seq_va[:mv], y_seq_va[:mv], **tft_kw)
                else:
                    results["tft"] = tft.train(X_seq_tr2, y_seq_tr2, **tft_kw)
                tft.save(symbol)
                _prog("tft", f"TFT done: {_t.time()-t1:.1f}s", 70)
            else:
                results["tft"] = {"error": "insufficient_sequence_data"}
                _prog("tft", "TFT skipped (insufficient sequences)", 70)
        else:
            # CPU-only path: skip TFT, jump straight to ARIMA
            results["tft"] = {"error": "cpu_only_skipped",
                              "note": "TFT requires CUDA GPU (compute ≥3.0). NVS315 not supported."}
            _prog("tft", "TFT skipped (CPU-only system — NVS315 needs CUDA 3.0+)", 70)

        # ── 6. ARIMA ────────────────────────────────────────────
        _prog("arima", "Fitting ARIMA — stepwise order search (p≤2, q≤2, d≤1, last 50 bars, 20s cap)...", 72)
        t1 = _t.time()
        from models.arima_model import ARIMAModel
        arima = ARIMAModel()
        results["arima"] = arima.train(df["close"].values)
        arima.save(symbol)
        _prog("arima", f"ARIMA: {_t.time()-t1:.1f}s | order: {results['arima'].get('order', '?')}", 83)

        # ── 7. Prophet ─────────────────────────────────────────
        _prog("prophet", "Prophet skipped (not available on Python 3.13)", 85)
        results["prophet"] = {"error": "not_installed", "probability": 0.5}

        # ── 8. Meta-learner ────────────────────────────────────
        _prog("meta", "Training stacking meta-learner (LogisticRegression)...", 87)
        t1 = _t.time()
        try:
            from models.meta_learner import MetaLearner
            meta = MetaLearner()
            lgbm_proba = lgbm.model.predict_proba(X)[:, 1] if lgbm.is_trained else _np.full(n, 0.5)
            xgb_proba  = xgb_m.model.predict_proba(X)[:, 1] if xgb_m.is_trained else _np.full(n, 0.5)
            meta_X = _np.column_stack([
                _np.clip(lgbm_proba, 0, 1),
                _np.clip(xgb_proba, 0, 1),
                _np.full(n, 0.5),   # lstm (not installed)
                _np.full(n, 0.5),   # tft predictions
                _np.full(n, 0.5),   # arima placeholder
                _np.full(n, 0.5),   # prophet (not installed)
                _np.full(n, 0.0),   # pattern_confidence
                _np.zeros(n),       # regime_id
            ])
            meta.train(meta_X, y)
            meta.save(symbol)
            results["meta_learner"] = {"trained": True, "time_sec": round(_t.time() - t1, 1)}
        except Exception as me:
            results["meta_learner"] = {"error": str(me)}
        _prog("meta", f"Meta-learner: {_t.time()-t1:.1f}s", 96)

        # ── Done ───────────────────────────────────────────────
        elapsed = round(_t.time() - t0, 1)
        results.update({"total_time_sec": elapsed, "symbol": symbol, "candles": candles, "days": days})
        model_manager.save_metadata(symbol, results)
        model_manager.invalidate(symbol)

        lgbm_acc_str = f"LGB {acc_lgbm*100:.1f}% (AUC {auc_lgbm:.3f})"
        xgb_acc_str  = f"XGB {acc_xgb*100:.1f}% (AUC {auc_xgb:.3f})"
        _prog("complete", f"All models trained in {elapsed}s — {lgbm_acc_str}  {xgb_acc_str}", 100)
        socketio.emit("training_complete", {
            "symbol": symbol, "days": days, "candles": candles,
            "elapsed_sec": elapsed, "results": results,
        })
        logger.info(f"Training complete: {ticker} in {elapsed}s  {lgbm_acc_str}  {xgb_acc_str}")

    except Exception as e:
        logger.error(f"Training failed for {symbol}: {e}", exc_info=True)
        socketio.emit("training_error", {"symbol": symbol, "error": str(e)})


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    logger.info(f"Starting Equity Trading Terminal on port {config.PORT}")

    # Auto-start scanner if token is already valid (user re-launched app)
    if fyers.is_authenticated():
        logger.info("Token already valid — auto-starting scanner")
        _start_poller()

    # Watchdog: silently restart poller if it ever crashes
    threading.Thread(target=_watchdog_loop, daemon=True).start()

    socketio.run(app, host=config.HOST, port=config.PORT, debug=config.DEBUG,
                 allow_unsafe_werkzeug=True)
