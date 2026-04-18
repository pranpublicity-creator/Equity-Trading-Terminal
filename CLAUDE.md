# Equity Trading Terminal — NSE Cash Market

## Project Purpose
Flask web application for **intraday equity trading on NSE cash market** with
AI-powered pattern detection, ML signal fusion, and rate-limited execution
through the Fyers API v3.

- **NIFTY 50 / NIFTY 100 / NIFTY 200** (or custom watchlist)
- **15-minute primary timeframe** with multi-TF context (5/15/60)
- **7-model fusion**: LightGBM + XGBoost + LSTM + TFT + ARIMA + Prophet + chart-pattern detector
- **6-point trade-quality scorecard** (A–F) gates every AUTO trade
- **7 market regimes** detected per-symbol; weights re-balanced by regime
- **6 strategy presets** (Pattern Breakout / Trend Follow / Mean Reversion /
  Momentum / ML Ensemble / Full Fusion)
- **Multi-pattern detector**: H&S, double/triple top/bottom, rounding, wedge,
  diamond, broadening, ascending/descending/symmetric triangle, rectangle,
  flag, pennant, cup-and-handle, measured move
- **Bayesian strategy selector** that learns per (sector, regime) which preset
  has the best historical win-rate
- **Live dashboard** with Lightweight Charts pattern overlays, scanner table,
  top-signals strip, position cards, performance panel, Telegram alerts

## Pipeline (3 layers)
```
                ┌────────────────────────────────────────────┐
                │  GATE   — market hours / holiday / circuit │
                └────────────────────────────────────────────┘
                                      │
                ┌────────────────────────────────────────────┐
                │  QUAL   — ADX floor / volume / gap /       │
                │           market-breadth (A/D ratio)       │
                └────────────────────────────────────────────┘
                                      │
                ┌────────────────────────────────────────────┐
                │  TRIG   — pattern detector                 │
                │       → feature pipeline                   │
                │       → 7-model ensemble                   │
                │       → SignalFusion (regime-weighted)     │
                │       → cooldown / R:R / entry-gate        │
                │       → pattern-instance dedup (per day)   │
                │       → TradeQuality scorecard (6 checks)  │
                │       → TradeEngine (AUTO gate by grade)   │
                └────────────────────────────────────────────┘
```

## Architecture
```
app.py (Flask + SocketIO :5005)
    │
    ├── core/
    │     ├── fyers_manager.py        (OAuth2, quotes, history, orders)
    │     ├── data_engine.py          (OHLCV + indicator pipeline + SQLite cache)
    │     ├── market_data_enricher.py (PCR, FII/DII, A/D, delivery%, sector)
    │     ├── stock_universe.py       (NIFTY_50 / 100 / 200 / CUSTOM)
    │     ├── watchlist_manager.py    (active scan rotation)
    │     └── rate_limiter.py         (8/sec, 150/min — under Fyers cap)
    │
    ├── patterns/
    │     ├── swing_detector.py       (fractal pivot detection)
    │     ├── trendline_engine.py     (linear-regression trendlines)
    │     ├── reversal_patterns.py    (H&S, double/triple top/bottom, rounding)
    │     ├── continuation_patterns.py(flag, pennant, measured_move)
    │     ├── breakout_patterns.py    (triangles, rectangle, cup_handle)
    │     ├── volatility_patterns.py  (wedge, diamond, broadening)
    │     ├── pattern_validator.py    (volume / breakout confirmation)
    │     ├── multi_tf_validator.py   (cross-timeframe agreement)
    │     └── pattern_detector.py     (orchestrator → returns ranked list)
    │
    ├── features/
    │     └── feature_pipeline.py     (RSI/MACD/BB/ATR/ADX/VWAP/OBV + lags)
    │
    ├── models/
    │     ├── lgbm_model.py           (LightGBM tabular classifier)
    │     ├── xgboost_model.py        (XGBoost tabular classifier)
    │     ├── lstm_model.py           (sequence model)
    │     ├── tft_model.py            (Temporal Fusion Transformer)
    │     ├── arima_model.py          (trend / direction)
    │     ├── prophet_model.py        (seasonality)
    │     ├── meta_learner.py         (logistic regression over base preds)
    │     ├── model_trainer.py        (parallel train + walk-forward)
    │     └── model_manager.py        (load/predict per symbol)
    │
    ├── signals/
    │     ├── signal_engine.py        (3-layer pipeline; THE entry point)
    │     ├── signal_fusion.py        (regime-weighted score + penalties)
    │     ├── regime_detector.py      (7 equity regimes)
    │     ├── entry_gate.py           (final pre-execution validation)
    │     ├── strategy_presets.py     (6 named weight schemes)
    │     └── trade_quality.py        (6-point pre-trade scorecard)
    │
    ├── trading/
    │     ├── trade_engine.py         (lifecycle: PENDING → ACTIVE → CLOSED)
    │     ├── risk_manager.py         (position sizing, exposure caps)
    │     ├── portfolio_allocator.py  (rank signals across portfolio)
    │     ├── charge_calculator.py    (STT / GST / brokerage / SEBI)
    │     ├── telegram_notifier.py    (signal + trade alerts)
    │     ├── webhook_executor.py     (TradingView webhook bridge)
    │     └── generate_report.py      (XLSX trade report)
    │
    ├── adaptive/
    │     ├── adaptive_bot.py         (auto-adapt strategy by regime)
    │     └── bayesian_selector.py    (per sector × regime priors)
    │
    ├── backtest/
    │     └── equity_backtest.py      (walk-forward replay)
    │
    ├── reports/
    │     └── generate_cheat_sheet.py (this XLSX cheat sheet generator)
    │
    ├── templates/index.html          (Tailwind dashboard, Lightweight Charts)
    └── static/                       (JS bundles, CSS, icons)
```

## Key Files
| File | Purpose |
|------|---------|
| `app.py` | Flask routes, SocketIO push, market poller (3 s) |
| `config.py` | All knobs — credentials, thresholds, charges, paths |
| `signals/signal_engine.py` | Heart of the pipeline (GATE → QUAL → TRIG) |
| `signals/signal_fusion.py` | Regime-weighted score across 7 models + patterns |
| `signals/trade_quality.py` | 6-point scorecard, A–F grade |
| `trading/trade_engine.py` | Position lifecycle + AUTO quality gate + cooldowns |
| `patterns/pattern_detector.py` | Orchestrates ~17 chart-pattern detectors |
| `models/model_manager.py` | Loads & runs per-symbol trained models |
| `core/data_engine.py` | OHLCV cache + indicator computation |
| `templates/index.html` | Single-page dashboard (Tailwind + LW Charts) |

## Risk & Safety Rules
1. **Pattern-instance dedup (per day)** — `(symbol, pattern_name, round(trigger,1), direction)`; the same neckline/breakdown fires only ONCE per IST day.
2. **Signal cooldown** — `SIGNAL_ENTRY_COOLDOWN = 1800 s` between any two signals on the same symbol.
3. **Post-trade cooldown** — `POST_TRADE_COOLDOWN_SEC = 1800 s` after close, **doubled per consecutive loss** (capped 4 h).
4. **Duplicate-symbol guard** — never two ACTIVE/PENDING positions on the same stock.
5. **AUTO quality gate** — `TRADE_QUALITY_MIN_SCORE = 0.60` (grade ≥ C). Failing signals are dropped in AUTO, queued in MANUAL.
6. **Daily loss limit** — `MAX_LOSS_PER_DAY = ₹5,000`; new trades blocked once breached.
7. **Entry-drift cap** — `MAX_ENTRY_DRIFT_PCT = 0.8 %`; suppress stale breakouts.
8. **Min R:R** — `MIN_RISK_REWARD = 1.0` for entry; quality scorer separately requires `TRADE_QUALITY_MIN_RR = 1.5`.
9. **Trailing stop** — activates at `1.5 × initial_risk` gained, trails by `1.0 × ATR`.
10. **Position size** — capped by `MAX_PORTFOLIO_RISK = 5 %` and `MAX_CAPITAL_PER_STOCK = 10 %`.
11. **Rate limit** — 8 req/sec, 150 req/min, batches of 15, 1.5 s delay (under Fyers' 10/sec / 200/min limits).
12. **No-signal windows** — first 5 min after open, last 10 min before close.

## How to Run
```bash
cd "C:\Users\pc\.claude\TRADING DATA\EQUITY TERMINAL"
python app.py
# Open http://localhost:5005
```
First run: open `/auth`, paste Fyers App ID + Secret, complete OAuth → redirect
copies the `auth_code` back to `/auth/callback`.

## Default Strategy
`FUSION_FULL` — balanced 7-model fusion, `min_confidence = 50`.

## Default Universe
`NIFTY_200` (200 large-cap NSE equities) scanned in batches.

## Generate the cheat-sheet
```bash
python reports/generate_cheat_sheet.py
# → reports/Strategy_Cheat_Sheet_Equity_Terminal.xlsx
```
