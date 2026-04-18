"""
Quick startup check -- run this before app.py to catch import errors.
Usage: python check_imports.py
"""
import sys
print(f"Python {sys.version}")
errors = []

def try_import(name):
    try:
        __import__(name)
        print(f"  [OK] {name}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {name}: {e}")
        errors.append(f"{name}: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        errors.append(f"{name}: {e}")
        return False

print("\n=== Core dependencies (required) ===")
try_import("flask")
try_import("flask_socketio")
try_import("fyers_apiv3")
try_import("numpy")
try_import("pandas")
try_import("sklearn")
try_import("lightgbm")
try_import("xgboost")
try_import("pmdarima")
try_import("requests")
try_import("bs4")
try_import("joblib")

print("\n=== Optional dependencies (missing = graceful fallback, not fatal) ===")
for name in ["talib", "torch", "tensorflow", "prophet"]:
    try:
        __import__(name)
        print(f"  [OK] {name}")
    except ImportError:
        print(f"  [--] {name}: not installed (fallback active - ok)")

print("\n=== App modules ===")
try_import("config")
try_import("core.rate_limiter")
try_import("core.fyers_manager")
try_import("core.data_engine")
try_import("core.watchlist_manager")
try_import("core.market_data_enricher")
try_import("core.stock_universe")
try_import("features.indicator_engine")
try_import("features.feature_pipeline")
try_import("patterns.pattern_detector")
try_import("models.model_manager")
try_import("signals.signal_engine")
try_import("signals.signal_fusion")
try_import("adaptive.adaptive_bot")
try_import("trading.trade_engine")
try_import("trading.risk_manager")
try_import("trading.charge_calculator")
try_import("trading.telegram_notifier")

print("\n" + ("=" * 40))
if errors:
    print(f"FAILED: {len(errors)} required import error(s):")
    for e in errors:
        print(f"   {e}")
    sys.exit(1)
else:
    print("ALL OK -- safe to run: python app.py")
