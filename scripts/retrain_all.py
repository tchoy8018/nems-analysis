"""
Retrain all models: XGBoost → Prophet → Ensemble → Spike Classifier
Prints comparison table: Model | RMSE before | RMSE after | improvement %
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import get_engine, setup_database
from sqlalchemy import text
from modules.forecasting import (
    train_xgboost,
    train_prophet,
    train_ensemble,
    train_spike_classifier,
)

engine = get_engine()
setup_database(engine)

print("Loading NEMS price data…")
with engine.connect() as conn:
    rows = conn.execute(text(
        "SELECT date, period, usep, demand_mw, solar_mw FROM nems_prices "
        "WHERE usep IS NOT NULL ORDER BY date, period"
    )).fetchall()
import pandas as pd
df = pd.DataFrame(rows, columns=["date", "period", "usep", "demand_mw", "solar_mw"])
df["date"] = pd.to_datetime(df["date"])
print(f"Loaded {len(df):,} rows  ({df['date'].min().date()} → {df['date'].max().date()})\n")

print("=" * 90)
print("RETRAINING — XGBoost (outlier_treatment=exclude) + enhanced holiday features")
print("=" * 90)
xgb_result = train_xgboost(df, engine, outlier_treatment="exclude")

print()
print("=" * 90)
print("RETRAINING — Prophet (daily aggregation + SG holidays + intraday scaling)")
print("=" * 90)
pro_result = train_prophet(df, engine, outlier_treatment="winsorize")

print()
print("=" * 90)
print("COMPUTING — Ensemble inverse-RMSE weights")
print("=" * 90)
ens_result = train_ensemble(engine)

print()
print("=" * 90)
print("TRAINING — Spike Classifier (threshold=200 SGD/MWh)")
print("=" * 90)
clf_result = train_spike_classifier(df, engine, threshold=200.0)

# ── Comparison table ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"{'Model':<18} {'RMSE Before':>12} {'RMSE After':>11} {'Δ %':>8}")
print("-" * 70)

for result, label in [
    (xgb_result, "XGBoost"),
    (pro_result,  "Prophet (daily)"),
]:
    before = result.get("rmse_before")
    after  = result["rmse"]
    if before:
        pct = (before - after) / before * 100
        delta = f"{pct:+.1f}%"
    else:
        delta = "—  (first run)"
    before_str = f"{before:.2f}" if before else "—"
    print(f"  {label:<16} {before_str:>12} {after:>11.2f} {delta:>8}")

if "w_xgboost" in ens_result:
    print(f"  {'Ensemble':<16} {'—':>12}  weights: XGB×{ens_result['w_xgboost']:.3f} + Prophet×{ens_result['w_prophet']:.3f}")

if "f1" in clf_result:
    print(f"  {'Spike Clf':<16} {'—':>12}  F1={clf_result['f1']:.3f}  prec={clf_result['precision']:.3f}  rec={clf_result['recall']:.3f}")

print("=" * 70)
