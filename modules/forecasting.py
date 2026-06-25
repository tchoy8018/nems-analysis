"""
NEMS Price Forecasting — XGBoost + Prophet + Ensemble + Spike Classifier
Chronological train/test split only. No lookahead.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from modules.utils import (
    HOLIDAY_NAMES,
    SG_PUBLIC_HOLIDAYS,
    _is_sg_public_holiday,
)

warnings.filterwarnings("ignore", category=FutureWarning)

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS_BASE = [
    "period", "hour", "day_of_week", "month", "is_weekend", "is_public_holiday_sg",
    "dt_public_holiday", "dt_saturday", "dt_sunday", "dt_weekday_wfh",
    "days_to_next_holiday",
    "period_sin", "period_cos", "month_sin", "month_cos",
    "usep_lag_48", "usep_lag_336", "usep_lag_672",
    "usep_rolling_mean_48", "usep_rolling_std_48",
    "demand_lag_48", "solar_lag_48",
]
GAS_FEATURE_COLS = [
    "gas_price_monthly", "gas_price_lag1m", "gas_price_lag2m",
    "implied_usep_floor", "lng_share",
]
FEATURE_COLS = FEATURE_COLS_BASE  # updated in build_features if gas data available
TARGET = "usep"

# Prophet holidays DataFrame built once at import
_PH_ROWS: list[dict] = [
    {"holiday": name, "ds": pd.to_datetime(ds), "lower_window": -1, "upper_window": 1}
    for ds, name in HOLIDAY_NAMES.items()
]
PROPHET_HOLIDAYS_DF = pd.DataFrame(_PH_ROWS)


def _load_gas_monthly(engine) -> pd.DataFrame:
    """Load monthly gas price data for feature joining. Returns empty df if unavailable."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT price_date, weighted_avg_sgd_mmbtu,
                       implied_usep_floor_sgd_mwh, lng_share_pct
                FROM gas_prices
                WHERE weighted_avg_sgd_mmbtu IS NOT NULL
                ORDER BY price_date
            """)).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["date", "gas_sgd", "implied_floor", "lng_share"])
        df["date"] = pd.to_datetime(df["date"])
        df["ym"]   = df["date"].dt.strftime("%Y-%m")
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, engine=None) -> pd.DataFrame:
    """
    Add time + lag + holiday + gas features to a NEMS DataFrame.

    Input columns: date, period (1-48), usep, demand_mw, solar_mw.
    If engine provided and gas_prices has data, gas features are added.
    Returns copy with NaN rows (from lags) dropped.
    """
    global FEATURE_COLS

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["date", "period"]).reset_index(drop=True)
    d["dt"] = d["date"] + pd.to_timedelta((d["period"] - 1) * 30, unit="m")

    # ── Time features ──
    d["hour"]        = d["period"].apply(lambda p: ((p - 1) * 30) // 60)
    d["day_of_week"] = d["date"].dt.dayofweek
    d["month"]       = d["date"].dt.month
    d["is_weekend"]  = d["day_of_week"].isin([5, 6]).astype(int)

    # ── Holiday features (hardcoded MOM dict) ──
    dates = d["date"].dt.date
    d["is_public_holiday_sg"] = dates.apply(lambda x: int(_is_sg_public_holiday(x)))

    # day_type one-hot (weekday_core is baseline, omitted)
    dow      = d["day_of_week"]
    is_ph    = d["is_public_holiday_sg"].astype(bool)
    d["dt_public_holiday"] = is_ph.astype(int)
    d["dt_saturday"]       = (~is_ph & (dow == 5)).astype(int)
    d["dt_sunday"]         = (~is_ph & (dow == 6)).astype(int)
    d["dt_weekday_wfh"]    = (~is_ph & dow.isin([0, 4])).astype(int)

    # days_to_next_holiday: vectorised via numpy searchsorted on precomputed ordinals
    from modules.utils import _PH_ORDS
    ords = dates.apply(lambda x: x.toordinal()).values.astype(np.int32)
    idx_next = np.searchsorted(_PH_ORDS, ords)
    idx_next = np.clip(idx_next, 0, len(_PH_ORDS) - 1)
    days_to  = np.where(idx_next < len(_PH_ORDS), _PH_ORDS[idx_next] - ords, 999)
    d["days_to_next_holiday"] = days_to.astype(int)

    d["period_sin"] = np.sin(2 * np.pi * d["period"] / 48)
    d["period_cos"] = np.cos(2 * np.pi * d["period"] / 48)
    d["month_sin"]  = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"]  = np.cos(2 * np.pi * d["month"] / 12)

    # ── Lag features ──
    d = d.set_index("dt").sort_index()

    d["usep_lag_48"]  = d["usep"].shift(48)
    d["usep_lag_336"] = d["usep"].shift(336)
    d["usep_lag_672"] = d["usep"].shift(672)

    d["usep_rolling_mean_48"] = d["usep"].rolling(48).mean().shift(1)
    d["usep_rolling_std_48"]  = d["usep"].rolling(48).std().shift(1)

    d["demand_lag_48"] = d["demand_mw"].shift(48) if "demand_mw" in d.columns else np.nan
    d["solar_lag_48"]  = d["solar_mw"].shift(48)  if "solar_mw"  in d.columns else np.nan

    d = d.reset_index()

    required_lags = ["usep_lag_48", "usep_lag_336", "usep_rolling_mean_48"]
    d = d.dropna(subset=required_lags + [TARGET]).reset_index(drop=True)

    # ── Gas features (optional) ──
    active_features = list(FEATURE_COLS_BASE)
    gas_df = _load_gas_monthly(engine) if engine is not None else pd.DataFrame()
    if not gas_df.empty:
        d["ym"] = d["date"].dt.strftime("%Y-%m")

        gas_l0 = gas_df[["ym", "gas_sgd", "implied_floor", "lng_share"]].rename(
            columns={"gas_sgd": "gas_price_monthly"})
        d = d.merge(gas_l0, on="ym", how="left")

        gas_df["ym_lag1"] = (gas_df["date"] + pd.DateOffset(months=1)).dt.strftime("%Y-%m")
        gas_l1 = gas_df[["ym_lag1", "gas_sgd"]].rename(
            columns={"ym_lag1": "ym", "gas_sgd": "gas_price_lag1m"})
        d = d.merge(gas_l1, on="ym", how="left")

        gas_df["ym_lag2"] = (gas_df["date"] + pd.DateOffset(months=2)).dt.strftime("%Y-%m")
        gas_l2 = gas_df[["ym_lag2", "gas_sgd"]].rename(
            columns={"ym_lag2": "ym", "gas_sgd": "gas_price_lag2m"})
        d = d.merge(gas_l2, on="ym", how="left")

        d["implied_usep_floor"] = d.get("implied_floor", np.nan)

        for gcol in GAS_FEATURE_COLS:
            if gcol in d.columns:
                d[gcol] = d[gcol].fillna(d[gcol].median())
                if gcol not in active_features:
                    active_features.append(gcol)

        FEATURE_COLS = active_features

    for col in active_features:
        if col in d.columns and d[col].isna().any():
            d[col] = d[col].fillna(d[col].median())

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 2. Price period classification (outlier handling)
# ─────────────────────────────────────────────────────────────────────────────

def classify_price_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify USEP periods into price regimes.

    Adds columns:
        price_regime      — 'normal' | 'elevated' | 'supply_shock' | 'map_cap_event'
        include_in_training — bool  (False for supply_shock / map_cap_event)
        spike_cluster_id  — int or NaN (sequential cluster ID for spike runs >200)

    MAP cap in SG is $4,500/MWh; supply_shock threshold 4× rolling mean & >300.
    """
    d = df.copy().sort_values(["date", "period"]).reset_index(drop=True)

    usep       = d["usep"].values.astype(float)
    roll_mean  = pd.Series(usep).rolling(48, min_periods=24).mean().values
    roll_mean  = np.where(roll_mean < 1.0, 1.0, roll_mean)
    shock_ratio = usep / roll_mean

    regime = np.full(len(d), "normal", dtype=object)
    regime[usep > 100]                              = "elevated"
    regime[(usep > 300) & (shock_ratio > 4.0)]     = "supply_shock"
    regime[(usep > 1000) | (shock_ratio > 10.0)]   = "map_cap_event"

    d["price_regime"]       = regime
    d["include_in_training"] = ~pd.Series(regime).isin(["supply_shock", "map_cap_event"])

    # Spike cluster IDs — contiguous runs where USEP > 200
    is_spike = (usep > 200).astype(int)
    cluster_change = pd.Series(is_spike).diff().fillna(0)
    cluster_id = (cluster_change > 0).cumsum()
    d["spike_cluster_id"] = np.where(is_spike, cluster_id.values, np.nan)

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metrics + registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    res  = y_true - y_pred
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae  = float(np.mean(np.abs(res)))
    nz   = y_true != 0
    mape = float(np.mean(np.abs(res[nz] / y_true[nz])) * 100) if nz.any() else np.nan
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 2)}


def _register_model(engine, model_name: str, metrics: dict,
                    training_rows: int, model_path: str) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE model_registry SET is_active = 0 WHERE model_name = :n"
        ), {"n": model_name})
        conn.execute(text("""
            INSERT INTO model_registry
              (model_name, trained_at, training_rows, rmse, mae, mape, model_path, is_active)
            VALUES (:n, :t, :tr, :rmse, :mae, :mape, :mp, 1)
        """), {
            "n": model_name, "t": now, "tr": training_rows,
            "rmse": metrics["rmse"], "mae": metrics["mae"], "mape": metrics["mape"],
            "mp": model_path,
        })


def _get_model_rmse(engine, model_name: str) -> float | None:
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT rmse FROM model_registry WHERE model_name=:n AND is_active=1"
            ), {"n": model_name}).fetchone()
        return float(row[0]) if row else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(
    df: pd.DataFrame,
    engine,
    outlier_treatment: str = "exclude",
) -> dict:
    """
    Train XGBoost on 80% chronological split.

    outlier_treatment:
        'exclude'     — remove supply_shock / map_cap_event rows from training
        'winsorize'   — cap usep at 99th percentile
        'include_all' — no treatment (original behaviour)
    """
    from xgboost import XGBRegressor

    feat_df = build_features(df, engine=engine)

    # Capture baseline RMSE before retraining
    baseline_rmse = _get_model_rmse(engine, "xgboost")

    # Outlier treatment on training data
    if outlier_treatment == "exclude":
        classified = classify_price_periods(feat_df)
        feat_df = classified[classified["include_in_training"]].copy()
    elif outlier_treatment == "winsorize":
        q99 = feat_df["usep"].quantile(0.99)
        feat_df = feat_df.copy()
        feat_df["usep"] = feat_df["usep"].clip(None, q99)

    split = int(len(feat_df) * 0.8)
    train = feat_df.iloc[:split]
    test  = feat_df.iloc[split:]

    avail = [c for c in FEATURE_COLS if c in feat_df.columns]
    X_tr, y_tr = train[avail].values, train[TARGET].values
    X_te, y_te = test[avail].values,  test[TARGET].values

    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred = model.predict(X_te)
    m = _metrics(y_te, y_pred)

    importance = sorted(
        zip(avail, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )[:15]
    top_features = [{"feature": f, "importance": round(float(v), 6)} for f, v in importance]

    path = str(MODELS_DIR / "xgboost_model.joblib")
    joblib.dump({"model": model, "features": avail}, path)
    _register_model(engine, "xgboost", m, len(train), path)

    # Print comparison table row
    before_str = f"{baseline_rmse:.2f}" if baseline_rmse else "—"
    improvement = ""
    if baseline_rmse:
        pct = (baseline_rmse - m["rmse"]) / baseline_rmse * 100
        improvement = f"{pct:+.1f}%"
    print(
        f"XGBoost  | treatment={outlier_treatment:12s} | "
        f"train={len(train):,}  test={len(test):,} | "
        f"RMSE before={before_str}  after={m['rmse']:.2f}  {improvement} | "
        f"MAE={m['mae']:.2f}  MAPE={m['mape']:.1f}%"
    )

    return {
        "model_name": "xgboost", "training_rows": len(train), "test_rows": len(test),
        "top_features": top_features,
        "rmse_before": baseline_rmse,
        **m,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Prophet (daily + intraday scaling)
# ─────────────────────────────────────────────────────────────────────────────

def train_prophet(
    df: pd.DataFrame,
    engine,
    outlier_treatment: str = "winsorize",
) -> dict:
    """
    Train Prophet on daily USEP averages with SG public holidays.
    Intraday profile (avg per-period ratio) is saved and applied at predict time.

    outlier_treatment: 'winsorize' (default) | 'exclude' | 'include_all'
    """
    from prophet import Prophet

    feat_df = build_features(df, engine=engine)
    baseline_rmse = _get_model_rmse(engine, "prophet")

    # Compute intraday profile from clean data (before treatment)
    period_mean  = feat_df.groupby("period")["usep"].mean()
    overall_mean = feat_df["usep"].mean()
    intraday_profile = (period_mean / overall_mean).to_dict()

    if outlier_treatment == "exclude":
        classified = classify_price_periods(feat_df)
        feat_df = classified[classified["include_in_training"]].copy()
    elif outlier_treatment == "winsorize":
        q99 = feat_df["usep"].quantile(0.99)
        feat_df = feat_df.copy()
        feat_df["usep"] = feat_df["usep"].clip(None, q99)

    # Aggregate to daily
    feat_df["date_only"] = feat_df["dt"].dt.date
    agg_dict: dict = {"usep": ("usep", "mean")}
    for gcol in ["gas_price_monthly", "implied_usep_floor"]:
        if gcol in feat_df.columns:
            agg_dict[gcol] = (gcol, "first")
    daily = feat_df.groupby("date_only").agg(**agg_dict).reset_index()
    daily["ds"] = pd.to_datetime(daily["date_only"])
    daily["y"]  = daily["usep"]

    split      = int(len(daily) * 0.8)
    train_day  = daily.iloc[:split]
    test_day   = daily.iloc[split:]

    regressors = [
        c for c in ["gas_price_monthly"]
        if c in daily.columns and daily[c].notna().mean() > 0.5
    ]

    model = Prophet(
        holidays=PROPHET_HOLIDAYS_DF,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
    )
    for reg in regressors:
        model.add_regressor(reg)

    fit_cols = ["ds", "y"] + regressors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_day[fit_cols])

    future = test_day[["ds"] + regressors].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = model.predict(future)

    m = _metrics(test_day["y"].values, forecast["yhat"].values)

    path = str(MODELS_DIR / "prophet_model.joblib")
    joblib.dump({
        "model":            model,
        "regressors":       regressors,
        "intraday_profile": intraday_profile,
    }, path)
    _register_model(engine, "prophet", m, len(train_day), path)

    before_str = f"{baseline_rmse:.2f}" if baseline_rmse else "—"
    improvement = ""
    if baseline_rmse:
        pct = (baseline_rmse - m["rmse"]) / baseline_rmse * 100
        improvement = f"{pct:+.1f}%"
    print(
        f"Prophet  | treatment={outlier_treatment:12s} | "
        f"train={len(train_day):,} days  test={len(test_day):,} days | "
        f"RMSE before={before_str}  after={m['rmse']:.2f}  {improvement}  (daily avg) | "
        f"MAE={m['mae']:.2f}  MAPE={m['mape']:.1f}%"
    )

    return {
        "model_name": "prophet", "training_rows": len(train_day), "test_rows": len(test_day),
        "top_features": regressors,
        "rmse_before": baseline_rmse,
        **m,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Ensemble (inverse-RMSE weights)
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble(engine) -> dict:
    """
    Compute inverse-RMSE blending weights from xgboost + prophet in model_registry.
    Registers 'ensemble' in model_registry. Returns {w_xgboost, w_prophet, blended_rmse}.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT model_name, rmse FROM model_registry
            WHERE model_name IN ('xgboost', 'prophet') AND is_active = 1
        """)).mappings().fetchall()

    rmses = {r["model_name"]: float(r["rmse"]) for r in rows if r["rmse"]}
    if len(rmses) < 2:
        print("Ensemble | SKIPPED — need both xgboost and prophet trained first")
        return {"error": "Need both xgboost and prophet trained first"}

    inv   = {k: 1.0 / max(v, 0.01) for k, v in rmses.items()}
    total = sum(inv.values())
    w     = {k: v / total for k, v in inv.items()}

    w_xgb = w.get("xgboost", 0.5)
    w_pro  = w.get("prophet",  0.5)
    blended_rmse = rmses["xgboost"] * w_xgb + rmses["prophet"] * w_pro

    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    with engine.begin() as conn:
        conn.execute(text("UPDATE model_registry SET is_active=0 WHERE model_name='ensemble'"))
        conn.execute(text("""
            INSERT INTO model_registry
              (model_name, trained_at, training_rows, rmse, mae, mape, model_path, is_active)
            VALUES ('ensemble', :t, 0, :rmse, 0, 0, 'inverse_rmse_blend', 1)
        """), {"t": now, "rmse": round(blended_rmse, 2)})

    print(
        f"Ensemble | w_xgboost={w_xgb:.3f}  w_prophet={w_pro:.3f}  "
        f"blended RMSE≈{blended_rmse:.2f}  "
        f"({rmses['xgboost']:.2f} × {w_xgb:.3f} + {rmses['prophet']:.2f} × {w_pro:.3f})"
    )

    return {
        "w_xgboost":    round(w_xgb, 4),
        "w_prophet":    round(w_pro, 4),
        "blended_rmse": round(blended_rmse, 2),
        "rmse_xgboost": rmses["xgboost"],
        "rmse_prophet": rmses["prophet"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Spike classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_spike_classifier(
    df: pd.DataFrame,
    engine,
    threshold: float = 200.0,
) -> dict:
    """
    XGBoost binary classifier: P(USEP > threshold).
    Stored as 'spike_classifier' in model_registry.
    Registry columns: rmse=1-F1, mae=1-precision, mape=1-recall (for display).
    """
    from xgboost import XGBClassifier

    feat_df = build_features(df, engine=engine)
    split   = int(len(feat_df) * 0.8)
    train   = feat_df.iloc[:split]
    test    = feat_df.iloc[split:]

    avail = [c for c in FEATURE_COLS if c in feat_df.columns]

    y_tr = (train[TARGET].values > threshold).astype(int)
    y_te = (test[TARGET].values  > threshold).astype(int)

    spike_rate       = float(y_tr.mean())
    scale_pos_weight = (1 - spike_rate) / max(spike_rate, 0.001)

    clf = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1, verbosity=0,
        eval_metric="logloss",
    )
    clf.fit(
        train[avail].values, y_tr,
        eval_set=[(test[avail].values, y_te)],
        verbose=False,
    )

    y_prob    = clf.predict_proba(test[avail].values)[:, 1]
    y_pred_b  = (y_prob > 0.5).astype(int)
    tp        = int(np.sum((y_pred_b == 1) & (y_te == 1)))
    precision = tp / max(int(y_pred_b.sum()), 1)
    recall    = tp / max(int(y_te.sum()), 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    path = str(MODELS_DIR / "spike_classifier.joblib")
    joblib.dump({"model": clf, "features": avail, "threshold": threshold}, path)

    # Store as pseudo-metrics so model status panel can display them
    m = {
        "rmse": round(1 - f1, 4),
        "mae":  round(1 - precision, 4),
        "mape": round(1 - recall, 4),
    }
    _register_model(engine, "spike_classifier", m, len(train), path)

    print(
        f"SpikeClf | threshold={threshold:.0f}  spike_rate={spike_rate:.3f}  "
        f"precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}"
    )

    return {
        "model_name":  "spike_classifier",
        "threshold":   threshold,
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "f1":          round(f1, 4),
        "spike_rate":  round(spike_rate, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Spike probability helpers
# ─────────────────────────────────────────────────────────────────────────────

def spike_probability(engine, period: int, month: int, threshold: float = 200) -> float:
    """Historical P(USEP > threshold) for a given (period, month)."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                SUM(CASE WHEN usep > :thr THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS spike_prob
            FROM nems_prices
            WHERE period = :p
              AND CAST(strftime('%m', date) AS INTEGER) = :m
              AND usep IS NOT NULL
        """), {"thr": threshold, "p": period, "m": month}).mappings().fetchone()
    return float(row["spike_prob"] or 0.0)


def spike_exceedance_curve(engine, period: int, month: int) -> pd.DataFrame:
    """
    Compute P(USEP > t) for a range of thresholds — used for the fan/risk chart.
    Returns: threshold, exceedance_prob
    """
    thresholds = [50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000]
    rows = [
        {"threshold": t, "exceedance_prob": spike_probability(engine, period, month, float(t))}
        for t in thresholds
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Predict
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_recent_history(engine, n: int = 700) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, period, usep, demand_mw, solar_mw
            FROM nems_prices WHERE usep IS NOT NULL
            ORDER BY date DESC, period DESC LIMIT :n
        """), {"n": n}).fetchall()
    df = pd.DataFrame(rows, columns=["date", "period", "usep", "demand_mw", "solar_mw"])
    return df.sort_values(["date", "period"]).reset_index(drop=True)


def predict(model_name: str, engine, n_periods: int = 48,
            spike_threshold: float = 200.0) -> pd.DataFrame:
    """
    Generate forecast for next n_periods using saved model.
    Returns: dt, period, time_label, predicted_usep, lower_bound, upper_bound, spike_prob
    """
    hist    = _fetch_recent_history(engine)
    feat_df = build_features(hist, engine=engine)
    if feat_df.empty:
        return pd.DataFrame()

    last_dt    = pd.Timestamp(feat_df["dt"].max())

    if model_name == "xgboost":
        payload    = joblib.load(MODELS_DIR / "xgboost_model.joblib")
        xgb_model  = payload["model"]
        feat_names = payload["features"]

        rmse_ci  = _get_model_rmse(engine, "xgboost") or 40.0
        last_row = feat_df.iloc[-1:][feat_names].values.copy()
        feat_idx = {f: j for j, f in enumerate(feat_names)}

        results = []
        for i in range(n_periods):
            dt_i     = last_dt + pd.Timedelta(minutes=30 * (i + 1))
            period_i = ((dt_i.hour * 60 + dt_i.minute) // 30) + 1
            row = last_row.copy()
            for feat, val in [
                ("period",      period_i),
                ("hour",        dt_i.hour),
                ("day_of_week", dt_i.dayofweek),
                ("month",       dt_i.month),
                ("is_weekend",  int(dt_i.dayofweek >= 5)),
                ("period_sin",  np.sin(2 * np.pi * period_i / 48)),
                ("period_cos",  np.cos(2 * np.pi * period_i / 48)),
                ("month_sin",   np.sin(2 * np.pi * dt_i.month / 12)),
                ("month_cos",   np.cos(2 * np.pi * dt_i.month / 12)),
            ]:
                if feat in feat_idx:
                    row[0, feat_idx[feat]] = val

            yhat = float(xgb_model.predict(row)[0])
            sp   = spike_probability(engine, period_i, dt_i.month, spike_threshold)
            results.append({
                "dt":             dt_i,
                "period":         period_i,
                "time_label":     f"{dt_i.hour:02d}:{dt_i.minute:02d}",
                "predicted_usep": round(max(0.0, yhat), 2),
                "lower_bound":    round(max(0.0, yhat - 1.5 * rmse_ci), 2),
                "upper_bound":    round(yhat + 1.5 * rmse_ci, 2),
                "spike_prob":     round(sp, 4),
            })
        return pd.DataFrame(results)

    elif model_name == "prophet":
        payload          = joblib.load(MODELS_DIR / "prophet_model.joblib")
        pro_model        = payload["model"]
        regressors       = payload.get("regressors", [])
        intraday_profile = payload.get("intraday_profile", {})

        last_gas = None
        if "gas_price_monthly" in regressors and "gas_price_monthly" in feat_df.columns:
            vals = feat_df["gas_price_monthly"].dropna()
            last_gas = float(vals.iloc[-1]) if not vals.empty else None

        # Generate daily forecasts covering required n_periods
        days_needed = (n_periods // 48) + 2
        daily_rows  = []
        for d in range(days_needed):
            row = {"ds": last_dt.normalize() + pd.Timedelta(days=d + 1)}
            if "gas_price_monthly" in regressors and last_gas is not None:
                row["gas_price_monthly"] = last_gas
            daily_rows.append(row)

        daily_future = pd.DataFrame(daily_rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            daily_fc = pro_model.predict(daily_future)

        yhat_map  = {row["ds"].date(): row["yhat"]       for _, row in daily_fc.iterrows()}
        lower_map = {row["ds"].date(): row["yhat_lower"] for _, row in daily_fc.iterrows()}
        upper_map = {row["ds"].date(): row["yhat_upper"] for _, row in daily_fc.iterrows()}

        mu_overall = float(feat_df["usep"].mean()) or 80.0

        results = []
        for i in range(n_periods):
            dt_i     = last_dt + pd.Timedelta(minutes=30 * (i + 1))
            period_i = ((dt_i.hour * 60 + dt_i.minute) // 30) + 1
            d_key    = dt_i.date()
            scale    = intraday_profile.get(period_i, 1.0)
            daily_mu = yhat_map.get(d_key, mu_overall)

            yhat  = max(0.0, daily_mu * scale)
            lower = max(0.0, lower_map.get(d_key, daily_mu * 0.85) * scale)
            upper = upper_map.get(d_key, daily_mu * 1.15) * scale

            sp = spike_probability(engine, period_i, dt_i.month, spike_threshold)
            results.append({
                "dt":             dt_i,
                "period":         period_i,
                "time_label":     f"{dt_i.hour:02d}:{dt_i.minute:02d}",
                "predicted_usep": round(yhat, 2),
                "lower_bound":    round(lower, 2),
                "upper_bound":    round(upper, 2),
                "spike_prob":     round(sp, 4),
            })
        return pd.DataFrame(results)

    else:
        raise ValueError(f"Unknown model: {model_name!r}")


def predict_ensemble(engine, n_periods: int = 48,
                     spike_threshold: float = 200.0) -> pd.DataFrame:
    """Inverse-RMSE weighted blend of XGBoost + Prophet."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT model_name, rmse FROM model_registry
                WHERE model_name IN ('xgboost', 'prophet') AND is_active = 1
            """)).mappings().fetchall()
        rmses = {r["model_name"]: float(r["rmse"]) for r in rows if r["rmse"]}
        if len(rmses) == 2:
            inv   = {k: 1.0 / max(v, 0.01) for k, v in rmses.items()}
            total = sum(inv.values())
            w_xgb = inv.get("xgboost", 0.5) / total
            w_pro  = inv.get("prophet",  0.5) / total
        else:
            w_xgb, w_pro = 0.5, 0.5
    except Exception:
        w_xgb, w_pro = 0.5, 0.5

    xgb = predict("xgboost", engine, n_periods, spike_threshold)
    pro = predict("prophet",  engine, n_periods, spike_threshold)
    if xgb.empty:
        return pro
    if pro.empty:
        return xgb

    merged = xgb.copy()
    merged["predicted_usep"] = (
        xgb["predicted_usep"] * w_xgb + pro["predicted_usep"] * w_pro
    ).round(2)
    merged["lower_bound"] = (
        xgb["lower_bound"] * w_xgb + pro["lower_bound"] * w_pro
    ).round(2)
    merged["upper_bound"] = (
        xgb["upper_bound"] * w_xgb + pro["upper_bound"] * w_pro
    ).round(2)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 10. Walk-forward backtest
# ─────────────────────────────────────────────────────────────────────────────

def _period_bucket(period: int) -> str:
    if period <= 13:
        return "off_peak"
    elif period <= 30:
        return "solar"
    elif period <= 42:
        return "evening_peak"
    else:
        return "night"


def backtest_walk_forward(
    df: pd.DataFrame,
    test_days: int = 90,
    progress_callback: Callable[[float], None] | None = None,
) -> pd.DataFrame:
    """
    Walk-forward backtest using lightweight XGBoost (n_estimators=200).
    Trains on all data before each test day, predicts that day's 48 periods.

    progress_callback: optional callable(fraction: float) called each day.
    Returns: date, period, bucket, actual, predicted, error, year
    """
    from xgboost import XGBRegressor

    feat_df = build_features(df)
    if feat_df.empty:
        return pd.DataFrame()

    feat_df["date_only"] = feat_df["dt"].dt.date
    all_dates  = sorted(feat_df["date_only"].unique())
    test_dates = all_dates[-test_days:]
    avail      = [c for c in FEATURE_COLS if c in feat_df.columns]
    n_total    = len(test_dates)

    results = []
    for i, test_date in enumerate(test_dates):
        tr_mask = feat_df["date_only"] < test_date
        te_mask = feat_df["date_only"] == test_date
        if tr_mask.sum() < 1000 or te_mask.sum() == 0:
            if progress_callback:
                progress_callback((i + 1) / n_total)
            continue

        X_tr, y_tr = feat_df.loc[tr_mask, avail].values, feat_df.loc[tr_mask, TARGET].values
        X_te, y_te = feat_df.loc[te_mask, avail].values, feat_df.loc[te_mask, TARGET].values
        periods    = feat_df.loc[te_mask, "period"].values

        mdl = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)

        for p, act, pred in zip(periods, y_te, y_pred):
            results.append({
                "date":      test_date,
                "period":    int(p),
                "bucket":    _period_bucket(int(p)),
                "actual":    round(float(act), 2),
                "predicted": round(float(pred), 2),
                "error":     round(float(act - pred), 2),
                "year":      test_date.year,
            })

        if progress_callback:
            progress_callback((i + 1) / n_total)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 additions
# ─────────────────────────────────────────────────────────────────────────────

def check_model_drift(engine, model_name: str = "xgboost", window_days: int = 30) -> dict:
    """
    Compare recent rolling RMSE vs baseline RMSE stored in model_registry.
    Returns {drift_detected, recent_rmse, baseline_rmse, ratio, n_obs}
    Drift threshold: recent_rmse > 1.5 × baseline_rmse
    """
    with engine.connect() as conn:
        reg_row = conn.execute(text("""
            SELECT rmse FROM model_registry
            WHERE model_name = :name AND is_active = 1
            ORDER BY trained_at DESC LIMIT 1
        """), {"name": model_name}).fetchone()

        recent_rows = conn.execute(text("""
            SELECT abs_error FROM forecast_actuals
            WHERE model_name = :name
              AND forecast_date >= DATE('now', :window)
              AND actual_usep IS NOT NULL
        """), {"name": model_name, "window": f"-{window_days} days"}).fetchall()

    if not reg_row or not recent_rows:
        return {"drift_detected": False, "recent_rmse": None,
                "baseline_rmse": None, "ratio": None, "n_obs": 0}

    baseline_rmse = float(reg_row[0])
    errors        = [float(r[0]) for r in recent_rows if r[0] is not None]
    recent_rmse   = float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else None
    ratio         = (recent_rmse / baseline_rmse) if (recent_rmse and baseline_rmse) else None

    return {
        "drift_detected": bool(ratio and ratio > 1.5),
        "recent_rmse":    round(recent_rmse, 2) if recent_rmse else None,
        "baseline_rmse":  round(baseline_rmse, 2),
        "ratio":          round(ratio, 3) if ratio else None,
        "n_obs":          len(errors),
    }


def save_predictions_to_db(
    predictions_df: pd.DataFrame,
    model_name: str,
    horizon: int,
    engine,
) -> int:
    """
    Upsert model predictions into forecast_actuals table.
    predictions_df must have: forecast_date, period, predicted_usep
    Returns number of rows inserted/updated.
    """
    if predictions_df.empty:
        return 0

    now_str  = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    dialect  = engine.dialect.name
    inserted = 0

    with engine.begin() as conn:
        for _, row in predictions_df.iterrows():
            try:
                actual  = row.get("actual_usep")
                error   = (float(actual) - float(row["predicted_usep"])) if pd.notna(actual) else None
                abs_err = abs(error) if error is not None else None

                if dialect == "sqlite":
                    sql = text("""
                        INSERT OR REPLACE INTO forecast_actuals
                        (model_name, forecast_date, period, predicted_usep, actual_usep,
                         error, abs_error, forecast_horizon_periods, created_at)
                        VALUES
                        (:model_name, :forecast_date, :period, :predicted_usep, :actual_usep,
                         :error, :abs_error, :horizon, :now)
                    """)
                else:
                    sql = text("""
                        INSERT INTO forecast_actuals
                        (model_name, forecast_date, period, predicted_usep, actual_usep,
                         error, abs_error, forecast_horizon_periods, created_at)
                        VALUES
                        (:model_name, :forecast_date, :period, :predicted_usep, :actual_usep,
                         :error, :abs_error, :horizon, :now)
                        ON CONFLICT (model_name, forecast_date, period, forecast_horizon_periods)
                        DO UPDATE SET
                            actual_usep = EXCLUDED.actual_usep,
                            error       = EXCLUDED.error,
                            abs_error   = EXCLUDED.abs_error
                    """)
                conn.execute(sql, {
                    "model_name":    model_name,
                    "forecast_date": str(row["forecast_date"]),
                    "period":        int(row["period"]),
                    "predicted_usep": float(row["predicted_usep"]),
                    "actual_usep":   float(actual) if pd.notna(actual) else None,
                    "error":         error,
                    "abs_error":     abs_err,
                    "horizon":       horizon,
                    "now":           now_str,
                })
                inserted += 1
            except Exception:
                pass

    return inserted


def forecast_monthly_scenarios(engine, target_months: int = 3) -> pd.DataFrame:
    """
    Generate monthly P10/P50/P90 USEP scenarios using seasonal pattern + gas regime.
    Returns: month_label, p10, p50, p90, gas_regime
    """
    with engine.connect() as conn:
        hist = conn.execute(text("""
            SELECT strftime('%m', date) AS month, usep
            FROM nems_prices WHERE usep IS NOT NULL
        """)).fetchall()

        gas_latest = conn.execute(text("""
            SELECT AVG(weighted_avg_usd_mmbtu) AS avg_price
            FROM gas_prices
            WHERE price_date >= DATE('now', '-90 days')
              AND weighted_avg_usd_mmbtu IS NOT NULL
        """)).fetchone()

    if not hist:
        return pd.DataFrame()

    hist_df         = pd.DataFrame(hist, columns=["month", "usep"])
    hist_df["usep"] = pd.to_numeric(hist_df["usep"], errors="coerce")
    hist_df["month"] = hist_df["month"].astype(int)

    seasonal = (
        hist_df.groupby("month")["usep"]
        .quantile([0.1, 0.5, 0.9])
        .unstack()
        .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
    )

    avg_price = float(gas_latest[0]) if (gas_latest and gas_latest[0]) else None
    if avg_price and avg_price > 15:
        gas_regime, mult = "high", 1.15
    elif avg_price and avg_price < 8:
        gas_regime, mult = "low", 0.90
    else:
        gas_regime, mult = "neutral", 1.0

    from datetime import date
    today = date.today()
    rows  = []
    for i in range(1, target_months + 1):
        mo    = ((today.month - 1 + i) % 12) + 1
        yr    = today.year + ((today.month - 1 + i) // 12)
        label = f"{yr}-{mo:02d}"
        if mo in seasonal.index:
            rows.append({
                "month_label": label,
                "p10":         round(float(seasonal.loc[mo, "p10"]) * mult, 2),
                "p50":         round(float(seasonal.loc[mo, "p50"]) * mult, 2),
                "p90":         round(float(seasonal.loc[mo, "p90"]) * mult, 2),
                "gas_regime":  gas_regime,
            })

    return pd.DataFrame(rows)
