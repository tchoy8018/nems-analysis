"""
NEMS Price Forecasting — XGBoost + Prophet
Chronological train/test split only. No lookahead.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

warnings.filterwarnings("ignore", category=FutureWarning)

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SG_HOLIDAYS = holidays.country_holidays("SG")

FEATURE_COLS = [
    "period", "hour", "day_of_week", "month", "is_weekend", "is_public_holiday_sg",
    "period_sin", "period_cos", "month_sin", "month_cos",
    "usep_lag_48", "usep_lag_336", "usep_lag_672",
    "usep_rolling_mean_48", "usep_rolling_std_48",
    "demand_lag_48", "solar_lag_48",
]
TARGET = "usep"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time + lag features to a NEMS DataFrame.

    Input columns expected: date (date/datetime), period (1-48),
    usep, demand_mw, solar_mw.

    Returns a copy with NaN rows (from lags) dropped.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["date", "period"]).reset_index(drop=True)

    d["dt"] = d["date"] + pd.to_timedelta((d["period"] - 1) * 30, unit="m")

    # ── Time features ──
    d["hour"]        = d["period"].apply(lambda p: ((p - 1) * 30) // 60)
    d["day_of_week"] = d["date"].dt.dayofweek
    d["month"]       = d["date"].dt.month
    d["is_weekend"]  = d["day_of_week"].isin([5, 6]).astype(int)
    d["is_public_holiday_sg"] = d["date"].dt.date.apply(
        lambda x: int(x in SG_HOLIDAYS)
    )

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

    # Fill remaining sparse NaNs with column median
    for col in FEATURE_COLS:
        if col in d.columns and d[col].isna().any():
            d[col] = d[col].fillna(d[col].median())

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metrics + registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    res = y_true - y_pred
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


# ─────────────────────────────────────────────────────────────────────────────
# 3. XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(df: pd.DataFrame, engine) -> dict:
    """Train XGBoost on 80% chronological split; save + register model."""
    from xgboost import XGBRegressor

    feat_df = build_features(df)
    split   = int(len(feat_df) * 0.8)
    train   = feat_df.iloc[:split]
    test    = feat_df.iloc[split:]

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

    print(f"XGBoost  | train={len(train):,}  test={len(test):,}  "
          f"RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  MAPE={m['mape']:.1f}%")

    return {
        "model_name": "xgboost", "training_rows": len(train), "test_rows": len(test),
        "top_features": top_features, **m,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Prophet
# ─────────────────────────────────────────────────────────────────────────────

def train_prophet(df: pd.DataFrame, engine) -> dict:
    """Train Prophet on 80% chronological split; save + register model."""
    from prophet import Prophet

    feat_df = build_features(df)
    split   = int(len(feat_df) * 0.8)
    train   = feat_df.iloc[:split].copy()
    test    = feat_df.iloc[split:].copy()

    for part in (train, test):
        part["ds"] = part["dt"]
        part["y"]  = part[TARGET]

    regressors = [
        c for c in ["demand_lag_48", "solar_lag_48"]
        if c in feat_df.columns and feat_df[c].notna().mean() > 0.5
    ]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.add_seasonality(name="half_hourly", period=1, fourier_order=8)
    for reg in regressors:
        model.add_regressor(reg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train[["ds", "y"] + regressors])

    future = test[["ds"] + regressors].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = model.predict(future)

    m = _metrics(test["y"].values, forecast["yhat"].values)

    path = str(MODELS_DIR / "prophet_model.joblib")
    joblib.dump({"model": model, "regressors": regressors}, path)
    _register_model(engine, "prophet", m, len(train), path)

    print(f"Prophet  | train={len(train):,}  test={len(test):,}  "
          f"RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  MAPE={m['mape']:.1f}%")

    return {
        "model_name": "prophet", "training_rows": len(train), "test_rows": len(test),
        "top_features": [], **m,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Spike probability
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


# ─────────────────────────────────────────────────────────────────────────────
# 6. Predict next N periods
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


def predict(model_name: str, engine, n_periods: int = 48) -> pd.DataFrame:
    """
    Generate forecast for next n_periods using saved model.
    Returns: dt, period, time_label, predicted_usep, lower_bound, upper_bound, spike_prob
    """
    hist     = _fetch_recent_history(engine)
    feat_df  = build_features(hist)
    if feat_df.empty:
        return pd.DataFrame()

    last_dt    = pd.Timestamp(feat_df["dt"].max())
    last_month = last_dt.month

    if model_name == "xgboost":
        payload     = joblib.load(MODELS_DIR / "xgboost_model.joblib")
        xgb_model   = payload["model"]
        feat_names  = payload["features"]

        with engine.connect() as conn:
            reg = conn.execute(text(
                "SELECT rmse FROM model_registry WHERE model_name='xgboost' AND is_active=1"
            )).fetchone()
        rmse_ci = float(reg[0]) if reg else 40.0

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
            sp   = spike_probability(engine, period_i, dt_i.month)
            results.append({
                "dt":            dt_i,
                "period":        period_i,
                "time_label":    f"{dt_i.hour:02d}:{dt_i.minute:02d}",
                "predicted_usep": round(max(0.0, yhat), 2),
                "lower_bound":   round(max(0.0, yhat - 1.5 * rmse_ci), 2),
                "upper_bound":   round(yhat + 1.5 * rmse_ci, 2),
                "spike_prob":    round(sp, 4),
            })
        return pd.DataFrame(results)

    elif model_name == "prophet":
        payload    = joblib.load(MODELS_DIR / "prophet_model.joblib")
        pro_model  = payload["model"]
        regressors = payload.get("regressors", [])

        future_rows = []
        for i in range(n_periods):
            dt_i     = last_dt + pd.Timedelta(minutes=30 * (i + 1))
            row      = {"ds": dt_i}
            for reg in regressors:
                row[reg] = float(feat_df[reg].iloc[-1]) if reg in feat_df.columns else 0.0
            future_rows.append(row)

        future_df = pd.DataFrame(future_rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = pro_model.predict(future_df)

        results = []
        for i, fc in forecast.iterrows():
            dt_i     = future_rows[i]["ds"]
            period_i = ((dt_i.hour * 60 + dt_i.minute) // 30) + 1
            sp       = spike_probability(engine, period_i, dt_i.month)
            results.append({
                "dt":             dt_i,
                "period":         period_i,
                "time_label":     f"{dt_i.hour:02d}:{dt_i.minute:02d}",
                "predicted_usep": round(max(0.0, float(fc["yhat"])), 2),
                "lower_bound":    round(max(0.0, float(fc["yhat_lower"])), 2),
                "upper_bound":    round(float(fc["yhat_upper"]), 2),
                "spike_prob":     round(sp, 4),
            })
        return pd.DataFrame(results)

    else:
        raise ValueError(f"Unknown model: {model_name!r}")


def predict_ensemble(engine, n_periods: int = 48) -> pd.DataFrame:
    """Average XGBoost + Prophet forecasts."""
    xgb = predict("xgboost", engine, n_periods)
    pro = predict("prophet", engine, n_periods)
    if xgb.empty:
        return pro
    if pro.empty:
        return xgb
    merged = xgb.copy()
    merged["predicted_usep"] = (xgb["predicted_usep"] + pro["predicted_usep"]) / 2
    merged["lower_bound"]    = (xgb["lower_bound"]    + pro["lower_bound"])    / 2
    merged["upper_bound"]    = (xgb["upper_bound"]    + pro["upper_bound"])    / 2
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 7. Walk-forward backtest (XGBoost)
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


def backtest_walk_forward(df: pd.DataFrame, test_days: int = 90) -> pd.DataFrame:
    """
    Walk-forward backtest using lightweight XGBoost (n_estimators=200).
    Trains on all data before each test day, predicts that day's 48 periods.
    Returns: date, period, bucket, actual, predicted, error
    """
    from xgboost import XGBRegressor

    feat_df = build_features(df)
    if feat_df.empty:
        return pd.DataFrame()

    feat_df["date_only"] = feat_df["dt"].dt.date
    all_dates  = sorted(feat_df["date_only"].unique())
    test_dates = all_dates[-test_days:]
    avail      = [c for c in FEATURE_COLS if c in feat_df.columns]

    results = []
    for test_date in test_dates:
        tr_mask = feat_df["date_only"] < test_date
        te_mask = feat_df["date_only"] == test_date
        if tr_mask.sum() < 1000 or te_mask.sum() == 0:
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
            })

    return pd.DataFrame(results)
