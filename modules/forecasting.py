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

FEATURE_COLS_BASE = [
    "period", "hour", "day_of_week", "month", "is_weekend", "is_public_holiday_sg",
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
    Add time + lag features to a NEMS DataFrame.

    Input columns expected: date (date/datetime), period (1-48),
    usep, demand_mw, solar_mw.

    If engine is provided and gas_prices table has data, gas features are added.
    Returns a copy with NaN rows (from lags) dropped.
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

    # ── Gas features (optional) ──
    active_features = list(FEATURE_COLS_BASE)
    gas_df = _load_gas_monthly(engine) if engine is not None else pd.DataFrame()
    if not gas_df.empty:
        d["ym"] = d["date"].dt.strftime("%Y-%m")

        # lag-0: same month gas price
        gas_l0 = gas_df[["ym", "gas_sgd", "implied_floor", "lng_share"]].rename(
            columns={"gas_sgd": "gas_price_monthly", "lng_share": "lng_share"})
        d = d.merge(gas_l0, on="ym", how="left")

        # lag-1m
        gas_df["ym_lag1"] = (gas_df["date"] + pd.DateOffset(months=1)).dt.strftime("%Y-%m")
        gas_l1 = gas_df[["ym_lag1", "gas_sgd"]].rename(
            columns={"ym_lag1": "ym", "gas_sgd": "gas_price_lag1m"})
        d = d.merge(gas_l1, on="ym", how="left")

        # lag-2m
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

    # Fill remaining sparse NaNs with column median
    for col in active_features:
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

    feat_df = build_features(df, engine=engine)
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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 additions
# ─────────────────────────────────────────────────────────────────────────────

def check_model_drift(engine, model_name: str = "xgboost_v1", window_days: int = 30) -> dict:
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
    errors = [float(r[0]) for r in recent_rows if r[0] is not None]
    recent_rmse = float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else None
    ratio = (recent_rmse / baseline_rmse) if (recent_rmse and baseline_rmse) else None

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

    now_str = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    dialect = engine.dialect.name
    inserted = 0

    with engine.begin() as conn:
        for _, row in predictions_df.iterrows():
            try:
                actual = row.get("actual_usep")
                error  = (float(actual) - float(row["predicted_usep"])) if pd.notna(actual) else None
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
                            error = EXCLUDED.error,
                            abs_error = EXCLUDED.abs_error
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
    Returns DataFrame: month_label, p10, p50, p90, gas_regime
    """
    with engine.connect() as conn:
        hist = conn.execute(text("""
            SELECT strftime('%m', date) AS month, usep
            FROM nems_prices
            WHERE usep IS NOT NULL
        """)).fetchall()

        gas_latest = conn.execute(text("""
            SELECT AVG(jkm_usd_mmbtu) AS avg_jkm
            FROM gas_prices
            WHERE price_date >= DATE('now', '-90 days')
              AND jkm_usd_mmbtu IS NOT NULL
        """)).fetchone()

    if not hist:
        return pd.DataFrame()

    hist_df = pd.DataFrame(hist, columns=["month", "usep"])
    hist_df["usep"] = pd.to_numeric(hist_df["usep"], errors="coerce")
    hist_df["month"] = hist_df["month"].astype(int)

    seasonal = (
        hist_df.groupby("month")["usep"]
        .quantile([0.1, 0.5, 0.9])
        .unstack()
        .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
    )

    avg_jkm = float(gas_latest[0]) if (gas_latest and gas_latest[0]) else None
    if avg_jkm and avg_jkm > 20:
        gas_regime = "high"
        mult = 1.15
    elif avg_jkm and avg_jkm < 10:
        gas_regime = "low"
        mult = 0.90
    else:
        gas_regime = "neutral"
        mult = 1.0

    from datetime import date
    today = date.today()
    rows = []
    for i in range(1, target_months + 1):
        mo = ((today.month - 1 + i) % 12) + 1
        yr = today.year + ((today.month - 1 + i) // 12)
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
