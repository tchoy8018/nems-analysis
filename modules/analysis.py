"""Statistical analysis functions for NEMS price and demand data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

from modules.utils import period_to_time_label


# ---------------------------------------------------------------------------
# 1. Summary statistics
# ---------------------------------------------------------------------------

def summary_statistics(df: pd.DataFrame) -> dict:
    """Return a dict of key descriptive stats for the given NEMS DataFrame."""
    usep = df["usep"].dropna()
    demand = df["demand_mw"].dropna() if "demand_mw" in df.columns else pd.Series(dtype=float)
    solar = df["solar_mw"].dropna() if "solar_mw" in df.columns else pd.Series(dtype=float)

    spike_mask = usep > 200

    return {
        "count": len(df),
        "date_min": df["date"].min() if "date" in df.columns else None,
        "date_max": df["date"].max() if "date" in df.columns else None,
        "usep_mean": usep.mean(),
        "usep_median": usep.median(),
        "usep_std": usep.std(),
        "usep_min": usep.min(),
        "usep_max": usep.max(),
        "usep_p10": usep.quantile(0.10),
        "usep_p90": usep.quantile(0.90),
        "spike_count": int(spike_mask.sum()),
        "spike_pct": 100.0 * spike_mask.sum() / len(usep) if len(usep) else 0.0,
        "demand_mean": demand.mean() if len(demand) else None,
        "demand_max": demand.max() if len(demand) else None,
        "solar_mean": solar.mean() if len(solar) else None,
        "solar_max": solar.max() if len(solar) else None,
    }


# ---------------------------------------------------------------------------
# 2. Half-hourly profile
# ---------------------------------------------------------------------------

def half_hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return average USEP, demand, and solar for each of the 48 half-hourly periods.

    Columns: period, time_label, avg_usep, avg_demand_mw, avg_solar_mw
    """
    agg: dict[str, tuple] = {"usep": ("usep", "mean")}
    if "demand_mw" in df.columns:
        agg["demand_mw"] = ("demand_mw", "mean")
    if "solar_mw" in df.columns:
        agg["solar_mw"] = ("solar_mw", "mean")

    profile = df.groupby("period").agg(**agg).reset_index()
    profile.columns = (
        ["period"]
        + [f"avg_{c}" if c != "period" else c for c in profile.columns[1:]]
    )
    profile["time_label"] = profile["period"].apply(period_to_time_label)
    return profile.sort_values("period").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Duck curve
# ---------------------------------------------------------------------------

def duck_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the duck curve: average demand, solar, and net demand by period.

    Columns: period, time_label, demand_mw, solar_mw, net_demand
    """
    d = df.copy()
    d["net_demand"] = d["demand_mw"] - d["solar_mw"].fillna(0)

    result = (
        d.groupby("period")
        .agg(
            demand_mw=("demand_mw", "mean"),
            solar_mw=("solar_mw", "mean"),
            net_demand=("net_demand", "mean"),
        )
        .reset_index()
    )
    result["time_label"] = result["period"].apply(period_to_time_label)
    return result.sort_values("period").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Correlation analysis
# ---------------------------------------------------------------------------

def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations for solar, demand, and net_demand vs USEP.

    Returns a DataFrame with columns:
        variable, pearson_r, pearson_p, spearman_r, spearman_p
    """
    d = df.copy()
    if "demand_mw" in d.columns and "solar_mw" in d.columns:
        d["net_demand"] = d["demand_mw"] - d["solar_mw"].fillna(0)

    candidates = [
        ("solar_mw", "Solar (MW)"),
        ("demand_mw", "Demand (MW)"),
        ("net_demand", "Net Demand (MW)"),
    ]

    rows = []
    for col, label in candidates:
        if col not in d.columns:
            continue
        sub = d[["usep", col]].dropna().astype(float)
        if len(sub) < 10:
            continue
        p_r, p_p = stats.pearsonr(sub["usep"].to_numpy(), sub[col].to_numpy())
        s_r, s_p = stats.spearmanr(sub["usep"].to_numpy(), sub[col].to_numpy())
        rows.append(
            {
                "variable": label,
                "pearson_r": round(p_r, 4),
                "pearson_p": round(p_p, 4),
                "spearman_r": round(s_r, 4),
                "spearman_p": round(s_p, 4),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Spike analysis
# ---------------------------------------------------------------------------

def spike_analysis(df: pd.DataFrame, threshold: float = 200) -> pd.DataFrame:
    """
    Return all half-hourly periods where USEP exceeded the threshold.

    Columns: date, period, time_label, usep, demand_mw
    """
    spikes = df[df["usep"] > threshold].copy()
    spikes["time_label"] = spikes["period"].apply(period_to_time_label)

    cols = ["date", "period", "time_label", "usep"]
    if "demand_mw" in spikes.columns:
        cols.append("demand_mw")

    return spikes[cols].sort_values(["date", "period"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Arbitrage windows
# ---------------------------------------------------------------------------

def arbitrage_windows(
    df: pd.DataFrame,
    capacity_mw: float = 560,
    efficiency: float = 0.75,
    utilization: float = 0.75,
    n_windows: int = 5,
) -> dict:
    """
    Identify optimal daily charge / discharge windows for a BESS.

    Strategy: rank all 48 periods by average USEP across the dataset;
    take the n_windows lowest as charge periods and n_windows highest
    as discharge periods.  Then compute an estimated daily revenue for
    every day in df.

    Revenue formula (per period):
        Revenue = USEP × (capacity_mw × utilization × efficiency) × 0.5 h
        Cost    = USEP × (capacity_mw × utilization)               × 0.5 h
        Net     = Revenue − Cost

    Returns a dict with:
        period_profile   — DataFrame: period, avg_usep, time_label
        charge_windows   — DataFrame: top-n cheapest periods (sorted asc)
        discharge_windows— DataFrame: top-n priciest periods (sorted desc)
        daily_revenue    — DataFrame: date, revenue_sgd
        monthly_revenue  — DataFrame: year_month, revenue_sgd
    """
    effective_mw = capacity_mw * utilization

    # --- Period averages across the full dataset ---
    period_avg = (
        df.groupby("period")["usep"]
        .mean()
        .reset_index()
        .rename(columns={"usep": "avg_usep"})
    )
    period_avg["time_label"] = period_avg["period"].apply(period_to_time_label)

    charge_windows = (
        period_avg.nsmallest(n_windows, "avg_usep")
        .sort_values("avg_usep")
        .reset_index(drop=True)
    )
    discharge_windows = (
        period_avg.nlargest(n_windows, "avg_usep")
        .sort_values("avg_usep", ascending=False)
        .reset_index(drop=True)
    )

    charge_periods = set(charge_windows["period"])
    discharge_periods = set(discharge_windows["period"])

    # --- Daily revenue ---
    def _daily_revenue(day_df: pd.DataFrame) -> float:
        charge_usep = day_df.loc[
            day_df["period"].isin(charge_periods), "usep"
        ].dropna()
        discharge_usep = day_df.loc[
            day_df["period"].isin(discharge_periods), "usep"
        ].dropna()
        discharge_rev = discharge_usep.sum() * effective_mw * efficiency * 0.5
        charge_cost = charge_usep.sum() * effective_mw * 0.5
        return max(0.0, discharge_rev - charge_cost)

    daily = (
        df.groupby("date")
        .apply(_daily_revenue)
        .reset_index()
    )
    daily.columns = ["date", "revenue_sgd"]
    daily["date"] = pd.to_datetime(daily["date"])

    # --- Monthly aggregation ---
    monthly = (
        daily.set_index("date")
        .resample("ME")["revenue_sgd"]
        .sum()
        .reset_index()
    )
    monthly["year_month"] = monthly["date"].dt.strftime("%Y-%m")

    return {
        "period_profile": period_avg,
        "charge_windows": charge_windows,
        "discharge_windows": discharge_windows,
        "daily_revenue": daily,
        "monthly_revenue": monthly[["year_month", "revenue_sgd"]],
    }


# ---------------------------------------------------------------------------
# 7. Year-on-year monthly comparison
# ---------------------------------------------------------------------------

def yoy_monthly_comparison(engine, month: int) -> pd.DataFrame:
    """
    Return average USEP by calendar year for a given month (1–12).

    Columns: year (str), avg_usep (float), period_count (int)
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT
                    CAST(strftime('%Y', date) AS INTEGER) AS year,
                    AVG(usep)  AS avg_usep,
                    COUNT(*)   AS period_count
                FROM nems_prices
                WHERE CAST(strftime('%m', date) AS INTEGER) = :month
                  AND usep IS NOT NULL
                GROUP BY year
                ORDER BY year
            """),
            {"month": month},
        ).fetchall()

    return pd.DataFrame(rows, columns=["year", "avg_usep", "period_count"])


# ---------------------------------------------------------------------------
# 8. Gas price ↔ USEP correlation  (Phase 4 — uses real customs data)
# ---------------------------------------------------------------------------

def gas_usep_correlation(engine) -> dict:
    """
    Join gas_prices (monthly, from SG Customs) with monthly avg USEP.
    Computes Pearson r, Spearman rho, R² at lags 0–3 months.
    Also computes pass-through regression slope and rolling 12-month Pearson r.

    Returns {
        lags: [{lag_months, pearson_r, spearman_rho, r2, n_obs}],
        best_lag: int,
        pass_through_slope: float,   # delta USEP per delta gas (SGD/MMBtu)
        regression_r2: float,
        regime_note: str,
        monthly_df: DataFrame with date, avg_usep, weighted_gas_usd, weighted_gas_sgd,
                    implied_floor, rolling_12m_pearson
    }
    """
    with engine.connect() as conn:
        usep_rows = conn.execute(text("""
            SELECT strftime('%Y-%m', date) AS ym, AVG(usep) AS avg_usep
            FROM nems_prices
            WHERE usep IS NOT NULL
            GROUP BY ym
            ORDER BY ym
        """)).fetchall()
        gas_rows = conn.execute(text("""
            SELECT price_date,
                   weighted_avg_usd_mmbtu,
                   weighted_avg_sgd_mmbtu,
                   implied_usep_floor_sgd_mwh,
                   lng_share_pct
            FROM gas_prices
            WHERE weighted_avg_usd_mmbtu IS NOT NULL
            ORDER BY price_date
        """)).fetchall()

    if not usep_rows or not gas_rows:
        return {"lags": [], "best_lag": 0, "pass_through_slope": None,
                "regression_r2": None, "regime_note": "", "monthly_df": pd.DataFrame()}

    usep_df = pd.DataFrame(usep_rows, columns=["ym", "avg_usep"])
    usep_df["date"] = pd.to_datetime(usep_df["ym"] + "-01")

    gas_df = pd.DataFrame(gas_rows,
        columns=["date", "weighted_usd", "weighted_sgd", "implied_floor", "lng_share"])
    gas_df["date"] = pd.to_datetime(gas_df["date"])
    gas_df["ym"]   = gas_df["date"].dt.strftime("%Y-%m")

    merged = usep_df.merge(gas_df[["ym","weighted_usd","weighted_sgd","implied_floor","lng_share"]],
                           on="ym", how="inner").sort_values("date")

    # Rolling 12-month Pearson r
    def _rolling_corr(df, x_col, y_col, window=12):
        return df[x_col].rolling(window).corr(df[y_col])

    merged["rolling_12m_pearson"] = _rolling_corr(merged, "weighted_usd", "avg_usep")

    # Regime note: compare correlation pre/post-2022
    pre  = merged[merged["date"] < "2022-01-01"]
    post = merged[merged["date"] >= "2022-01-01"]
    try:
        r_pre,  _ = stats.pearsonr(pre["weighted_usd"].to_numpy(dtype=float),
                                   pre["avg_usep"].to_numpy(dtype=float))
        r_post, _ = stats.pearsonr(post["weighted_usd"].to_numpy(dtype=float),
                                   post["avg_usep"].to_numpy(dtype=float))
        if abs(r_post) > abs(r_pre) + 0.1:
            regime_note = f"Correlation stronger post-2022 (r={r_post:.2f} vs r={r_pre:.2f})"
        else:
            regime_note = f"Correlation stable across regimes (pre-2022: r={r_pre:.2f}, post-2022: r={r_post:.2f})"
    except Exception:
        regime_note = ""

    # Lag analysis (months)
    lag_results = []
    best_r2 = -1
    best_lag = 0
    for lag in range(4):
        gas_lagged = gas_df[["ym", "weighted_usd"]].copy()
        gas_lagged["date"] = gas_df["date"] + pd.DateOffset(months=lag)
        gas_lagged["ym"] = gas_lagged["date"].dt.strftime("%Y-%m")
        m = usep_df.merge(gas_lagged[["ym","weighted_usd"]], on="ym", how="inner")
        if len(m) < 12:
            continue
        x = m["weighted_usd"].to_numpy(dtype=float)
        y = m["avg_usep"].to_numpy(dtype=float)
        try:
            p_r, _ = stats.pearsonr(x, y)
            s_r, _ = stats.spearmanr(x, y)
        except Exception:
            continue
        r2 = float(p_r ** 2)
        lag_results.append({
            "lag_months":   lag,
            "pearson_r":    round(float(p_r), 4),
            "spearman_rho": round(float(s_r), 4),
            "r2":           round(r2, 4),
            "n_obs":        len(m),
        })
        if r2 > best_r2:
            best_r2  = r2
            best_lag = lag

    # Pass-through regression at best lag
    pass_through_slope = None
    regression_r2      = None
    try:
        x = merged["weighted_sgd"].to_numpy(dtype=float)
        y = merged["avg_usep"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 10:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            pass_through_slope = round(float(coeffs[0]), 3)
            p_r, _ = stats.pearsonr(x[mask], y[mask])
            regression_r2 = round(float(p_r**2), 4)
    except Exception:
        pass

    return {
        "lags":                lag_results,
        "best_lag":            best_lag,
        "pass_through_slope":  pass_through_slope,
        "regression_r2":       regression_r2,
        "regime_note":         regime_note,
        "monthly_df":          merged,
    }


def gas_mix_evolution(engine) -> pd.DataFrame:
    """
    Monthly source shares over time.
    Returns: date, malaysia_share_pct, indonesia_share_pct, lng_share_pct,
             weighted_avg_usd_mmbtu, implied_usep_floor_sgd_mwh
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT price_date,
                   malaysia_share_pct, indonesia_share_pct, lng_share_pct,
                   weighted_avg_usd_mmbtu, implied_usep_floor_sgd_mwh
            FROM gas_prices
            WHERE malaysia_share_pct IS NOT NULL
            ORDER BY price_date
        """)).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "date", "malaysia_share_pct", "indonesia_share_pct", "lng_share_pct",
        "weighted_avg_usd_mmbtu", "implied_usep_floor_sgd_mwh",
    ])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# 9. Analyst forecast vs actuals  (Phase 4)
# ---------------------------------------------------------------------------

def analyst_vs_actuals(
    engine,
    source_ids: list[int] | None = None,
    date_from=None,
    date_to=None,
) -> pd.DataFrame:
    """
    Compare each analyst forecast source vs actual USEP.
    Returns DataFrame: source_name, vintage_year, n_overlap,
                       mae, rmse, bias, pearson_r
    """
    where_clauses = ["fd.price IS NOT NULL", "np.usep IS NOT NULL"]
    params: dict = {}
    if source_ids:
        ids_str = ",".join(str(i) for i in source_ids)
        where_clauses.append(f"fd.source_id IN ({ids_str})")
    if date_from:
        where_clauses.append("fd.date >= :dfrom")
        params["dfrom"] = str(date_from)
    if date_to:
        where_clauses.append("fd.date <= :dto")
        params["dto"] = str(date_to)

    where = " AND ".join(where_clauses)

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT
                fs.source_name,
                fs.vintage_year,
                fs.granularity,
                fd.date,
                fd.period,
                fd.price AS forecast_usep,
                AVG(np.usep) AS actual_usep
            FROM forecast_data fd
            JOIN forecast_sources fs ON fs.id = fd.source_id
            JOIN nems_prices np ON np.date = fd.date
                AND (fd.period IS NULL OR np.period = fd.period)
            WHERE {where}
            GROUP BY fs.source_name, fs.vintage_year, fd.date, fd.period
            ORDER BY fs.source_name, fs.vintage_year, fd.date
        """), params).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "source_name", "vintage_year", "granularity",
        "date", "period", "forecast_usep", "actual_usep",
    ])
    df["error"] = df["actual_usep"] - df["forecast_usep"]

    results = []
    for (src, vint), grp in df.groupby(["source_name", "vintage_year"], dropna=False):
        grp = grp.dropna(subset=["forecast_usep", "actual_usep"])
        if len(grp) < 2:
            continue
        rmse = float(np.sqrt(np.mean(grp["error"] ** 2)))
        mae  = float(np.mean(np.abs(grp["error"])))
        bias = float(grp["error"].mean())
        try:
            p_r, _ = stats.pearsonr(
                grp["forecast_usep"].to_numpy(dtype=float),
                grp["actual_usep"].to_numpy(dtype=float),
            )
        except Exception:
            p_r = float("nan")
        results.append({
            "source_name":  src,
            "vintage_year": vint,
            "n_overlap":    len(grp),
            "mae":          round(mae, 2),
            "rmse":         round(rmse, 2),
            "bias":         round(bias, 2),
            "pearson_r":    round(float(p_r), 4),
        })
    return pd.DataFrame(results)


def vintage_comparison(engine, source_name: str) -> pd.DataFrame:
    """
    Show how one analyst's forecasts evolved across vintages vs actuals.
    Returns: date, actual_avg_usep, forecast_2023, forecast_2024, ...
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                fs.vintage_year,
                fd.date,
                AVG(fd.price) AS forecast_usep,
                AVG(np.usep)  AS actual_usep
            FROM forecast_data fd
            JOIN forecast_sources fs ON fs.id = fd.source_id
            JOIN nems_prices np ON np.date = fd.date
                AND (fd.period IS NULL OR np.period = fd.period)
            WHERE fs.source_name = :src
            GROUP BY fs.vintage_year, fd.date
            ORDER BY fd.date, fs.vintage_year
        """), {"src": source_name}).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["vintage_year", "date", "forecast_usep", "actual_usep"])
    df["date"] = pd.to_datetime(df["date"])

    pivot = df.pivot_table(
        index="date", columns="vintage_year", values="forecast_usep"
    ).reset_index()
    pivot.columns = ["date"] + [f"forecast_{int(c)}" for c in pivot.columns[1:]]

    actual = df.groupby("date")["actual_usep"].mean().reset_index()
    return pivot.merge(actual, on="date", how="left")


# ---------------------------------------------------------------------------
# 10. Day-type USEP profile  (Phase 4)
# ---------------------------------------------------------------------------

def day_type_usep_profile(engine, date_from=None, date_to=None) -> pd.DataFrame:
    """
    Average USEP by (period × day_type) across 5 day types.
    Returns heatmap-ready DataFrame: 48 periods × 5 day types.
    Columns: period, weekday_core, weekday_wfh, saturday, sunday, public_holiday
    """
    from modules.utils import get_sg_calendar_features

    where = "usep IS NOT NULL"
    params: dict = {}
    if date_from:
        where += " AND date >= :dfrom"
        params["dfrom"] = str(date_from)
    if date_to:
        where += " AND date <= :dto"
        params["dto"] = str(date_to)

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT date, period, usep FROM nems_prices
            WHERE {where} ORDER BY date, period
        """), params).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "period", "usep"])
    df["date"] = pd.to_datetime(df["date"])

    cal = get_sg_calendar_features(df["date"])
    df["day_type"] = cal["day_type"].values

    pivot = (
        df.groupby(["period", "day_type"])["usep"]
        .mean()
        .reset_index()
        .pivot(index="period", columns="day_type", values="usep")
        .reset_index()
    )
    pivot.columns.name = None
    return pivot
