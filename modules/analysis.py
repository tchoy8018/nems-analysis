"""Statistical analysis functions for NEMS price and demand data."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

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


# ---------------------------------------------------------------------------
# 11. Demand–USEP threshold analysis  (Phase 5)
# ---------------------------------------------------------------------------

VESTING_PRICE = 170.0  # S$/MWh — approximate current vesting contract level


def demand_usep_threshold_analysis(
    engine,
    date_from=None,
    date_to=None,
) -> dict:
    """
    Quantify the demand threshold above which USEP rises nonlinearly.

    Uses piecewise-linear regression (scipy.optimize.curve_fit) to find
    the inflection point in the demand → USEP relationship.

    Returns:
      demand_bins          — DataFrame(demand_range, avg_usep, median_usep,
                              p90_usep, p99_usep, spike_freq_200,
                              spike_freq_300, spike_freq_500, n_periods)
      inflection_mw        — float  (demand at which price accelerates)
      vesting_price        — 170.0
      pct_above_vesting    — float
      demand_at_vesting_breach — float  (median demand when USEP > vesting)
      spearman_r           — float
      spearman_p           — float
      interpretation       — str
    """
    from scipy.optimize import curve_fit

    where  = "usep IS NOT NULL AND demand_mw IS NOT NULL"
    params: dict = {}
    if date_from:
        where += " AND date >= :dfrom"
        params["dfrom"] = str(date_from)
    if date_to:
        where += " AND date <= :dto"
        params["dto"] = str(date_to)

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT date, period, usep, demand_mw
            FROM nems_prices WHERE {where} ORDER BY date, period
        """), params).fetchall()

    if not rows:
        return {"demand_bins": pd.DataFrame(), "inflection_mw": None,
                "vesting_price": VESTING_PRICE, "pct_above_vesting": None,
                "demand_at_vesting_breach": None,
                "spearman_r": None, "spearman_p": None, "interpretation": ""}

    df = pd.DataFrame(rows, columns=["date", "period", "usep", "demand_mw"])
    df = df.dropna(subset=["usep", "demand_mw"])

    # ── Demand bins (200 MW) ──
    d_min = (df["demand_mw"].min() // 200) * 200
    d_max = (df["demand_mw"].max() // 200 + 1) * 200
    bins  = np.arange(d_min, d_max + 200, 200)
    labels = [f"{int(b)}–{int(b+200)}" for b in bins[:-1]]
    df["demand_bin_mid"] = (df["demand_mw"] // 200) * 200 + 100
    df["demand_bin"]     = pd.cut(df["demand_mw"], bins=bins, labels=labels, right=False)

    def _spike_freq(g, threshold):
        return 100.0 * (g > threshold).sum() / len(g) if len(g) else 0.0

    bin_rows = []
    for label, grp in df.groupby("demand_bin", observed=True):
        if len(grp) < 5:
            continue
        u = grp["usep"]
        bin_rows.append({
            "demand_range":   str(label),
            "avg_usep":       round(u.mean(), 2),
            "median_usep":    round(u.median(), 2),
            "p90_usep":       round(u.quantile(0.90), 2),
            "p99_usep":       round(u.quantile(0.99), 2),
            "spike_freq_200": round(_spike_freq(u, 200), 2),
            "spike_freq_300": round(_spike_freq(u, 300), 2),
            "spike_freq_500": round(_spike_freq(u, 500), 2),
            "n_periods":      len(grp),
        })
    demand_bins = pd.DataFrame(bin_rows)

    # ── Piecewise linear inflection (curve_fit on bin midpoints) ──
    inflection_mw = None
    if len(bin_rows) >= 4:
        # Parse midpoints directly from demand_range strings ("4600–4800" → 4700)
        def _parse_mid(rng: str) -> float:
            try:
                lo, hi = (float(x.strip()) for x in rng.replace("–", "-").split("-", 1))
                return (lo + hi) / 2
            except Exception:
                return float("nan")

        bin_mids  = np.array([_parse_mid(r) for r in demand_bins["demand_range"]], dtype=float)
        avg_useps = demand_bins["avg_usep"].values.astype(float)
        mask      = np.isfinite(bin_mids) & np.isfinite(avg_useps)
        bin_mids  = bin_mids[mask]
        avg_useps = avg_useps[mask]

        def _piecewise(x, xb, m1, m2, b0):
            return np.where(x < xb, m1 * (x - xb) + b0, m2 * (x - xb) + b0)

        try:
            x_mid = float(np.median(bin_mids))
            p0    = [x_mid, 0.005, 0.05, float(np.median(avg_useps))]
            popt, _ = curve_fit(
                _piecewise, bin_mids, avg_useps,
                p0=p0,
                bounds=([bin_mids.min(), -1,  0,   0],
                        [bin_mids.max(),  1,   2, 1000]),
                maxfev=5000,
            )
            candidate = float(popt[0])
            if np.isfinite(candidate):
                inflection_mw = round(candidate, 0)
        except Exception:
            pass

        # Fallback: bin with largest step-up in avg USEP
        if inflection_mw is None or not np.isfinite(inflection_mw):
            if len(avg_useps) > 2:
                diffs = np.diff(avg_useps)
                idx   = int(np.argmax(diffs))
                inflection_mw = round(float(bin_mids[idx + 1]), 0) if idx + 1 < len(bin_mids) else None

    # ── Vesting price analysis ──
    pct_above_vesting      = round(100.0 * (df["usep"] > VESTING_PRICE).mean(), 2)
    above_vesting          = df[df["usep"] > VESTING_PRICE]["demand_mw"]
    demand_at_vesting_breach = round(float(above_vesting.median()), 0) if len(above_vesting) else None

    # ── Spearman correlation ──
    s_r, s_p = stats.spearmanr(
        df["demand_mw"].to_numpy(dtype=float),
        df["usep"].to_numpy(dtype=float),
    )

    interpretation = (
        f"Demand–USEP Spearman r={s_r:.3f} (p<0.001). "
    )
    if inflection_mw:
        interpretation += (
            f"Price accelerates above {inflection_mw:,.0f} MW. "
        )
    interpretation += (
        f"USEP exceeds vesting price (S${VESTING_PRICE}/MWh) in "
        f"{pct_above_vesting:.1f}% of periods, typically when demand "
        f"> {demand_at_vesting_breach:,.0f} MW."
        if demand_at_vesting_breach else ""
    )

    # Cache result in DB
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO demand_analysis_cache
                    (computed_at, inflection_mw, spearman_r, pct_above_vesting,
                     demand_at_vesting_breach, date_from, date_to, n_periods)
                VALUES (:computed_at, :inflection_mw, :spearman_r, :pct_above_vesting,
                        :demand_at_vesting_breach, :date_from, :date_to, :n_periods)
            """), {
                "computed_at":              datetime.utcnow(),
                "inflection_mw":            inflection_mw,
                "spearman_r":               round(float(s_r), 4),
                "pct_above_vesting":        pct_above_vesting,
                "demand_at_vesting_breach": demand_at_vesting_breach,
                "date_from":                str(date_from) if date_from else None,
                "date_to":                  str(date_to)   if date_to   else None,
                "n_periods":                len(df),
            })
    except Exception:
        pass

    return {
        "demand_bins":             demand_bins,
        "inflection_mw":           inflection_mw,
        "vesting_price":           VESTING_PRICE,
        "pct_above_vesting":       pct_above_vesting,
        "demand_at_vesting_breach": demand_at_vesting_breach,
        "spearman_r":              round(float(s_r), 4),
        "spearman_p":              round(float(s_p), 6),
        "interpretation":          interpretation,
        "_raw_df":                 df,  # kept for scatter chart reuse
    }


# ---------------------------------------------------------------------------
# 12. Demand profile analysis  (Phase 5)
# ---------------------------------------------------------------------------

def demand_profile_analysis(engine, date_from=None, date_to=None) -> dict:
    """
    Average demand by period, day_type, and year-on-year growth.

    Returns:
      period_profile    — DataFrame(period, time_label, avg_demand)
      day_type_profile  — DataFrame(period, weekday_core, weekday_wfh,
                           saturday, sunday, public_holiday)
      monthly_profile   — DataFrame(year, month, avg_peak_demand)
      demand_yoy        — DataFrame(year, avg_demand, max_demand)
      peak_period       — int   (period with highest avg demand)
      min_demand_period — int
      demand_cagr       — float or None (2019–latest)
    """
    from modules.utils import get_sg_calendar_features, period_to_time_label

    where  = "demand_mw IS NOT NULL"
    params: dict = {}
    if date_from:
        where += " AND date >= :dfrom"
        params["dfrom"] = str(date_from)
    if date_to:
        where += " AND date <= :dto"
        params["dto"] = str(date_to)

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT date, period, demand_mw
            FROM nems_prices WHERE {where} ORDER BY date, period
        """), params).fetchall()

    if not rows:
        return {}

    df = pd.DataFrame(rows, columns=["date", "period", "demand_mw"])
    df["date"] = pd.to_datetime(df["date"])

    # ── Period profile (all day-types combined) ──
    period_profile = (
        df.groupby("period")["demand_mw"].mean()
        .reset_index()
        .rename(columns={"demand_mw": "avg_demand"})
    )
    period_profile["time_label"] = period_profile["period"].apply(period_to_time_label)

    peak_period       = int(period_profile.loc[period_profile["avg_demand"].idxmax(), "period"])
    min_demand_period = int(period_profile.loc[period_profile["avg_demand"].idxmin(), "period"])

    # ── Day-type profile ──
    cal = get_sg_calendar_features(df["date"])
    df["day_type"] = cal["day_type"].values

    dt_profile = (
        df.groupby(["period", "day_type"])["demand_mw"]
        .mean()
        .reset_index()
        .pivot(index="period", columns="day_type", values="demand_mw")
        .reset_index()
    )
    dt_profile.columns.name = None

    # ── Monthly peak demand ──
    monthly = df.copy()
    monthly["year"]  = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    peak_per_day = (
        monthly.groupby(["year", "month", monthly["date"].dt.date])["demand_mw"]
        .max()
        .reset_index(level=[0, 1, 2])
    )
    peak_per_day.columns = ["year", "month", "date", "peak_demand"]
    monthly_profile = (
        peak_per_day.groupby(["year", "month"])["peak_demand"]
        .mean()
        .reset_index()
        .rename(columns={"peak_demand": "avg_peak_demand"})
    )

    # ── YoY demand ──
    demand_yoy = (
        df.groupby(df["date"].dt.year)["demand_mw"]
        .agg(avg_demand="mean", max_demand="max")
        .reset_index()
        .rename(columns={"date": "year"})
    )

    # ── CAGR 2019 → latest ──
    demand_cagr = None
    yoy_clean = demand_yoy.dropna(subset=["avg_demand"])
    if len(yoy_clean) >= 2:
        first = yoy_clean.iloc[0]
        last  = yoy_clean.iloc[-1]
        years = float(last["year"] - first["year"])
        if years > 0 and first["avg_demand"] > 0:
            demand_cagr = round(
                (last["avg_demand"] / first["avg_demand"]) ** (1 / years) - 1, 4
            )

    return {
        "period_profile":    period_profile,
        "day_type_profile":  dt_profile,
        "monthly_profile":   monthly_profile,
        "demand_yoy":        demand_yoy,
        "peak_period":       peak_period,
        "min_demand_period": min_demand_period,
        "demand_cagr":       demand_cagr,
    }


# ---------------------------------------------------------------------------
# 13. Demand features for forecasting  (Phase 5)
# ---------------------------------------------------------------------------

def build_demand_features(df: pd.DataFrame, inflection_mw: Optional[float] = None) -> pd.DataFrame:
    """
    Add demand-based features to a NEMS DataFrame already containing demand_mw.
    Intended to be called from forecasting.build_features().

    Features added:
      demand_lag_336           — demand same period last week
      demand_rolling_mean_48   — rolling 24h avg demand (lagged 1 period)
      demand_pct_of_daily_peak — demand_lag_48 / that day's peak demand_lag_48
      is_above_inflection      — 1 if demand_lag_48 > inflection_mw
      demand_usep_regime       — 0=low, 1=normal, 2=high (demand percentiles)
    """
    d = df.copy()
    if "demand_mw" not in d.columns:
        return d

    # Expects df already sorted and indexed by dt (as in build_features)
    d["demand_lag_336"]        = d["demand_mw"].shift(336)
    d["demand_rolling_mean_48"] = d["demand_mw"].rolling(48).mean().shift(1)

    # Percentile regime from demand_lag_48
    if "demand_lag_48" in d.columns:
        dl48 = d["demand_lag_48"].dropna()
        if len(dl48):
            p33 = dl48.quantile(0.33)
            p67 = dl48.quantile(0.67)
            d["demand_usep_regime"] = np.where(
                d["demand_lag_48"] < p33, 0,
                np.where(d["demand_lag_48"] < p67, 1, 2)
            )
            if inflection_mw is not None:
                d["is_above_inflection"] = (
                    d["demand_lag_48"] > inflection_mw
                ).astype(int)
            else:
                d["is_above_inflection"] = (
                    d["demand_lag_48"] > p67
                ).astype(int)

    return d
