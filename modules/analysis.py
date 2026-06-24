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
