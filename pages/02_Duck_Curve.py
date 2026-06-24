from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import get_engine, setup_database
from modules.analysis import duck_curve, yoy_monthly_comparison
from modules.theme import apply_theme_css, get_chart_layout, render_theme_toggle
from config import COLOR_DEMAND, COLOR_SOLAR, COLOR_USEP


@st.cache_resource
def _get_engine():
    engine = get_engine()
    setup_database(engine)
    return engine


@st.cache_data(ttl=300)
def load_duck_data(_engine, start: date, end: date) -> pd.DataFrame:
    from sqlalchemy import text
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, period, usep, demand_mw, solar_mw
            FROM nems_prices
            WHERE date >= :s AND date <= :e
              AND demand_mw IS NOT NULL
            ORDER BY date, period
        """), {"s": start, "e": end}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "period", "usep", "demand_mw", "solar_mw"])
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_duck_for_year(_engine, year: int) -> pd.DataFrame:
    from sqlalchemy import text
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT period, AVG(demand_mw) AS demand_mw,
                   AVG(solar_mw) AS solar_mw,
                   AVG(demand_mw - COALESCE(solar_mw, 0)) AS net_demand
            FROM nems_prices
            WHERE strftime('%Y', date) = :yr AND demand_mw IS NOT NULL
            GROUP BY period
            ORDER BY period
        """), {"yr": str(year)}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["period", "demand_mw", "solar_mw", "net_demand"])
    df["time_label"] = df["period"].apply(_period_to_hhmm)
    return df


@st.cache_data(ttl=300)
def load_db_status(_engine):
    from sqlalchemy import text
    with _engine.connect() as conn:
        row = conn.execute(text("""
            SELECT MIN(date) AS min_d, MAX(date) AS max_d FROM nems_prices
        """)).mappings().fetchone()
    return dict(row) if row else {}


def _period_to_hhmm(period: int) -> str:
    h, m = divmod((period - 1) * 30, 60)
    return f"{h:02d}:{m:02d}"


def _x_axis_ticks():
    """Return tick positions (period numbers) and labels every 6 periods."""
    tick_periods = list(range(1, 49, 6))
    return tick_periods, [_period_to_hhmm(p) for p in tick_periods]


st.set_page_config(page_title="Duck Curve — NEMS", layout="wide")

engine = _get_engine()
apply_theme_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ NEMS Analytics")
    st.caption("Singa Renewables")
    render_theme_toggle()
    st.divider()

    status = load_db_status(engine)
    db_min = pd.to_datetime(status.get("min_d", "2019-01-01")).date()
    db_max = pd.to_datetime(status.get("max_d", date.today())).date()

    st.markdown("**Date range**")
    start_date = st.date_input("From", value=db_max - timedelta(days=365),
                               min_value=db_min, max_value=db_max)
    end_date   = st.date_input("To",   value=db_max,
                               min_value=db_min, max_value=db_max)

    st.divider()
    st.markdown("**YoY comparison years**")
    available_years = list(range(db_min.year, db_max.year + 1))
    yoy_years = st.multiselect(
        "Select years", options=available_years,
        default=[y for y in [2020, 2022, 2024, 2026] if y in available_years],
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🦆 Duck Curve")
st.caption("Average daily load shape: how solar generation hollows out mid-day demand.")

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

df = load_duck_data(engine, start_date, end_date)
if df.empty:
    st.warning("No data for selected period.")
    st.stop()

dc = duck_curve(df)
cl = get_chart_layout()
tick_periods, tick_labels = _x_axis_ticks()

# ── Duck curve chart ──────────────────────────────────────────────────────────
st.subheader("Duck curve — Demand, Solar, Net Demand")

fig_dc = go.Figure()
fig_dc.update_layout(**cl)

fig_dc.add_trace(go.Scatter(
    x=dc["period"], y=dc["demand_mw"],
    mode="lines", name="Demand",
    line=dict(color=COLOR_DEMAND, width=2.5),
    hovertemplate="Period %{x} (%{customdata})<br>Demand: %{y:.0f} MW<extra></extra>",
    customdata=dc["time_label"],
))
fig_dc.add_trace(go.Scatter(
    x=dc["period"], y=dc["solar_mw"],
    mode="lines", name="Solar",
    line=dict(color=COLOR_SOLAR, width=2.5),
    hovertemplate="Period %{x} (%{customdata})<br>Solar: %{y:.0f} MW<extra></extra>",
    customdata=dc["time_label"],
))
fig_dc.add_trace(go.Scatter(
    x=dc["period"], y=dc["net_demand"],
    mode="lines", name="Net Demand",
    line=dict(color=COLOR_USEP, width=2.5),
    fill="tozeroy", fillcolor="rgba(0,156,234,0.08)",
    hovertemplate="Period %{x} (%{customdata})<br>Net: %{y:.0f} MW<extra></extra>",
    customdata=dc["time_label"],
))

fig_dc.update_layout(
    height=400,
    xaxis=dict(
        tickmode="array", tickvals=tick_periods, ticktext=tick_labels,
        title="Time of day (SGT)", **cl.get("xaxis", {}),
    ),
    yaxis=dict(title="MW", **cl.get("yaxis", {})),
    hovermode="x unified", showlegend=True,
)
st.plotly_chart(fig_dc, use_container_width=True)

st.divider()

# ── USEP by period bar chart ──────────────────────────────────────────────────
st.subheader("Average USEP by period")

usep_profile = (
    df.groupby("period")["usep"]
    .mean()
    .reset_index()
    .rename(columns={"usep": "avg_usep"})
)
usep_profile["time_label"] = usep_profile["period"].apply(_period_to_hhmm)
usep_profile["color"] = usep_profile["avg_usep"].apply(
    lambda v: "#2ecc71" if v < 100 else ("#f0b429" if v < 200 else "#e74c3c")
)

fig_bar = go.Figure()
fig_bar.update_layout(**cl)
fig_bar.add_trace(go.Bar(
    x=usep_profile["period"],
    y=usep_profile["avg_usep"],
    marker_color=usep_profile["color"],
    name="Avg USEP",
    hovertemplate="Period %{x} (%{customdata})<br>Avg USEP: S$%{y:.2f}/MWh<extra></extra>",
    customdata=usep_profile["time_label"],
))
fig_bar.update_layout(
    height=320,
    xaxis=dict(
        tickmode="array", tickvals=tick_periods, ticktext=tick_labels,
        title="Time of day (SGT)", **cl.get("xaxis", {}),
    ),
    yaxis=dict(title="Avg USEP (S$/MWh)", **cl.get("yaxis", {})),
)
# Legend annotation for colour bands
for color, label in [("#2ecc71", "< $100"), ("#f0b429", "$100–200"), ("#e74c3c", "> $200")]:
    fig_bar.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color=color),
        name=label,
    ))
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Solar suppression panel ───────────────────────────────────────────────────
st.subheader("Solar suppression — USEP by period: high-solar vs low-solar months")

df["month"] = df["date"].dt.month
# High-solar: Oct–Feb; low-solar: May–Aug
df_high = df[df["month"].isin([10, 11, 12, 1, 2])]
df_low  = df[df["month"].isin([5, 6, 7, 8])]

usep_high = df_high.groupby("period")["usep"].mean().reset_index().rename(columns={"usep": "avg_usep"})
usep_low  = df_low.groupby("period")["usep"].mean().reset_index().rename(columns={"usep": "avg_usep"})

fig_sup = go.Figure()
fig_sup.update_layout(**cl)
fig_sup.add_trace(go.Scatter(
    x=usep_high["period"], y=usep_high["avg_usep"],
    mode="lines", name="High-solar months (Oct–Feb)",
    line=dict(color=COLOR_SOLAR, width=2),
    hovertemplate="Period %{x}<br>Avg USEP: S$%{y:.2f}/MWh<extra></extra>",
))
fig_sup.add_trace(go.Scatter(
    x=usep_low["period"], y=usep_low["avg_usep"],
    mode="lines", name="Low-solar months (May–Aug)",
    line=dict(color=COLOR_DEMAND, width=2),
    hovertemplate="Period %{x}<br>Avg USEP: S$%{y:.2f}/MWh<extra></extra>",
))
fig_sup.update_layout(
    height=340,
    xaxis=dict(
        tickmode="array", tickvals=tick_periods, ticktext=tick_labels,
        title="Time of day (SGT)", **cl.get("xaxis", {}),
    ),
    yaxis=dict(title="Avg USEP (S$/MWh)", **cl.get("yaxis", {})),
    hovermode="x unified",
)
st.plotly_chart(fig_sup, use_container_width=True)
st.caption(
    "Solar suppression effect: during high-solar months, mid-day USEP (periods 14–28) "
    "is pushed lower by excess solar generation feeding into the grid."
)

st.divider()

# ── YoY duck curve evolution ──────────────────────────────────────────────────
st.subheader("Year-on-year duck curve evolution")

if not yoy_years:
    st.info("Select years in the sidebar to compare.")
else:
    YOY_COLORS = ["#009CEA", "#f0b429", "#2ecc71", "#e74c3c", "#9b59b6", "#1abc9c"]
    fig_yoy = go.Figure()
    fig_yoy.update_layout(**cl)

    for i, yr in enumerate(sorted(yoy_years)):
        yr_df = load_duck_for_year(engine, yr)
        if yr_df.empty:
            continue
        color = YOY_COLORS[i % len(YOY_COLORS)]
        fig_yoy.add_trace(go.Scatter(
            x=yr_df["period"], y=yr_df["net_demand"],
            mode="lines", name=str(yr),
            line=dict(color=color, width=2),
            hovertemplate=f"{yr} — Period %{{x}} (%{{customdata}})<br>Net Demand: %{{y:.0f}} MW<extra></extra>",
            customdata=yr_df["time_label"],
        ))

    fig_yoy.update_layout(
        height=380,
        xaxis=dict(
            tickmode="array", tickvals=tick_periods, ticktext=tick_labels,
            title="Time of day (SGT)", **cl.get("xaxis", {}),
        ),
        yaxis=dict(title="Net Demand (MW)", **cl.get("yaxis", {})),
        hovermode="x unified",
    )
    st.plotly_chart(fig_yoy, use_container_width=True)
    st.caption(
        "Net Demand = System Demand − Solar. The deepening mid-day trough from 2020→2026 "
        "reflects Singapore's rapidly growing rooftop and utility-scale solar capacity."
    )
