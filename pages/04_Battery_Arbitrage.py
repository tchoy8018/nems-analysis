from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import get_engine, setup_database
from modules.analysis import arbitrage_windows
from modules.theme import apply_theme_css, get_chart_layout, render_theme_toggle
from config import CAPACITY_MW, EFFICIENCY, UTILIZATION


@st.cache_resource
def _get_engine():
    engine = get_engine()
    setup_database(engine)
    return engine


@st.cache_data(ttl=300)
def load_prices(_engine, start: date, end: date) -> pd.DataFrame:
    from sqlalchemy import text
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, period, usep
            FROM nems_prices
            WHERE date >= :s AND date <= :e AND usep IS NOT NULL
            ORDER BY date, period
        """), {"s": start, "e": end}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "period", "usep"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
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


st.set_page_config(page_title="Battery Arbitrage — NEMS", layout="wide")

engine = _get_engine()
apply_theme_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ NEMS Analytics")
    st.caption("Singa Renewables")
    render_theme_toggle()
    st.divider()

    st.markdown("**BESS Parameters**")
    capacity_mw = st.slider(
        "Capacity (MW)", min_value=100, max_value=1000,
        value=int(CAPACITY_MW), step=10,
    )
    efficiency_pct = st.slider(
        "Round-trip Efficiency (%)", min_value=50, max_value=98,
        value=int(EFFICIENCY * 100), step=1,
    )
    utilization_pct = st.slider(
        "Utilization (%)", min_value=10, max_value=100,
        value=int(UTILIZATION * 100), step=5,
    )
    efficiency   = efficiency_pct / 100
    utilization  = utilization_pct / 100
    effective_mw = capacity_mw * utilization

    st.divider()
    status = load_db_status(engine)
    db_min = pd.to_datetime(status.get("min_d", "2019-01-01")).date()
    db_max = pd.to_datetime(status.get("max_d", date.today())).date()

    st.markdown("**Analysis period**")
    start_date = st.date_input("From", value=db_max - timedelta(days=365),
                               min_value=db_min, max_value=db_max)
    end_date   = st.date_input("To",   value=db_max,
                               min_value=db_min, max_value=db_max)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🔋 Battery Arbitrage")
st.caption(
    f"Optimal charge/discharge strategy for a {capacity_mw} MW BESS "
    f"({efficiency_pct}% RT efficiency, {utilization_pct}% utilization → "
    f"{effective_mw:.0f} MW effective)"
)

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

df = load_prices(engine, start_date, end_date)
if df.empty:
    st.warning("No price data for selected period.")
    st.stop()

# Formula callout
with st.expander("📐 Revenue formula", expanded=False):
    st.latex(
        r"\text{Revenue per period} = \text{USEP} \times "
        r"\underbrace{(" + str(capacity_mw) + r"\,\text{MW} \times "
        + f"{utilization:.2f}" + r")}_{\text{eff. capacity}} \times \eta_{\text{RT}} \times 0.5\,\text{h}"
    )
    st.markdown(
        f"Net daily P&L = Σ discharge revenue − Σ charge cost  \n"
        f"> Discharge: USEP × {effective_mw:.0f} MW × {efficiency:.2f} × 0.5 h  \n"
        f"> Charge cost: USEP × {effective_mw:.0f} MW × 0.5 h"
    )

result = arbitrage_windows(df, capacity_mw=capacity_mw,
                           efficiency=efficiency, utilization=utilization)

charge_w    = result["charge_windows"]
discharge_w = result["discharge_windows"]
daily_rev   = result["daily_revenue"]
monthly_rev = result["monthly_revenue"]

st.divider()

# ── Optimal windows ───────────────────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("🔌 Top 5 charge periods (lowest avg USEP)")
    charge_display = charge_w[["period", "time_label", "avg_usep"]].copy()
    charge_display.columns = ["Period", "Time (SGT)", "Avg USEP (S$/MWh)"]
    charge_display["Avg USEP (S$/MWh)"] = charge_display["Avg USEP (S$/MWh)"].round(2)
    st.dataframe(charge_display, use_container_width=True, hide_index=True)

with col_d:
    st.subheader("⚡ Top 5 discharge periods (highest avg USEP)")
    discharge_display = discharge_w[["period", "time_label", "avg_usep"]].copy()
    discharge_display.columns = ["Period", "Time (SGT)", "Avg USEP (S$/MWh)"]
    discharge_display["Avg USEP (S$/MWh)"] = discharge_display["Avg USEP (S$/MWh)"].round(2)
    st.dataframe(discharge_display, use_container_width=True, hide_index=True)

# Spread headline
avg_charge    = charge_w["avg_usep"].mean()
avg_discharge = discharge_w["avg_usep"].mean()
avg_spread    = avg_discharge - avg_charge
st.metric(
    "Average daily price spread",
    f"S${avg_spread:.2f}/MWh",
    help=f"Discharge avg: S${avg_discharge:.2f}  |  Charge avg: S${avg_charge:.2f}",
)

st.divider()

# ── Monthly revenue bar chart ─────────────────────────────────────────────────
st.subheader("Monthly revenue estimate")

cl = get_chart_layout()

fig_month = go.Figure()
fig_month.update_layout(**cl)
fig_month.add_trace(go.Bar(
    x=monthly_rev["year_month"],
    y=monthly_rev["revenue_sgd"] / 1e6,
    marker_color="#009CEA",
    name="Revenue",
    hovertemplate="%{x}<br>Revenue: S$%{y:.2f}M<extra></extra>",
))
fig_month.update_layout(
    height=340,
    xaxis=dict(title="Month", **cl.get("xaxis", {})),
    yaxis=dict(title="Estimated revenue (S$ million)", **cl.get("yaxis", {})),
)
st.plotly_chart(fig_month, use_container_width=True)

# Monthly revenue table (last 12 months)
last_12 = monthly_rev.tail(12).copy()
last_12["revenue_sgd"] = (last_12["revenue_sgd"] / 1e6).round(3)
last_12.columns = ["Month", "Revenue (S$M)"]
st.dataframe(last_12, use_container_width=True, hide_index=True)

st.divider()

# ── Annual projection ─────────────────────────────────────────────────────────
st.subheader("Annual revenue projection")

if len(daily_rev) >= 30:
    avg_daily   = daily_rev["revenue_sgd"].mean()
    p10_daily   = daily_rev["revenue_sgd"].quantile(0.10)
    p90_daily   = daily_rev["revenue_sgd"].quantile(0.90)

    annual_exp  = avg_daily * 365
    annual_p10  = p10_daily * 365
    annual_p90  = p90_daily * 365

    a1, a2, a3 = st.columns(3)
    a1.metric("Expected (P50)", f"S${annual_exp/1e6:.1f}M/yr")
    a2.metric("Downside (P10)", f"S${annual_p10/1e6:.1f}M/yr",
              delta=f"{(annual_p10-annual_exp)/1e6:+.1f}M vs P50")
    a3.metric("Upside (P90)",   f"S${annual_p90/1e6:.1f}M/yr",
              delta=f"{(annual_p90-annual_exp)/1e6:+.1f}M vs P50")

    st.caption(
        f"Based on {len(daily_rev):,} trading days in selected period. "
        f"Avg daily revenue: S${avg_daily:,.0f} · "
        f"P10: S${p10_daily:,.0f} · P90: S${p90_daily:,.0f}"
    )

    # Daily revenue distribution
    fig_dist = go.Figure()
    fig_dist.update_layout(**cl)
    fig_dist.add_trace(go.Histogram(
        x=daily_rev["revenue_sgd"] / 1000,
        nbinsx=60,
        marker_color="#009CEA",
        opacity=0.8,
        name="Daily revenue",
        hovertemplate="S$%{x:.0f}k<br>Days: %{y}<extra></extra>",
    ))
    fig_dist.add_vline(
        x=avg_daily / 1000, line_color="#f0b429", line_dash="dot",
        annotation_text=f"P50 S${avg_daily/1000:.0f}k", annotation_font_color="#f0b429",
    )
    fig_dist.add_vline(
        x=p10_daily / 1000, line_color="#e74c3c", line_dash="dash",
        annotation_text=f"P10", annotation_font_color="#e74c3c",
    )
    fig_dist.add_vline(
        x=p90_daily / 1000, line_color="#2ecc71", line_dash="dash",
        annotation_text=f"P90", annotation_font_color="#2ecc71",
    )
    fig_dist.update_layout(
        height=300,
        xaxis=dict(title="Daily revenue (S$'000)", **cl.get("xaxis", {})),
        yaxis=dict(title="Number of days", **cl.get("yaxis", {})),
        showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True)
else:
    st.info("Select at least 30 days to generate a projection.")
