"""
Battery Dispatch & Market Upside — Singa Renewables
Correct commercial framing: CfD base revenue + intraday BESS optimization upside.
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import get_engine, setup_database
from modules.theme import (
    apply_theme_css, get_chart_layout, get_rangeselector_style,
    render_theme_toggle,
)

SOLAR_MW = 560  # nameplate solar capacity


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


def _compute_dispatch(
    df: pd.DataFrame,
    contracted_daily_mwh: float,
    bess_energy_mwh: float,
    n_charge: int,
    n_discharge: int,
    usep_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    For each date in df, compute:
      - daily_avg_usep
      - discharge_avg_usep  (top n_discharge periods)
      - charge_avg_usep     (bottom n_charge periods)
      - bess_shift_mwh      = min(bess_energy_mwh, contracted_daily_mwh)
      - cfd_revenue         = contracted_daily_mwh × daily_avg_usep
      - market_upside       = bess_shift_mwh × (discharge_avg_usep − daily_avg_usep)
    """
    rows = []
    for d, day_df in df.groupby("date"):
        usep = day_df["usep"].dropna() * usep_multiplier
        if len(usep) < 10:
            continue
        daily_avg = usep.mean()
        sorted_u = usep.sort_values()
        charge_avg = sorted_u.head(n_charge).mean()
        discharge_avg = sorted_u.tail(n_discharge).mean()
        shift = min(bess_energy_mwh, contracted_daily_mwh)
        cfd = contracted_daily_mwh * daily_avg
        upside = shift * (discharge_avg - daily_avg)
        rows.append({
            "date": d,
            "daily_avg_usep": daily_avg,
            "charge_avg_usep": charge_avg,
            "discharge_avg_usep": discharge_avg,
            "bess_shift_mwh": shift,
            "cfd_revenue": cfd,
            "market_upside": max(0.0, upside),
        })
    return pd.DataFrame(rows)


# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Battery Dispatch — NEMS", layout="wide")

engine = _get_engine()
apply_theme_css()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ NEMS Analytics")
    st.caption("Singa Renewables")
    render_theme_toggle()
    st.divider()

    st.markdown("**Contracted volume mode**")
    commit_mode = st.radio(
        "Settlement basis",
        ["Daily commitment", "Weekly commitment"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("**Plant Load Factor (PLF)**")
    plf_pct = st.radio(
        "PLF", [50, 60, 65, 70, 75],
        index=2,
        horizontal=True,
        format_func=lambda x: f"{x}%",
        label_visibility="collapsed",
    )
    plf = plf_pct / 100

    st.markdown("**BESS round-trip efficiency**")
    efficiency_pct = st.slider("Efficiency (%)", 60, 95, 75, step=5)
    efficiency = efficiency_pct / 100

    st.markdown("**BESS storage duration**")
    storage_hours = st.radio(
        "Duration", [2, 4, 6],
        index=1,
        horizontal=True,
        format_func=lambda x: f"{x}h",
        label_visibility="collapsed",
    )

    st.divider()
    status = load_db_status(engine)
    db_min = pd.to_datetime(status.get("min_d", "2019-01-01")).date()
    db_max = pd.to_datetime(status.get("max_d", date.today())).date()

    st.markdown("**Analysis period**")
    start_date = st.date_input("From", value=db_max - timedelta(days=365),
                               min_value=db_min, max_value=db_max)
    end_date   = st.date_input("To",   value=db_max,
                               min_value=db_min, max_value=db_max)

# ── Derived parameters ─────────────────────────────────────────────────────────
contracted_daily_mwh = plf * SOLAR_MW * 24          # e.g. 0.65 × 560 × 24 = 8,736 MWh
contracted_annual_mwh = contracted_daily_mwh * 365

n_discharge = storage_hours * 2                     # periods (each 0.5h)
n_charge    = storage_hours * 2
bess_energy_mwh = SOLAR_MW * efficiency * storage_hours  # BESS discharge capacity MWh

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("⚡ Battery Dispatch & Market Upside")
st.caption("Intraday optimization within daily/weekly contracted volume")

st.info(
    f"**Commercial framing:** Singa Renewables targets a PPA with a "
    f"{'daily' if commit_mode == 'Daily commitment' else 'weekly'} volume commitment "
    f"settled on the daily average USEP. This gives full intraday dispatch flexibility — "
    f"the BESS optimizes **when** to deliver within the day, capturing the spread between "
    f"peak discharge USEP and the daily average CfD settlement price. "
    f"This page quantifies that incremental upside.  \n\n"
    f"**PLF {plf_pct}%** → Contracted: **{contracted_daily_mwh:,.0f} MWh/day** · "
    f"BESS shift capacity: **{bess_energy_mwh:,.0f} MWh** ({storage_hours}h × "
    f"{SOLAR_MW} MW × {efficiency_pct}% η)"
)

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

df = load_prices(engine, start_date, end_date)
if df.empty:
    st.warning("No data for selected period.")
    st.stop()

daily = _compute_dispatch(df, contracted_daily_mwh, bess_energy_mwh, n_charge, n_discharge)
if daily.empty:
    st.warning("Not enough data to compute dispatch model.")
    st.stop()

n_days = len(daily)
cl  = get_chart_layout()
rs  = get_rangeselector_style()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CfD Base Revenue
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Contracted Layer — CfD Revenue")

st.markdown(
    r"**Formula:** Daily CfD Revenue = Contracted MWh × USEP_daily_average"
)

avg_daily_usep = daily["daily_avg_usep"].mean()
annual_cfd = daily["cfd_revenue"].sum() * (365 / n_days)
monthly_cfd = (
    daily.copy()
    .assign(date=lambda d: pd.to_datetime(d["date"]))
    .set_index("date")
    .resample("ME")
    .agg(
        avg_daily_usep=("daily_avg_usep", "mean"),
        cfd_revenue=("cfd_revenue", "sum"),
    )
    .reset_index()
)
monthly_cfd["month"] = monthly_cfd["date"].dt.strftime("%Y-%m")
monthly_cfd["cfd_revenue_sgd_m"] = monthly_cfd["cfd_revenue"] / 1e6

s1_c1, s1_c2, s1_c3 = st.columns(3)
s1_c1.metric("Avg Daily USEP", f"S${avg_daily_usep:.2f}/MWh")
s1_c2.metric("Contracted MWh/day", f"{contracted_daily_mwh:,.0f} MWh")
s1_c3.metric(
    "Annual CfD Revenue (floor)",
    f"S${annual_cfd/1e6:.1f}M",
    help="Annualised from selected period. This is the base revenue — fully de-risked vs intraday volatility.",
)

with st.expander("Monthly CfD revenue table"):
    cfd_display = monthly_cfd[["month", "avg_daily_usep", "cfd_revenue_sgd_m"]].copy()
    cfd_display.columns = ["Month", "Avg Daily USEP (S$/MWh)", "Monthly CfD Revenue (S$M)"]
    cfd_display["Avg Daily USEP (S$/MWh)"] = cfd_display["Avg Daily USEP (S$/MWh)"].round(2)
    cfd_display["Monthly CfD Revenue (S$M)"] = cfd_display["Monthly CfD Revenue (S$M)"].round(2)
    st.dataframe(cfd_display, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Intraday USEP Profile
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Intraday Price Spread")

period_avg = df.groupby("period")["usep"].mean().reset_index().rename(columns={"usep": "avg_usep"})
period_avg["time_label"] = period_avg["period"].apply(_period_to_hhmm)

tick_ps = list(range(1, 49, 6))
xax = dict(
    tickmode="array",
    tickvals=tick_ps,
    ticktext=[_period_to_hhmm(p) for p in tick_ps],
    title="Time of day (SGT)",
    **cl.get("xaxis", {}),
)

fig_profile = go.Figure()
fig_profile.update_layout(**cl)

# Fill green where USEP > daily avg (discharge zone), red where < (charge zone)
fig_profile.add_trace(go.Scatter(
    x=period_avg["period"], y=[avg_daily_usep] * 48,
    mode="lines", name=f"Daily avg (S${avg_daily_usep:.2f})",
    line=dict(color="#f0b429", width=1.5, dash="dot"),
    hoverinfo="skip",
))
above = period_avg["avg_usep"].clip(lower=avg_daily_usep)
below = period_avg["avg_usep"].clip(upper=avg_daily_usep)

fig_profile.add_trace(go.Scatter(
    x=period_avg["period"], y=above,
    fill="tonexty", fillcolor="rgba(46,204,113,0.18)",
    mode="none", name="Discharge opportunity (USEP > avg)",
    showlegend=True,
))
fig_profile.add_trace(go.Scatter(
    x=period_avg["period"], y=below,
    fill="tonexty", fillcolor="rgba(231,76,60,0.18)",
    mode="none", name="Charge opportunity (USEP < avg)",
    showlegend=True,
))
# Reset reference line so fills work correctly
fig_profile.data = (fig_profile.data[1], fig_profile.data[2], fig_profile.data[0])

fig_profile.add_trace(go.Scatter(
    x=period_avg["period"], y=period_avg["avg_usep"],
    mode="lines", name="Avg USEP by period",
    line=dict(color="#009CEA", width=2.5),
    hovertemplate="Period %{x} (%{customdata})<br>Avg USEP: S$%{y:.2f}/MWh<extra></extra>",
    customdata=period_avg["time_label"],
))

fig_profile.update_layout(
    height=360, xaxis=xax,
    yaxis=dict(title="Avg USEP (S$/MWh)", **cl.get("yaxis", {})),
    hovermode="x unified",
)
st.plotly_chart(fig_profile, use_container_width=True)

# Best charge / discharge windows
col_ch, col_dis = st.columns(2)
with col_ch:
    st.markdown("**Best 6 charge periods** (lowest avg USEP)")
    best_charge = period_avg.nsmallest(6, "avg_usep").sort_values("avg_usep").copy()
    best_charge["premium"] = (best_charge["avg_usep"] - avg_daily_usep).round(2)
    best_charge_disp = best_charge[["period", "time_label", "avg_usep", "premium"]].copy()
    best_charge_disp.columns = ["Period", "Time (SGT)", "Avg USEP", "vs Daily Avg"]
    best_charge_disp["Avg USEP"] = best_charge_disp["Avg USEP"].round(2)
    st.dataframe(best_charge_disp, use_container_width=True, hide_index=True)

with col_dis:
    st.markdown("**Best 6 discharge periods** (highest avg USEP)")
    best_discharge = period_avg.nlargest(6, "avg_usep").sort_values("avg_usep", ascending=False).copy()
    best_discharge["premium"] = (best_discharge["avg_usep"] - avg_daily_usep).round(2)
    best_discharge_disp = best_discharge[["period", "time_label", "avg_usep", "premium"]].copy()
    best_discharge_disp.columns = ["Period", "Time (SGT)", "Avg USEP", "Premium vs Daily Avg"]
    best_discharge_disp["Avg USEP"] = best_discharge_disp["Avg USEP"].round(2)
    st.dataframe(best_discharge_disp, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Dispatch Optimization
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔋 BESS Temporal Optimization")

avg_upside = daily["market_upside"].mean()
annual_upside = avg_upside * 365
total_annual = annual_cfd + annual_upside

s3_c1, s3_c2, s3_c3 = st.columns(3)
s3_c1.metric("Avg Daily Market Upside", f"S${avg_upside:,.0f}")
s3_c2.metric("Annual Market Upside", f"S${annual_upside/1e6:.1f}M")
s3_c3.metric("Total Annual Revenue", f"S${total_annual/1e6:.1f}M",
             delta=f"+S${annual_upside/1e6:.1f}M vs CfD floor")

# Daily upside time-series
daily_plot = daily.copy()
daily_plot["date"] = pd.to_datetime(daily_plot["date"])

fig_daily = go.Figure()
fig_daily.update_layout(**cl)
fig_daily.add_trace(go.Scatter(
    x=daily_plot["date"], y=daily_plot["market_upside"] / 1000,
    mode="lines+markers", name="Daily upside",
    line=dict(color="#2ecc71", width=1),
    marker=dict(size=3, color="#2ecc71"),
    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Upside: S$%{y:.0f}k<extra></extra>",
))

# 30-day rolling average
rolling_upside = (
    daily_plot.set_index("date")["market_upside"]
    .rolling("30D").mean()
    .reset_index()
)
fig_daily.add_trace(go.Scatter(
    x=rolling_upside["date"], y=rolling_upside["market_upside"] / 1000,
    mode="lines", name="30-day avg",
    line=dict(color="#f0b429", width=2, dash="dot"),
    hovertemplate="30-day avg: S$%{y:.0f}k<extra></extra>",
))

fig_daily.update_layout(
    height=340,
    xaxis=dict(
        rangeselector=dict(
            buttons=[
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
            **rs,
        ),
        rangeslider=dict(visible=True, thickness=0.04),
        type="date",
        **cl.get("xaxis", {}),
    ),
    yaxis=dict(title="Daily market upside (S$'000)", **cl.get("yaxis", {})),
    hovermode="x unified",
)
st.plotly_chart(fig_daily, use_container_width=True)

# Monthly average upside bar
monthly_upside = (
    daily_plot.set_index("date")
    .resample("ME")["market_upside"]
    .sum()
    .reset_index()
)
monthly_upside["month"] = monthly_upside["date"].dt.strftime("%Y-%m")

fig_mu = go.Figure()
fig_mu.update_layout(**cl)
fig_mu.add_trace(go.Bar(
    x=monthly_upside["month"],
    y=monthly_upside["market_upside"] / 1e6,
    marker_color="#2ecc71",
    hovertemplate="%{x}<br>Monthly upside: S$%{y:.2f}M<extra></extra>",
))
fig_mu.update_layout(
    height=260,
    xaxis=dict(title="Month", **cl.get("xaxis", {})),
    yaxis=dict(title="Market upside (S$M)", **cl.get("yaxis", {})),
    showlegend=False,
)
st.plotly_chart(fig_mu, use_container_width=True)

# Distribution of daily upside
fig_hist = go.Figure()
fig_hist.update_layout(**cl)
fig_hist.add_trace(go.Histogram(
    x=daily["market_upside"] / 1000,
    nbinsx=50,
    marker_color="#009CEA",
    opacity=0.85,
    hovertemplate="S$%{x:.0f}k<br>Days: %{y}<extra></extra>",
))
p10 = daily["market_upside"].quantile(0.10) / 1000
p90 = daily["market_upside"].quantile(0.90) / 1000
avg  = daily["market_upside"].mean() / 1000
fig_hist.add_vline(x=avg,  line_color="#f0b429", line_dash="dot",
                   annotation_text=f"P50 S${avg:.0f}k", annotation_font_color="#f0b429")
fig_hist.add_vline(x=p10,  line_color="#e74c3c", line_dash="dash",
                   annotation_text="P10", annotation_font_color="#e74c3c")
fig_hist.add_vline(x=p90,  line_color="#2ecc71", line_dash="dash",
                   annotation_text="P90", annotation_font_color="#2ecc71")
fig_hist.update_layout(
    height=270,
    xaxis=dict(title="Daily market upside (S$'000)", **cl.get("xaxis", {})),
    yaxis=dict(title="Number of days", **cl.get("yaxis", {})),
    showlegend=False,
)
st.plotly_chart(fig_hist, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Annual Revenue Summary by PLF
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("💰 Revenue Stack — Annual Estimate")

plf_scenarios = [50, 60, 65, 70, 75]

@st.cache_data(ttl=300)
def _revenue_by_plf(daily_usep_mean: float, daily_upside_mean: float,
                    bess_e: float, eff: float, stor_h: int, scenarios: tuple) -> list:
    results = []
    for p in scenarios:
        cdm = (p / 100) * SOLAR_MW * 24
        ann_cfd = cdm * daily_usep_mean * 365
        shift = min(bess_e, cdm)
        avg_spread = daily_upside_mean / min(bess_energy_mwh, contracted_daily_mwh) if min(bess_energy_mwh, contracted_daily_mwh) > 0 else 0
        ann_up = shift * avg_spread * 365
        results.append({
            "PLF": f"{p}%",
            "Contracted MWh/yr": f"{cdm * 365 / 1e6:.2f}M",
            "CfD Revenue (S$M)": round(ann_cfd / 1e6, 1),
            "Market Upside (S$M)": round(ann_up / 1e6, 1),
            "Total (S$M)": round((ann_cfd + ann_up) / 1e6, 1),
        })
    return results

# Use actual spread per shifted MWh from the computed daily data
shift_mwh = daily["bess_shift_mwh"].mean()
avg_spread_per_mwh = (daily["market_upside"] / shift_mwh.clip(lower=1)).mean()

plf_rows = []
plf_cfd_vals, plf_up_vals, plf_labels = [], [], []
for p in plf_scenarios:
    cdm = (p / 100) * SOLAR_MW * 24
    ann_cfd_p = cdm * avg_daily_usep * 365
    shift_p = min(bess_energy_mwh, cdm)
    ann_up_p = shift_p * avg_spread_per_mwh * 365
    plf_rows.append({
        "PLF": f"{p}%",
        "Contracted MWh/yr": f"{cdm*365/1e6:.2f}M MWh",
        "CfD Revenue (S$M)": round(ann_cfd_p / 1e6, 1),
        "Market Upside (S$M)": round(ann_up_p / 1e6, 1),
        "Total (S$M)": round((ann_cfd_p + ann_up_p) / 1e6, 1),
    })
    plf_cfd_vals.append(ann_cfd_p / 1e6)
    plf_up_vals.append(ann_up_p / 1e6)
    plf_labels.append(f"{p}%")

fig_stack = go.Figure()
fig_stack.update_layout(**cl)
fig_stack.add_trace(go.Bar(
    x=plf_labels, y=plf_cfd_vals,
    name="CfD Base Revenue",
    marker_color="#009CEA",
    hovertemplate="PLF %{x}<br>CfD: S$%{y:.1f}M<extra></extra>",
))
fig_stack.add_trace(go.Bar(
    x=plf_labels, y=plf_up_vals,
    name="Market Upside (BESS)",
    marker_color="#2ecc71",
    hovertemplate="PLF %{x}<br>Upside: S$%{y:.1f}M<extra></extra>",
))
for i, (c, u) in enumerate(zip(plf_cfd_vals, plf_up_vals)):
    fig_stack.add_annotation(
        x=plf_labels[i], y=c + u + 2,
        text=f"S${c+u:.0f}M",
        showarrow=False,
        font=dict(color=cl["font"]["color"], size=11),
    )
fig_stack.update_layout(
    barmode="stack", height=360,
    xaxis=dict(title="Plant Load Factor", **cl.get("xaxis", {})),
    yaxis=dict(title="Annual Revenue (S$M)", **cl.get("yaxis", {})),
)
st.plotly_chart(fig_stack, use_container_width=True)

plf_df = pd.DataFrame(plf_rows)
st.dataframe(plf_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Sensitivity: PLF × USEP scenario
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Market Upside Sensitivity")
st.caption("Annual market upside (S$M) — PLF rows × USEP scenario columns")

usep_scenarios = {"−20% USEP": 0.80, "Base USEP": 1.00, "+20% USEP": 1.20}
sens_rows = []
for p in plf_scenarios:
    cdm = (p / 100) * SOLAR_MW * 24
    shift_p = min(bess_energy_mwh, cdm)
    row = {"PLF": f"{p}%"}
    for label, factor in usep_scenarios.items():
        scaled_upside = shift_p * avg_spread_per_mwh * factor * 365
        row[label] = round(scaled_upside / 1e6, 1)
    sens_rows.append(row)

sens_df = pd.DataFrame(sens_rows).set_index("PLF")

# Colour the cells
col_min = sens_df.values.min()
col_max = sens_df.values.max()

def _cell_style(val):
    if col_max == col_min:
        pct = 0.5
    else:
        pct = (val - col_min) / (col_max - col_min)
    r = int(255 * (1 - pct))
    g = int(180 + 75 * pct)
    b = int(100 * (1 - pct))
    return f"background-color: rgb({r},{g},{b}); color: #000"

st.dataframe(
    sens_df.style.applymap(_cell_style).format("S${:.1f}M"),
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Disclaimer:** Model assumes BESS can charge/discharge freely within the daily "
    "contracted volume. Actual dispatch depends on HVDC cable scheduling, grid operator "
    "requirements, and final CfD terms. Ancillary services revenue excluded. "
    f"BESS sized at {SOLAR_MW} MW × {storage_hours}h × {efficiency_pct}% η = "
    f"{bess_energy_mwh:,.0f} MWh shift capacity."
)
