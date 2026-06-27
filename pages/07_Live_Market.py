"""
📡 Live Market — real-time USEP monitoring, intraday charts, demand-price scatter.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import get_engine, setup_database
from modules.scraper import fetch_live_api, get_last_data_timestamp, detect_gaps_and_fill
from modules.theme import (
    add_copy_button, apply_theme_css, get_chart_layout,
    get_yaxis2_style, render_theme_toggle,
)
from modules.utils import _axis
from config import COLOR_USEP, COLOR_DEMAND, COLOR_SPIKE

VESTING_PRICE = 170.0   # S$/MWh
DATA_DIR      = Path(__file__).parent.parent / "data" / "raw"


@st.cache_resource
def _get_engine():
    engine = get_engine()
    setup_database(engine)
    return engine


def _period_to_hhmm(p: int) -> str:
    h, m = divmod((p - 1) * 30, 60)
    return f"{h:02d}:{m:02d}"


def _current_period() -> int:
    now = datetime.now()
    return min(48, now.hour * 2 + now.minute // 30 + 1)


# ── DB loaders ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_today(_engine) -> pd.DataFrame:
    from sqlalchemy import text
    today = date.today()
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT period, usep, demand_mw, solar_mw
            FROM nems_prices
            WHERE date = :d AND usep IS NOT NULL
            ORDER BY period
        """), {"d": today}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["period", "usep", "demand_mw", "solar_mw"])
    df["time_label"] = df["period"].apply(_period_to_hhmm)
    return df


@st.cache_data(ttl=60)
def load_yesterday(_engine) -> pd.DataFrame:
    from sqlalchemy import text
    yesterday = date.today() - timedelta(days=1)
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT period, usep, demand_mw
            FROM nems_prices
            WHERE date = :d AND usep IS NOT NULL
            ORDER BY period
        """), {"d": yesterday}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["period", "usep", "demand_mw"])
    df["time_label"] = df["period"].apply(_period_to_hhmm)
    return df


@st.cache_data(ttl=60)
def load_last_week_same_day(_engine) -> pd.DataFrame:
    from sqlalchemy import text
    last_week = date.today() - timedelta(days=7)
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT period, usep FROM nems_prices
            WHERE date = :d AND usep IS NOT NULL ORDER BY period
        """), {"d": last_week}).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["period", "usep"])


@st.cache_data(ttl=300)
def load_7day_heatmap(_engine) -> pd.DataFrame:
    from sqlalchemy import text
    cutoff = date.today() - timedelta(days=7)
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, period, usep FROM nems_prices
            WHERE date >= :c AND usep IS NOT NULL
            ORDER BY date DESC, period
        """), {"c": cutoff}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "period", "usep"])
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_scatter_data(_engine, days: int = 30) -> pd.DataFrame:
    from sqlalchemy import text
    cutoff = date.today() - timedelta(days=days)
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, period, usep, demand_mw
            FROM nems_prices
            WHERE date >= :c AND usep IS NOT NULL AND demand_mw IS NOT NULL
            ORDER BY date, period
        """), {"c": cutoff}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "period", "usep", "demand_mw"])
    return df


@st.cache_data(ttl=300)
def load_live_log_summary(_engine) -> pd.DataFrame:
    from sqlalchemy import text
    try:
        with _engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT source, MAX(fetched_at) AS last_fetched,
                       SUM(periods_new) AS total_new,
                       MAX(error) AS last_error
                FROM live_data_log
                GROUP BY source
                ORDER BY last_fetched DESC
            """)).mappings().fetchall()
        return pd.DataFrame([dict(r) for r in rows])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_inflection_mw(_engine) -> float:
    from sqlalchemy import text
    try:
        with _engine.connect() as conn:
            row = conn.execute(text("""
                SELECT inflection_mw FROM demand_analysis_cache
                ORDER BY computed_at DESC LIMIT 1
            """)).fetchone()
        if row and row[0]:
            return float(row[0])
    except Exception:
        pass
    return 6800.0   # sensible default from our analysis


def _bucket(p: int) -> str:
    if p <= 13:
        return "Off-peak"
    if p <= 28:
        return "Solar"
    if p <= 42:
        return "Evening peak"
    return "Night"


BUCKET_COLORS = {
    "Off-peak": "#009CEA",
    "Solar": "#2ecc71",
    "Evening peak": "#e74c3c",
    "Night": "#9b59b6",
}


# ── Page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Live Market — NEMS", layout="wide")
engine = _get_engine()
apply_theme_css()

with st.sidebar:
    st.title("⚡ NEMS Analytics")
    st.caption("Singa Renewables")
    render_theme_toggle()
    st.divider()

    st.markdown("**📡 Live Data**")
    last_ts = get_last_data_timestamp(engine)
    if last_ts:
        minutes_ago = int((datetime.now() - last_ts).total_seconds() // 60)
        if minutes_ago < 35:
            st.success(f"✅ Live — updated {minutes_ago}m ago")
        elif minutes_ago < 120:
            st.warning(f"⚠️ Last update: {minutes_ago}m ago")
        else:
            st.error(f"❌ Stale — {minutes_ago // 60}h ago")
    else:
        st.warning("No timestamp available")

    cp = _current_period()
    st.caption(f"Current period: P{cp} ({_period_to_hhmm(cp)} SGT)")
    st.divider()

    if st.button("🔄 Fetch latest data", key="fetch_live"):
        with st.spinner("Fetching from EMC..."):
            result = asyncio.run(fetch_live_api(engine))
            pn = result.get("periods_new", 0)
            if pn > 0:
                st.success(f"✅ +{pn} new periods")
                st.cache_data.clear()
                st.rerun()
            elif result.get("error"):
                st.error(result["error"])
            else:
                st.info("Already up to date")

    if st.button("🔍 Fill gaps (7 days)", key="fill_gaps"):
        with st.spinner("Scanning for gaps…"):
            g = detect_gaps_and_fill(engine, lookback_days=7)
            st.info(f"Gaps: {g['gaps_found']} found, {g['gaps_filled']} filled, "
                    f"{g['remaining_gaps']} remaining")
            if g["gaps_filled"]:
                st.cache_data.clear()
                st.rerun()

st.title("📡 Live Market")
st.caption(f"Real-time USEP monitoring — {date.today().strftime('%d %b %Y')} SGT")

# ── Auto-refresh (30 min) ──────────────────────────────────────────────────────
@st.fragment(run_every=1800)
def _auto_refresh():
    try:
        result = asyncio.run(fetch_live_api(engine))
        if result.get("periods_new", 0) > 0:
            st.cache_data.clear()
    except Exception:
        pass

_auto_refresh()

# ── Load data ────────────────────────────────────────────────────────────────
today_df      = load_today(engine)
yesterday_df  = load_yesterday(engine)
lastweek_df   = load_last_week_same_day(engine)
inflection_mw = get_inflection_mw(engine)
cur_period    = _current_period()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — KPI rows
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Today at a Glance")

if today_df.empty:
    st.warning("No data for today yet. Click **Fetch latest data** in the sidebar.")
else:
    # Closest completed period
    past_today  = today_df[today_df["period"] <= cur_period]
    cur_row     = past_today.iloc[-1] if not past_today.empty else today_df.iloc[-1]
    cur_usep    = cur_row["usep"]
    cur_demand  = cur_row["demand_mw"]
    cur_p       = int(cur_row["period"])

    today_avg   = past_today["usep"].mean() if not past_today.empty else None
    today_peak_d = past_today["demand_mw"].dropna().max() if not past_today.empty else None

    # Yesterday comparison
    yd_row   = yesterday_df[yesterday_df["period"] == cur_p]
    yd_usep  = float(yd_row["usep"].iloc[0]) if not yd_row.empty else None
    yd_d     = float(yd_row["demand_mw"].iloc[0]) if not yd_row.empty and "demand_mw" in yd_row.columns else None

    delta_usep   = (cur_usep  - yd_usep)  if yd_usep  else None
    delta_demand = (cur_demand - yd_d)     if yd_d and cur_demand else None

    # 7-day avg USEP for same period
    week_usep = yesterday_df["usep"].mean() if not yesterday_df.empty else None
    delta_week = ((today_avg - week_usep) / week_usep * 100) if (today_avg and week_usep) else None

    # Demand regime
    if cur_demand and cur_demand > 0:
        if cur_demand > inflection_mw:
            regime_label, regime_color = "🔴 High", "#e74c3c"
        elif cur_demand > inflection_mw * 0.92:
            regime_label, regime_color = "🟡 Elevated", "#f0b429"
        else:
            regime_label, regime_color = "🟢 Normal", "#2ecc71"
    else:
        regime_label, regime_color = "—", "#888"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        f"Current USEP (P{cur_p} — {_period_to_hhmm(cur_p)} SGT)",
        f"S${cur_usep:.2f}/MWh" if cur_usep else "N/A",
        delta=f"vs yesterday: {delta_usep:+.2f}" if delta_usep is not None else None,
    )
    col2.metric(
        "Today's avg so far",
        f"S${today_avg:.2f}/MWh" if today_avg else "N/A",
        delta=f"vs 7-day avg: {delta_week:+.1f}%" if delta_week is not None else None,
    )
    col3.metric(
        f"Current demand (P{cur_p})",
        f"{cur_demand:,.0f} MW" if cur_demand else "N/A",
        delta=f"vs yesterday: {delta_demand:+.0f} MW" if delta_demand is not None else None,
    )
    col4.metric(
        "Today's peak demand",
        f"{today_peak_d:,.0f} MW" if today_peak_d else "N/A",
        help=f"Demand regime threshold: {inflection_mw:,.0f} MW",
    )
    st.caption(
        f"Demand regime: **{regime_label}** "
        f"{'(inflection at ' + f'{inflection_mw:,.0f}' + ' MW)' if inflection_mw else ''}"
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Intraday chart
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Intraday USEP — Today vs Yesterday")

cl = get_chart_layout()

fig_intra = go.Figure()
fig_intra.update_layout(**cl)

if not today_df.empty:
    past_t = today_df[today_df["period"] <= cur_period]
    fig_intra.add_trace(go.Scatter(
        x=past_t["period"], y=past_t["usep"],
        mode="lines", name="Today",
        line=dict(color=COLOR_USEP, width=2.5),
        hovertemplate="P%{x} (%{customdata})<br>USEP: S$%{y:.2f}/MWh<extra></extra>",
        customdata=past_t["time_label"],
    ))
    # Demand on secondary axis
    fig_intra.add_trace(go.Scatter(
        x=past_t["period"], y=past_t["demand_mw"],
        mode="lines", name="Demand (MW)",
        line=dict(color=COLOR_DEMAND, width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="P%{x}<br>Demand: %{y:,.0f} MW<extra></extra>",
    ))

if not yesterday_df.empty:
    fig_intra.add_trace(go.Scatter(
        x=yesterday_df["period"], y=yesterday_df["usep"],
        mode="lines", name="Yesterday",
        line=dict(color="#888", width=1.5, dash="dash"),
        opacity=0.7,
        hovertemplate="P%{x} (%{customdata})<br>Yesterday: S$%{y:.2f}/MWh<extra></extra>",
        customdata=yesterday_df["time_label"],
    ))

if not lastweek_df.empty:
    fig_intra.add_trace(go.Scatter(
        x=lastweek_df["period"], y=lastweek_df["usep"],
        mode="lines", name="Last week same day",
        line=dict(color="#f0b429", width=1.5, dash="dot"),
        opacity=0.7,
        hovertemplate="P%{x}<br>Last week: S$%{y:.2f}/MWh<extra></extra>",
    ))

# Vesting price line
fig_intra.add_hline(
    y=VESTING_PRICE, line_dash="dash", line_color="#e74c3c",
    annotation_text=f"Vesting S${VESTING_PRICE}/MWh",
    annotation_position="bottom right",
)

# "Now" marker
if cur_period <= 48:
    fig_intra.add_vline(
        x=cur_period, line_dash="dot", line_color="#2ecc71",
        annotation_text="Now",
        annotation_position="top right",
    )

tick_periods = list(range(1, 49, 6))
tick_labels  = [_period_to_hhmm(p) for p in tick_periods]

fig_intra.update_layout(
    height=420,
    xaxis=_axis(cl.get("xaxis", {}), {
        "tickmode": "array", "tickvals": tick_periods, "ticktext": tick_labels,
        "title": "Time of day (SGT)",
    }),
    yaxis=_axis(cl.get("yaxis", {}), {"title": "USEP (S$/MWh)"}),
    yaxis2=_axis(get_yaxis2_style(), {"title": "Demand (MW)", "side": "right", "overlaying": "y"}),
    hovermode="x unified",
    showlegend=True,
)

st.plotly_chart(fig_intra, use_container_width=True, key="intraday_chart")
add_copy_button("intraday_chart")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — 7-day USEP heatmap
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("7-day USEP Heatmap")

hm_df = load_7day_heatmap(engine)
if hm_df.empty:
    st.info("Insufficient data for heatmap.")
else:
    dates_uniq = sorted(hm_df["date"].dt.date.unique())
    dates_str  = [d.strftime("%a %d %b") for d in dates_uniq]
    z_matrix   = []
    for d in dates_uniq:
        day_df  = hm_df[hm_df["date"].dt.date == d].set_index("period")
        row_vals = [day_df.loc[p, "usep"] if p in day_df.index else None for p in range(1, 49)]
        z_matrix.append(row_vals)

    fig_hm = go.Figure(go.Heatmap(
        z=z_matrix,
        x=list(range(1, 49)),
        y=dates_str,
        colorscale=[
            [0.0, "#2ecc71"],
            [0.3, "#f0b429"],
            [0.7, "#e74c3c"],
            [1.0, "#6c0000"],
        ],
        hovertemplate="Date: %{y}<br>Period: %{x}<br>USEP: S$%{z:.2f}/MWh<extra></extra>",
        colorbar=dict(title="S$/MWh"),
    ))
    fig_hm.update_layout(**cl)
    fig_hm.update_layout(
        height=280,
        xaxis=_axis(cl.get("xaxis", {}), {
            "tickmode": "array", "tickvals": tick_periods, "ticktext": tick_labels,
            "title": "Time of day (SGT)",
        }),
        yaxis=_axis(cl.get("yaxis", {}), {"title": "", "autorange": "reversed"}),
    )
    st.plotly_chart(fig_hm, use_container_width=True, key="heatmap_7day")
    add_copy_button("heatmap_7day")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Demand-price scatter
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Demand vs USEP — last 30 days")

scatter_df = load_scatter_data(engine, days=30)
if scatter_df.empty:
    st.info("Insufficient data for scatter chart.")
else:
    scatter_df["bucket"] = scatter_df["period"].apply(_bucket)

    fig_sc = go.Figure()
    fig_sc.update_layout(**cl)

    for bucket, color in BUCKET_COLORS.items():
        sub = scatter_df[scatter_df["bucket"] == bucket]
        if sub.empty:
            continue
        fig_sc.add_trace(go.Scatter(
            x=sub["demand_mw"], y=sub["usep"],
            mode="markers",
            name=bucket,
            marker=dict(color=color, size=4, opacity=0.55),
            hovertemplate=(
                "Demand: %{x:,.0f} MW<br>"
                "USEP: S$%{y:.2f}/MWh<extra></extra>"
            ),
        ))

    # Inflection vertical line
    fig_sc.add_vline(
        x=inflection_mw, line_dash="dash", line_color="#f0b429",
        annotation_text=f"Inflection {inflection_mw:,.0f} MW",
        annotation_position="top right",
    )
    # Vesting horizontal line
    fig_sc.add_hline(
        y=VESTING_PRICE, line_dash="dash", line_color="#e74c3c",
        annotation_text=f"Vesting S${VESTING_PRICE}/MWh",
        annotation_position="bottom right",
    )

    # Quadrant annotations
    xmid = inflection_mw
    fig_sc.add_annotation(x=scatter_df["demand_mw"].quantile(0.25), y=VESTING_PRICE * 0.5,
                          text="Normal operation", showarrow=False, font=dict(size=11, color="#888"))
    fig_sc.add_annotation(x=scatter_df["demand_mw"].quantile(0.9), y=VESTING_PRICE * 2,
                          text="Market stress", showarrow=False, font=dict(size=11, color="#e74c3c"))

    fig_sc.update_layout(
        height=420,
        xaxis=_axis(cl.get("xaxis", {}), {"title": "Demand (MW)"}),
        yaxis=_axis(cl.get("yaxis", {}), {"title": "USEP (S$/MWh)"}),
        hovermode="closest",
    )
    st.plotly_chart(fig_sc, use_container_width=True, key="demand_scatter")
    add_copy_button("demand_scatter")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Data freshness & status
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Data Freshness & Auto-Update Status")

# Counts for today
today_count = len(today_df) if today_df is not None else 0

col1, col2, col3 = st.columns(3)
col1.metric("Today's periods loaded", f"{today_count}/48")
last_ts = get_last_data_timestamp(engine)
col2.metric("Last DB timestamp",
            last_ts.strftime("%H:%M SGT") if last_ts else "—")
col3.metric("Inflection threshold", f"{inflection_mw:,.0f} MW")

# Live fetch log
log_df = load_live_log_summary(engine)
if not log_df.empty:
    log_df = log_df.rename(columns={
        "source": "Source", "last_fetched": "Last Fetched",
        "total_new": "Periods Added", "last_error": "Last Error",
    })
    log_df["Last Fetched"] = pd.to_datetime(log_df["Last Fetched"]).dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(log_df, use_container_width=True)
else:
    st.info("No fetch history yet. Use the sidebar button to fetch live data.")
