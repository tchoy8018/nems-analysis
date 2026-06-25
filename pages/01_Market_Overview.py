import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import get_engine, setup_database
from modules.analysis import spike_analysis
from modules.ingestion import import_csv_to_db
from modules.theme import (
    apply_theme_css, get_chart_layout, get_rangeselector_style,
    render_theme_toggle,
)
from modules.utils import get_holidays_in_range
from config import COLOR_USEP, COLOR_SPIKE


@st.cache_resource
def _get_engine():
    engine = get_engine()
    setup_database(engine)
    return engine


@st.cache_data(ttl=300)
def load_usep_range(_engine, start: date, end: date) -> pd.DataFrame:
    from sqlalchemy import text
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, period, usep, demand_mw
            FROM nems_prices
            WHERE date >= :s AND date <= :e AND usep IS NOT NULL
            ORDER BY date, period
        """), {"s": start, "e": end}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "period", "usep", "demand_mw"])
    df["date"] = pd.to_datetime(df["date"])
    df["datetime"] = df["date"] + pd.to_timedelta((df["period"] - 1) * 30, unit="m")
    return df


@st.cache_data(ttl=300)
def load_heatmap_data(_engine, start: date, end: date) -> pd.DataFrame:
    from sqlalchemy import text
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                strftime('%Y-%m', date) AS year_month,
                period,
                AVG(usep) AS avg_usep
            FROM nems_prices
            WHERE date >= :s AND date <= :e AND usep IS NOT NULL
            GROUP BY year_month, period
            ORDER BY year_month, period
        """), {"s": start, "e": end}).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["year_month", "period", "avg_usep"])


@st.cache_data(ttl=300)
def load_db_status(_engine):
    from sqlalchemy import text
    with _engine.connect() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) AS n, MIN(date) AS min_d, MAX(date) AS max_d
            FROM nems_prices
        """)).mappings().fetchone()
    return dict(row) if row else {}


def _period_to_hhmm(period: int) -> str:
    h, m = divmod((period - 1) * 30, 60)
    return f"{h:02d}:{m:02d}"


st.set_page_config(page_title="Market Overview — NEMS", layout="wide")

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

    st.markdown("**Analysis period**")
    start_date = st.date_input("From", value=db_max - timedelta(days=365),
                               min_value=db_min, max_value=db_max)
    end_date   = st.date_input("To",   value=db_max,
                               min_value=db_min, max_value=db_max)

    st.divider()
    st.markdown("**Spike Threshold**")
    spike_threshold = st.select_slider(
        "Threshold (S$/MWh)",
        options=[100, 150, 200, 250, 300, 400, 500, 1000],
        value=200,
        label_visibility="collapsed",
        key="spike_thr_01",
    )

    st.divider()
    st.markdown("**Database**")
    if status.get("n"):
        st.metric("Total rows", f"{status['n']:,}")
        st.caption(f"{status['min_d']} → {status['max_d']}")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📊 Market Overview")

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

df = load_usep_range(engine, start_date, end_date)
if df.empty:
    st.warning("No data for selected period.")
    st.stop()

cl = get_chart_layout()
rs = get_rangeselector_style()

# ── KPI row ───────────────────────────────────────────────────────────────────
avg_usep   = df["usep"].mean()
max_usep   = df["usep"].max()
spikes_n   = int((df["usep"] > spike_threshold).sum())
spike_pct  = 100.0 * spikes_n / len(df)
avg_demand = df["demand_mw"].mean() if "demand_mw" in df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg USEP", f"S${avg_usep:.2f}/MWh")
c2.metric("Peak USEP", f"S${max_usep:.2f}/MWh")
c3.metric(f"Spike Freq (>S${spike_threshold})", f"{spike_pct:.1f}%",
          help=f"{spikes_n:,} periods above S${spike_threshold}/MWh")
c4.metric("Avg Demand", f"{avg_demand:,.0f} MW")

st.divider()

# ── USEP time-series ──────────────────────────────────────────────────────────
st.subheader("USEP over selected period")

rolling = (
    df.set_index("datetime")["usep"]
    .rolling("30D").mean()
    .reset_index()
    .rename(columns={"usep": "rolling_avg"})
)

spike_mask = df["usep"] > spike_threshold
fig_ts = go.Figure()
fig_ts.update_layout(**cl)

fig_ts.add_trace(go.Scattergl(
    x=df.loc[~spike_mask, "datetime"], y=df.loc[~spike_mask, "usep"],
    mode="lines", name="USEP",
    line=dict(color=COLOR_USEP, width=0.8),
    hovertemplate="<b>%{x}</b><br>USEP: S$%{y:.2f}/MWh<extra></extra>",
))
if spike_mask.any():
    fig_ts.add_trace(go.Scattergl(
        x=df.loc[spike_mask, "datetime"], y=df.loc[spike_mask, "usep"],
        mode="markers", name=f"Spike >S${spike_threshold}",
        marker=dict(color=COLOR_SPIKE, size=4),
        hovertemplate="<b>%{x}</b><br>USEP: S$%{y:.2f}/MWh<extra></extra>",
    ))

# Holiday vertical dashed lines (amber)
_holidays = get_holidays_in_range(start_date, end_date)
for _h in _holidays:
    fig_ts.add_vline(
        x=pd.Timestamp(_h["date"]).timestamp() * 1000,
        line_dash="dash", line_color="#f0b429", line_width=1, opacity=0.6,
    )
if _holidays:
    _h_x    = [pd.Timestamp(_h["date"]) + pd.Timedelta(hours=12) for _h in _holidays]
    _h_y    = [float(df["usep"].quantile(0.92))] * len(_holidays)
    _h_text = [_h["name"] for _h in _holidays]
    fig_ts.add_trace(go.Scatter(
        x=_h_x, y=_h_y, mode="markers",
        marker=dict(color="#f0b429", size=7, symbol="triangle-down"),
        name="Public Holiday",
        customdata=_h_text,
        hovertemplate="<b>%{customdata}</b><extra></extra>",
    ))
fig_ts.add_trace(go.Scatter(
    x=rolling["datetime"], y=rolling["rolling_avg"],
    mode="lines", name="30-day avg",
    line=dict(color="#f0b429", width=1.5, dash="dot"),
    hovertemplate="30-day avg: S$%{y:.2f}/MWh<extra></extra>",
))

fig_ts.update_layout(
    height=400,
    yaxis_title="USEP (S$/MWh)",
    hovermode="x unified",
    showlegend=True,
    xaxis=dict(
        rangeselector=dict(
            buttons=[
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=1,  label="YTD", step="year",  stepmode="todate"),
                dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
            **rs,
        ),
        rangeslider=dict(visible=True, thickness=0.04),
        type="date",
        **cl.get("xaxis", {}),
    ),
)
st.plotly_chart(fig_ts, use_container_width=True)

st.divider()

# ── Monthly heatmap ───────────────────────────────────────────────────────────
st.subheader("Monthly heatmap — avg USEP by period")

hm_df = load_heatmap_data(engine, start_date, end_date)
if not hm_df.empty:
    pivot = hm_df.pivot(index="year_month", columns="period", values="avg_usep")
    tick_periods = list(range(1, 49, 6))
    tick_labels  = [_period_to_hhmm(p) for p in tick_periods]
    tick_indices = [p - 1 for p in tick_periods]  # 0-based column indices

    fig_hm = px.imshow(
        pivot,
        color_continuous_scale="RdYlBu_r",
        aspect="auto",
        labels={"x": "Period", "y": "Month", "color": "Avg USEP (S$/MWh)"},
    )
    # Apply theme base styles first, then override axes separately (avoids duplicate key error)
    cl_base = {k: v for k, v in cl.items() if k not in ("xaxis", "yaxis")}
    fig_hm.update_layout(**cl_base)
    fig_hm.update_layout(
        height=max(300, len(pivot) * 18 + 80),
        xaxis=dict(
            tickmode="array",
            tickvals=tick_indices,
            ticktext=tick_labels,
            title="Time of day",
            **cl.get("xaxis", {}),
        ),
        yaxis=dict(title="Month", **cl.get("yaxis", {})),
        coloraxis_colorbar=dict(title="S$/MWh"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("No heatmap data for selected period.")

st.divider()

# ── Spike event table ─────────────────────────────────────────────────────────
st.subheader(f"Spike events — USEP > S${spike_threshold}/MWh")

spike_df = spike_analysis(df, threshold=spike_threshold)
if spike_df.empty:
    st.success("No spikes in the selected period.")
else:
    spike_df["date"] = pd.to_datetime(spike_df["date"]).dt.date
    display_cols = {
        "date": "Date",
        "period": "Period",
        "time_label": "Time (SGT)",
        "usep": "USEP (S$/MWh)",
    }
    if "demand_mw" in spike_df.columns:
        display_cols["demand_mw"] = "Demand (MW)"

    display_df = spike_df[list(display_cols.keys())].rename(columns=display_cols)
    display_df["USEP (S$/MWh)"] = display_df["USEP (S$/MWh)"].round(2)
    if "Demand (MW)" in display_df.columns:
        display_df["Demand (MW)"] = display_df["Demand (MW)"].round(1)

    st.caption(f"{len(display_df):,} spike periods in selected range")
    st.dataframe(display_df, use_container_width=True, height=320)

st.divider()

# ── CSV upload ────────────────────────────────────────────────────────────────
st.subheader("Upload EMC monthly CSV")
st.caption(
    "Upload a raw EMC market data CSV (same column format as the master Excel). "
    "New rows are inserted; duplicates are silently skipped."
)

uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")
if uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    with st.spinner("Importing…"):
        result = import_csv_to_db(tmp_path, engine)
        tmp_path.unlink(missing_ok=True)

    if result["errors"]:
        st.error("Import failed: " + "; ".join(result["errors"]))
    else:
        n  = result["rows_imported"]
        sk = result["rows_skipped"]
        dr = result.get("date_range")
        st.success(
            f"✅ {n:,} new rows added"
            + (f", {sk:,} duplicates skipped" if sk else "")
            + (f"  —  {dr['min']} → {dr['max']}" if dr else "")
        )
        load_usep_range.clear()
        load_heatmap_data.clear()
        load_db_status.clear()
        st.rerun()
