import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from db import get_engine, setup_database
from modules.theme import apply_theme_css, add_copy_button, get_chart_layout, render_theme_toggle
from config import COLOR_USEP, COLOR_SPIKE


@st.cache_resource
def _get_engine():
    engine = get_engine()
    setup_database(engine)
    return engine


@st.cache_data(ttl=86400)
def _check_drift_daily(_engine) -> dict:
    """Check XGBoost drift once per day; returns {drift_detected, recent_rmse, baseline_rmse}."""
    try:
        from modules.forecasting import check_model_drift
        return check_model_drift(_engine, "xgboost", window_days=30)
    except Exception:
        return {"drift_detected": False, "recent_rmse": None, "baseline_rmse": None}


@st.cache_data(ttl=300)
def load_kpis(_engine):
    from sqlalchemy import text
    today = date.today()
    this_month_start = today.replace(day=1)
    last_month_start = (this_month_start - timedelta(days=1)).replace(day=1)

    with _engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                AVG(CASE WHEN date >= :cm  THEN usep END) AS cur_avg,
                AVG(CASE WHEN date >= :lm AND date < :cm THEN usep END) AS prev_avg,
                SUM(CASE WHEN date >= :cm AND usep > 200 THEN 1 ELSE 0 END) AS spikes,
                COUNT(CASE WHEN date >= :cm THEN 1 END) AS cur_periods,
                AVG(CASE WHEN date >= :cm THEN demand_mw END) AS avg_demand
            FROM nems_prices
        """), {"cm": this_month_start, "lm": last_month_start}).mappings().fetchone()
    return dict(row) if row else {}


@st.cache_data(ttl=300)
def load_recent_usep(_engine, days: int = 90):
    from sqlalchemy import text
    cutoff = date.today() - timedelta(days=days)
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, period, usep
            FROM nems_prices
            WHERE date >= :cutoff AND usep IS NOT NULL
            ORDER BY date, period
        """), {"cutoff": cutoff}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "period", "usep"])
    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta((df["period"] - 1) * 30, unit="m")
    return df


@st.cache_data(ttl=300)
def load_db_status(_engine):
    from sqlalchemy import text
    try:
        with _engine.connect() as conn:
            row = conn.execute(text("""
                SELECT COUNT(*) AS n, MIN(date) AS min_d, MAX(date) AS max_d
                FROM nems_prices
            """)).mappings().fetchone()
        return dict(row)
    except Exception:
        return {"n": 0, "min_d": None, "max_d": None}


def main():
    st.set_page_config(
        page_title="NEMS Analytics — Singa Renewables",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    engine = _get_engine()
    apply_theme_css()

    # --- Sidebar ---
    with st.sidebar:
        st.title("⚡ NEMS Analytics")
        st.caption("Singa Renewables Intelligence Platform")
        render_theme_toggle()
        st.divider()

        status = load_db_status(engine)
        db_url = str(engine.url)
        conn_type = "PostgreSQL" if "postgresql" in db_url else "SQLite (local)"

        st.markdown("**Database**")
        st.caption(conn_type)
        if status["n"]:
            st.metric("Total Periods", f"{status['n']:,}")
            st.caption(f"{status['min_d']} → {status['max_d']}")
        else:
            st.warning("No data loaded yet.")

        st.divider()
        drift = _check_drift_daily(engine)
        if drift.get("drift_detected"):
            _rr = drift.get("recent_rmse")
            _br = drift.get("baseline_rmse")
            st.warning(
                f"⚠️ **Model drift detected**  \n"
                f"Recent RMSE: {'S$' + f'{_rr:.1f}/MWh' if _rr is not None else 'N/A'}  \n"
                f"Baseline: {'S$' + f'{_br:.1f}/MWh' if _br is not None else 'N/A'}  \n"
                "Go to Forecast → Retrain All Models"
            )

        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/01_Market_Overview.py", label="Market Overview", icon="📊")
        st.page_link("pages/02_Duck_Curve.py", label="Duck Curve", icon="🦆")
        st.page_link("pages/03_Forecast.py", label="Forecast", icon="🔮")
        st.page_link("pages/04_Battery_Arbitrage.py", label="Battery Arbitrage", icon="🔋")
        st.page_link("pages/05_Scenario_Comparison.py", label="Scenario Comparison", icon="📈")

    # --- Home ---
    st.title("NEMS Analytics — Singa Renewables Intelligence Platform")
    st.caption("560 MW Solar + BESS · Riau Islands → Singapore · COD 2029 · TotalEnergies + RGE")

    kpis = load_kpis(engine)

    if not kpis or not kpis.get("cur_avg"):
        st.info("No data found in the database. Run `scripts/bootstrap_db.py` to load the master Excel file.")
        return

    cur_avg = kpis.get("cur_avg") or 0.0
    prev_avg = kpis.get("prev_avg") or 0.0
    delta = cur_avg - prev_avg if prev_avg else None
    spikes = kpis.get("spikes") or 0
    cur_periods = kpis.get("cur_periods") or 1
    spike_pct = 100.0 * spikes / cur_periods
    avg_demand = kpis.get("avg_demand") or 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("This Month Avg USEP", f"S${cur_avg:.2f}/MWh",
                delta=f"{delta:+.2f}" if delta is not None else None)
    col2.metric("vs Prior Month", f"S${prev_avg:.2f}/MWh" if prev_avg else "—")
    col3.metric("Spike Frequency (>200)", f"{spike_pct:.1f}%",
                help="% of half-hourly periods this month with USEP > S$200/MWh")
    col4.metric("Avg System Demand", f"{avg_demand:,.0f} MW")

    st.divider()
    st.subheader("USEP — Last 90 Days")

    df = load_recent_usep(engine)
    if df.empty:
        st.warning("No recent data available.")
        return

    rolling = df.set_index("datetime")["usep"].rolling("30D").mean().reset_index()
    rolling.columns = ["datetime", "rolling_avg"]

    fig = go.Figure()
    fig.update_layout(**get_chart_layout())

    spike_mask = df["usep"] > 200

    fig.add_trace(go.Scattergl(
        x=df.loc[~spike_mask, "datetime"],
        y=df.loc[~spike_mask, "usep"],
        mode="lines",
        name="USEP",
        line=dict(color=COLOR_USEP, width=0.8),
        hovertemplate="<b>%{x}</b><br>USEP: S$%{y:.2f}/MWh<extra></extra>",
    ))

    if spike_mask.any():
        fig.add_trace(go.Scattergl(
            x=df.loc[spike_mask, "datetime"],
            y=df.loc[spike_mask, "usep"],
            mode="markers",
            name="Spike >$200",
            marker=dict(color=COLOR_SPIKE, size=4),
            hovertemplate="<b>%{x}</b><br>USEP: S$%{y:.2f}/MWh<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=rolling["datetime"],
        y=rolling["rolling_avg"],
        mode="lines",
        name="30-day avg",
        line=dict(color="#f0b429", width=1.5, dash="dot"),
        hovertemplate="30-day avg: S$%{y:.2f}/MWh<extra></extra>",
    ))

    fig.update_layout(
        height=420,
        xaxis_title="Date",
        yaxis_title="USEP (S$/MWh)",
        hovermode="x unified",
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True, key="home_usep")
    add_copy_button("home_usep")


if __name__ == "__main__":
    main()
