from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import get_engine, setup_database
from modules.analysis import duck_curve, demand_usep_threshold_analysis, demand_profile_analysis
from modules.theme import add_copy_button, apply_theme_css, get_chart_layout, render_theme_toggle
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
            WHERE date >= :s AND date <= :e AND demand_mw IS NOT NULL
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
            SELECT period,
                   AVG(demand_mw) AS demand_mw,
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


@st.cache_data(ttl=3600, show_spinner="Running demand-threshold analysis…")
def _demand_threshold(_engine, dfrom, dto):
    return demand_usep_threshold_analysis(_engine, date_from=dfrom, date_to=dto)


@st.cache_data(ttl=3600, show_spinner="Loading demand profiles…")
def _demand_profile(_engine, dfrom, dto):
    return demand_profile_analysis(_engine, date_from=dfrom, date_to=dto)


def _period_to_hhmm(period: int) -> str:
    h, m = divmod((period - 1) * 30, 60)
    return f"{h:02d}:{m:02d}"


def _make_period_xaxis(cl: dict, p_min: int, p_max: int) -> dict:
    """Build an xaxis dict with time-of-day tick labels."""
    tick_periods = [p for p in range(1, 49, 6) if p_min <= p <= p_max]
    if not tick_periods:
        tick_periods = [p_min, p_max]
    return dict(
        tickmode="array",
        tickvals=tick_periods,
        ticktext=[_period_to_hhmm(p) for p in tick_periods],
        title="Time of day (SGT)",
        range=[p_min - 0.5, p_max + 0.5],
        **cl.get("xaxis", {}),
    )


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
    st.markdown("**Period filter (time of day)**")
    p_min, p_max = st.slider(
        "Select period range",
        min_value=1, max_value=48,
        value=(1, 48),
        help="Each period = 30 min. Period 1 = 00:00, Period 48 = 23:30 SGT",
    )
    st.caption(f"{_period_to_hhmm(p_min)} — {_period_to_hhmm(p_max)} SGT")

    st.divider()
    st.markdown("**YoY comparison years**")
    available_years = list(range(db_min.year, db_max.year + 1))
    yoy_years = st.multiselect(
        "Select years", options=available_years,
        default=[y for y in [2020, 2022, 2024, 2026] if y in available_years],
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🦆 Duck Curve & Demand Analysis")
st.caption("Load shape, solar suppression, and demand–price threshold analysis.")

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

df = load_duck_data(engine, start_date, end_date)
if df.empty:
    st.warning("No data for selected period.")
    st.stop()

tab1, tab2 = st.tabs(["🦆 Duck Curve", "📊 Demand–Price Analysis"])

with tab1:
    dc = duck_curve(df)
    dc_view = dc[(dc["period"] >= p_min) & (dc["period"] <= p_max)]
    cl = get_chart_layout()
    xax = _make_period_xaxis(cl, p_min, p_max)

    # ── Duck curve chart ──────────────────────────────────────────────────────
    st.subheader("Duck curve — Demand, Solar, Net Demand")

    fig_dc = go.Figure()
    fig_dc.update_layout(**cl)
    fig_dc.add_trace(go.Scatter(
        x=dc_view["period"], y=dc_view["demand_mw"],
        mode="lines", name="Demand",
        line=dict(color=COLOR_DEMAND, width=2.5),
        hovertemplate="Period %{x} (%{customdata})<br>Demand: %{y:.0f} MW<extra></extra>",
        customdata=dc_view["time_label"],
    ))
    fig_dc.add_trace(go.Scatter(
        x=dc_view["period"], y=dc_view["solar_mw"],
        mode="lines", name="Solar",
        line=dict(color=COLOR_SOLAR, width=2.5),
        hovertemplate="Period %{x} (%{customdata})<br>Solar: %{y:.0f} MW<extra></extra>",
        customdata=dc_view["time_label"],
    ))
    fig_dc.add_trace(go.Scatter(
        x=dc_view["period"], y=dc_view["net_demand"],
        mode="lines", name="Net Demand",
        line=dict(color=COLOR_USEP, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,156,234,0.08)",
        hovertemplate="Period %{x} (%{customdata})<br>Net: %{y:.0f} MW<extra></extra>",
        customdata=dc_view["time_label"],
    ))
    fig_dc.update_layout(
        height=400, xaxis=xax,
        yaxis=dict(title="MW", **cl.get("yaxis", {})),
        hovermode="x unified", showlegend=True,
    )
    st.plotly_chart(fig_dc, use_container_width=True, key="duck_curve")
    add_copy_button("duck_curve")

    st.divider()

    # ── USEP by period bar chart ──────────────────────────────────────────────
    st.subheader("Average USEP by period")

    df_view = df[(df["period"] >= p_min) & (df["period"] <= p_max)]
    usep_profile = (
        df_view.groupby("period")["usep"]
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
    for _color, _label in [("#2ecc71", "< $100/MWh"), ("#f0b429", "$100–200/MWh"), ("#e74c3c", "> $200/MWh")]:
        fig_bar.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=_color, symbol="square"), name=_label,
        ))
    fig_bar.update_layout(
        height=320, xaxis=xax,
        yaxis=dict(title="Avg USEP (S$/MWh)", **cl.get("yaxis", {})),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Solar suppression panel ───────────────────────────────────────────────
    st.subheader("Solar suppression — USEP by period: high-solar vs low-solar months")

    df["month"] = df["date"].dt.month
    df_hs = df[(df["month"].isin([10, 11, 12, 1, 2])) & (df["period"] >= p_min) & (df["period"] <= p_max)]
    df_ls = df[(df["month"].isin([5, 6, 7, 8]))        & (df["period"] >= p_min) & (df["period"] <= p_max)]

    usep_hs = df_hs.groupby("period")["usep"].mean().reset_index().rename(columns={"usep": "avg_usep"})
    usep_ls = df_ls.groupby("period")["usep"].mean().reset_index().rename(columns={"usep": "avg_usep"})

    fig_sup = go.Figure()
    fig_sup.update_layout(**cl)
    fig_sup.add_trace(go.Scatter(
        x=usep_hs["period"], y=usep_hs["avg_usep"],
        mode="lines", name="High-solar months (Oct–Feb)",
        line=dict(color=COLOR_SOLAR, width=2),
        hovertemplate="Period %{x}<br>Avg USEP: S$%{y:.2f}/MWh<extra></extra>",
    ))
    fig_sup.add_trace(go.Scatter(
        x=usep_ls["period"], y=usep_ls["avg_usep"],
        mode="lines", name="Low-solar months (May–Aug)",
        line=dict(color=COLOR_DEMAND, width=2),
        hovertemplate="Period %{x}<br>Avg USEP: S$%{y:.2f}/MWh<extra></extra>",
    ))
    fig_sup.update_layout(
        height=340, xaxis=xax,
        yaxis=dict(title="Avg USEP (S$/MWh)", **cl.get("yaxis", {})),
        hovermode="x unified",
    )
    st.plotly_chart(fig_sup, use_container_width=True)
    st.caption(
        "Solar suppression: during high-solar months, mid-day USEP (periods 14–28) is "
        "pushed lower by excess solar generation. The BESS captures this intraday spread."
    )

    st.divider()

    # ── YoY duck curve evolution ──────────────────────────────────────────────
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
            yr_view = yr_df[(yr_df["period"] >= p_min) & (yr_df["period"] <= p_max)]
            _color = YOY_COLORS[i % len(YOY_COLORS)]
            fig_yoy.add_trace(go.Scatter(
                x=yr_view["period"], y=yr_view["net_demand"],
                mode="lines", name=str(yr),
                line=dict(color=_color, width=2),
                hovertemplate=(
                    f"{yr} — Period %{{x}} (%{{customdata}})<br>"
                    "Net Demand: %{y:.0f} MW<extra></extra>"
                ),
                customdata=yr_view["time_label"],
            ))

        fig_yoy.update_layout(
            height=380, xaxis=xax,
            yaxis=dict(title="Net Demand (MW)", **cl.get("yaxis", {})),
            hovermode="x unified",
        )
        st.plotly_chart(fig_yoy, use_container_width=True)
        st.caption(
            "Net Demand = System Demand − Solar. The deepening mid-day trough from "
            "2020→2026 reflects Singapore's growing solar capacity. Our 560 MW "
            "solar + BESS will deepen this further post-2029."
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Demand–Price Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    thresh = _demand_threshold(engine, start_date, end_date)
    prof   = _demand_profile(engine, start_date, end_date)
    cl2    = get_chart_layout()

    tick_ps = list(range(1, 49, 6))
    tick_ls = [_period_to_hhmm(p) for p in tick_ps]

    # ── Section A: Demand threshold bar chart ────────────────────────────────
    st.subheader("A — Demand–Price Threshold")
    bins_df = thresh.get("demand_bins", pd.DataFrame())
    infl_mw = thresh.get("inflection_mw")

    if bins_df.empty:
        st.info("No demand-threshold data available for selected period.")
    else:
        # Color bars green→yellow→red
        n = len(bins_df)
        bar_colors = [
            f"hsl({int(120 * (1 - i / max(n - 1, 1)))}, 80%, 45%)"
            for i in range(n)
        ]
        fig_thresh = go.Figure()
        fig_thresh.update_layout(**cl2)
        fig_thresh.add_trace(go.Bar(
            x=bins_df["demand_range"],
            y=bins_df["avg_usep"],
            marker_color=bar_colors,
            name="Avg USEP",
            hovertemplate=(
                "Demand: %{x}<br>"
                "Avg USEP: S$%{y:.2f}/MWh<br>"
                "P90: S$%{customdata[0]:.2f}/MWh<br>"
                "Spike freq (>200): %{customdata[1]:.1f}%<extra></extra>"
            ),
            customdata=list(zip(bins_df["p90_usep"], bins_df["spike_freq_200"])),
        ))
        if infl_mw and not (isinstance(infl_mw, float) and infl_mw != infl_mw):
            # Find x-axis index of inflection
            fig_thresh.add_vline(
                x=infl_mw // 200 - 0.5,
                line_dash="dash", line_color="#f0b429",
                annotation_text=f"Inflection ≈ {infl_mw:,.0f} MW",
                annotation_position="top right",
            )
        fig_thresh.update_layout(
            height=360,
            xaxis=dict(title="Demand bucket (MW)", tickangle=45, **cl2.get("xaxis", {})),
            yaxis=dict(title="Avg USEP (S$/MWh)", **cl2.get("yaxis", {})),
        )
        st.plotly_chart(fig_thresh, use_container_width=True, key="demand_threshold")
        add_copy_button("demand_threshold")

        if infl_mw:
            st.caption(
                f"Above **{infl_mw:,.0f} MW** system demand, USEP rises rapidly. "
                f"Spearman r = {thresh.get('spearman_r', 0):.3f}. "
                f"{thresh.get('interpretation', '')}"
            )

    st.divider()

    # ── Section B: Vesting price KPIs ────────────────────────────────────────
    st.subheader("B — Vesting Price Analysis (S$170/MWh)")
    col1, col2, col3 = st.columns(3)
    pct_v  = thresh.get("pct_above_vesting")
    dav    = thresh.get("demand_at_vesting_breach")
    col1.metric("% periods above vesting", f"{pct_v:.1f}%" if pct_v else "N/A")
    col2.metric("Demand at vesting breach (median)", f"{dav:,.0f} MW" if dav else "N/A")
    col3.metric("Inflection point", f"{infl_mw:,.0f} MW" if infl_mw else "N/A")

    if dav and infl_mw:
        st.info(
            f"USEP breaches the vesting price (S$170/MWh) when system demand exceeds "
            f"approximately **{dav:,.0f} MW** — around the inflection threshold "
            f"of **{infl_mw:,.0f} MW**. Evening peak periods (P31–42) carry the highest risk."
        )

    st.divider()

    # ── Section C: Day-type demand profiles ──────────────────────────────────
    st.subheader("C — Day-type Demand Profiles")
    dt_df = prof.get("day_type_profile", pd.DataFrame())
    if not dt_df.empty:
        DAY_TYPE_COLORS = {
            "weekday_core": "#009CEA",
            "weekday_wfh":  "#2ecc71",
            "saturday":     "#f0b429",
            "sunday":       "#e74c3c",
            "public_holiday": "#9b59b6",
        }
        fig_dt = go.Figure()
        fig_dt.update_layout(**cl2)
        for col in [c for c in DAY_TYPE_COLORS if c in dt_df.columns]:
            fig_dt.add_trace(go.Scatter(
                x=dt_df["period"], y=dt_df[col],
                mode="lines", name=col.replace("_", " ").title(),
                line=dict(color=DAY_TYPE_COLORS[col], width=2),
                hovertemplate=f"{col} — P%{{x}}: %{{y:,.0f}} MW<extra></extra>",
            ))
        fig_dt.update_layout(
            height=360,
            xaxis=dict(
                tickmode="array", tickvals=tick_ps, ticktext=tick_ls,
                title="Time of day (SGT)",
                **cl2.get("xaxis", {}),
            ),
            yaxis=dict(title="Avg Demand (MW)", **cl2.get("yaxis", {})),
            hovermode="x unified",
        )
        st.plotly_chart(fig_dt, use_container_width=True, key="daytype_demand")
        add_copy_button("daytype_demand")
        st.caption(
            "Mon/Fri (WFH) show ~3% lower demand than Tue–Thu core weekdays. "
            "Public holidays track Sunday but slightly lower."
        )
    else:
        st.info("Day-type profile unavailable.")

    st.divider()

    # ── Section D: Demand growth trend ───────────────────────────────────────
    st.subheader("D — Annual Demand Growth")
    yoy_df = prof.get("demand_yoy", pd.DataFrame())
    cagr   = prof.get("demand_cagr")

    if not yoy_df.empty:
        fig_yoy_d = go.Figure()
        fig_yoy_d.update_layout(**cl2)
        fig_yoy_d.add_trace(go.Bar(
            x=yoy_df["year"], y=yoy_df["avg_demand"],
            name="Avg Demand",
            marker_color="#009CEA",
            hovertemplate="Year: %{x}<br>Avg Demand: %{y:,.0f} MW<extra></extra>",
        ))
        # Trend line via numpy polyfit
        if len(yoy_df) >= 3:
            x_n  = yoy_df["year"].to_numpy(dtype=float)
            y_n  = yoy_df["avg_demand"].to_numpy(dtype=float)
            mask = ~(pd.isnull(y_n))
            if mask.sum() >= 3:
                import numpy as np
                coeffs = np.polyfit(x_n[mask], y_n[mask], 1)
                trend  = np.polyval(coeffs, x_n)
                fig_yoy_d.add_trace(go.Scatter(
                    x=yoy_df["year"], y=trend,
                    mode="lines", name="Trend",
                    line=dict(color="#f0b429", dash="dash", width=2),
                ))
        fig_yoy_d.update_layout(
            height=320,
            xaxis=dict(title="Year", **cl2.get("xaxis", {})),
            yaxis=dict(title="Avg Demand (MW)", **cl2.get("yaxis", {})),
        )
        st.plotly_chart(fig_yoy_d, use_container_width=True, key="demand_yoy_trend")
        if cagr:
            st.caption(
                f"Demand CAGR {yoy_df['year'].iloc[0]}–{yoy_df['year'].iloc[-1]}: "
                f"**{cagr*100:.1f}%** — driven by data center expansion and electrification."
            )
    else:
        st.info("Demand growth data unavailable.")
