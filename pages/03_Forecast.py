"""
NEMS Price Forecast — XGBoost + Prophet + Ensemble
Phase 3
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
from modules.forecasting import (
    backtest_walk_forward, build_features, predict, predict_ensemble,
    train_prophet, train_xgboost, _period_bucket, MODELS_DIR,
)
from modules.theme import (
    apply_theme_css, get_chart_layout, get_rangeselector_style,
    render_theme_toggle,
)
from config import COLOR_USEP, COLOR_FORECAST, COLOR_CONFIDENCE


@st.cache_resource
def _get_engine():
    engine = get_engine()
    setup_database(engine)
    return engine


@st.cache_data(ttl=300)
def load_full_df(_engine) -> pd.DataFrame:
    from sqlalchemy import text
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, period, usep, demand_mw, solar_mw
            FROM nems_prices WHERE usep IS NOT NULL
            ORDER BY date, period
        """)).fetchall()
    df = pd.DataFrame(rows, columns=["date", "period", "usep", "demand_mw", "solar_mw"])
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_recent_history(_engine, days: int = 7) -> pd.DataFrame:
    from sqlalchemy import text
    cutoff = date.today() - timedelta(days=days)
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, period, usep
            FROM nems_prices
            WHERE date >= :c AND usep IS NOT NULL
            ORDER BY date, period
        """), {"c": cutoff}).fetchall()
    df = pd.DataFrame(rows, columns=["date", "period", "usep"])
    df["dt"] = (
        pd.to_datetime(df["date"])
        + pd.to_timedelta((df["period"] - 1) * 30, unit="m")
    )
    return df


@st.cache_data(ttl=60)
def load_model_registry(_engine):
    from sqlalchemy import text
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT model_name, trained_at, training_rows, rmse, mae, mape
            FROM model_registry WHERE is_active = 1
            ORDER BY model_name
        """)).mappings().fetchall()
    return [dict(r) for r in rows]


def _model_file_exists(name: str) -> bool:
    return (MODELS_DIR / f"{name}_model.joblib").exists()


def _bucket_color(bucket: str) -> str:
    return {
        "off_peak":     "#009CEA",
        "solar":        "#2ecc71",
        "evening_peak": "#e74c3c",
        "night":        "#9b59b6",
    }.get(bucket, "#888")


BUCKET_ORDER  = ["off_peak", "solar", "evening_peak", "night"]
BUCKET_LABELS = {
    "off_peak":     "Off-peak (P1–13, 00:00–06:30)",
    "solar":        "Solar hours (P14–30, 06:30–14:30)",
    "evening_peak": "Evening peak (P31–42, 15:00–21:00)",
    "night":        "Night (P43–48, 21:00–23:30)",
}

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Forecast — NEMS", layout="wide")

engine = _get_engine()
apply_theme_css()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ NEMS Analytics")
    st.caption("Singa Renewables")
    render_theme_toggle()
    st.divider()

    st.markdown("**Forecast horizon**")
    horizon = st.radio(
        "Horizon",
        ["24h (48 periods)", "48h (96 periods)", "7 days (336 periods)"],
        index=0, label_visibility="collapsed",
    )
    n_periods = {
        "24h (48 periods)":    48,
        "48h (96 periods)":    96,
        "7 days (336 periods)": 336,
    }[horizon]

    st.markdown("**Model**")
    model_choice = st.radio(
        "Model", ["XGBoost", "Prophet", "Ensemble"],
        index=0, label_visibility="collapsed",
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔮 NEMS Price Forecast")

st.info(
    "**Disclaimer:** Price spike events (USEP > S$300/MWh) are caused by unplanned "
    "plant outages and gas supply disruptions — these are unpredictable from price "
    "history alone. This model forecasts expected price levels. Spike risk is shown "
    "separately as a **historical frequency estimate**, not a prediction."
)

cl = get_chart_layout()
rs = get_rangeselector_style()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Model status
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🗂 Model Status")

registry  = load_model_registry(engine)
reg_map   = {r["model_name"]: r for r in registry}
xgb_ready = "xgboost" in reg_map and _model_file_exists("xgboost")
pro_ready  = "prophet"  in reg_map and _model_file_exists("prophet")

col_xgb, col_pro = st.columns(2)
for col, mname, label in [(col_xgb, "xgboost", "XGBoost"), (col_pro, "prophet", "Prophet")]:
    with col:
        st.markdown(f"**{label}**")
        if mname in reg_map and _model_file_exists(mname):
            r = reg_map[mname]
            st.success("✅ Ready")
            st.caption(f"Trained: {str(r['trained_at'])[:19]}")
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{r['rmse']:.1f}")
            m2.metric("MAE",  f"{r['mae']:.1f}")
            m3.metric("MAPE", f"{r['mape']:.1f}%")
            st.caption(f"Training rows: {r['training_rows']:,}")
        else:
            st.warning("⚠️ Not trained")

if st.button("🔄 Retrain All Models", type="secondary"):
    with st.spinner("Loading full dataset…"):
        full_df = load_full_df(engine)
    prog = st.progress(0, text="Training XGBoost…")
    try:
        r_xgb = train_xgboost(full_df, engine)
        prog.progress(50, text="Training Prophet…")
        r_pro = train_prophet(full_df, engine)
        prog.progress(100, text="Done.")
        load_model_registry.clear()
        st.success(
            f"✅ XGBoost RMSE={r_xgb['rmse']:.1f}  |  Prophet RMSE={r_pro['rmse']:.1f}"
        )
        st.rerun()
    except Exception as e:
        st.error(f"Training failed: {e}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Forecast chart
# ─────────────────────────────────────────────────────────────────────────────
st.subheader(f"📈 {model_choice} Forecast — {horizon}")

can_run = {
    "XGBoost":  xgb_ready,
    "Prophet":  pro_ready,
    "Ensemble": xgb_ready and pro_ready,
}

if not can_run[model_choice]:
    st.warning(f"**{model_choice}** is not trained yet. Click Retrain above.")
else:
    with st.spinner(f"Generating {model_choice} forecast…"):
        if model_choice == "XGBoost":
            fc_df = predict("xgboost", engine, n_periods)
        elif model_choice == "Prophet":
            fc_df = predict("prophet", engine, n_periods)
        else:
            fc_df = predict_ensemble(engine, n_periods)

    hist_df = load_recent_history(engine, days=7)

    fig_fc = go.Figure()
    fig_fc.update_layout(**cl)

    if not hist_df.empty:
        fig_fc.add_trace(go.Scattergl(
            x=hist_df["dt"], y=hist_df["usep"],
            mode="lines", name="Historical USEP",
            line=dict(color=COLOR_USEP, width=1),
            hovertemplate="<b>%{x}</b><br>USEP: S$%{y:.2f}/MWh<extra></extra>",
        ))

    if not fc_df.empty:
        # Confidence band
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fc_df["dt"], fc_df["dt"].iloc[::-1]]),
            y=pd.concat([fc_df["upper_bound"], fc_df["lower_bound"].iloc[::-1]]),
            fill="toself", fillcolor=COLOR_CONFIDENCE,
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence band", hoverinfo="skip",
        ))

        fc_color = {
            "XGBoost":  COLOR_FORECAST,
            "Prophet":  "#1abc9c",
            "Ensemble": "#e67e22",
        }[model_choice]
        fig_fc.add_trace(go.Scatter(
            x=fc_df["dt"], y=fc_df["predicted_usep"],
            mode="lines", name=f"{model_choice} forecast",
            line=dict(color=fc_color, width=2, dash="dash"),
            hovertemplate="<b>%{x}</b><br>Forecast: S$%{y:.2f}/MWh<extra></extra>",
        ))

        now_dt = fc_df["dt"].min()
        fig_fc.add_vline(
            x=now_dt.timestamp() * 1000,
            line_dash="dot", line_color="#555",
            annotation_text="Now", annotation_font_color="#888",
        )

        spike_periods = fc_df[fc_df["spike_prob"] > 0.25]
        if not spike_periods.empty:
            times = ", ".join(spike_periods["time_label"].tolist()[:6])
            st.warning(
                f"⚠️ **Elevated spike risk (hist. freq > 25%)** at: {times} SGT  \n"
                "Historical frequency only — not a causal prediction."
            )

    fig_fc.update_layout(
        height=430,
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=12, label="12h", step="hour",  stepmode="backward"),
                    dict(count=1,  label="1D",  step="day",   stepmode="backward"),
                    dict(count=3,  label="3D",  step="day",   stepmode="backward"),
                    dict(count=7,  label="7D",  step="day",   stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                **rs,
            ),
            rangeslider=dict(visible=True, thickness=0.04),
            type="date",
            **cl.get("xaxis", {}),
        ),
        yaxis=dict(title="USEP (S$/MWh)", **cl.get("yaxis", {})),
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    if not fc_df.empty and "spike_prob" in fc_df.columns:
        with st.expander("Spike probability by forecast period"):
            fig_sp = go.Figure()
            fig_sp.update_layout(**cl)
            fig_sp.add_trace(go.Bar(
                x=fc_df["time_label"],
                y=fc_df["spike_prob"] * 100,
                marker_color=fc_df["spike_prob"].apply(
                    lambda v: "#2ecc71" if v < 0.15 else ("#f0b429" if v < 0.30 else "#e74c3c")
                ),
                hovertemplate="%{x}<br>Spike freq: %{y:.1f}%<extra></extra>",
            ))
            cl_base = {k: v for k, v in cl.items() if k not in ("xaxis", "yaxis")}
            fig_sp.update_layout(
                **cl_base, height=220,
                xaxis=dict(title="Time (SGT)", **cl.get("xaxis", {})),
                yaxis=dict(title="P(USEP > $200) %", **cl.get("yaxis", {})),
                showlegend=False,
            )
            st.plotly_chart(fig_sp, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Performance metrics
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Model Performance")

if not (xgb_ready or pro_ready):
    st.info("Train at least one model to see performance metrics.")
else:
    perf_rows = []
    for mname, mlabel in [("xgboost", "XGBoost"), ("prophet", "Prophet")]:
        if mname not in reg_map or not _model_file_exists(mname):
            continue
        r = reg_map[mname]
        perf_rows.append({
            "Model":         mlabel,
            "Overall RMSE":  f"S${r['rmse']:.1f}/MWh",
            "Overall MAE":   f"S${r['mae']:.1f}/MWh",
            "MAPE":          f"{r['mape']:.1f}%",
            "Training rows": f"{r['training_rows']:,}",
        })

    if perf_rows:
        st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

    st.caption(
        "High RMSE is driven by unpredictable spike events (USEP can reach "
        "S$4,500/MWh during grid stress). MAE reflects typical off-peak accuracy. "
        "Run the backtest below for per-bucket analysis."
    )

    # Reference bucket RMSE guide
    ref_df = pd.DataFrame({
        "Period bucket":   ["Off-peak (P1–13)", "Solar hours (P14–30)", "Evening peak (P31–42)", "Night (P43–48)"],
        "Typical RMSE":    ["~30–50",          "~40–70",              "~80–150",               "~40–60"],
        "Volatility":      ["Low",             "Moderate",            "High (spike-prone)",    "Low"],
    })
    st.dataframe(ref_df, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Feature importance
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🔑 Feature Importance (XGBoost)")

if not xgb_ready:
    st.info("Train XGBoost to see feature importance.")
else:
    import joblib as _jl
    payload = _jl.load(MODELS_DIR / "xgboost_model.joblib")
    model   = payload["model"]
    feats   = payload["features"]

    importance = sorted(
        zip(feats, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )[:15]

    names  = [f for f, _ in importance]
    values = [v for _, v in importance]

    def _feat_color(name: str) -> str:
        if "lag" in name or "rolling" in name:
            return "#009CEA"
        if any(x in name for x in ["sin", "cos", "hour", "period", "month", "day", "weekend", "holiday"]):
            return "#f0b429"
        return "#2ecc71"

    colors = [_feat_color(f) for f in names]

    fig_imp = go.Figure()
    cl_base = {k: v for k, v in cl.items() if k not in ("xaxis", "yaxis")}
    fig_imp.update_layout(**cl_base)
    fig_imp.add_trace(go.Bar(
        x=values[::-1], y=names[::-1],
        orientation="h",
        marker_color=colors[::-1],
        hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
    ))
    for color, label in [
        ("#009CEA", "Lag / rolling features"),
        ("#f0b429", "Time / cyclical features"),
        ("#2ecc71", "Market features"),
    ]:
        fig_imp.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color, symbol="square"), name=label,
        ))
    fig_imp.update_layout(
        height=420, bargap=0.25,
        xaxis=dict(title="Feature importance (gain)", **cl.get("xaxis", {})),
        yaxis=dict(title="", **cl.get("yaxis", {})),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Walk-forward backtest
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🔁 Walk-forward Backtest (last 90 days)")

st.warning(
    "⏱ The 90-day walk-forward backtest re-trains XGBoost for each test day "
    "and takes **10–20 minutes**. Results are cached in session state."
)

if "backtest_df" not in st.session_state:
    st.session_state["backtest_df"] = None

if st.button("▶ Run 90-day Backtest", type="primary"):
    if not xgb_ready:
        st.error("Train XGBoost first.")
    else:
        with st.spinner("Running walk-forward backtest (10–20 min)…"):
            full_df = load_full_df(engine)
            bt_df   = backtest_walk_forward(full_df, test_days=90)
            st.session_state["backtest_df"] = bt_df

bt_df = st.session_state.get("backtest_df")

if bt_df is not None and not bt_df.empty:
    overall_rmse = float(np.sqrt(np.mean(bt_df["error"] ** 2)))
    overall_mae  = float(np.mean(np.abs(bt_df["error"])))

    c1, c2, c3 = st.columns(3)
    c1.metric("Backtest RMSE", f"S${overall_rmse:.1f}/MWh")
    c2.metric("Backtest MAE",  f"S${overall_mae:.1f}/MWh")
    c3.metric("Days tested",   str(bt_df["date"].nunique()))

    # Scatter: actual vs predicted, colored by period bucket
    fig_bt = go.Figure()
    fig_bt.update_layout(**cl)

    max_val = float(max(bt_df["actual"].max(), bt_df["predicted"].max()))
    fig_bt.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", name="Perfect forecast",
        line=dict(color="#555", dash="dash", width=1.5), hoverinfo="skip",
    ))

    for bucket in BUCKET_ORDER:
        sub = bt_df[bt_df["bucket"] == bucket]
        if sub.empty:
            continue
        rmse_b = float(np.sqrt(np.mean(sub["error"] ** 2)))
        fig_bt.add_trace(go.Scattergl(
            x=sub["actual"], y=sub["predicted"],
            mode="markers",
            name=f"{BUCKET_LABELS.get(bucket, bucket)} (RMSE {rmse_b:.0f})",
            marker=dict(color=_bucket_color(bucket), size=3, opacity=0.55),
            hovertemplate="Actual: S$%{x:.2f}<br>Predicted: S$%{y:.2f}<extra></extra>",
        ))

    fig_bt.update_layout(
        height=480,
        xaxis=dict(title="Actual USEP (S$/MWh)", **cl.get("xaxis", {})),
        yaxis=dict(title="Predicted USEP (S$/MWh)", **cl.get("yaxis", {})),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # Per-bucket table
    bucket_rows = []
    for bucket in BUCKET_ORDER:
        sub = bt_df[bt_df["bucket"] == bucket]
        if sub.empty:
            continue
        bucket_rows.append({
            "Period bucket": BUCKET_LABELS.get(bucket, bucket),
            "Periods":       len(sub),
            "RMSE (S$/MWh)": round(float(np.sqrt(np.mean(sub["error"] ** 2))), 1),
            "MAE (S$/MWh)":  round(float(np.mean(np.abs(sub["error"]))), 1),
        })
    st.dataframe(pd.DataFrame(bucket_rows), use_container_width=True, hide_index=True)
