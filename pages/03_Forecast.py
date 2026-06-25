"""
NEMS Price Forecast — XGBoost + Prophet + Ensemble + Spike Classifier
4-model status panel | configurable spike threshold | extended backtest
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
    MODELS_DIR,
    _period_bucket,
    backtest_walk_forward,
    build_features,
    forecast_monthly_scenarios,
    predict,
    predict_ensemble,
    spike_exceedance_curve,
    train_ensemble,
    train_prophet,
    train_spike_classifier,
    train_xgboost,
)
from modules.theme import (
    add_copy_button,
    apply_theme_css,
    get_chart_layout,
    get_rangeselector_style,
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
def load_recent_history(_engine, days: int = 14) -> pd.DataFrame:
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


def _spike_clf_exists() -> bool:
    return (MODELS_DIR / "spike_classifier.joblib").exists()


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

_RMSE_HELP = (
    "Root Mean Square Error — penalises large errors heavily. Our RMSE is elevated "
    "by rare spike events (>$300/MWh). Off-peak periods achieve RMSE ≈ $8–15/MWh."
)
_MAE_HELP  = (
    "Mean Absolute Error — typical magnitude of forecast error. "
    "More representative than RMSE for operational planning."
)
_MAPE_HELP = (
    "Mean Absolute Percentage Error — error relative to actual price. "
    "High MAPE at low-price periods is mathematically expected."
)

@st.cache_data(ttl=300)
def _load_evolution(_engine):
    from sqlalchemy import text
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT model_name, trained_at, rmse, mae, mape, training_rows, notes
            FROM model_evolution ORDER BY trained_at
        """)).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["model_name", "trained_at", "rmse", "mae", "mape",
                                       "training_rows", "notes"])


@st.cache_data(ttl=300)
def _load_backtest_history(_engine):
    from sqlalchemy import text
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT run_at, model_name, test_days, rmse, mae, mape, n_periods
            FROM backtest_runs ORDER BY run_at DESC LIMIT 20
        """)).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["run_at", "model_name", "test_days",
                                       "rmse", "mae", "mape", "n_periods"])


def _load_model_history(engine, cl: dict) -> None:
    evo_df = _load_evolution(engine)
    bt_df  = _load_backtest_history(engine)

    if evo_df.empty and bt_df.empty:
        st.info("No model history yet. Run a backtest or retrain a model to start tracking.")
        return

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**RMSE evolution by model**")
        if not evo_df.empty:
            evo_df["trained_at"] = pd.to_datetime(evo_df["trained_at"])
            cl_base = {k: v for k, v in cl.items() if k not in ("xaxis", "yaxis")}
            fig_evo = go.Figure()
            fig_evo.update_layout(**cl_base)
            colors = {"xgboost": "#009CEA", "prophet": "#f0b429", "ensemble": "#e67e22"}
            for mn in evo_df["model_name"].unique():
                sub = evo_df[evo_df["model_name"] == mn]
                fig_evo.add_trace(go.Scatter(
                    x=sub["trained_at"], y=sub["rmse"],
                    mode="lines+markers", name=mn,
                    line=dict(color=colors.get(mn, "#888"), width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>RMSE: S$%{y:.2f}<extra></extra>",
                ))
            fig_evo.update_layout(
                height=280,
                xaxis=dict(title="Train date"),
                yaxis=dict(title="RMSE (S$/MWh)"),
            )
            st.plotly_chart(fig_evo, use_container_width=True)
        else:
            st.info("No evolution data yet.")

    with col_b:
        st.markdown("**Last 10 backtest runs**")
        if not bt_df.empty:
            display = bt_df.head(10).copy()
            display["rmse"]  = display["rmse"].apply(lambda v: f"S${v:.1f}" if pd.notna(v) else "N/A")
            display["mae"]   = display["mae"].apply(lambda v: f"S${v:.1f}"  if pd.notna(v) else "N/A")
            display["run_at"] = display["run_at"].astype(str).str[:16]
            display = display.rename(columns={
                "run_at": "Run", "model_name": "Model", "test_days": "Days",
                "rmse": "RMSE", "mae": "MAE", "n_periods": "Periods",
            })
            st.dataframe(display[["Run", "Model", "Days", "RMSE", "MAE", "Periods"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No backtest runs saved yet.")


# ── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Forecast — NEMS", layout="wide")

engine = _get_engine()

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
apply_theme_css()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ NEMS Analytics")
    st.caption("Singa Renewables")
    render_theme_toggle()
    st.divider()

    st.markdown("**Model**")
    model_choice = st.radio(
        "Model", ["XGBoost", "Prophet", "Ensemble"],
        index=0, label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Spike Alert Threshold**")
    spike_threshold = st.select_slider(
        "Threshold (S$/MWh)",
        options=[100, 150, 200, 250, 300, 400, 500, 1000],
        value=200,
        label_visibility="collapsed",
        key="spike_threshold",
        help="Historical spike frequency alert above this USEP level.",
    )
    st.caption(f"Alerts when historical P(USEP > S${spike_threshold}/MWh) > 25%")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🔮 NEMS Price Forecast")

st.info(
    "**Disclaimer:** Price spike events are caused by unplanned plant outages and "
    "gas supply disruptions — unpredictable from price history alone. This model "
    "forecasts expected price levels. Spike risk is shown as **historical frequency**, "
    "not a causal prediction."
)

cl = get_chart_layout()
rs = get_rangeselector_style()

# ─────────────────────────────────────────────────────────────────────────────
# Model status — 4 cards
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🗂 Model Status")

registry = load_model_registry(engine)
reg_map  = {r["model_name"]: r for r in registry}

xgb_ready = "xgboost" in reg_map and _model_file_exists("xgboost")
pro_ready  = "prophet"  in reg_map and _model_file_exists("prophet")
ens_ready  = "ensemble" in reg_map
clf_ready  = "spike_classifier" in reg_map and _spike_clf_exists()

col_xgb, col_pro, col_ens, col_clf = st.columns(4)

# XGBoost card
with col_xgb:
    st.markdown("**XGBoost**")
    if xgb_ready:
        r = reg_map["xgboost"]
        st.success("✅ Ready")
        st.caption(f"Trained: {str(r.get('trained_at', ''))[:19]}")
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"S${r['rmse']:.1f}" if r.get("rmse") else "N/A", help=_RMSE_HELP)
        m2.metric("MAE",  f"S${r['mae']:.1f}"  if r.get("mae")  else "N/A", help=_MAE_HELP)
        m3.metric("MAPE", f"{r['mape']:.1f}%"  if r.get("mape") else "N/A", help=_MAPE_HELP)
        st.caption(f"Rows: {r.get('training_rows', 0):,} | outlier exclusion")
    else:
        st.warning("⚠️ Not trained")

# Prophet card
with col_pro:
    st.markdown("**Prophet**")
    if pro_ready:
        r = reg_map["prophet"]
        st.success("✅ Ready")
        st.caption(f"Trained: {str(r.get('trained_at', ''))[:19]}")
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"S${r['rmse']:.1f}" if r.get("rmse") else "N/A",
                  help=_RMSE_HELP + " Prophet RMSE is daily avg.")
        m2.metric("MAE",  f"S${r['mae']:.1f}"  if r.get("mae")  else "N/A", help=_MAE_HELP)
        m3.metric("MAPE", f"{r['mape']:.1f}%"  if r.get("mape") else "N/A", help=_MAPE_HELP)
        st.caption(f"Days: {r.get('training_rows', 0):,} | daily + intraday scaling")
    else:
        st.warning("⚠️ Not trained")

# Ensemble card
with col_ens:
    st.markdown("**Ensemble**")
    if ens_ready:
        r = reg_map["ensemble"]
        st.success("✅ Ready")
        st.caption(f"Computed: {str(r['trained_at'])[:19]}")
        st.metric("Blended RMSE", f"{r['rmse']:.1f}",
                  help="Weighted average of XGBoost and Prophet RMSE (inverse-RMSE weights).")
        # Derive weights from component RMSEs
        if xgb_ready and pro_ready:
            rmse_x = reg_map["xgboost"]["rmse"]
            rmse_p = reg_map["prophet"]["rmse"]
            if rmse_x and rmse_p:
                inv_x = 1 / max(rmse_x, 0.01)
                inv_p = 1 / max(rmse_p, 0.01)
                tot   = inv_x + inv_p
                w_x   = inv_x / tot
                w_p   = inv_p / tot
                st.caption(f"XGBoost × {w_x:.2f} + Prophet × {w_p:.2f}")
    else:
        st.warning("⚠️ Not computed")

# Spike Classifier card
with col_clf:
    st.markdown("**Spike Classifier**")
    if clf_ready:
        r = reg_map["spike_classifier"]
        st.success("✅ Ready")
        st.caption(f"Trained: {str(r['trained_at'])[:19]}")
        # Stored as rmse=1-F1, mae=1-precision, mape=1-recall
        f1_val  = round(1 - float(r["rmse"]),  3) if r.get("rmse")  is not None else 0.0
        prec    = round(1 - float(r["mae"]),   3) if r.get("mae")   is not None else 0.0
        rec_val = round(1 - float(r["mape"]),  3) if r.get("mape")  is not None else 0.0
        m1, m2, m3 = st.columns(3)
        m1.metric("F1",    f"{f1_val:.3f}",
                  help="Harmonic mean of precision & recall for spike detection.")
        m2.metric("Prec.", f"{prec:.3f}",
                  help="Fraction of predicted spikes that actually exceeded the threshold.")
        m3.metric("Recall", f"{rec_val:.3f}",
                  help="Fraction of actual spike events that the classifier caught.")
        st.caption(f"Rows: {r['training_rows']:,} | threshold S$200/MWh")
    else:
        st.warning("⚠️ Not trained")

if st.button("🔄 Retrain All Models", type="secondary"):
    with st.spinner("Loading full dataset…"):
        full_df = load_full_df(engine)
    prog = st.progress(0, text="Training XGBoost…")
    try:
        r_xgb = train_xgboost(full_df, engine, outlier_treatment="exclude")
        prog.progress(25, text="Training Prophet…")
        r_pro = train_prophet(full_df, engine, outlier_treatment="winsorize")
        prog.progress(55, text="Computing ensemble weights…")
        r_ens = train_ensemble(engine)
        prog.progress(70, text="Training spike classifier…")
        r_clf = train_spike_classifier(full_df, engine, threshold=200.0)
        prog.progress(100, text="Done.")
        load_model_registry.clear()
        st.success(
            f"✅ XGBoost RMSE={r_xgb['rmse']:.1f}  |  Prophet RMSE={r_pro['rmse']:.1f}  |  "
            f"Ensemble RMSE≈{r_ens.get('blended_rmse', '—')}  |  Spike F1={r_clf.get('f1', '—')}"
        )
        st.rerun()
    except Exception as e:
        st.error(f"Training failed: {e}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 3-horizon tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_st, tab_mt, tab_sc = st.tabs([
    "⚡ Short-term (24–48 h)",
    "📅 Medium-term (7–14 d)",
    "📊 Monthly Scenarios",
])


def _forecast_chart(model_choice, engine, n_periods, hist_days, cl, rs,
                    spike_thr: float = 200.0):
    """Shared forecast chart renderer for short-term and medium-term tabs."""
    can_run = {
        "XGBoost":  xgb_ready,
        "Prophet":  pro_ready,
        "Ensemble": xgb_ready and pro_ready,
    }
    if not can_run[model_choice]:
        st.warning(f"**{model_choice}** is not trained yet. Click Retrain above.")
        return

    with st.spinner(f"Generating {model_choice} forecast ({n_periods} periods)…"):
        if model_choice == "XGBoost":
            fc_df = predict("xgboost", engine, n_periods, spike_threshold=spike_thr)
        elif model_choice == "Prophet":
            fc_df = predict("prophet", engine, n_periods, spike_threshold=spike_thr)
        else:
            fc_df = predict_ensemble(engine, n_periods, spike_threshold=spike_thr)

    hist_df = load_recent_history(engine, days=hist_days)

    fig = go.Figure()
    fig.update_layout(**cl)

    if not hist_df.empty:
        fig.add_trace(go.Scattergl(
            x=hist_df["dt"], y=hist_df["usep"],
            mode="lines", name="Historical USEP",
            line=dict(color=COLOR_USEP, width=1),
            hovertemplate="<b>%{x}</b><br>USEP: S$%{y:.2f}/MWh<extra></extra>",
        ))

    spike_alert_periods = []
    if not fc_df.empty:
        fig.add_trace(go.Scatter(
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
        fig.add_trace(go.Scatter(
            x=fc_df["dt"], y=fc_df["predicted_usep"],
            mode="lines", name=f"{model_choice} forecast",
            line=dict(color=fc_color, width=2, dash="dash"),
            hovertemplate="<b>%{x}</b><br>Forecast: S$%{y:.2f}/MWh<extra></extra>",
        ))
        now_dt = fc_df["dt"].min()
        fig.add_vline(
            x=now_dt.timestamp() * 1000,
            line_dash="dot", line_color="#555",
            annotation_text="Now", annotation_font_color="#888",
        )
        spike_alert_periods = fc_df[fc_df["spike_prob"] > 0.25]["time_label"].tolist()

    fig.update_layout(
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
        ),
        yaxis=dict(title="USEP (S$/MWh)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"fc_{model_choice}_{n_periods}")
    add_copy_button(f"fc_{model_choice}_{n_periods}")

    # Dynamic spike alert banner
    if spike_alert_periods:
        times = ", ".join(spike_alert_periods[:8])
        st.warning(
            f"⚠️ **Elevated spike risk** (hist. freq > 25% above S${spike_thr:.0f}/MWh) "
            f"at: {times} SGT  \n"
            "Historical frequency only — not a causal prediction."
        )

    if not fc_df.empty and "spike_prob" in fc_df.columns:
        with st.expander(f"Spike probability & exceedance curve (threshold S${spike_thr:.0f})"):
            left, right = st.columns(2)
            with left:
                # Bar chart: historical spike freq per forecast period
                fig_sp = go.Figure()
                cl_base = {k: v for k, v in cl.items() if k not in ("xaxis", "yaxis")}
                fig_sp.update_layout(**cl_base)
                fig_sp.add_trace(go.Bar(
                    x=fc_df["time_label"],
                    y=fc_df["spike_prob"] * 100,
                    marker_color=fc_df["spike_prob"].apply(
                        lambda v: "#2ecc71" if v < 0.15
                        else ("#f0b429" if v < 0.30 else "#e74c3c")
                    ),
                    hovertemplate="%{x}<br>Spike freq: %{y:.1f}%<extra></extra>",
                ))
                fig_sp.update_layout(
                    height=240, title=f"P(USEP > S${spike_thr:.0f}) by period",
                    xaxis=dict(title="Time (SGT)"),
                    yaxis=dict(title="Historical freq. (%)"),
                    showlegend=False,
                )
                st.plotly_chart(fig_sp, use_container_width=True)

            with right:
                # Exceedance / fan chart
                peak_period = int(fc_df.loc[fc_df["predicted_usep"].idxmax(), "period"])
                peak_month  = date.today().month
                exc_df = spike_exceedance_curve(engine, peak_period, peak_month)
                fig_ex = go.Figure()
                fig_ex.update_layout(**cl_base)
                fig_ex.add_trace(go.Scatter(
                    x=exc_df["threshold"],
                    y=exc_df["exceedance_prob"] * 100,
                    mode="lines+markers",
                    line=dict(color="#e74c3c", width=2),
                    marker=dict(size=6),
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
                    hovertemplate="S$%{x}/MWh → P(exceed)=%{y:.1f}%<extra></extra>",
                ))
                fig_ex.add_vline(
                    x=spike_thr, line_dash="dash", line_color="#f0b429",
                    annotation_text=f"S${spike_thr:.0f}", annotation_font_color="#f0b429",
                )
                fig_ex.update_layout(
                    height=240,
                    title=f"Exceedance curve — P{peak_period} (current month)",
                    xaxis=dict(title="USEP threshold (S$/MWh)"),
                    yaxis=dict(title="P(USEP > threshold) %"),
                    showlegend=False,
                )
                st.plotly_chart(fig_ex, use_container_width=True)


# ── Tab 1: Short-term 24–48h ─────────────────────────────────────────────────
with tab_st:
    st.subheader(f"⚡ Short-term Forecast — {model_choice}")
    st.caption("XGBoost is primary model for 24–48h. High precision, lower confidence band width.")

    n_st = st.radio(
        "Horizon", ["24h (48 periods)", "48h (96 periods)"],
        index=0, horizontal=True, key="n_st",
    )
    n_periods_st = 48 if "24h" in n_st else 96

    _forecast_chart(model_choice, engine, n_periods_st,
                    hist_days=7, cl=cl, rs=rs, spike_thr=float(spike_threshold))

    st.divider()
    st.subheader("📊 Model Performance")

    if not (xgb_ready or pro_ready):
        st.info("Train at least one model to see performance metrics.")
    else:
        perf_rows = []
        for mname, mlabel in [("xgboost", "XGBoost"), ("prophet", "Prophet"),
                               ("ensemble", "Ensemble"), ("spike_classifier", "Spike Clf")]:
            if mname not in reg_map:
                continue
            r = reg_map[mname]
            if mname == "spike_classifier":
                perf_rows.append({
                    "Model":    mlabel,
                    "F1":       f"{1 - r['rmse']:.3f}" if r.get("rmse") is not None else "N/A",
                    "Precision": f"{1 - r['mae']:.3f}" if r.get("mae")  is not None else "N/A",
                    "Recall":   f"{1 - r['mape']:.3f}" if r.get("mape") is not None else "N/A",
                    "Training rows": f"{r.get('training_rows') or 0:,}",
                })
            elif mname == "ensemble":
                perf_rows.append({
                    "Model":    mlabel,
                    "RMSE":     f"S${r['rmse']:.1f}/MWh" if r.get("rmse") else "N/A",
                    "MAE":      "—",
                    "MAPE":     "—",
                    "Training rows": "—",
                })
            else:
                perf_rows.append({
                    "Model":    mlabel,
                    "RMSE":     f"S${r['rmse']:.1f}/MWh" if r.get("rmse") else "N/A",
                    "MAE":      f"S${r['mae']:.1f}/MWh"  if r.get("mae")  else "N/A",
                    "MAPE":     f"{r['mape']:.1f}%"       if r.get("mape") else "N/A",
                    "Training rows": f"{r.get('training_rows') or 0:,}",
                })
        if perf_rows:
            st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)
        st.caption(
            "High RMSE is driven by unpredictable spike events (USEP can reach "
            "S$4,500/MWh during grid stress). MAE reflects typical off-peak accuracy. "
            "Prophet RMSE is on daily averages — multiply by ~1.5× for half-hourly equivalent."
        )

    st.divider()
    st.subheader("🔑 Feature Importance (XGBoost)")

    if not xgb_ready:
        st.info("Train XGBoost to see feature importance.")
    else:
        import joblib as _jl
        payload   = _jl.load(MODELS_DIR / "xgboost_model.joblib")
        model_obj = payload["model"]
        feats     = payload["features"]
        importance = sorted(
            zip(feats, model_obj.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )[:15]
        names  = [f for f, _ in importance]
        values = [v for _, v in importance]

        def _feat_color(name: str) -> str:
            if "lag" in name or "rolling" in name:
                return "#009CEA"
            if any(x in name for x in ["sin", "cos", "hour", "period", "month", "day",
                                        "weekend", "holiday", "dt_", "days_to"]):
                return "#f0b429"
            return "#2ecc71"

        fig_imp = go.Figure()
        cl_base = {k: v for k, v in cl.items() if k not in ("xaxis", "yaxis")}
        fig_imp.update_layout(**cl_base)
        fig_imp.add_trace(go.Bar(
            x=values[::-1], y=names[::-1], orientation="h",
            marker_color=[_feat_color(f) for f in names[::-1]],
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        ))
        for color, label in [
            ("#009CEA", "Lag / rolling features"),
            ("#f0b429", "Time / holiday features"),
            ("#2ecc71", "Market / gas features"),
        ]:
            fig_imp.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color, symbol="square"), name=label,
            ))
        fig_imp.update_layout(
            height=420, bargap=0.25,
            xaxis=dict(title="Feature importance (gain)"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    st.subheader("🔁 Walk-forward Backtest")

    col_sl, col_est = st.columns([3, 2])
    with col_sl:
        bt_days = st.slider(
            "Backtest window (days)", min_value=30, max_value=730, value=90, step=10,
            key="bt_days",
        )
    with col_est:
        est_min = bt_days * 10 / 60
        st.info(
            f"**{bt_days} days** selected  \n"
            f"Est. runtime: **~{est_min:.0f} min**  \n"
            f"(≈ 10 sec/day, one model fit per day)"
        )

    st.warning(
        "⏱ Re-trains XGBoost for each test day. Results cached in session."
    )

    if "backtest_df" not in st.session_state:
        st.session_state["backtest_df"] = None
    if "backtest_days" not in st.session_state:
        st.session_state["backtest_days"] = None

    if st.button(f"▶ Run {bt_days}-day Backtest", type="primary"):
        if not xgb_ready:
            st.error("Train XGBoost first.")
        else:
            full_df = load_full_df(engine)
            prog    = st.progress(0, text=f"Running {bt_days}-day backtest…")

            def _cb(frac: float) -> None:
                day_num = int(frac * bt_days)
                prog.progress(
                    min(frac, 1.0),
                    text=f"Day {day_num}/{bt_days} — {frac*100:.0f}%",
                )

            bt_df = backtest_walk_forward(full_df, test_days=bt_days,
                                          progress_callback=_cb, engine=engine)
            st.session_state["backtest_df"]   = bt_df
            st.session_state["backtest_days"] = bt_days
            prog.empty()

    bt_df = st.session_state.get("backtest_df")

    if bt_df is not None and not bt_df.empty:
        bt_label = st.session_state.get("backtest_days", "?")
        overall_rmse = float(np.sqrt(np.mean(bt_df["error"] ** 2)))
        overall_mae  = float(np.mean(np.abs(bt_df["error"])))

        st.caption(f"Showing {bt_label}-day backtest results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Backtest RMSE", f"S${overall_rmse:.1f}/MWh")
        c2.metric("Backtest MAE",  f"S${overall_mae:.1f}/MWh")
        c3.metric("Days tested",   str(bt_df["date"].nunique()))

        # Actual vs predicted scatter
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
            xaxis=dict(title="Actual USEP (S$/MWh)"),
            yaxis=dict(title="Predicted USEP (S$/MWh)"),
        )
        st.plotly_chart(fig_bt, use_container_width=True, key="backtest_scatter")
        add_copy_button("backtest_scatter")

        # Breakdown tables side by side
        col_bucket, col_year = st.columns(2)

        with col_bucket:
            st.markdown("**By period bucket**")
            bucket_rows = []
            for bucket in BUCKET_ORDER:
                sub = bt_df[bt_df["bucket"] == bucket]
                if sub.empty:
                    continue
                bucket_rows.append({
                    "Period bucket":  BUCKET_LABELS.get(bucket, bucket),
                    "Periods":        len(sub),
                    "RMSE (S$/MWh)":  round(float(np.sqrt(np.mean(sub["error"] ** 2))), 1),
                    "MAE (S$/MWh)":   round(float(np.mean(np.abs(sub["error"]))), 1),
                })
            st.dataframe(pd.DataFrame(bucket_rows), use_container_width=True, hide_index=True)

        with col_year:
            st.markdown("**Year-by-year breakdown**")
            if "year" in bt_df.columns:
                year_rows = []
                for yr in sorted(bt_df["year"].unique()):
                    sub = bt_df[bt_df["year"] == yr]
                    year_rows.append({
                        "Year":          yr,
                        "Days":          sub["date"].nunique(),
                        "RMSE (S$/MWh)": round(float(np.sqrt(np.mean(sub["error"] ** 2))), 1),
                        "MAE (S$/MWh)":  round(float(np.mean(np.abs(sub["error"]))), 1),
                    })
                st.dataframe(pd.DataFrame(year_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📈 Model History")
    _load_model_history(engine, cl)


# ── Tab 2: Medium-term 7–14d ─────────────────────────────────────────────────
with tab_mt:
    st.subheader(f"📅 Medium-term Forecast — {model_choice}")
    st.caption(
        "Ensemble model preferred for 7–14d. Confidence band widens with horizon — "
        "treat as directional guidance, not point estimates."
    )

    n_mt = st.radio(
        "Horizon", ["7 days (336 periods)", "14 days (672 periods)"],
        index=0, horizontal=True, key="n_mt",
    )
    n_periods_mt = 336 if "7 days" in n_mt else 672

    _forecast_chart(model_choice, engine, n_periods_mt,
                    hist_days=14, cl=cl, rs=rs, spike_thr=float(spike_threshold))

    st.caption(
        "⚠️ Beyond 3–4 days, XGBoost relies on repeated lag patterns. "
        "RMSE typically grows to 2–3× the 24h value by day 7. "
        "Use for trend direction, not intraday dispatch decisions."
    )


# ── Tab 3: Monthly Scenarios ──────────────────────────────────────────────────
with tab_sc:
    st.subheader("📊 Monthly USEP Scenarios — P10 / P50 / P90")
    st.caption(
        "Scenarios from seasonal percentiles of historical USEP adjusted by current "
        "gas price regime (90-day weighted average). Scenario ranges, not point forecasts."
    )

    target_months = st.slider("Months ahead", 1, 6, 3)

    with st.spinner("Computing monthly scenarios…"):
        scen_df = forecast_monthly_scenarios(engine, target_months=target_months)

    if scen_df.empty:
        st.info("Insufficient data for monthly scenarios.")
    else:
        gas_regime = scen_df["gas_regime"].iloc[0] if "gas_regime" in scen_df.columns else "neutral"
        regime_icon = {"high": "🔴", "low": "🟢", "neutral": "🟡"}.get(gas_regime, "⚪")
        st.info(
            f"{regime_icon} **Gas regime: {gas_regime.title()}** — "
            "P10/P50/P90 are adjusted from historical seasonal distributions. "
            "High gas regime applies +15% multiplier; low applies -10%."
        )

        cl_base = {k: v for k, v in cl.items() if k not in ("xaxis", "yaxis")}
        fig_sc = go.Figure()
        fig_sc.update_layout(**cl_base)

        x = scen_df["month_label"].tolist()
        fig_sc.add_trace(go.Bar(
            x=x, y=scen_df["p50"],
            name="P50 (median)",
            marker_color="#009CEA",
            error_y=dict(
                type="data", symmetric=False,
                array=(scen_df["p90"] - scen_df["p50"]).tolist(),
                arrayminus=(scen_df["p50"] - scen_df["p10"]).tolist(),
                color="#555", thickness=2, width=6,
            ),
            hovertemplate="%{x}<br>P50: S$%{y:.0f}/MWh<extra></extra>",
        ))
        fig_sc.add_trace(go.Scatter(
            x=x, y=scen_df["p10"], mode="markers+lines", name="P10 (optimistic)",
            marker=dict(color="#2ecc71", size=10, symbol="triangle-down"),
            line=dict(color="#2ecc71", width=1, dash="dot"),
            hovertemplate="%{x}<br>P10: S$%{y:.0f}/MWh<extra></extra>",
        ))
        fig_sc.add_trace(go.Scatter(
            x=x, y=scen_df["p90"], mode="markers+lines", name="P90 (stressed)",
            marker=dict(color="#e74c3c", size=10, symbol="triangle-up"),
            line=dict(color="#e74c3c", width=1, dash="dot"),
            hovertemplate="%{x}<br>P90: S$%{y:.0f}/MWh<extra></extra>",
        ))
        fig_sc.update_layout(
            height=420, barmode="overlay",
            title="Monthly USEP Scenario Range (P10 / P50 / P90)",
            xaxis=dict(title="Month"),
            yaxis=dict(title="USEP (S$/MWh)"),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        st.dataframe(
            scen_df[["month_label", "p10", "p50", "p90", "gas_regime"]].rename(columns={
                "month_label": "Month", "p10": "P10 (S$/MWh)",
                "p50": "P50 (S$/MWh)", "p90": "P90 (S$/MWh)",
                "gas_regime": "Gas Regime",
            }),
            use_container_width=True, hide_index=True,
        )

        st.caption(
            "**Methodology:** Historical USEP percentiles by calendar month (2019–present), "
            "adjusted by 90-day rolling gas price average. No ML model used for monthly "
            "scenarios — designed to avoid false precision at multi-month horizons."
        )
