"""
Market Intelligence Hub — Data Hub
5 sections: status, universal upload, forecast accuracy, gas analysis, analyst comparison.
"""
import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from db import get_engine, setup_database
from modules.ingestion import (
    detect_file_type,
    get_database_summary,
    ingest_analyst_forecast,
    ingest_and_retrain,
    ingest_gas_prices,
)
from modules.analysis import gas_usep_correlation, analyst_vs_actuals, vintage_comparison
from modules.theme import apply_theme_css, get_chart_layout, get_rangeselector_style, render_theme_toggle

st.set_page_config(page_title="Data Hub", layout="wide", page_icon="🗄️")

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
apply_theme_css(st.session_state["theme"])

render_theme_toggle()

engine = get_engine()
setup_database(engine)

st.title("🗄️ Market Intelligence Hub")

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Data Sources Status
# ─────────────────────────────────────────────────────────────────────────────
st.header("1 · Data Sources Status")


@st.cache_data(ttl=300)
def _get_source_status(_engine):
    summary = get_database_summary(_engine)

    with _engine.connect() as conn:
        gas_row = conn.execute(text("""
            SELECT COUNT(*) AS n, MIN(price_date) AS min_d, MAX(price_date) AS max_d
            FROM gas_prices
        """)).mappings().fetchone()

        src_rows = conn.execute(text("""
            SELECT fs.source_name, fs.vintage_year, fs.granularity, fs.row_count, fs.uploaded_at
            FROM forecast_sources fs
            ORDER BY fs.uploaded_at DESC
        """)).fetchall()

        fa_row = conn.execute(text("""
            SELECT COUNT(*) AS n,
                   SUM(CASE WHEN actual_usep IS NOT NULL THEN 1 ELSE 0 END) AS matched
            FROM forecast_actuals
        """)).mappings().fetchone()

        model_row = conn.execute(text("""
            SELECT model_name, trained_at, rmse, mae FROM model_registry
            WHERE is_active = 1 ORDER BY trained_at DESC LIMIT 1
        """)).fetchone()

    return summary, gas_row, src_rows, fa_row, model_row


summary, gas_row, src_rows, fa_row, model_row = _get_source_status(engine)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("NEMS Rows", f"{summary['total_rows']:,}")
    st.caption(f"{summary['min_date']} → {summary['max_date']}")
with col2:
    gas_n = gas_row["n"] if gas_row else 0
    st.metric("Gas Price Days", f"{gas_n:,}")
    if gas_n:
        st.caption(f"{gas_row['min_d']} → {gas_row['max_d']}")
with col3:
    st.metric("Analyst Sources", str(len(src_rows)))
    st.caption(f"{len(src_rows)} file(s) loaded")
with col4:
    fa_total = fa_row["n"] if fa_row else 0
    fa_matched = fa_row["matched"] if fa_row else 0
    st.metric("Forecast–Actual Pairs", f"{fa_matched:,} / {fa_total:,}")

if model_row:
    st.info(
        f"**Active model:** {model_row[0]} | "
        f"Trained: {str(model_row[1])[:10]} | "
        f"RMSE: {model_row[2]:.1f} | MAE: {model_row[3]:.1f}"
    )

if src_rows:
    src_df = pd.DataFrame(
        src_rows,
        columns=["Source Name", "Vintage Year", "Granularity", "Rows", "Uploaded At"],
    )
    st.dataframe(src_df, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Universal Drag & Drop Upload
# ─────────────────────────────────────────────────────────────────────────────
st.header("2 · Universal Upload")
st.caption("Supports NEMS price CSV/Excel, gas price CSV, and analyst forecast CSV/Excel.")

uploaded = st.file_uploader(
    "Drop file here",
    type=["csv", "xlsx", "xls"],
    key="hub_upload",
)

if uploaded:
    try:
        if uploaded.name.endswith((".xlsx", ".xls")):
            raw_df = pd.read_excel(uploaded, engine="openpyxl")
        else:
            raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        raw_df = None

    if raw_df is not None:
        detected_type = detect_file_type(raw_df)
        st.info(f"Detected file type: **{detected_type}**")

        # Save to temp path for ingestion functions
        tmp_path = Path(f"/tmp/_hub_upload_{uploaded.name}")
        tmp_path.write_bytes(uploaded.getvalue())

        if detected_type == "nems_prices":
            if st.button("Ingest NEMS Prices + Check Drift"):
                with st.spinner("Ingesting..."):
                    result = ingest_and_retrain(tmp_path, engine)
                st.success(
                    f"Imported {result['rows_imported']:,} rows | "
                    f"Skipped {result['rows_skipped']:,} | "
                    f"Drift detected: {result['drift_detected']} | "
                    f"Retrained: {result['retrained']}"
                )
                if result["errors"]:
                    st.warning("Errors: " + "; ".join(result["errors"][:3]))
                st.cache_data.clear()

        elif detected_type == "gas_prices":
            if st.button("Ingest Gas Prices"):
                with st.spinner("Ingesting gas prices..."):
                    result = ingest_gas_prices(tmp_path, engine)
                st.success(
                    f"Imported {result['rows_imported']:,} | "
                    f"Skipped {result['rows_skipped']:,}"
                )
                if result["errors"]:
                    st.warning("; ".join(result["errors"][:3]))
                st.cache_data.clear()

        elif detected_type == "analyst_forecast":
            with st.form("analyst_meta"):
                c1, c2, c3 = st.columns(3)
                source_name  = c1.text_input("Source Name", value=uploaded.name.rsplit(".", 1)[0])
                vintage_year = c2.number_input("Vintage Year", min_value=2020, max_value=2040, value=2025)
                granularity  = c3.selectbox("Granularity", ["annual", "monthly", "daily", "half_hourly"])
                submitted    = st.form_submit_button("Ingest Analyst Forecast")

            if submitted:
                with st.spinner("Ingesting..."):
                    result = ingest_analyst_forecast(
                        tmp_path, engine, source_name, int(vintage_year), granularity
                    )
                st.success(
                    f"Imported {result['rows_imported']:,} expanded rows | "
                    f"Source ID: {result.get('source_id')}"
                )
                if result["errors"]:
                    st.warning("; ".join(result["errors"][:3]))
                st.cache_data.clear()

        else:
            st.warning("Unknown file type. Please map columns manually.")
            st.dataframe(raw_df.head(5), use_container_width=True)

            with st.expander("Manual column mapper"):
                st.caption("Map your columns to NEMS schema if auto-detect failed.")
                col_opts = ["(skip)"] + list(raw_df.columns)
                date_col   = st.selectbox("date",   col_opts)
                period_col = st.selectbox("period", col_opts)
                usep_col   = st.selectbox("usep",   col_opts)
                if st.button("Try manual NEMS import"):
                    mapped = {}
                    if date_col != "(skip)":   mapped[date_col]   = "DATE"
                    if period_col != "(skip)": mapped[period_col] = "PERIOD"
                    if usep_col != "(skip)":   mapped[usep_col]   = "USEP ($/MWh)"
                    remapped = raw_df.rename(columns={v: k for k, v in mapped.items()})
                    tmp2 = tmp_path.with_suffix(".mapped.csv")
                    remapped.to_csv(tmp2, index=False)
                    from modules.ingestion import import_csv_to_db
                    res = import_csv_to_db(tmp2, engine)
                    st.write(res)
                    st.cache_data.clear()

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Forecast Accuracy Tracker
# ─────────────────────────────────────────────────────────────────────────────
st.header("3 · Forecast Accuracy Tracker")


@st.cache_data(ttl=300)
def _get_rolling_rmse(_engine):
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT model_name, forecast_date,
                   SQRT(AVG(abs_error * abs_error)) AS rmse,
                   COUNT(*) AS n
            FROM forecast_actuals
            WHERE actual_usep IS NOT NULL
            GROUP BY model_name, forecast_date
            ORDER BY forecast_date
        """)).fetchall()
    return pd.DataFrame(rows, columns=["model_name", "forecast_date", "rmse", "n"])


@st.cache_data(ttl=300)
def _get_baseline_rmse(_engine):
    with _engine.connect() as conn:
        row = conn.execute(text("""
            SELECT rmse FROM model_registry
            WHERE is_active = 1 ORDER BY trained_at DESC LIMIT 1
        """)).fetchone()
    return float(row[0]) if row else None


rmse_df   = _get_rolling_rmse(engine)
base_rmse = _get_baseline_rmse(engine)

if rmse_df.empty:
    st.info("No forecast–actual pairs yet. Run a forecast and save predictions to populate.")
else:
    cl = get_chart_layout(st.session_state["theme"])
    fig = go.Figure()

    rs_style = get_rangeselector_style(st.session_state["theme"])
    for model, grp in rmse_df.groupby("model_name"):
        fig.add_trace(go.Scatter(
            x=grp["forecast_date"], y=grp["rmse"],
            mode="lines", name=model, line=dict(width=2),
        ))

    if base_rmse:
        drift_thresh = base_rmse * 1.5
        fig.add_hline(y=base_rmse,    line_dash="dash", line_color="#2ecc71",
                      annotation_text=f"Baseline {base_rmse:.0f}")
        fig.add_hline(y=drift_thresh, line_dash="dot",  line_color="#e74c3c",
                      annotation_text=f"Drift threshold {drift_thresh:.0f}")

    fig.update_layout(**cl)
    fig.update_layout(
        title="Rolling Daily RMSE by Model",
        xaxis=dict(
            title="Date",
            rangeselector=rs_style,
            rangeslider=dict(visible=True, thickness=0.04),
            type="date",
        ),
        yaxis=dict(title="RMSE ($/MWh)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-bucket breakdown
    with engine.connect() as conn:
        bucket_rows = conn.execute(text("""
            SELECT model_name,
                   CASE
                     WHEN period <= 13 THEN 'off_peak'
                     WHEN period <= 30 THEN 'solar'
                     WHEN period <= 42 THEN 'evening_peak'
                     ELSE 'night'
                   END AS bucket,
                   SQRT(AVG(abs_error * abs_error)) AS rmse,
                   AVG(error) AS bias,
                   COUNT(*) AS n
            FROM forecast_actuals
            WHERE actual_usep IS NOT NULL
            GROUP BY model_name, bucket
            ORDER BY model_name, bucket
        """)).fetchall()

    if bucket_rows:
        bucket_df = pd.DataFrame(bucket_rows, columns=["Model", "Bucket", "RMSE", "Bias", "N"])
        st.dataframe(bucket_df.style.format({"RMSE": "{:.1f}", "Bias": "{:.1f}"}),
                     use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Gas Price Analysis
# ─────────────────────────────────────────────────────────────────────────────
st.header("4 · Gas Price Analysis")


@st.cache_data(ttl=300)
def _get_gas_series(_engine):
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT price_date, jkm_usd_mmbtu, piped_gas_sgd_mmbtu
            FROM gas_prices ORDER BY price_date
        """)).fetchall()
    return pd.DataFrame(rows, columns=["date", "jkm", "piped"])


@st.cache_data(ttl=300)
def _get_daily_usep(_engine):
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, AVG(usep) AS avg_usep FROM nems_prices
            WHERE usep IS NOT NULL GROUP BY date ORDER BY date
        """)).fetchall()
    return pd.DataFrame(rows, columns=["date", "avg_usep"])


gas_df  = _get_gas_series(engine)
usep_df = _get_daily_usep(engine)

if gas_df.empty:
    st.info("No gas price data yet. Upload a gas price CSV/Excel above.")
else:
    gas_df["date"]  = pd.to_datetime(gas_df["date"])
    usep_df["date"] = pd.to_datetime(usep_df["date"])
    merged = gas_df.merge(usep_df, on="date", how="inner")

    cl = get_chart_layout(st.session_state["theme"])
    rs_style = get_rangeselector_style(st.session_state["theme"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gas_df["date"], y=gas_df["jkm"],
        name="JKM (USD/MMBtu)", yaxis="y1",
        line=dict(color="#f0b429", width=2),
    ))
    if not merged.empty:
        fig.add_trace(go.Scatter(
            x=merged["date"], y=merged["avg_usep"],
            name="Daily Avg USEP ($/MWh)", yaxis="y2",
            line=dict(color="#009CEA", width=1.5, dash="dot"),
        ))

    fig.update_layout(**cl)
    fig.update_layout(
        title="JKM Gas Price vs Daily Average USEP",
        xaxis=dict(
            rangeselector=rs_style,
            rangeslider=dict(visible=True, thickness=0.04),
            type="date",
        ),
        yaxis=dict(title="JKM (USD/MMBtu)", side="left"),
        yaxis2=dict(title="Avg USEP ($/MWh)", side="right", overlaying="y"),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Lag correlation bar chart
    with st.spinner("Computing lag correlations..."):
        corr_result = gas_usep_correlation(engine, lag_days=[0, 1, 3, 7, 14, 30])

    if corr_result["by_lag"]:
        lag_df = pd.DataFrame(corr_result["by_lag"])
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[f"Lag {r['lag_days']}d" for r in corr_result["by_lag"]],
            y=[r["pearson_r"] for r in corr_result["by_lag"]],
            marker_color=["#2ecc71" if r["pearson_r"] > 0 else "#e74c3c"
                          for r in corr_result["by_lag"]],
            name="Pearson r",
        ))
        fig2.update_layout(**cl)
        fig2.update_layout(
            title="JKM → USEP Pearson Correlation by Lag",
            xaxis=dict(title="Lag"),
            yaxis=dict(title="Pearson r", range=[-1, 1]),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(
            lag_df.rename(columns={
                "lag_days": "Lag (days)", "pearson_r": "Pearson r",
                "spearman_rho": "Spearman ρ", "r2": "R²", "n_obs": "N obs",
            }),
            use_container_width=True, hide_index=True,
        )

    if not corr_result["rolling"].empty:
        roll = corr_result["rolling"]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=roll["date"], y=roll["corr_30d"],
            mode="lines", name="30-day rolling Pearson r",
            line=dict(color="#9b59b6", width=1.5),
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="#555")
        fig3.update_layout(**cl)
        fig3.update_layout(
            title="Rolling 30-Day Gas–USEP Correlation",
            xaxis=dict(title="Date", type="date"),
            yaxis=dict(title="Pearson r (30d)", range=[-1, 1]),
        )
        st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Analyst Forecast Comparison
# ─────────────────────────────────────────────────────────────────────────────
st.header("5 · Analyst Forecast Comparison")

if not src_rows:
    st.info("No analyst forecasts loaded. Upload a forecast file in Section 2.")
else:
    all_source_names = list({r[0] for r in src_rows})
    selected_sources = st.multiselect("Select sources to compare", all_source_names, default=all_source_names[:3])

    if selected_sources:
        with st.spinner("Computing accuracy metrics..."):
            acc_df = analyst_vs_actuals(engine)

        if not acc_df.empty:
            display_acc = acc_df[acc_df["source_name"].isin(selected_sources)]
            if not display_acc.empty:
                st.dataframe(
                    display_acc.rename(columns={
                        "source_name": "Source", "vintage_year": "Vintage",
                        "n_overlap": "N", "mae": "MAE", "rmse": "RMSE",
                        "bias": "Bias", "pearson_r": "Pearson r",
                    }),
                    use_container_width=True, hide_index=True,
                )

        # Vintage comparison for single source
        if len(selected_sources) == 1:
            vint_df = vintage_comparison(engine, selected_sources[0])
            if not vint_df.empty:
                cl = get_chart_layout(st.session_state["theme"])
                fig = go.Figure()
                forecast_cols = [c for c in vint_df.columns if c.startswith("forecast_")]
                for col in forecast_cols:
                    fig.add_trace(go.Scatter(
                        x=vint_df["date"], y=vint_df[col],
                        mode="lines", name=col.replace("forecast_", "Vintage "),
                        line=dict(dash="dot"),
                    ))
                if "actual_usep" in vint_df.columns:
                    fig.add_trace(go.Scatter(
                        x=vint_df["date"], y=vint_df["actual_usep"],
                        mode="lines", name="Actual USEP",
                        line=dict(color="#009CEA", width=2),
                    ))
                fig.update_layout(**cl)
                fig.update_layout(
                    title=f"Vintage Comparison — {selected_sources[0]}",
                    xaxis=dict(title="Date", type="date"),
                    yaxis=dict(title="USEP ($/MWh)"),
                )
                st.plotly_chart(fig, use_container_width=True)

        # Multi-source forecast lines vs actuals
        else:
            @st.cache_data(ttl=300)
            def _get_source_daily(_engine, source_name):
                with _engine.connect() as conn:
                    rows = conn.execute(text("""
                        SELECT fd.date, AVG(fd.price) AS forecast_usep
                        FROM forecast_data fd
                        JOIN forecast_sources fs ON fs.id = fd.source_id
                        WHERE fs.source_name = :src
                        GROUP BY fd.date ORDER BY fd.date
                    """), {"src": source_name}).fetchall()
                return pd.DataFrame(rows, columns=["date", "forecast_usep"])

            cl = get_chart_layout(st.session_state["theme"])
            rs_style = get_rangeselector_style(st.session_state["theme"])
            fig = go.Figure()

            usep_daily = _get_daily_usep(engine)
            if not usep_daily.empty:
                fig.add_trace(go.Scatter(
                    x=usep_daily["date"], y=usep_daily["avg_usep"],
                    mode="lines", name="Actual USEP",
                    line=dict(color="#009CEA", width=2),
                ))

            for src in selected_sources:
                sdf = _get_source_daily(engine, src)
                if not sdf.empty:
                    fig.add_trace(go.Scatter(
                        x=sdf["date"], y=sdf["forecast_usep"],
                        mode="lines", name=src,
                        line=dict(dash="dot", width=1.5),
                    ))

            fig.update_layout(**cl)
            fig.update_layout(
                title="Analyst Forecasts vs Actual Daily USEP",
                xaxis=dict(
                    title="Date",
                    rangeselector=rs_style,
                    rangeslider=dict(visible=True, thickness=0.04),
                    type="date",
                ),
                yaxis=dict(title="USEP ($/MWh)"),
            )
            st.plotly_chart(fig, use_container_width=True)
