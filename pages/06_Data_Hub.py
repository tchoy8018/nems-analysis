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
from modules.analysis import analyst_vs_actuals, vintage_comparison
from modules.theme import apply_theme_css, get_chart_layout, get_rangeselector_style, render_theme_toggle

st.set_page_config(page_title="Data Hub", layout="wide", page_icon="🗄️")

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
apply_theme_css()

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
    cl = get_chart_layout()
    fig = go.Figure()

    rs_style = get_rangeselector_style()
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
# Section 4: Gas & USEP Intelligence
# ─────────────────────────────────────────────────────────────────────────────
st.header("4 · ⛽ Gas & USEP Intelligence")
st.caption("Source: Singapore Customs / S&P Global Energy Gas Trade Data Tables, June 2026")

from modules.analysis import gas_usep_correlation as _gas_usep_corr, gas_mix_evolution


@st.cache_data(ttl=300)
def _get_gas_full(_engine):
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT price_date,
                   malaysia_price_usd_mmbtu, indonesia_price_usd_mmbtu, lng_price_usd_mmbtu,
                   weighted_avg_usd_mmbtu, weighted_avg_sgd_mmbtu, implied_usep_floor_sgd_mwh,
                   malaysia_share_pct, indonesia_share_pct, lng_share_pct
            FROM gas_prices
            WHERE weighted_avg_usd_mmbtu IS NOT NULL
            ORDER BY price_date
        """)).fetchall()
    return pd.DataFrame(rows, columns=[
        "date", "my_price", "id_price", "lng_price",
        "weighted_usd", "weighted_sgd", "implied_floor",
        "my_share", "id_share", "lng_share",
    ])


@st.cache_data(ttl=300)
def _get_monthly_usep(_engine):
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT strftime('%Y-%m', date) AS ym, AVG(usep) AS avg_usep
            FROM nems_prices WHERE usep IS NOT NULL
            GROUP BY ym ORDER BY ym
        """)).fetchall()
    df = pd.DataFrame(rows, columns=["ym", "avg_usep"])
    df["date"] = pd.to_datetime(df["ym"] + "-01")
    return df


@st.cache_data(ttl=600)
def _get_gas_corr(_engine):
    return _gas_usep_corr(_engine)


gas_full = _get_gas_full(engine)
monthly_usep = _get_monthly_usep(engine)
corr_result = _get_gas_corr(engine)

if gas_full.empty:
    st.info("No gas price data. Import the S&P Global workbook in Section 2.")
else:
    gas_full["date"] = pd.to_datetime(gas_full["date"])
    latest = gas_full.iloc[-1]

    # KPI tiles
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Latest Weighted Gas",
        f"${latest['weighted_usd']:.2f}/MMBtu",
        help=f"S${latest['weighted_sgd']:.2f}/MMBtu after FX conversion",
    )
    k2.metric(
        "LNG Share",
        f"{latest['lng_share']:.0f}%",
        help="LNG's share of total gas volume (MT basis)",
    )
    k3.metric(
        "Implied CCGT Floor",
        f"S${latest['implied_floor']:.0f}/MWh",
        help="Weighted gas price × 7.5 MMBtu/MWh heat rate",
    )
    best_lag = corr_result.get("best_lag", 0)
    best_r = next((l["pearson_r"] for l in corr_result.get("lags", [])
                   if l["lag_months"] == best_lag), None)
    r2_pct = f"{corr_result['regression_r2']*100:.0f}%" if corr_result.get("regression_r2") else "—"
    k4.metric(
        "Gas→USEP Correlation",
        f"r = {best_r:.2f}" if best_r else "—",
        help=f"Pearson r at lag {best_lag}m. R² = {r2_pct}",
    )

    cl = get_chart_layout()
    rs_style = get_rangeselector_style()

    # ── Chart 1: Gas Price by Source ──────────────────────────────────────────
    fig1 = go.Figure()
    fig1.update_layout(**cl)

    # Filter from 2019 for cleaner display
    gf19 = gas_full[gas_full["date"] >= "2019-01-01"]

    fig1.add_trace(go.Scatter(x=gf19["date"], y=gf19["my_price"],
        name="Malaysia piped", line=dict(color="#009CEA", width=2)))
    fig1.add_trace(go.Scatter(x=gf19["date"], y=gf19["id_price"],
        name="Indonesia piped", line=dict(color="#f0b429", width=2)))
    fig1.add_trace(go.Scatter(x=gf19["date"], y=gf19["lng_price"],
        name="LNG", line=dict(color="#e74c3c", width=2)))

    for ann_date, ann_text in [("2022-02-01", "2022 energy crisis"),
                                ("2023-01-01", "Indonesia decline begins")]:
        if pd.to_datetime(ann_date) >= gf19["date"].min():
            fig1.add_vline(x=pd.to_datetime(ann_date).timestamp()*1000,
                           line_dash="dot", line_color="#555", line_width=1,
                           annotation_text=ann_text, annotation_font_size=10)

    fig1.update_layout(
        title="Gas Import Price by Source (USD/MMBtu)",
        xaxis=dict(title="Month",
                   rangeselector=rs_style,
                   rangeslider=dict(visible=True, thickness=0.04),
                   type="date"),
        yaxis=dict(title="USD/MMBtu"),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Weighted Gas vs Monthly USEP (dual axis) ────────────────────
    monthly_merged = gas_full.merge(
        monthly_usep[["date", "avg_usep"]], on="date", how="inner"
    )

    if not monthly_merged.empty:
        r2_str = f"{corr_result['regression_r2']:.2f}" if corr_result.get("regression_r2") else "N/A"
        fig2 = go.Figure()
        fig2.update_layout(**cl)
        fig2.add_trace(go.Scatter(
            x=monthly_merged["date"], y=monthly_merged["weighted_usd"],
            name="Weighted gas (USD/MMBtu)", yaxis="y1",
            line=dict(color="#009CEA", width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=monthly_merged["date"], y=monthly_merged["avg_usep"],
            name="Monthly avg USEP (S$/MWh)", yaxis="y2",
            line=dict(color="#f0b429", width=2, dash="dot"),
        ))
        fig2.add_trace(go.Scatter(
            x=monthly_merged["date"], y=monthly_merged["implied_floor"],
            name="Implied CCGT floor (S$/MWh)", yaxis="y2",
            line=dict(color="#e74c3c", width=1.5, dash="dash"),
        ))
        fig2.update_layout(
            title=f"Weighted Gas Price vs Monthly Avg USEP  (R²={r2_str} — gas explains {r2_str} of USEP variance)",
            xaxis=dict(title="Month",
                       rangeselector=rs_style,
                       rangeslider=dict(visible=True, thickness=0.04),
                       type="date"),
            yaxis=dict(title="Weighted gas (USD/MMBtu)", side="left"),
            yaxis2=dict(title="USEP / Floor (S$/MWh)", side="right", overlaying="y"),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig2, use_container_width=True)

        slope = corr_result.get("pass_through_slope")
        if slope:
            st.caption(
                f"**Pass-through:** S${slope:.1f} USEP change per S$1/MMBtu gas change "
                f"(expected ~7–10 × heat rate; higher value reflects market power + spike premium). "
                f"{corr_result.get('regime_note', '')}"
            )

    # ── Chart 3: Gas Supply Mix Evolution (stacked area) ─────────────────────
    mix_df = gas_mix_evolution(engine)
    if not mix_df.empty:
        mix19 = mix_df[mix_df["date"] >= "2019-01-01"]
        fig3 = go.Figure()
        cl_base = {k: v for k, v in cl.items() if k not in ("xaxis", "yaxis")}
        fig3.update_layout(**cl_base)
        fig3.add_trace(go.Scatter(
            x=mix19["date"], y=mix19["malaysia_share_pct"],
            name="Malaysia piped", stackgroup="one",
            fillcolor="rgba(0,156,234,0.7)", line=dict(color="#009CEA", width=0),
        ))
        fig3.add_trace(go.Scatter(
            x=mix19["date"], y=mix19["indonesia_share_pct"],
            name="Indonesia piped", stackgroup="one",
            fillcolor="rgba(240,180,41,0.7)", line=dict(color="#f0b429", width=0),
        ))
        fig3.add_trace(go.Scatter(
            x=mix19["date"], y=mix19["lng_share_pct"],
            name="LNG", stackgroup="one",
            fillcolor="rgba(231,76,60,0.7)", line=dict(color="#e74c3c", width=0),
        ))
        fig3.add_annotation(
            x="2026-01-01", y=95,
            text="Full LNG dependency forecast from 2029<br>(S&P Global)",
            showarrow=False, font=dict(size=10, color="#aaa"),
            align="left",
        )
        fig3.update_layout(
            title="Gas Supply Mix Evolution (% volume share)",
            xaxis=dict(title="Month", type="date"),
            yaxis=dict(title="Share (%)", range=[0, 100]),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Chart 4: Lag Correlation Bar ──────────────────────────────────────────
    lags = corr_result.get("lags", [])
    if lags:
        best_lag = corr_result.get("best_lag", 0)
        fig4 = go.Figure()
        cl_base = {k: v for k, v in cl.items() if k not in ("xaxis", "yaxis")}
        fig4.update_layout(**cl_base)
        fig4.add_trace(go.Bar(
            x=[f"Lag {l['lag_months']}m" for l in lags],
            y=[l["pearson_r"] for l in lags],
            marker_color=["#2ecc71" if l["lag_months"] == best_lag else "#009CEA"
                          for l in lags],
            hovertemplate="Lag %{x}<br>r = %{y:.3f}<extra></extra>",
        ))
        fig4.update_layout(
            title="Gas Price → USEP Lag Correlation (monthly, Pearson r)",
            xaxis=dict(title="Lag"),
            yaxis=dict(title="Pearson r", range=[0, 1]),
        )
        st.plotly_chart(fig4, use_container_width=True)

        lag_table = pd.DataFrame(lags).rename(columns={
            "lag_months": "Lag (months)", "pearson_r": "Pearson r",
            "spearman_rho": "Spearman ρ", "r2": "R²", "n_obs": "N obs",
        })
        st.dataframe(lag_table, use_container_width=True, hide_index=True)

    # ── S&P Global Reference Panel ────────────────────────────────────────────
    with st.expander("📋 Long-Term Outlook — S&P Global Energy, May 2026"):
        st.caption("For internal reference only. Not for distribution.")
        ref_data = {
            "Year": ["2026", "2027", "2028", "2030", "2035"],
            "Gas Forecast (USD/MMBtu)": [
                "~$14 (Q3 peak, Jan–Apr actuals $8–11)",
                "~$11 (declining LNG spot)",
                "~$11–12",
                "~$12 + aging plant retirements",
                "~$13 (structural tightness)",
            ],
            "USEP Forecast (S$/MWh)": [
                "~$243 (Jul peak); avg ~$155",
                "~$126 avg",
                "~$130 avg",
                "Upward pressure from retirements",
                "+65% vs 2025 avg",
            ],
            "Key Risk": [
                "LNG spot price spike (Q3 peak demand)",
                "Indonesia supply decline accelerating",
                "CCGT retirement schedule uncertainty",
                "Cross-border imports (Singa COD 2029)",
                "Grid decarbonisation + storage uptake",
            ],
        }
        st.table(pd.DataFrame(ref_data))
        st.caption(
            "Source: S&P Global Energy, *Singapore Power Price Outlook*, May 2026. "
            "Reproduced for internal research purposes only. © 2026 S&P Global Inc."
        )

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
                cl = get_chart_layout()
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

            cl = get_chart_layout()
            rs_style = get_rangeselector_style()
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
