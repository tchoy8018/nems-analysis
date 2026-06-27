# NEMS Analytics — Singa Renewables Intelligence Platform
# Project memory file — auto-updated on every git commit

## Project Identity
- Asset: 560 MW solar + BESS, TotalEnergies + RGE (Royal Golden Eagle)
- Location: Indonesia (Riau Islands) → Singapore via subsea HVDC cable
- COD target: 2029
- Regulatory framework: EMA RFP 2 (cross-border electricity import)
- Revenue streams: CfD energy + wholesale market upside + ancillary services

## Database
- Engine: SQLite (local dev) → PostgreSQL/Supabase (production)
- File: data/nems_master.db
- Rows: 131,060 | Range: 2019-01-01 → 2026-06-16 | Coverage: 99.96%
- Schema: nems_prices, forecast_sources, forecast_data, model_registry,
          forecast_actuals (Phase 4), gas_prices (Phase 4 — full schema)

## Gas Price Data
- Source: S&P Global Energy / Singapore Customs Gas Trade Data Tables, June 2026
- File: data/gas/sg_gas_trade_data_2026.xlsx (gitignored)
- 184 rows: 2011-01-01 → 2026-04-01 (monthly)
- Columns: malaysia/indonesia/lng volumes, values, prices + weighted avg + implied CCGT floor
- Conversion: LNG 1 MT = 52 MMBtu; piped gas 1 MT = 50 MMBtu; heat rate = 7.5 MMBtu/MWh
- FX default: 1.35 USD/SGD (fx_rate_usd_sgd column, overrideable)
- Apr 2026: weighted avg $11.24 USD/MMBtu → implied floor S$113.8/MWh; LNG share 48%
- Gas→USEP correlation: Pearson r=0.659 (lag 0m), Spearman=0.826; R²=0.435
- Pass-through slope: 18.8 SGD/MWh per SGD/MMBtu (higher than heat rate — market power premium)
- Regime: correlation stronger pre-2022 (r=0.75) when piped gas dominated; LNG spot volatility post-2022 weakens link

## Phase Status
- [x] Phase 1: Scaffold, DB ingestion, bootstrap, Streamlit app (dark theme)
- [x] Phase 2: Analysis module, Duck Curve, Dispatch/Arbitrage, Market Overview
- [x] Phase 3: Forecasting — XGBoost + Prophet + Ensemble, backtest, pages/03_Forecast.py
- [x] Phase 4: Market Intelligence Hub
  - [x] Module 1: forecast_actuals table, check_model_drift, save_predictions_to_db, ingest_and_retrain
  - [x] Module 2: gas_prices table (full schema), ingest_gas_prices (S&P Global/Customs workbook parser),
              gas_usep_correlation (monthly, 4 lags, pass-through regression, rolling 12m Pearson),
              gas_mix_evolution, gas features in build_features (implied_usep_floor, lng_share, lag1m/2m)
  - [x] Module 3: ingest_analyst_forecast (granularity expansion), analyst_vs_actuals, vintage_comparison
  - [x] Module 4: get_sg_calendar_features, day_type_usep_profile
  - [x] Module 5: pages/06_Data_Hub.py (5 sections)
  - [x] Module 6: 3-tab forecast (Short-term / Medium-term / Monthly Scenarios), forecast_monthly_scenarios
- [x] Phase 5: EMC live data integration + demand analysis
  - [x] modules/scraper.py: fetch_live_api() via TableChart?value=10,
        scrape_7day_chart() via Get?value=14 (HTTP only, no Playwright),
        check_and_download_monthly_csv() (Playwright form download),
        detect_gaps_and_fill(), get_last_data_timestamp()
  - [x] DB: live_data_log, demand_analysis_cache tables
  - [x] demand_usep_threshold_analysis(): inflection 6,485 MW, Spearman r=0.508
  - [x] demand_profile_analysis(): by period/day_type/year; CAGR 1.41%
  - [x] build_demand_features(): demand_lag_336, rolling_mean_48,
        is_above_inflection, demand_usep_regime
  - [x] forecasting.py: new demand features in FEATURE_COLS_BASE + build_features()
  - [x] pages/07_Live_Market.py: intraday chart, 7-day heatmap, demand-price
        scatter, KPIs, freshness table, 30-min auto-refresh fragment
  - [x] pages/02_Duck_Curve.py: Tab 2 Demand–Price Analysis
  - [x] app.py: live data sidebar widget, auto-refresh, daily CSV check
- [ ] Phase 6: Streamlit Cloud deployment

## Technical Stack
- Python 3.14 / Streamlit 1.58 / Plotly 6.8 / SQLAlchemy 2.0 / pandas 3.0
- pathlib.Path everywhere (no string path concatenation)
- @st.cache_data(ttl=300) on all DB query functions
- @st.cache_resource on get_engine()
- Light/dark mode toggle via CSS injection + st.session_state["theme"]
- Plotly dark: paper_bgcolor #0d1117, plot_bgcolor #161b22
- Colors: USEP #009CEA | Demand #f0b429 | Solar #2ecc71 | Spike #e74c3c

## Key Technical Decisions
- SQLAlchemy Core only (no raw sqlite3)
- Bootstrap script: scripts/bootstrap_db.py
- Pre-commit hook: .git/hooks/pre-commit (updates CONTEXT.md timestamp)
- Data files gitignored: data/*.db, data/raw/, data/exports/

## Commercial Structure — CRITICAL FOR ALL MODELING

### PPA / CfD Architecture
Singa Renewables targets a PPA with the following structure:
- Delivery obligation: daily or weekly volume (MWh), NOT period-by-period
- CfD settlement price: daily average USEP (or weekly average USEP)
- This gives Singa FULL INTRADAY DISPATCH FLEXIBILITY
- Singa commits to "how much" but NOT "when" within the day/week

### Revenue Stack
1. CONTRACTED LAYER (base revenue, de-risked):
   = Contracted_Volume_MWh × USEP_daily_average
   Settled via CfD — insulated from intraday price volatility

2. WHOLESALE MARKET UPSIDE (incremental revenue):
   = Contracted_Volume_MWh × (USEP_dispatch_weighted - USEP_daily_average)
   Generated by BESS temporal optimization:
   - Charge during low USEP periods (midday solar surplus, periods ~14–28)
   - Discharge during high USEP periods (evening peak, periods ~35–42)
   - The spread between actual dispatch timing and daily average = upside captured

3. ANCILLARY SERVICES (modeled separately, cost item for now):
   RUSEP reserve market — currently not a revenue stream for imported energy

### Battery Arbitrage Page — Correct Framing
The BESS is NOT a standalone arbitrage asset.
It operates within a contracted volume framework.
The "arbitrage" is intraday temporal optimization WITHIN the daily/weekly volume.
Key metric: Intraday spread = avg USEP of discharge periods - daily avg USEP
This spread × contracted volume = incremental annual revenue above CfD floor

### PLF Constraints
- PLF range modeled: 50–75% (base case 65%)
- PLF applies at the injection point (Singapore side of HVDC cable)
- BESS charges primarily from excess solar (midday surplus over contracted delivery)
- Available BESS dispatch window = periods where solar > contracted delivery profile

## EMC API (confirmed June 2026)
- Real-time today: GET /api/DataSync/TableChart?value=10&fromDate=DD-Mon-YYYY&toDate=DD-Mon-YYYY&tpcValue=0
  → data.data[0].datasets[].columns: [date_str, time_range, demand_mw, solar_mw, tcl_mw, usep, eheur, lcp]
  → tags: "past" | "current" | "future" (future = DA forecast, do not ingest)
- 7-day history: GET /api/DataSync/Get?value=14
  → data.data[0]: {labels: ["DD Mon HH:MM-HH:MM" ×336], datasets: [USEP, Demand, Solar, VCP]}
- Monthly CSV: form POST via Playwright (session token required)
  → endpoint hint: /api/sitecore/DataSync/DataDownload

## Demand Analysis Results (full dataset 2019–2026)
- Spearman r (demand vs USEP) = 0.508
- Inflection point: 6,485 MW (piecewise linear curve_fit)
- Vesting price S$170/MWh exceeded in 26.1% of periods
- Demand at vesting breach (median): 6,765 MW
- Demand CAGR 2019–2026: 1.41% (data center-driven)
- Peak period: P39 (19:00–19:30 SGT)

## Last Updated
2026-06-27 (Phase 5: EMC live data + demand analysis)
