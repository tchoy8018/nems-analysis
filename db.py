from pathlib import Path
import os

from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, Float, String, Date, DateTime, Boolean, Text,
    UniqueConstraint, Index, text
)

_DEFAULT_DB_URL = f"sqlite:///{Path(__file__).parent / 'data' / 'nems_master.db'}"


def get_engine():
    """
    Return a SQLAlchemy engine. Resolution order:
      1. st.secrets["DATABASE_URL"]
      2. os.environ["DATABASE_URL"]
      3. SQLite at data/nems_master.db
    """
    url = None
    try:
        import streamlit as st
        url = st.secrets.get("DATABASE_URL")
    except Exception:
        pass
    if not url:
        url = os.environ.get("DATABASE_URL")
    if not url:
        url = _DEFAULT_DB_URL
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


metadata = MetaData()

nems_prices = Table(
    "nems_prices", metadata,
    Column("id",          Integer, primary_key=True, autoincrement=True),
    Column("source_file", Text),
    Column("date",        Date,    nullable=False),
    Column("period",      Integer, nullable=False),
    Column("usep",        Float),
    Column("lcp",         Float),
    Column("demand_mw",   Float),
    Column("solar_mw",    Float),
    Column("tcl_mw",      Float),
    Column("rusep",       Float),
    Column("map_price",   Float),
    Column("mapt_price",  Float),
    Column("tpc_applied", Text),
    Column("imported_at", DateTime),
    UniqueConstraint("date", "period", name="uq_nems_date_period"),
)

# Extended forecast_sources (new columns added via ALTER TABLE in setup_database)
forecast_sources = Table(
    "forecast_sources", metadata,
    Column("id",           Integer,  primary_key=True, autoincrement=True),
    Column("source_name",  Text,     nullable=False),
    Column("color_hex",    Text,     default="#f0b429"),
    Column("uploaded_at",  DateTime),
    Column("row_count",    Integer),
    Column("vintage_year", Integer),
    Column("forecast_type", Text),   # 'annual_avg','monthly_avg','hourly'
    Column("granularity",  Text),    # 'annual','monthly','daily','half_hourly'
)

# Extended forecast_data
forecast_data = Table(
    "forecast_data", metadata,
    Column("id",        Integer, primary_key=True, autoincrement=True),
    Column("source_id", Integer),
    Column("date",      Date,    nullable=False),
    Column("period",    Integer),
    Column("price",     Float,   nullable=False),
    Column("unit",      Text,    default="$/MWh"),
)

model_registry = Table(
    "model_registry", metadata,
    Column("id",            Integer,  primary_key=True, autoincrement=True),
    Column("model_name",    Text,     nullable=False),
    Column("trained_at",    DateTime),
    Column("training_rows", Integer),
    Column("rmse",          Float),
    Column("mae",           Float),
    Column("mape",          Float),
    Column("model_path",    Text),
    Column("is_active",     Boolean,  default=True),
)

# NEW — Phase 4
forecast_actuals = Table(
    "forecast_actuals", metadata,
    Column("id",                       Integer, primary_key=True, autoincrement=True),
    Column("model_name",               Text,    nullable=False),
    Column("forecast_date",            Date,    nullable=False),
    Column("period",                   Integer, nullable=False),
    Column("predicted_usep",           Float),
    Column("actual_usep",              Float),
    Column("error",                    Float),      # actual - predicted
    Column("abs_error",                Float),
    Column("forecast_horizon_periods", Integer),    # how far ahead was this forecast?
    Column("created_at",               DateTime),
    UniqueConstraint(
        "model_name", "forecast_date", "period", "forecast_horizon_periods",
        name="uq_forecast_actuals",
    ),
)

gas_prices = Table(
    "gas_prices", metadata,
    Column("id",                        Integer, primary_key=True, autoincrement=True),
    Column("price_date",                Date,    nullable=False),
    # Volumes (metric tons)
    Column("malaysia_vol_mt",           Float),
    Column("indonesia_vol_mt",          Float),
    Column("lng_vol_mt",                Float),
    Column("total_vol_mt",              Float),
    # Values (USD)
    Column("malaysia_val_usd",          Float),
    Column("indonesia_val_usd",         Float),
    Column("lng_val_usd",               Float),
    # Prices from customs data (USD/MMBtu)
    Column("malaysia_price_usd_mmbtu",  Float),
    Column("indonesia_price_usd_mmbtu", Float),
    Column("lng_price_usd_mmbtu",       Float),
    # Volume-weighted average (LNG: 1 MT = 52 MMBtu; piped: 1 MT = 50 MMBtu)
    Column("weighted_avg_usd_mmbtu",    Float),
    # Source share (%)
    Column("malaysia_share_pct",        Float),
    Column("indonesia_share_pct",       Float),
    Column("lng_share_pct",             Float),
    # SGD conversion and CCGT cost floor
    Column("fx_rate_usd_sgd",           Float, default=1.35),
    Column("weighted_avg_sgd_mmbtu",    Float),
    Column("implied_usep_floor_sgd_mwh", Float),  # weighted_sgd × 7.5 MMBtu/MWh
    Column("source",                    Text, default="SG Customs"),
    Column("imported_at",               DateTime),
    UniqueConstraint("price_date", name="uq_gas_price_date"),
)

# NEW — Phase 5 persistent learning
backtest_runs = Table(
    "backtest_runs", metadata,
    Column("id",          Integer,  primary_key=True, autoincrement=True),
    Column("run_at",      DateTime, nullable=False),
    Column("model_name",  Text,     nullable=False),
    Column("test_days",   Integer),
    Column("rmse",        Float),
    Column("mae",         Float),
    Column("mape",        Float),
    Column("n_periods",   Integer),
)

prediction_log = Table(
    "prediction_log", metadata,
    Column("id",            Integer,  primary_key=True, autoincrement=True),
    Column("model_name",    Text,     nullable=False),
    Column("predicted_at",  DateTime, nullable=False),
    Column("forecast_date", Date,     nullable=False),
    Column("period",        Integer,  nullable=False),
    Column("predicted_usep", Float),
    Column("spike_prob",    Float),
    UniqueConstraint("model_name", "forecast_date", "period", name="uq_prediction_log"),
)

model_evolution = Table(
    "model_evolution", metadata,
    Column("id",            Integer,  primary_key=True, autoincrement=True),
    Column("model_name",    Text,     nullable=False),
    Column("trained_at",    DateTime, nullable=False),
    Column("rmse",          Float),
    Column("mae",           Float),
    Column("mape",          Float),
    Column("training_rows", Integer),
    Column("notes",         Text),
)

# NEW — Phase 5 live data integration
live_data_log = Table(
    "live_data_log", metadata,
    Column("id",              Integer,  primary_key=True, autoincrement=True),
    Column("fetched_at",      DateTime),
    Column("source",          Text),        # 'live_api' | 'playwright' | 'monthly_csv'
    Column("periods_fetched", Integer, default=0),
    Column("periods_new",     Integer, default=0),
    Column("latest_date",     Date),
    Column("latest_period",   Integer),
    Column("latest_usep",     Float),
    Column("latest_demand_mw", Float),
    Column("error",           Text),
    Column("duration_ms",     Integer),
)

demand_analysis_cache = Table(
    "demand_analysis_cache", metadata,
    Column("id",                       Integer, primary_key=True, autoincrement=True),
    Column("computed_at",              DateTime),
    Column("inflection_mw",            Float),
    Column("spearman_r",               Float),
    Column("pct_above_vesting",        Float),
    Column("demand_at_vesting_breach", Float),
    Column("date_from",                Date),
    Column("date_to",                  Date),
    Column("n_periods",                Integer),
)

# Indexes
Index("idx_nems_date",        nems_prices.c.date)
Index("idx_nems_period",      nems_prices.c.period)
Index("idx_nems_date_period", nems_prices.c.date, nems_prices.c.period)
Index("idx_forecast_source",  forecast_data.c.source_id)
Index("idx_forecast_date",    forecast_data.c.date)
Index("idx_fa_model_date",    forecast_actuals.c.model_name, forecast_actuals.c.forecast_date)
Index("idx_gas_date",         gas_prices.c.price_date)
Index("idx_bt_model",         backtest_runs.c.model_name)
Index("idx_plog_model_date",  prediction_log.c.model_name, prediction_log.c.forecast_date)
Index("idx_evo_model",        model_evolution.c.model_name)
Index("idx_ldl_fetched_at",   live_data_log.c.fetched_at)
Index("idx_dac_computed_at",  demand_analysis_cache.c.computed_at)


def _add_column_safe(conn, table: str, column: str, definition: str) -> None:
    """Add column to existing table; silently skip if already exists."""
    try:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
    except Exception:
        pass


def setup_database(engine) -> None:
    """Create all tables; migrate existing ones with new columns."""
    metadata.create_all(engine)

    with engine.begin() as conn:
        # forecast_sources Phase 4 columns
        _add_column_safe(conn, "forecast_sources", "vintage_year",  "INTEGER")
        _add_column_safe(conn, "forecast_sources", "forecast_type", "TEXT")
        _add_column_safe(conn, "forecast_sources", "granularity",   "TEXT")

        # gas_prices Phase 4 full schema migration
        new_gas_cols = [
            ("malaysia_vol_mt",            "REAL"),
            ("indonesia_vol_mt",           "REAL"),
            ("lng_vol_mt",                 "REAL"),
            ("total_vol_mt",               "REAL"),
            ("malaysia_val_usd",           "REAL"),
            ("indonesia_val_usd",          "REAL"),
            ("lng_val_usd",                "REAL"),
            ("malaysia_price_usd_mmbtu",   "REAL"),
            ("indonesia_price_usd_mmbtu",  "REAL"),
            ("lng_price_usd_mmbtu",        "REAL"),
            ("weighted_avg_usd_mmbtu",     "REAL"),
            ("malaysia_share_pct",         "REAL"),
            ("indonesia_share_pct",        "REAL"),
            ("lng_share_pct",              "REAL"),
            ("fx_rate_usd_sgd",            "REAL DEFAULT 1.35"),
            ("weighted_avg_sgd_mmbtu",     "REAL"),
            ("implied_usep_floor_sgd_mwh", "REAL"),
        ]
        for col, defn in new_gas_cols:
            _add_column_safe(conn, "gas_prices", col, defn)

        # Phase 5 — backtest_runs, prediction_log, model_evolution are created
        # via metadata.create_all above; no ALTER TABLE needed for new tables
