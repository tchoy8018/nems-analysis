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
    Column("id",                   Integer, primary_key=True, autoincrement=True),
    Column("price_date",           Date,    nullable=False),
    Column("jkm_usd_mmbtu",        Float),
    Column("piped_gas_sgd_mmbtu",  Float),
    Column("source",               Text),
    Column("imported_at",          DateTime),
    UniqueConstraint("price_date", name="uq_gas_price_date"),
)

# Indexes
Index("idx_nems_date",        nems_prices.c.date)
Index("idx_nems_period",      nems_prices.c.period)
Index("idx_nems_date_period", nems_prices.c.date, nems_prices.c.period)
Index("idx_forecast_source",  forecast_data.c.source_id)
Index("idx_forecast_date",    forecast_data.c.date)
Index("idx_fa_model_date",    forecast_actuals.c.model_name, forecast_actuals.c.forecast_date)
Index("idx_gas_date",         gas_prices.c.price_date)


def _add_column_safe(conn, table: str, column: str, definition: str) -> None:
    """Add column to existing table; silently skip if already exists."""
    try:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
    except Exception:
        pass


def setup_database(engine) -> None:
    """Create all tables; migrate existing ones with new columns."""
    metadata.create_all(engine)

    # Migrate forecast_sources with new Phase 4 columns
    with engine.begin() as conn:
        _add_column_safe(conn, "forecast_sources", "vintage_year",  "INTEGER")
        _add_column_safe(conn, "forecast_sources", "forecast_type", "TEXT")
        _add_column_safe(conn, "forecast_sources", "granularity",   "TEXT")
