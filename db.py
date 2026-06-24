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
    Return a SQLAlchemy engine. Safe to call from standalone scripts and
    from within Streamlit. Resolution order:
      1. st.secrets["DATABASE_URL"]
      2. os.environ["DATABASE_URL"]
      3. SQLite at data/nems_master.db (default)

    Do NOT decorate this with @st.cache_resource — it breaks outside Streamlit.
    Wrap it in app.py with a @st.cache_resource function if you need caching there.
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
    "nems_prices",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source_file", Text),
    Column("date", Date, nullable=False),
    Column("period", Integer, nullable=False),
    Column("usep", Float),
    Column("lcp", Float),
    Column("demand_mw", Float),
    Column("solar_mw", Float),
    Column("tcl_mw", Float),
    Column("rusep", Float),
    Column("map_price", Float),
    Column("mapt_price", Float),
    Column("tpc_applied", Text),
    Column("imported_at", DateTime),
    UniqueConstraint("date", "period", name="uq_nems_date_period"),
)

forecast_sources = Table(
    "forecast_sources",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source_name", Text, nullable=False),
    Column("color_hex", Text, default="#f0b429"),
    Column("uploaded_at", DateTime),
    Column("row_count", Integer),
)

forecast_data = Table(
    "forecast_data",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source_id", Integer),
    Column("date", Date, nullable=False),
    Column("period", Integer),
    Column("price", Float, nullable=False),
    Column("unit", Text, default="$/MWh"),
)

model_registry = Table(
    "model_registry",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("model_name", Text, nullable=False),
    Column("trained_at", DateTime),
    Column("training_rows", Integer),
    Column("rmse", Float),
    Column("mae", Float),
    Column("mape", Float),
    Column("model_path", Text),
    Column("is_active", Boolean, default=True),
)

Index("idx_nems_date", nems_prices.c.date)
Index("idx_nems_period", nems_prices.c.period)
Index("idx_nems_date_period", nems_prices.c.date, nems_prices.c.period)
Index("idx_forecast_source", forecast_data.c.source_id)
Index("idx_forecast_date", forecast_data.c.date)


def setup_database(engine):
    """Create all tables if they do not exist. Safe to call on every startup."""
    metadata.create_all(engine)
