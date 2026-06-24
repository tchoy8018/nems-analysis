from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text


EXCEL_COLUMN_MAP = {
    "Source.Name": "source_file",
    "DATE": "date",
    "PERIOD": "period",
    "USEP ($/MWh)": "usep",
    "LCP ($/MWh)": "lcp",
    "DEMAND (MW)": "demand_mw",
    "SOLAR(MW)": "solar_mw",
    "TCL (MW)": "tcl_mw",
    "RUSEP ($/MWh)": "rusep",
    "MAP ($/MWh)": "map_price",
    "MAPT ($/MWh)": "mapt_price",
    "TPC Applied": "tpc_applied",
}

NUMERIC_COLS = ["usep", "lcp", "demand_mw", "solar_mw", "tcl_mw",
                "rusep", "map_price", "mapt_price"]

CHUNKSIZE = 5000


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, coerce types, and drop rows missing date/period."""
    present = {k: v for k, v in EXCEL_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=present)

    required = {"date", "period"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing after rename: {missing}")

    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = df["date"].dt.date
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df = df.dropna(subset=["date", "period"])
    df["period"] = df["period"].astype(int)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")

    if "tpc_applied" not in df.columns:
        df["tpc_applied"] = None
    if "source_file" not in df.columns:
        df["source_file"] = None

    df["imported_at"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    keep = [
        "source_file", "date", "period", "usep", "lcp", "demand_mw",
        "solar_mw", "tcl_mw", "rusep", "map_price", "mapt_price",
        "tpc_applied", "imported_at",
    ]
    return df[[c for c in keep if c in df.columns]]


def _detect_dialect(engine) -> str:
    return engine.dialect.name  # "sqlite" or "postgresql"


def _insert_chunk(chunk: pd.DataFrame, engine) -> tuple[int, int]:
    """Insert a chunk, returning (rows_inserted, rows_skipped)."""
    dialect = _detect_dialect(engine)
    records = chunk.to_dict(orient="records")

    inserted = 0
    skipped = 0

    with engine.begin() as conn:
        for row in records:
            if dialect == "sqlite":
                sql = text("""
                    INSERT OR IGNORE INTO nems_prices
                    (source_file, date, period, usep, lcp, demand_mw, solar_mw,
                     tcl_mw, rusep, map_price, mapt_price, tpc_applied, imported_at)
                    VALUES
                    (:source_file, :date, :period, :usep, :lcp, :demand_mw, :solar_mw,
                     :tcl_mw, :rusep, :map_price, :mapt_price, :tpc_applied, :imported_at)
                """)
            else:
                sql = text("""
                    INSERT INTO nems_prices
                    (source_file, date, period, usep, lcp, demand_mw, solar_mw,
                     tcl_mw, rusep, map_price, mapt_price, tpc_applied, imported_at)
                    VALUES
                    (:source_file, :date, :period, :usep, :lcp, :demand_mw, :solar_mw,
                     :tcl_mw, :rusep, :map_price, :mapt_price, :tpc_applied, :imported_at)
                    ON CONFLICT (date, period) DO NOTHING
                """)

            result = conn.execute(sql, row)
            if result.rowcount == 1:
                inserted += 1
            else:
                skipped += 1

    return inserted, skipped


def import_excel_to_db(excel_path: Path, engine) -> dict:
    """
    Import the NEMS master Excel file into the database.

    Returns a summary dict with keys:
        rows_imported, rows_skipped, date_range (min, max), errors
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        return {
            "rows_imported": 0,
            "rows_skipped": 0,
            "date_range": None,
            "errors": [f"File not found: {excel_path}"],
        }

    errors = []
    total_imported = 0
    total_skipped = 0
    all_dates = []

    print(f"Reading {excel_path.name} ...")
    try:
        raw = pd.read_excel(
            excel_path,
            sheet_name="datasource Market USEP",
            engine="openpyxl",
        )
    except Exception as e:
        return {
            "rows_imported": 0,
            "rows_skipped": 0,
            "date_range": None,
            "errors": [f"Failed to read Excel: {e}"],
        }

    print(f"  Total rows read: {len(raw):,}")

    try:
        df = _clean_dataframe(raw)
    except Exception as e:
        return {
            "rows_imported": 0,
            "rows_skipped": 0,
            "date_range": None,
            "errors": [f"Data cleaning failed: {e}"],
        }

    print(f"  Rows after cleaning: {len(df):,}")

    for chunk_start in range(0, len(df), CHUNKSIZE):
        chunk = df.iloc[chunk_start : chunk_start + CHUNKSIZE]
        chunk_num = chunk_start // CHUNKSIZE + 1
        try:
            ins, skip = _insert_chunk(chunk, engine)
            total_imported += ins
            total_skipped += skip
            all_dates.extend(chunk["date"].tolist())
            print(f"  Chunk {chunk_num}: +{ins} inserted, {skip} skipped")
        except Exception as e:
            errors.append(f"Chunk {chunk_num}: {e}")
            print(f"  Chunk {chunk_num} ERROR: {e}")

    date_range = None
    if all_dates:
        valid_dates = [d for d in all_dates if d is not None]
        if valid_dates:
            date_range = {"min": min(valid_dates), "max": max(valid_dates)}

    return {
        "rows_imported": total_imported,
        "rows_skipped": total_skipped,
        "date_range": date_range,
        "errors": errors,
    }


def import_csv_to_db(csv_path: Path, engine) -> dict:
    """
    Append a single EMC monthly CSV to the database.
    Deduplicates by (date, period).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {
            "rows_imported": 0,
            "rows_skipped": 0,
            "date_range": None,
            "errors": [f"File not found: {csv_path}"],
        }

    errors = []
    try:
        raw = pd.read_csv(csv_path)
        df = _clean_dataframe(raw)
        ins, skip = _insert_chunk(df, engine)
        valid_dates = [d for d in df["date"].tolist() if d is not None]
        date_range = {"min": min(valid_dates), "max": max(valid_dates)} if valid_dates else None
        return {
            "rows_imported": ins,
            "rows_skipped": skip,
            "date_range": date_range,
            "errors": errors,
        }
    except Exception as e:
        return {
            "rows_imported": 0,
            "rows_skipped": 0,
            "date_range": None,
            "errors": [str(e)],
        }


def get_database_summary(engine) -> dict:
    """Return key statistics about the nems_prices table."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                MIN(date)        AS min_date,
                MAX(date)        AS max_date,
                COUNT(*)         AS total_rows,
                MAX(imported_at) AS last_import_at
            FROM nems_prices
        """)).mappings().fetchone()

    if row is None or row["total_rows"] == 0:
        return {
            "min_date": None,
            "max_date": None,
            "total_rows": 0,
            "last_import_at": None,
            "coverage_pct": 0.0,
        }

    min_date = row["min_date"]
    max_date = row["max_date"]
    total_rows = row["total_rows"]
    last_import_at = row["last_import_at"]

    from datetime import date as date_type
    if isinstance(min_date, str):
        min_date = date_type.fromisoformat(min_date)
    if isinstance(max_date, str):
        max_date = date_type.fromisoformat(max_date)

    days_in_range = (max_date - min_date).days + 1
    expected_periods = days_in_range * 48
    coverage_pct = round(100.0 * total_rows / expected_periods, 2) if expected_periods > 0 else 0.0

    return {
        "min_date": min_date,
        "max_date": max_date,
        "total_rows": total_rows,
        "last_import_at": last_import_at,
        "coverage_pct": coverage_pct,
    }
