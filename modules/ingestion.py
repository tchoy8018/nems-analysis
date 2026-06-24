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


def detect_file_type(df: pd.DataFrame) -> str:
    """
    Infer file type from column names.
    Returns: 'nems_prices' | 'gas_prices' | 'analyst_forecast' | 'unknown'
    """
    cols = {c.lower().strip() for c in df.columns}
    if any(c in cols for c in ("usep ($/mwh)", "usep", "lcp ($/mwh)")):
        return "nems_prices"
    if any(c in cols for c in ("jkm", "jkm_usd_mmbtu", "piped_gas", "piped_gas_sgd_mmbtu")):
        return "gas_prices"
    if any(c in cols for c in ("forecast_price", "price", "usep_forecast", "annual_avg", "monthly_avg")):
        return "analyst_forecast"
    return "unknown"


def ingest_gas_prices(file_path, engine) -> dict:
    """
    Import gas price data (JKM + optional piped gas) from CSV or Excel.

    Expected columns (flexible):
        date / price_date  — date
        jkm / jkm_usd_mmbtu
        piped_gas / piped_gas_sgd_mmbtu  (optional)
        source  (optional)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {"rows_imported": 0, "rows_skipped": 0, "errors": [f"File not found: {file_path}"]}

    try:
        if file_path.suffix.lower() in (".xlsx", ".xls"):
            raw = pd.read_excel(file_path, engine="openpyxl")
        else:
            raw = pd.read_csv(file_path)
    except Exception as e:
        return {"rows_imported": 0, "rows_skipped": 0, "errors": [str(e)]}

    col_map = {}
    for c in raw.columns:
        lc = c.lower().strip()
        if lc in ("date", "price_date"):
            col_map[c] = "price_date"
        elif lc in ("jkm", "jkm_usd_mmbtu"):
            col_map[c] = "jkm_usd_mmbtu"
        elif lc in ("piped_gas", "piped_gas_sgd_mmbtu"):
            col_map[c] = "piped_gas_sgd_mmbtu"
        elif lc == "source":
            col_map[c] = "source"
    df = raw.rename(columns=col_map)

    if "price_date" not in df.columns:
        return {"rows_imported": 0, "rows_skipped": 0, "errors": ["No date column found"]}

    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce").dt.date
    df = df.dropna(subset=["price_date"])
    if "jkm_usd_mmbtu" in df.columns:
        df["jkm_usd_mmbtu"] = pd.to_numeric(df["jkm_usd_mmbtu"], errors="coerce")
    else:
        df["jkm_usd_mmbtu"] = float("nan")
    if "piped_gas_sgd_mmbtu" not in df.columns:
        df["piped_gas_sgd_mmbtu"] = float("nan")
    if "source" not in df.columns:
        df["source"] = file_path.stem
    df["imported_at"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    dialect = _detect_dialect(engine)
    inserted = 0
    skipped = 0
    errors = []

    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
                if dialect == "sqlite":
                    sql = text("""
                        INSERT OR IGNORE INTO gas_prices
                        (price_date, jkm_usd_mmbtu, piped_gas_sgd_mmbtu, source, imported_at)
                        VALUES (:price_date, :jkm_usd_mmbtu, :piped_gas_sgd_mmbtu, :source, :imported_at)
                    """)
                else:
                    sql = text("""
                        INSERT INTO gas_prices
                        (price_date, jkm_usd_mmbtu, piped_gas_sgd_mmbtu, source, imported_at)
                        VALUES (:price_date, :jkm_usd_mmbtu, :piped_gas_sgd_mmbtu, :source, :imported_at)
                        ON CONFLICT (price_date) DO NOTHING
                    """)
                result = conn.execute(sql, {
                    "price_date": str(row["price_date"]),
                    "jkm_usd_mmbtu": None if pd.isna(row["jkm_usd_mmbtu"]) else float(row["jkm_usd_mmbtu"]),
                    "piped_gas_sgd_mmbtu": None if pd.isna(row["piped_gas_sgd_mmbtu"]) else float(row["piped_gas_sgd_mmbtu"]),
                    "source": row["source"],
                    "imported_at": row["imported_at"],
                })
                if result.rowcount == 1:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                errors.append(str(e))

    return {"rows_imported": inserted, "rows_skipped": skipped, "errors": errors}


def ingest_analyst_forecast(
    file_path,
    engine,
    source_name: str,
    vintage_year: int | None = None,
    granularity: str = "annual",
) -> dict:
    """
    Import external analyst price forecast into forecast_sources + forecast_data.

    granularity options: 'annual' | 'monthly' | 'daily' | 'half_hourly'
    For 'annual': single value replicated to all periods for the year.
    For 'monthly': monthly avg replicated to all periods in each month.
    For 'daily' / 'half_hourly': inserted as-is.

    Expected columns (flexible):
        For annual:      year, price
        For monthly:     year, month, price  OR  date, price
        For daily/hh:    date, period (hh only), price
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {"rows_imported": 0, "rows_skipped": 0, "errors": [f"File not found: {file_path}"]}

    try:
        if file_path.suffix.lower() in (".xlsx", ".xls"):
            raw = pd.read_excel(file_path, engine="openpyxl")
        else:
            raw = pd.read_csv(file_path)
    except Exception as e:
        return {"rows_imported": 0, "rows_skipped": 0, "errors": [str(e)]}

    # Normalise column names
    col_map = {}
    for c in raw.columns:
        lc = c.lower().strip()
        if lc in ("price", "usep", "forecast_price", "usep_forecast", "annual_avg", "monthly_avg"):
            col_map[c] = "price"
        elif lc in ("year",):
            col_map[c] = "year"
        elif lc in ("month",):
            col_map[c] = "month"
        elif lc in ("date", "price_date"):
            col_map[c] = "date"
        elif lc == "period":
            col_map[c] = "period"
    df = raw.rename(columns=col_map)
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df = df.dropna(subset=["price"])

    # Expand to per-row records ready for forecast_data
    records: list[dict] = []

    if granularity == "annual":
        for _, row in df.iterrows():
            yr = int(row.get("year", vintage_year or 2025))
            for month in range(1, 13):
                import calendar
                days_in_month = calendar.monthrange(yr, month)[1]
                for day in range(1, days_in_month + 1):
                    dt = f"{yr:04d}-{month:02d}-{day:02d}"
                    records.append({"date": dt, "period": None, "price": float(row["price"])})

    elif granularity == "monthly":
        for _, row in df.iterrows():
            if "date" in df.columns and "date" in row:
                dt = pd.to_datetime(row["date"], errors="coerce")
                yr, mo = dt.year, dt.month
            else:
                yr = int(row.get("year", vintage_year or 2025))
                mo = int(row.get("month", 1))
            import calendar
            days_in_month = calendar.monthrange(yr, mo)[1]
            for day in range(1, days_in_month + 1):
                dt_str = f"{yr:04d}-{mo:02d}-{day:02d}"
                records.append({"date": dt_str, "period": None, "price": float(row["price"])})

    else:  # daily or half_hourly
        for _, row in df.iterrows():
            dt = pd.to_datetime(row.get("date", ""), errors="coerce")
            if pd.isna(dt):
                continue
            period = int(row["period"]) if "period" in df.columns and pd.notna(row.get("period")) else None
            records.append({"date": dt.date().isoformat(), "period": period, "price": float(row["price"])})

    if not records:
        return {"rows_imported": 0, "rows_skipped": 0, "errors": ["No valid rows after expansion"]}

    now_str = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    dialect = _detect_dialect(engine)

    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO forecast_sources (source_name, vintage_year, granularity, forecast_type, uploaded_at, row_count)
            VALUES (:name, :vy, :gran, :ftype, :now, :rc)
        """), {
            "name": source_name,
            "vy": vintage_year,
            "gran": granularity,
            "ftype": granularity,
            "now": now_str,
            "rc": len(records),
        })
        source_id = result.lastrowid

    inserted = 0
    skipped = 0
    errors: list[str] = []

    with engine.begin() as conn:
        for rec in records:
            try:
                sql = text("""
                    INSERT INTO forecast_data (source_id, date, period, price)
                    VALUES (:source_id, :date, :period, :price)
                """)
                conn.execute(sql, {"source_id": source_id, "date": rec["date"],
                                   "period": rec["period"], "price": rec["price"]})
                inserted += 1
            except Exception as e:
                skipped += 1
                if len(errors) < 5:
                    errors.append(str(e))

    return {"rows_imported": inserted, "rows_skipped": skipped, "errors": errors, "source_id": source_id}


def ingest_and_retrain(file_path, engine, model_registry_table=None) -> dict:
    """
    Auto-learning pipeline:
      1. Detect file type
      2. Ingest to appropriate table
      3. Check model drift vs last 30 days of forecast_actuals
      4. Retrain XGBoost if drift > 1.5× baseline RMSE
      5. Return summary dict

    Returns: {
        file_type, rows_imported, rows_skipped,
        drift_detected (bool), retrained (bool),
        retrain_metrics (dict or None), errors
    }
    """
    file_path = Path(file_path)
    errors: list[str] = []

    # Step 1: Read & detect
    try:
        if file_path.suffix.lower() in (".xlsx", ".xls"):
            raw = pd.read_excel(file_path, engine="openpyxl")
        else:
            raw = pd.read_csv(file_path)
    except Exception as e:
        return {"file_type": "unknown", "rows_imported": 0, "rows_skipped": 0,
                "drift_detected": False, "retrained": False,
                "retrain_metrics": None, "errors": [str(e)]}

    file_type = detect_file_type(raw)

    # Step 2: Ingest
    if file_type == "nems_prices":
        result = import_csv_to_db(file_path, engine)
    elif file_type == "gas_prices":
        result = ingest_gas_prices(file_path, engine)
    else:
        result = {"rows_imported": 0, "rows_skipped": 0,
                  "errors": [f"Auto-ingest not supported for type: {file_type}"]}
        return {"file_type": file_type, **result,
                "drift_detected": False, "retrained": False, "retrain_metrics": None}

    rows_imported = result.get("rows_imported", 0)
    errors.extend(result.get("errors", []))

    # Steps 3–4: Only attempt for NEMS price files (model relevance)
    drift_detected = False
    retrained = False
    retrain_metrics = None

    if file_type == "nems_prices" and rows_imported > 0:
        try:
            from modules.forecasting import check_model_drift, train_xgboost
            drift_info = check_model_drift(engine, "xgboost_v1", window_days=30)
            drift_detected = drift_info.get("drift_detected", False)

            if drift_detected:
                metrics = train_xgboost(engine)
                retrained = True
                retrain_metrics = metrics
        except Exception as e:
            errors.append(f"Drift/retrain check failed: {e}")

    return {
        "file_type": file_type,
        "rows_imported": rows_imported,
        "rows_skipped": result.get("rows_skipped", 0),
        "drift_detected": drift_detected,
        "retrained": retrained,
        "retrain_metrics": retrain_metrics,
        "errors": errors,
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
