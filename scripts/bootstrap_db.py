"""
Bootstrap script: create tables and import the master Excel file.
Run from the repo root:
    python scripts/bootstrap_db.py [path/to/usep_from2019_toJun2026.xlsx]
"""
import sys
from pathlib import Path

# Ensure repo root is on the path so db / modules are importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from db import get_engine, setup_database
from modules.ingestion import import_excel_to_db, get_database_summary

DEFAULT_EXCEL = ROOT / "data" / "usep_from2019_toJun2026.xlsx"


def main():
    excel_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXCEL

    print("=" * 60)
    print("NEMS Analytics — Database Bootstrap")
    print("=" * 60)

    engine = get_engine()
    print(f"Database URL: {engine.url}")

    print("\nCreating tables ...")
    setup_database(engine)
    print("  Tables created (or already exist).")

    print(f"\nImporting Excel: {excel_path}")
    result = import_excel_to_db(excel_path, engine)

    print("\n--- Import Result ---")
    print(f"  Rows imported : {result['rows_imported']:,}")
    print(f"  Rows skipped  : {result['rows_skipped']:,}")
    if result["date_range"]:
        print(f"  Date range    : {result['date_range']['min']} → {result['date_range']['max']}")
    if result["errors"]:
        print(f"  Errors ({len(result['errors'])}):")
        for e in result["errors"]:
            print(f"    - {e}")

    print("\n--- Database Summary ---")
    summary = get_database_summary(engine)
    print(f"  Total rows    : {summary['total_rows']:,}")
    print(f"  Date range    : {summary['min_date']} → {summary['max_date']}")
    print(f"  Coverage      : {summary['coverage_pct']}%")
    print(f"  Last imported : {summary['last_import_at']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
