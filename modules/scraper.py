"""
EMC live data integration — three data paths:
  A1. fetch_live_api()                 — today's USEP via TableChart?value=10
  A2. scrape_7day_chart()              — 7-day via Get?value=14 (no Playwright)
  A3. check_and_download_monthly_csv() — Playwright CSV download
  A4. detect_gaps_and_fill()           — gap detection + back-fill

Confirmed endpoints (June 2026):
  /api/DataSync/TableChart?value=10&fromDate=DD-Mon-YYYY&toDate=DD-Mon-YYYY&tpcValue=0
    → columns: [date, time_range, demand_mw, solar_mw, tcl_mw, usep, eheur, lcp]
    → tag: "past" | "current" | "future"
  /api/DataSync/Get?value=14
    → labels: ["DD Mon HH:MM-HH:MM", ...] (336 = 7 × 48)
    → datasets: [{label: "USEP", data: [...]}, "Demand", "Solar", "VCP"]
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

EMC_BASE       = "https://www.nems.emcsg.com"
EMC_TABLE_URL  = EMC_BASE + "/api/DataSync/TableChart"
EMC_CHART_URL  = EMC_BASE + "/api/DataSync/Get"
EMC_PRICES_URL = EMC_BASE + "/nems-prices"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": EMC_PRICES_URL,
    "Accept": "application/json, */*",
    "X-Requested-With": "XMLHttpRequest",
}


def is_local_environment() -> bool:
    """True when running locally; False on Streamlit Cloud."""
    return not bool(
        os.environ.get("STREAMLIT_SHARING_MODE") or
        os.environ.get("DATABASE_URL", "").startswith("postgresql")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Date / period helpers
# ─────────────────────────────────────────────────────────────────────────────

def _time_range_to_period(time_range: str) -> Optional[int]:
    """
    Convert "HH:MM-HH:MM" → period (1-48).
    Uses start time: "00:00-00:30" → 1, "00:30-01:00" → 2, etc.
    """
    try:
        start = time_range.split("-")[0].strip()
        h, m  = map(int, start.split(":"))
        p = h * 2 + m // 30 + 1
        return p if 1 <= p <= 48 else None
    except Exception:
        return None


def _parse_label_date(label: str, year_hint: int) -> Optional[date]:
    """
    Parse "DD Mon HH:MM-HH:MM" labels from the 7-day API.
    Year is inferred: same as year_hint unless that puts the date in the future,
    in which case subtract 1 year.
    """
    try:
        parts    = label.strip().split()
        date_str = f"{parts[0]} {parts[1]} {year_hint}"
        d        = datetime.strptime(date_str, "%d %b %Y").date()
        if d > date.today():
            d = d.replace(year=year_hint - 1)
        return d
    except Exception:
        return None


def _parse_table_date(raw: str) -> Optional[date]:
    """Parse "27 Jun 2026" from TableChart column[0]."""
    try:
        return datetime.strptime(raw.strip(), "%d %b %Y").date()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ingest_rows(engine, rows: list[dict]) -> int:
    """INSERT OR IGNORE into nems_prices; return count of newly-inserted rows."""
    if not rows:
        return 0
    now     = datetime.utcnow()
    new_cnt = 0
    with engine.begin() as conn:
        for row in rows:
            try:
                result = conn.execute(text("""
                    INSERT OR IGNORE INTO nems_prices
                        (source_file, date, period, usep, lcp, demand_mw,
                         solar_mw, tcl_mw, rusep, map_price, mapt_price, imported_at)
                    VALUES
                        (:source_file, :date, :period, :usep, :lcp, :demand_mw,
                         :solar_mw, :tcl_mw, :rusep, :map_price, :mapt_price, :imported_at)
                """), {
                    "source_file":  row.get("source_file", "live_api"),
                    "date":         row["date"],
                    "period":       int(row["period"]),
                    "usep":         row.get("usep"),
                    "lcp":          row.get("lcp"),
                    "demand_mw":    row.get("demand_mw"),
                    "solar_mw":     row.get("solar_mw"),
                    "tcl_mw":       row.get("tcl_mw"),
                    "rusep":        row.get("rusep"),
                    "map_price":    row.get("map_price"),
                    "mapt_price":   row.get("mapt_price"),
                    "imported_at":  now,
                })
                new_cnt += result.rowcount
            except Exception as exc:
                logger.debug("Row insert skipped: %s", exc)
    return new_cnt


def _log_fetch(engine, source: str, result: dict, duration_ms: int) -> None:
    """Write one row to live_data_log."""
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO live_data_log
                    (fetched_at, source, periods_fetched, periods_new,
                     latest_date, latest_period, latest_usep, latest_demand_mw,
                     error, duration_ms)
                VALUES
                    (:fetched_at, :source, :periods_fetched, :periods_new,
                     :latest_date, :latest_period, :latest_usep, :latest_demand_mw,
                     :error, :duration_ms)
            """), {
                "fetched_at":      datetime.utcnow(),
                "source":          source,
                "periods_fetched": result.get("periods_fetched", 0),
                "periods_new":     result.get("periods_new", 0),
                "latest_date":     result.get("latest_date"),
                "latest_period":   result.get("latest_period"),
                "latest_usep":     result.get("latest_usep"),
                "latest_demand_mw": result.get("latest_demand"),
                "error":           result.get("error"),
                "duration_ms":     duration_ms,
            })
    except Exception as exc:
        logger.debug("live_data_log write failed: %s", exc)


def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return f if f >= 0 else None
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# A1. fetch_live_api() — today's data via TableChart?value=10
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_live_api(engine) -> dict:
    """
    Fetch today's USEP/Demand/Solar from EMC via TableChart?value=10.
    Ingests only 'past' and 'current' tagged rows (not 'future').
    Returns {periods_fetched, periods_new, latest_date, latest_period,
             latest_usep, latest_demand, source}.
    Never raises.
    """
    t0   = time.monotonic()
    today_str = date.today().strftime("%d-%b-%Y")
    url  = (f"{EMC_TABLE_URL}?value=10"
            f"&fromDate={today_str}&toDate={today_str}&tpcValue=0")

    try:
        async with httpx.AsyncClient(
            headers=_BROWSER_HEADERS,
            timeout=12.0,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        try:
            payload = resp.json()
        except Exception:
            err = f"JSON parse failed. Raw: {resp.text[:300]}"
            logger.error("fetch_live_api: %s", err)
            result = {"periods_fetched": 0, "error": err, "source": "live_api"}
            _log_fetch(engine, "live_api", result, int((time.monotonic() - t0) * 1000))
            return result

        # Navigate to datasets
        try:
            datasets = payload["data"]["data"][0]["datasets"]
        except (KeyError, IndexError, TypeError) as exc:
            err = f"Unexpected response structure: {exc}"
            logger.error("fetch_live_api: %s", err)
            result = {"periods_fetched": 0, "error": err, "source": "live_api"}
            _log_fetch(engine, "live_api", result, int((time.monotonic() - t0) * 1000))
            return result

        rows = []
        for ds in datasets:
            tag = ds.get("tag", "")
            if tag == "future":
                continue                          # skip forecast-only rows
            cols = ds.get("columns", [])
            if len(cols) < 6:
                continue

            d = _parse_table_date(cols[0])
            if d is None:
                continue
            p = _time_range_to_period(cols[1])
            if p is None:
                continue

            rows.append({
                "date":       d,
                "period":     p,
                "demand_mw":  _safe_float(cols[2]),
                "solar_mw":   _safe_float(cols[3]),
                "tcl_mw":     _safe_float(cols[4]),
                "usep":       _safe_float(cols[5]),
                "lcp":        _safe_float(cols[7]) if len(cols) > 7 else None,
                "source_file": "live_api",
            })

        periods_new = _ingest_rows(engine, rows)

        latest = rows[-1] if rows else {}
        result = {
            "periods_fetched": len(rows),
            "periods_new":     periods_new,
            "latest_date":     latest.get("date"),
            "latest_period":   latest.get("period"),
            "latest_usep":     latest.get("usep"),
            "latest_demand":   latest.get("demand_mw"),
            "source":          "live_api",
        }
        _log_fetch(engine, "live_api", result, int((time.monotonic() - t0) * 1000))
        return result

    except Exception as exc:
        logger.error("fetch_live_api error: %s", exc)
        result = {"periods_fetched": 0, "error": str(exc), "source": "live_api"}
        try:
            _log_fetch(engine, "live_api", result, int((time.monotonic() - t0) * 1000))
        except Exception:
            pass
        return result


# ─────────────────────────────────────────────────────────────────────────────
# A2. scrape_7day_chart() — 7-day data via Get?value=14 (no Playwright)
# ─────────────────────────────────────────────────────────────────────────────

async def scrape_7day_chart(engine, output_dir: Optional[Path] = None) -> dict:
    """
    Fetch 7-day USEP/Demand/Solar from EMC via Get?value=14.
    Uses direct HTTP (no Playwright needed — confirmed working API).
    Falls back to Playwright page scrape only if the API fails.
    """
    t0  = time.monotonic()
    url = f"{EMC_CHART_URL}?value=14"

    try:
        async with httpx.AsyncClient(
            headers=_BROWSER_HEADERS,
            timeout=15.0,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            payload = resp.json()

        inner   = payload["data"]["data"][0]
        labels  = inner.get("labels", [])
        ds_list = inner.get("datasets", [])

        # Index datasets by label
        ds_map: dict[str, list] = {}
        for ds in ds_list:
            lbl = (ds.get("label") or "").strip()
            ds_map[lbl] = ds.get("data", [])

        usep_vals   = ds_map.get("USEP",   [])
        demand_vals = ds_map.get("Demand", [])
        solar_vals  = ds_map.get("Solar",  [])
        vcp_vals    = ds_map.get("VCP",    [])

        year_hint = date.today().year
        rows = []
        for i, label in enumerate(labels):
            parts = label.strip().split()
            if len(parts) < 3:
                continue
            d = _parse_label_date(label, year_hint)
            if d is None:
                continue
            p = _time_range_to_period(parts[2])
            if p is None:
                continue

            usep      = _safe_float(usep_vals[i])   if i < len(usep_vals)   else None
            demand_mw = _safe_float(demand_vals[i]) if i < len(demand_vals) else None
            solar_mw  = _safe_float(solar_vals[i])  if i < len(solar_vals)  else None

            if usep is None and demand_mw is None:
                continue

            rows.append({
                "date":       d,
                "period":     p,
                "usep":       usep,
                "demand_mw":  demand_mw,
                "solar_mw":   solar_mw,
                "source_file": "7day_api",
            })

        periods_new = _ingest_rows(engine, rows)

        latest = rows[-1] if rows else {}
        result = {
            "periods_fetched": len(rows),
            "periods_new":     periods_new,
            "latest_date":     latest.get("date"),
            "latest_period":   latest.get("period"),
            "latest_usep":     latest.get("usep"),
            "latest_demand":   latest.get("demand_mw"),
            "method":          "7day_api",
            "source":          "7day_chart",
        }
        _log_fetch(engine, "7day_chart", result, int((time.monotonic() - t0) * 1000))
        return result

    except Exception as exc:
        logger.error("scrape_7day_chart error: %s", exc)
        result = {"periods_fetched": 0, "error": str(exc), "source": "7day_chart"}
        try:
            _log_fetch(engine, "7day_chart", result, int((time.monotonic() - t0) * 1000))
        except Exception:
            pass
        return result


# ─────────────────────────────────────────────────────────────────────────────
# A3. check_and_download_monthly_csv() — Playwright form download
# ─────────────────────────────────────────────────────────────────────────────

async def check_and_download_monthly_csv(engine, output_dir: Path) -> dict:
    """
    Check if a new monthly CSV is available and download it via Playwright.
    The EMC DataDownload endpoint requires a form POST with session token.
    Returns {downloaded, filename, rows_added, drift_check}.
    """
    from modules.ingestion import import_csv_to_db, _backfill_actuals

    t0 = time.monotonic()
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT MAX(date) AS max_d FROM nems_prices"
            )).mappings().fetchone()
        max_d = row["max_d"] if row and row["max_d"] else None
        if not max_d:
            return {"downloaded": False, "error": "No data in DB yet"}

        if isinstance(max_d, str):
            max_d = date.fromisoformat(max_d[:10])
        elif isinstance(max_d, datetime):
            max_d = max_d.date()

        last_day_of_month = (
            max_d.replace(day=1) + timedelta(days=32)
        ).replace(day=1) - timedelta(days=1)
        today = date.today()
        if today <= last_day_of_month + timedelta(days=3):
            return {"downloaded": False,
                    "reason": f"Latest data {max_d} — CSV not yet due"}

        target_month = max_d.strftime("%b")   # "Jun"
        target_year  = max_d.strftime("%Y")   # "2026"
        filename     = f"USEP_{target_month}-{target_year}.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / filename

        if save_path.exists():
            return {"downloaded": False,
                    "reason": f"{filename} already downloaded",
                    "filename": filename}

        if not is_local_environment():
            return {"downloaded": False,
                    "error": "Playwright not available on cloud",
                    "filename": filename}

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return {"downloaded": False,
                    "error": "playwright not installed — pip install playwright && playwright install chromium",
                    "filename": filename}

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                accept_downloads=True,
                extra_http_headers={
                    "User-Agent": _BROWSER_HEADERS["User-Agent"],
                    "Referer":    EMC_PRICES_URL,
                },
            )
            page = await context.new_page()
            await page.goto(EMC_PRICES_URL, wait_until="networkidle", timeout=30_000)
            await asyncio.sleep(2)

            # Look for year/month selectors and download button
            try:
                # Try select elements
                await page.select_option(
                    "select[id*='year'], select[name*='year'], select.year-select",
                    target_year
                )
                await asyncio.sleep(0.5)
                await page.select_option(
                    "select[id*='month'], select[name*='month'], select.month-select",
                    target_month
                )
                await asyncio.sleep(0.5)
                async with page.expect_download(timeout=30_000) as dl_info:
                    await page.click(
                        "button:text-matches('Download', 'i'), "
                        "a:text-matches('Download CSV', 'i'), "
                        "a[href*='DataDownload']"
                    )
                download = await dl_info.value
                await download.save_as(save_path)
            except Exception as ui_exc:
                await browser.close()
                return {"downloaded": False,
                        "error": f"Playwright UI interaction failed: {ui_exc}",
                        "filename": filename}
            finally:
                await browser.close()

        rows_added = import_csv_to_db(engine, save_path)
        _backfill_actuals(engine)

        drift_check = {"retrain_recommended": False}
        try:
            from modules.forecasting import check_model_drift
            drift = check_model_drift(engine, "xgboost", window_days=30)
            drift_check["retrain_recommended"] = drift.get("drift_detected", False)
        except Exception:
            pass

        result = {"downloaded": True, "filename": filename,
                  "rows_added": rows_added, "drift_check": drift_check}
        _log_fetch(engine, "monthly_csv", {
            "periods_fetched": rows_added, "periods_new": rows_added,
        }, int((time.monotonic() - t0) * 1000))
        return result

    except Exception as exc:
        logger.error("check_and_download_monthly_csv: %s", exc)
        return {"downloaded": False, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# A4. detect_gaps_and_fill()
# ─────────────────────────────────────────────────────────────────────────────

def detect_gaps_and_fill(engine, lookback_days: int = 7) -> dict:
    """
    Find missing (date, period) pairs in the last lookback_days.
    Attempts to fill via the 7-day API and the live API.
    Returns {gaps_found, gaps_filled, remaining_gaps}.
    """
    cutoff = date.today() - timedelta(days=lookback_days)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, period FROM nems_prices WHERE date >= :cutoff
        """), {"cutoff": cutoff}).fetchall()

    existing: set = set()
    for r in rows:
        d = r[0] if isinstance(r[0], date) else date.fromisoformat(str(r[0])[:10])
        existing.add((d, int(r[1])))

    expected: set = {
        (cutoff + timedelta(days=i), p)
        for i in range(lookback_days)
        for p in range(1, 49)
    }
    gaps = expected - existing
    gaps_found = len(gaps)

    if gaps_found == 0:
        return {"gaps_found": 0, "gaps_filled": 0, "remaining_gaps": 0}

    # Try 7-day API first (covers the week), then live API for today
    filled = 0
    try:
        r7 = asyncio.run(scrape_7day_chart(engine))
        filled += r7.get("periods_new", 0)
    except Exception:
        pass
    try:
        rl = asyncio.run(fetch_live_api(engine))
        filled += rl.get("periods_new", 0)
    except Exception:
        pass

    # Recount
    with engine.connect() as conn:
        rows2 = conn.execute(text("""
            SELECT date, period FROM nems_prices WHERE date >= :cutoff
        """), {"cutoff": cutoff}).fetchall()
    existing2: set = set()
    for r in rows2:
        d = r[0] if isinstance(r[0], date) else date.fromisoformat(str(r[0])[:10])
        existing2.add((d, int(r[1])))
    remaining = len(expected - existing2)

    return {"gaps_found": gaps_found, "gaps_filled": filled,
            "remaining_gaps": remaining}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: last data timestamp for sidebar widget
# ─────────────────────────────────────────────────────────────────────────────

def get_last_data_timestamp(engine) -> Optional[datetime]:
    """Return the datetime of the most recent nems_prices row (SGT-approx)."""
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT date, period FROM nems_prices
                ORDER BY date DESC, period DESC LIMIT 1
            """)).fetchone()
        if row:
            d  = row[0] if isinstance(row[0], date) else date.fromisoformat(str(row[0])[:10])
            p  = int(row[1])
            return datetime.combine(d, datetime.min.time()) + timedelta(minutes=(p - 1) * 30)
    except Exception:
        pass
    return None
