from datetime import time, date as date_type, timedelta

import holidays
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Official SG public holidays — MOM (mom.gov.sg) confirmed, 2019–2027
# ─────────────────────────────────────────────────────────────────────────────

SG_PUBLIC_HOLIDAYS: set[str] = {
    # 2019
    "2019-01-01", "2019-02-05", "2019-02-06", "2019-04-19",
    "2019-05-01", "2019-05-19", "2019-06-05", "2019-07-31",
    "2019-08-09", "2019-10-27", "2019-12-25",
    # 2020
    "2020-01-01", "2020-01-25", "2020-01-26", "2020-04-10",
    "2020-05-01", "2020-05-07", "2020-05-24", "2020-07-31",
    "2020-08-09", "2020-11-14", "2020-12-25",
    # 2021
    "2021-01-01", "2021-02-12", "2021-02-13", "2021-05-01",
    "2021-05-13", "2021-05-26", "2021-07-20", "2021-08-09",
    "2021-11-04", "2021-12-25",
    # 2022
    "2022-01-01", "2022-02-01", "2022-02-02", "2022-04-15",
    "2022-05-01", "2022-05-03", "2022-05-15", "2022-07-10",
    "2022-08-09", "2022-10-24", "2022-12-25",
    # 2023
    "2023-01-01", "2023-01-02", "2023-01-22", "2023-01-23",
    "2023-04-07", "2023-04-22", "2023-05-01", "2023-06-02",
    "2023-06-29", "2023-08-09", "2023-11-13", "2023-12-25",
    # 2024
    "2024-01-01", "2024-02-10", "2024-02-11", "2024-04-10",
    "2024-04-19", "2024-05-01", "2024-05-22", "2024-06-17",
    "2024-08-09", "2024-10-31", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-29", "2025-01-30", "2025-03-31",
    "2025-04-18", "2025-05-01", "2025-05-03",
    "2025-05-12", "2025-06-07", "2025-08-09", "2025-10-20", "2025-12-25",
    # 2026
    "2026-01-01", "2026-02-17", "2026-02-18", "2026-03-21",
    "2026-04-03", "2026-05-01", "2026-05-27", "2026-05-31",
    "2026-06-01", "2026-08-09", "2026-08-10",
    "2026-11-08", "2026-11-09", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-27", "2027-01-28", "2027-03-10",
    "2027-04-02", "2027-05-01", "2027-05-16", "2027-06-17",
    "2027-08-09", "2027-10-28", "2027-12-25",
}

HOLIDAY_NAMES: dict[str, str] = {
    "2019-01-01": "New Year's Day",        "2019-02-05": "Chinese New Year (Day 1)",
    "2019-02-06": "Chinese New Year (Day 2)", "2019-04-19": "Good Friday",
    "2019-05-01": "Labour Day",            "2019-05-19": "Vesak Day",
    "2019-06-05": "Hari Raya Puasa",       "2019-07-31": "Hari Raya Haji",
    "2019-08-09": "National Day",          "2019-10-27": "Deepavali",
    "2019-12-25": "Christmas Day",
    "2020-01-01": "New Year's Day",        "2020-01-25": "Chinese New Year (Day 1)",
    "2020-01-26": "Chinese New Year (Day 2)", "2020-04-10": "Good Friday",
    "2020-05-01": "Labour Day",            "2020-05-07": "Vesak Day",
    "2020-05-24": "Hari Raya Puasa",       "2020-07-31": "Hari Raya Haji",
    "2020-08-09": "National Day",          "2020-11-14": "Deepavali",
    "2020-12-25": "Christmas Day",
    "2021-01-01": "New Year's Day",        "2021-02-12": "Chinese New Year (Day 1)",
    "2021-02-13": "Chinese New Year (Day 2)", "2021-05-01": "Labour Day",
    "2021-05-13": "Hari Raya Puasa",       "2021-05-26": "Vesak Day",
    "2021-07-20": "Hari Raya Haji",        "2021-08-09": "National Day",
    "2021-11-04": "Deepavali",             "2021-12-25": "Christmas Day",
    "2022-01-01": "New Year's Day",        "2022-02-01": "Chinese New Year (Day 1)",
    "2022-02-02": "Chinese New Year (Day 2)", "2022-04-15": "Good Friday",
    "2022-05-01": "Labour Day",            "2022-05-03": "Hari Raya Puasa",
    "2022-05-15": "Vesak Day",             "2022-07-10": "Polling Day",
    "2022-08-09": "National Day",          "2022-10-24": "Deepavali",
    "2022-12-25": "Christmas Day",
    "2023-01-01": "New Year's Day",        "2023-01-02": "New Year's Day (sub.)",
    "2023-01-22": "Chinese New Year (Day 1)", "2023-01-23": "Chinese New Year (Day 2)",
    "2023-04-07": "Good Friday",           "2023-04-22": "Hari Raya Puasa",
    "2023-05-01": "Labour Day",            "2023-06-02": "Vesak Day",
    "2023-06-29": "Hari Raya Haji",        "2023-08-09": "National Day",
    "2023-11-13": "Deepavali",             "2023-12-25": "Christmas Day",
    "2024-01-01": "New Year's Day",        "2024-02-10": "Chinese New Year (Day 1)",
    "2024-02-11": "Chinese New Year (Day 2)", "2024-04-10": "Hari Raya Puasa",
    "2024-04-19": "Good Friday",           "2024-05-01": "Labour Day",
    "2024-05-22": "Vesak Day",             "2024-06-17": "Hari Raya Haji",
    "2024-08-09": "National Day",          "2024-10-31": "Deepavali",
    "2024-12-25": "Christmas Day",
    "2025-01-01": "New Year's Day",        "2025-01-29": "Chinese New Year (Day 1)",
    "2025-01-30": "Chinese New Year (Day 2)", "2025-03-31": "Hari Raya Puasa",
    "2025-04-18": "Good Friday",           "2025-05-01": "Labour Day",
    "2025-05-03": "Polling Day",           "2025-05-12": "Vesak Day",
    "2025-06-07": "Hari Raya Haji",        "2025-08-09": "National Day",
    "2025-10-20": "Deepavali",             "2025-12-25": "Christmas Day",
    "2026-01-01": "New Year's Day",        "2026-02-17": "Chinese New Year (Day 1)",
    "2026-02-18": "Chinese New Year (Day 2)", "2026-03-21": "Hari Raya Puasa",
    "2026-04-03": "Good Friday",           "2026-05-01": "Labour Day",
    "2026-05-27": "Vesak Day",             "2026-05-31": "Hari Raya Haji",
    "2026-06-01": "Vesak Day (sub.)",      "2026-08-09": "National Day",
    "2026-08-10": "National Day (sub.)",   "2026-11-08": "Deepavali",
    "2026-11-09": "Deepavali (sub.)",      "2026-12-25": "Christmas Day",
    "2027-01-01": "New Year's Day",        "2027-01-27": "Chinese New Year (Day 1)",
    "2027-01-28": "Chinese New Year (Day 2)", "2027-03-10": "Hari Raya Puasa",
    "2027-04-02": "Good Friday",           "2027-05-01": "Labour Day",
    "2027-05-16": "Vesak Day",             "2027-06-17": "Hari Raya Haji",
    "2027-08-09": "National Day",          "2027-10-28": "Deepavali",
    "2027-12-25": "Christmas Day",
}

# Fallback library for years outside hardcoded range
_SG_HOLIDAYS_LIB = holidays.country_holidays("SG")

# Sorted numpy array of holiday ordinals (days since epoch) for fast searchsorted
_PH_DATES: list[date_type] = sorted(
    date_type.fromisoformat(s) for s in SG_PUBLIC_HOLIDAYS
)
_PH_ORDS = np.array([d.toordinal() for d in _PH_DATES], dtype=np.int32)


def _is_sg_public_holiday(d: date_type) -> bool:
    ds = d.strftime("%Y-%m-%d")
    if ds in SG_PUBLIC_HOLIDAYS:
        return True
    if d.year < 2019 or d.year > 2027:
        return d in _SG_HOLIDAYS_LIB
    return False


def _days_to_next_holiday(ord_: int) -> int:
    """Days from ordinal ord_ to the next (or same-day) public holiday."""
    idx = int(np.searchsorted(_PH_ORDS, ord_))
    if idx >= len(_PH_ORDS):
        return 999
    return int(_PH_ORDS[idx] - ord_)


def _days_since_last_holiday(ord_: int) -> int:
    """Days from the last public holiday to ordinal ord_."""
    idx = int(np.searchsorted(_PH_ORDS, ord_, side="right")) - 1
    if idx < 0:
        return 999
    return int(ord_ - _PH_ORDS[idx])


# Approximate Singapore school holiday periods (for get_sg_calendar_features)
_SCHOOL_HOLIDAY_RANGES = [
    ((3, 12), (3, 20)),
    ((5, 28), (6, 26)),
    ((9, 3),  (9, 11)),
    ((11, 19),(12, 31)),
    ((1, 1),  (1, 2)),
]


def _is_school_holiday(d: date_type) -> bool:
    for (sm, sd), (em, ed) in _SCHOOL_HOLIDAY_RANGES:
        if (d.month, d.day) >= (sm, sd) and (d.month, d.day) <= (em, ed):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────────────────────────────────────

def period_to_time_label(period: int) -> str:
    """Convert NEMS half-hourly period number to human-readable SGT range."""
    start_minutes = (period - 1) * 30
    end_minutes   = period * 30
    start_h, start_m = divmod(start_minutes, 60)
    end_h,   end_m   = divmod(end_minutes % 1440, 60)
    return f"{start_h:02d}:{start_m:02d}–{end_h:02d}:{end_m:02d} SGT"


def period_to_start_time(period: int) -> time:
    """Return the start time (datetime.time) for a given NEMS period."""
    total_minutes = (period - 1) * 30
    h, m = divmod(total_minutes, 60)
    return time(h, m)


# ─────────────────────────────────────────────────────────────────────────────
# Calendar feature generator
# ─────────────────────────────────────────────────────────────────────────────

def get_sg_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """
    Return SG-specific calendar features for a Series of dates.

    Columns:
        date                    — original date (date type)
        is_public_holiday       — bool (MOM hardcoded 2019–2027, library fallback)
        holiday_name            — str or None
        day_of_week             — 0=Mon … 6=Sun
        day_type                — 'public_holiday' | 'sunday' | 'saturday' |
                                  'weekday_wfh' (Mon/Fri) | 'weekday_core' (Tue–Thu)
        is_school_holiday       — bool (approximate SG school terms)
        month_type              — 'peak' (Nov–Jan) | 'shoulder' (Feb–Mar, Sep–Oct) | 'low'
        days_to_next_holiday    — int (captures pre-holiday demand dip)
        days_since_last_holiday — int
    """
    dates = pd.to_datetime(dates)
    rows = []
    for d in dates:
        d_date = d.date()
        ds     = d_date.strftime("%Y-%m-%d")
        is_ph  = _is_sg_public_holiday(d_date)
        dow    = d.dayofweek
        month  = d.month
        ord_   = d_date.toordinal()

        if is_ph:
            day_type = "public_holiday"
        elif dow == 6:
            day_type = "sunday"
        elif dow == 5:
            day_type = "saturday"
        elif dow in (0, 4):
            day_type = "weekday_wfh"
        else:
            day_type = "weekday_core"

        if month in (11, 12, 1):
            month_type = "peak"
        elif month in (2, 3, 9, 10):
            month_type = "shoulder"
        else:
            month_type = "low"

        rows.append({
            "date":                    d_date,
            "is_public_holiday":       is_ph,
            "holiday_name":            HOLIDAY_NAMES.get(ds),
            "day_of_week":             dow,
            "day_type":                day_type,
            "is_school_holiday":       _is_school_holiday(d_date),
            "month_type":              month_type,
            "days_to_next_holiday":    _days_to_next_holiday(ord_),
            "days_since_last_holiday": _days_since_last_holiday(ord_),
        })
    return pd.DataFrame(rows)


def get_holidays_in_range(start: date_type, end: date_type) -> list[dict]:
    """
    Return list of {date, name} for public holidays in [start, end].
    Used for chart annotations.
    """
    result = []
    for ds in SG_PUBLIC_HOLIDAYS:
        d = date_type.fromisoformat(ds)
        if start <= d <= end:
            result.append({"date": d, "name": HOLIDAY_NAMES.get(ds, "Public Holiday")})
    return sorted(result, key=lambda x: x["date"])


# ─────────────────────────────────────────────────────────────────────────────
# Chart template (dark default)
# ─────────────────────────────────────────────────────────────────────────────

def _axis(base: dict, overrides: dict | None = None) -> dict:
    """
    Merge a base axis style dict (from get_chart_layout()["xaxis" / "yaxis"])
    with per-chart overrides without producing duplicate keyword arguments.

    Usage:
        cl = get_chart_layout()
        xax = _axis(cl["xaxis"], {"title": "Time (SGT)", "tickvals": [1,13,25,37,48]})
        yax = _axis(cl["yaxis"], {"title": "USEP (S$/MWh)", "rangemode": "tozero"})
    """
    result = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    if not overrides:
        return result
    for k, v in overrides.items():
        if k == "title":
            if isinstance(result.get("title"), dict):
                result["title"] = {**result["title"], **({"text": v} if isinstance(v, str) else v)}
            else:
                result["title"] = {"text": v} if isinstance(v, str) else v
        elif k in ("tickfont", "font") and isinstance(result.get(k), dict):
            result[k] = {**result[k], **v}
        else:
            result[k] = v
    return result


CHART_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#0d1117",
        "plot_bgcolor":  "#161b22",
        "font": {
            "color": "#e6edf3",
            "family": "Inter, sans-serif",
            "size": 12,
        },
        "xaxis": {
            "gridcolor": "#21262d",
            "linecolor": "#30363d",
            "zerolinecolor": "#21262d",
        },
        "yaxis": {
            "gridcolor": "#21262d",
            "linecolor": "#30363d",
            "zerolinecolor": "#21262d",
        },
        "legend": {
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "borderwidth": 1,
        },
        "hoverlabel": {
            "bgcolor": "#21262d",
            "bordercolor": "#009CEA",
            "font": {"color": "#e6edf3"},
        },
        "colorway": [
            "#009CEA", "#f0b429", "#2ecc71",
            "#e74c3c", "#9b59b6", "#1abc9c", "#e67e22",
        ],
    }
}
