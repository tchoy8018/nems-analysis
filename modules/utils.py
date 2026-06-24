from datetime import time, date as date_type

import holidays
import pandas as pd

_SG_HOLIDAYS = holidays.country_holidays("SG")

# Approximate Singapore school holiday periods (Jan–Dec)
# Term 1: ~Jan 3 – Mar 11 | Holiday: ~Mar 12 – Mar 20
# Term 2: ~Mar 21 – May 27 | Holiday: ~May 28 – Jun 26
# Term 3: ~Jun 27 – Sep 2  | Holiday: ~Sep 3 – Sep 11
# Term 4: ~Sep 12 – Nov 18 | Holiday: ~Nov 19 – Jan 2 (next yr)
_SCHOOL_HOLIDAY_RANGES = [
    (3, 12), (3, 20),   # Mar mid-term
    (5, 28), (6, 26),   # Jun long break
    (9, 3),  (9, 11),   # Sep mid-term
    (11, 19),(12, 31),  # Nov–Dec year-end
    (1, 1),  (1, 2),    # New Year carry-over
]


def _is_school_holiday(d: date_type) -> bool:
    ranges = [
        ((3, 12), (3, 20)),
        ((5, 28), (6, 26)),
        ((9, 3),  (9, 11)),
        ((11, 19),(12, 31)),
        ((1, 1),  (1, 2)),
    ]
    for (sm, sd), (em, ed) in ranges:
        if (d.month, d.day) >= (sm, sd) and (d.month, d.day) <= (em, ed):
            return True
    return False


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


def get_sg_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """
    Return a DataFrame of Singapore-specific calendar features for a Series of dates.

    Columns:
        date                — original date
        is_public_holiday   — bool (SG public holidays via `holidays` library)
        day_of_week         — 0=Mon … 6=Sun
        day_type            — 'weekday_core' (Tue-Thu) | 'weekday_wfh' (Mon, Fri) |
                              'saturday' | 'sunday' | 'public_holiday'
        is_school_holiday   — bool (approximate SG school term calendar)
        month_type          — 'peak' (Nov-Jan) | 'shoulder' (Feb-Mar, Sep-Oct) | 'low'
    """
    dates = pd.to_datetime(dates)
    rows = []
    for d in dates:
        d_date = d.date()
        is_ph  = d_date in _SG_HOLIDAYS
        dow    = d.dayofweek   # 0=Mon
        month  = d.month

        if is_ph:
            day_type = "public_holiday"
        elif dow in (1, 2, 3):   # Tue Wed Thu
            day_type = "weekday_core"
        elif dow in (0, 4):      # Mon Fri — WFH effect
            day_type = "weekday_wfh"
        elif dow == 5:
            day_type = "saturday"
        else:
            day_type = "sunday"

        if month in (11, 12, 1):
            month_type = "peak"
        elif month in (2, 3, 9, 10):
            month_type = "shoulder"
        else:
            month_type = "low"

        rows.append({
            "date":              d_date,
            "is_public_holiday": is_ph,
            "day_of_week":       dow,
            "day_type":          day_type,
            "is_school_holiday": _is_school_holiday(d_date),
            "month_type":        month_type,
        })
    return pd.DataFrame(rows)


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
