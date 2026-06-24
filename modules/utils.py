from datetime import time


def period_to_time_label(period: int) -> str:
    """Convert NEMS half-hourly period number to human-readable SGT range.

    Period 1 → '00:00–00:30 SGT', Period 32 → '15:30–16:00 SGT'
    """
    start_minutes = (period - 1) * 30
    end_minutes = period * 30

    start_h, start_m = divmod(start_minutes, 60)
    end_h, end_m = divmod(end_minutes % 1440, 60)

    return f"{start_h:02d}:{start_m:02d}–{end_h:02d}:{end_m:02d} SGT"


def period_to_start_time(period: int) -> time:
    """Return the start time (as datetime.time) for a given NEMS period."""
    total_minutes = (period - 1) * 30
    h, m = divmod(total_minutes, 60)
    return time(h, m)


CHART_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#0d1117",
        "plot_bgcolor": "#161b22",
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
            "#009CEA",
            "#f0b429",
            "#2ecc71",
            "#e74c3c",
            "#9b59b6",
            "#1abc9c",
            "#e67e22",
        ],
    }
}
