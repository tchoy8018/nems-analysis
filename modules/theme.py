import streamlit as st

THEMES = {
    "dark": {
        "backgroundColor": "#0d1117",
        "secondaryBackgroundColor": "#161b22",
        "textColor": "#e6edf3",
        "chart": {
            "paper_bgcolor": "#0d1117",
            "plot_bgcolor": "#161b22",
            "font": {"color": "#e6edf3", "family": "Inter, sans-serif", "size": 12},
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
            "legend": {"bgcolor": "#161b22", "bordercolor": "#30363d", "borderwidth": 1},
            "hoverlabel": {
                "bgcolor": "#21262d",
                "bordercolor": "#009CEA",
                "font": {"color": "#e6edf3"},
            },
            "colorway": [
                "#009CEA", "#f0b429", "#2ecc71",
                "#e74c3c", "#9b59b6", "#1abc9c", "#e67e22",
            ],
        },
    },
    "light": {
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#262730",
        "chart": {
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#f0f2f6",
            "font": {"color": "#262730", "family": "Inter, sans-serif", "size": 12},
            "xaxis": {
                "gridcolor": "#dde1e9",
                "linecolor": "#c4c9d4",
                "zerolinecolor": "#dde1e9",
            },
            "yaxis": {
                "gridcolor": "#dde1e9",
                "linecolor": "#c4c9d4",
                "zerolinecolor": "#dde1e9",
            },
            "legend": {"bgcolor": "#ffffff", "bordercolor": "#c4c9d4", "borderwidth": 1},
            "hoverlabel": {
                "bgcolor": "#f0f2f6",
                "bordercolor": "#009CEA",
                "font": {"color": "#262730"},
            },
            "colorway": [
                "#007bbf", "#c8920a", "#1d8348",
                "#c0392b", "#7d3c98", "#117a65", "#b7510a",
            ],
        },
    },
}


def get_theme() -> str:
    return st.session_state.get("theme", "dark")


def render_theme_toggle():
    """Render toggle button in sidebar. Call inside a `with st.sidebar:` block."""
    theme = get_theme()
    label = "☀️ Light mode" if theme == "dark" else "🌙 Dark mode"
    if st.button(label, key="theme_toggle_btn", use_container_width=True):
        st.session_state["theme"] = "light" if theme == "dark" else "dark"
        st.rerun()


def apply_theme_css():
    """Inject CSS overrides based on current theme selection."""
    theme = get_theme()
    cfg = THEMES[theme]
    bg = cfg["backgroundColor"]
    sbg = cfg["secondaryBackgroundColor"]
    tc = cfg["textColor"]

    st.markdown(
        f"""
<style>
/* Main backgrounds */
.stApp, .stApp > header {{
    background-color: {bg} !important;
}}
.main .block-container {{
    background-color: {bg} !important;
}}
section[data-testid="stSidebar"] > div {{
    background-color: {sbg} !important;
}}
/* Text */
body, p, span, label, div, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown p,
div[data-testid="stMetricValue"],
div[data-testid="stMetricLabel"],
div[data-testid="stMetricDelta"],
.stCaption {{
    color: {tc} !important;
}}
/* Sidebar text */
section[data-testid="stSidebar"] * {{
    color: {tc} !important;
}}
/* Inputs */
.stTextInput input, .stNumberInput input, .stDateInput input,
.stSelectbox select {{
    background-color: {sbg} !important;
    color: {tc} !important;
    border-color: #555 !important;
}}
/* Dataframes / tables */
.stDataFrame, .stDataFrame * {{
    background-color: {sbg} !important;
    color: {tc} !important;
}}
/* Divider */
hr {{
    border-color: {'#30363d' if theme == 'dark' else '#c4c9d4'} !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def get_chart_layout() -> dict:
    """Return Plotly layout dict for the current theme."""
    return dict(THEMES[get_theme()]["chart"])


def get_rangeselector_style() -> dict:
    """Return Plotly rangeselector kwargs styled for the current theme."""
    cfg = THEMES[get_theme()]
    return dict(
        bgcolor=cfg["secondaryBackgroundColor"],
        activecolor="#009CEA",
        font=dict(color=cfg["textColor"], size=11),
        borderwidth=1,
        bordercolor="#009CEA",
    )
