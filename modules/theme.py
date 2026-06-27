import streamlit as st
import streamlit.components.v1 as components


def get_theme() -> str:
    return st.session_state.get("theme", "dark")


def render_theme_toggle():
    """Render toggle button in sidebar. Call inside a `with st.sidebar:` block."""
    theme = get_theme()
    label = "☀️ Light mode" if theme == "dark" else "🌙 Dark mode"
    if st.button(label, key="theme_toggle_btn", use_container_width=True):
        st.session_state["theme"] = "light" if theme == "dark" else "dark"
        st.cache_data.clear()
        st.rerun()


def apply_theme_css() -> None:
    """Inject full CSS overrides for current theme."""
    theme = get_theme()
    is_dark = theme == "dark"

    bg      = "#0d1117" if is_dark else "#ffffff"
    sbg     = "#161b22" if is_dark else "#f0f2f6"
    tc      = "#e6edf3" if is_dark else "#1c2128"
    border  = "#30363d" if is_dark else "#d0d7de"
    primary = "#009CEA"
    inp_bg  = "#21262d" if is_dark else "#f6f8fa"

    st.markdown(f"""
<style>
/* ── Backgrounds ─────────────────────────────────────── */
.stApp, .stApp > header {{
    background-color: {bg} !important;
}}
.main .block-container {{
    background-color: {bg} !important;
}}
section[data-testid="stSidebar"] > div {{
    background-color: {sbg} !important;
}}

/* ── Text ────────────────────────────────────────────── */
body, p, span, label, div, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown p,
div[data-testid="stMetricValue"],
div[data-testid="stMetricLabel"],
div[data-testid="stMetricDelta"],
.stCaption {{
    color: {tc} !important;
}}
section[data-testid="stSidebar"] * {{
    color: {tc} !important;
}}

/* ── Inputs ──────────────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stDateInput input,
.stSelectbox select {{
    background-color: {inp_bg} !important;
    color: {tc} !important;
    border-color: {border} !important;
}}

/* ── Dataframes / tables ─────────────────────────────── */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] * {{
    color: {tc} !important;
}}
[data-testid="stDataFrame"] .dvn-scroller {{
    background-color: {sbg} !important;
}}

/* ── Divider ─────────────────────────────────────────── */
hr {{
    border-color: {border} !important;
}}

/* ── Expanders ───────────────────────────────────────── */
.streamlit-expanderHeader,
[data-testid="stExpander"] > div:first-child {{
    background-color: {sbg} !important;
    color: {tc} !important;
    border: 1px solid {border} !important;
    border-radius: 6px !important;
}}
[data-testid="stExpander"] {{
    background-color: {sbg} !important;
    border: 1px solid {border} !important;
}}
[data-testid="stExpander"] * {{
    color: {tc} !important;
}}

/* ── Alert / info / warning / error boxes ────────────── */
[data-testid="stAlert"] {{
    background-color: {sbg} !important;
    color: {tc} !important;
}}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {{
    color: {tc} !important;
}}

/* ── Tabs ────────────────────────────────────────────── */
button[data-baseweb="tab"] {{
    color: {tc} !important;
    background-color: transparent !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {primary} !important;
    border-bottom: 2px solid {primary} !important;
}}

/* ── Radio buttons ───────────────────────────────────── */
[data-testid="stRadio"] label {{
    color: {tc} !important;
}}

/* ── Sliders ─────────────────────────────────────────── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] p {{
    color: {tc} !important;
}}

/* ── File uploader ───────────────────────────────────── */
[data-testid="stFileUploader"] {{
    background-color: {sbg} !important;
    border: 1px dashed {border} !important;
    color: {tc} !important;
}}
[data-testid="stFileUploader"] * {{
    color: {tc} !important;
}}

/* ── Select sliders ──────────────────────────────────── */
[data-testid="stSelectSlider"] label,
[data-testid="stSelectSlider"] p,
[data-testid="stSelectSlider"] span {{
    color: {tc} !important;
}}

/* ── Progress bar label ──────────────────────────────── */
[data-testid="stProgress"] p {{
    color: {tc} !important;
}}
</style>
""", unsafe_allow_html=True)


def get_chart_layout() -> dict:
    """Return Plotly layout dict for the current theme with full font/color coverage."""
    is_dark = get_theme() == "dark"
    tc      = "#e6edf3" if is_dark else "#1c2128"
    grid    = "#21262d" if is_dark else "#e5e7eb"
    line    = "#30363d" if is_dark else "#d0d7de"
    hover_bg = "#21262d" if is_dark else "#ffffff"

    axis_style = dict(
        gridcolor=grid,
        linecolor=line,
        zerolinecolor=grid,
        tickfont=dict(color=tc),
        title=dict(font=dict(color=tc)),
    )

    return {
        "paper_bgcolor": "#0d1117" if is_dark else "#ffffff",
        "plot_bgcolor":  "#161b22" if is_dark else "#f6f8fa",
        "font": dict(color=tc, family="Inter, sans-serif", size=12),
        "title": dict(font=dict(color=tc, size=15)),
        "legend": dict(
            bgcolor="#161b22" if is_dark else "#f0f2f5",
            bordercolor=line,
            borderwidth=1,
            font=dict(color=tc),
        ),
        "xaxis": dict(**axis_style),
        "yaxis": dict(**axis_style),
        "hoverlabel": dict(
            bgcolor=hover_bg,
            bordercolor="#009CEA",
            font=dict(color=tc),
        ),
        "colorway": [
            "#009CEA", "#f0b429", "#2ecc71",
            "#e74c3c", "#9b59b6", "#1abc9c", "#e67e22",
        ],
    }


def get_yaxis2_style() -> dict:
    """Return theme-aware yaxis2 dict for dual-axis charts."""
    is_dark = get_theme() == "dark"
    tc      = "#e6edf3" if is_dark else "#1c2128"
    grid    = "#21262d" if is_dark else "#e5e7eb"
    return dict(
        tickfont=dict(color=tc),
        title=dict(font=dict(color=tc)),
        gridcolor=grid,
    )


def get_rangeselector_style() -> dict:
    """Return Plotly rangeselector kwargs styled for the current theme."""
    is_dark = get_theme() == "dark"
    return dict(
        bgcolor="#161b22" if is_dark else "#f0f2f6",
        activecolor="#009CEA",
        font=dict(color="#e6edf3" if is_dark else "#1c2128", size=11),
        borderwidth=1,
        bordercolor="#009CEA",
    )


def add_copy_button(fig_key: str, label: str = "Copy chart") -> None:
    """PNG copy-to-clipboard via Plotly.toImage. Falls back to download with message."""
    components.html(f"""
<div style="margin:6px 0 10px 0;">
  <button id="cb_{fig_key}" onclick="doCopy_{fig_key}()"
    style="background:#009CEA;color:#fff;border:none;
           padding:5px 14px;border-radius:5px;cursor:pointer;
           font-size:13px;font-family:Inter,sans-serif;">
    📋 {label}
  </button>
  <span id="msg_{fig_key}"
    style="margin-left:10px;font-size:12px;font-family:Inter,sans-serif;">
  </span>
</div>
<script>
async function doCopy_{fig_key}() {{
  const btn = document.getElementById('cb_{fig_key}');
  const msg = document.getElementById('msg_{fig_key}');
  btn.disabled = true; btn.textContent = '⏳ Copying...';
  try {{
    const p      = window.parent || window;
    const plots  = p.document.querySelectorAll('.js-plotly-plot');
    if (!plots || !plots.length) throw new Error('No chart found on page');
    const div    = plots[plots.length - 1];
    const Plotly = p.Plotly;
    const url    = await Plotly.toImage(div, {{
      format: 'png',
      width:  div.offsetWidth  || 1400,
      height: div.offsetHeight || 700,
      scale:  2,
    }});
    const blob = await (await fetch(url)).blob();
    if (navigator.clipboard?.write) {{
      await navigator.clipboard.write([new ClipboardItem({{'image/png': blob}})]);
      msg.style.color = '#2ecc71';
      msg.textContent = '✅ Copied!';
    }} else {{
      const a = document.createElement('a');
      a.href = url; a.download = '{fig_key}.png'; a.click();
      msg.style.color = '#f0b429';
      msg.textContent = '⬇️ Downloaded (clipboard needs HTTPS)';
    }}
  }} catch(e) {{
    msg.style.color = '#e74c3c';
    msg.textContent = '❌ ' + e.message;
  }} finally {{
    btn.disabled = false; btn.textContent = '📋 {label}';
    setTimeout(() => {{ msg.textContent = ''; }}, 3500);
  }}
}}
</script>
""", height=48, scrolling=False)
