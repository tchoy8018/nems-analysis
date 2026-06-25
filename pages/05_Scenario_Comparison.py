from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from db import get_engine, setup_database
from modules.theme import apply_theme_css, render_theme_toggle

st.set_page_config(page_title="Scenario Comparison — NEMS", layout="wide")


@st.cache_resource
def _get_engine():
    engine = get_engine()
    setup_database(engine)
    return engine


engine = _get_engine()
apply_theme_css()

with st.sidebar:
    st.title("⚡ NEMS Analytics")
    st.caption("Singa Renewables")
    render_theme_toggle()

st.title("📈 Scenario Comparison")
st.info("Phase 5 — coming soon.")
