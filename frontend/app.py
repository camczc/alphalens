"""
frontend/app.py

AlphaLens Streamlit Dashboard

Run:
    python -m streamlit run frontend/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="AlphaLens",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Sidebar nav
# ------------------------------------------------------------------

st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
st.sidebar.title("AlphaLens")
st.sidebar.caption("AI-Powered Stock Research")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Research", "📊 Backtest", "⚖️ Compare Strategies"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption("Powered by Claude · Built by Cameron Cooper")

# ------------------------------------------------------------------
# Page routing — imports from views/ (not pages/) to avoid
# Streamlit's auto multi-page detection
# ------------------------------------------------------------------

if page == "🔍 Research":
    from frontend.views.research import render
    render()
elif page == "📊 Backtest":
    from frontend.views.backtest import render
    render()
elif page == "⚖️ Compare Strategies":
    from frontend.views.compare import render
    render()
