from __future__ import annotations

import streamlit as st

from pages import (
    backtesting_lab_page,
    comparison_news_page,
    dashboard_page,
    portfolio_watchlist_page,
    prediction_page,
    technicals_risk_page,
)


st.set_page_config(page_title="AI Stock Intelligence Platform", page_icon="📈", layout="wide")
st.markdown(
    """
<style>
.stApp { background: linear-gradient(180deg, #0b1220, #0f172a); color: #e2e8f0; }
div[data-testid="stMetric"] { background: #111827; padding: 10px; border-radius: 12px; border: 1px solid #1f2937; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="padding:0.25rem 0 0.75rem 0;">
      <h1 style="margin-bottom:0.15rem;">AI Stock Intelligence Platform</h1>
      <p style="margin-top:0;color:#94a3b8;">Advanced market analytics, forecasting, and risk intelligence using free data and open-source AI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose page",
    [
        "Dashboard",
        "Prediction Engine",
        "Technicals and Risk",
        "Portfolio and Watchlist",
        "Comparison and News",
        "Backtesting Lab",
    ],
)
st.sidebar.caption("Built with free APIs: yFinance + open-source ML.")

try:
    if page == "Dashboard":
        dashboard_page.render()
    elif page == "Prediction Engine":
        prediction_page.render()
    elif page == "Technicals and Risk":
        technicals_risk_page.render()
    elif page == "Portfolio and Watchlist":
        portfolio_watchlist_page.render()
    elif page == "Backtesting Lab":
        backtesting_lab_page.render()
    else:
        comparison_news_page.render()
except Exception as exc:
    st.error(f"Something went wrong: {exc}")
    st.caption("Check backend server is running at http://127.0.0.1:8000")

