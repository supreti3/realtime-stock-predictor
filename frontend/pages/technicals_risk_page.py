from __future__ import annotations

import pandas as pd
import streamlit as st

from components import api_client
from components.charts import multi_line_chart


def render() -> None:
    st.header("Technical Indicators and Risk Analysis")
    ticker = st.text_input("Ticker for technicals/risk", value="MSFT").upper().strip()
    if not ticker:
        return
    with st.spinner("Calculating indicators and risk metrics..."):
        indicators = api_client.get_indicators(ticker)
        risk = api_client.get_risk(ticker)

    ind_df = pd.DataFrame(indicators["series"])
    ind_df["Date"] = pd.to_datetime(ind_df["Date"])
    st.plotly_chart(
        multi_line_chart(ind_df, "Date", ["Close", "SMA_7", "SMA_30", "SMA_90"], f"{ticker} Price + SMA"),
        use_container_width=True,
    )
    st.plotly_chart(
        multi_line_chart(ind_df, "Date", ["EMA_12", "EMA_26", "MACD", "MACD_SIGNAL"], f"{ticker} EMA + MACD"),
        use_container_width=True,
    )
    st.plotly_chart(
        multi_line_chart(ind_df, "Date", ["BB_UPPER", "BB_MID", "BB_LOWER", "Close"], f"{ticker} Bollinger Bands"),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Volatility", f"{risk['volatility']:.2f}")
    c2.metric("Max Drawdown", f"{risk['max_drawdown']:.2%}")
    c3.metric("Sharpe Ratio", f"{risk['sharpe_ratio']:.2f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("VaR (95%)", f"{risk['value_at_risk_95']:.2%}")
    c5.metric("Beta", f"{risk['beta']:.2f}" if risk["beta"] else "N/A")
    c6.metric("Risk Score", f"{risk['risk_score']}/100")


if __name__ == "__main__":
    render()

