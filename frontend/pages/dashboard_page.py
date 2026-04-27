from __future__ import annotations

import pandas as pd
import streamlit as st

from components import api_client
from components.charts import line_chart


def render() -> None:
    st.title("AI Stock Intelligence Platform")
    st.caption("Educational analytics only. Not financial advice.")
    strategy_mode = st.selectbox("Strategy Mode", options=["Conservative", "Balanced", "Aggressive"], index=1)
    ticker = st.text_input("Search ticker", value="AAPL").upper().strip()
    if not ticker:
        return
    with st.spinner("Loading stock summary..."):
        summary = api_client.get_summary(ticker)
        history_resp = api_client.get_history(ticker, period="1y")
        insight_payload = api_client.get_insights_with_mode(ticker, strategy_mode=strategy_mode)
        insights = insight_payload["insights"]
        signal = insight_payload["signal"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${summary['current_price']:.2f}" if summary["current_price"] else "N/A")
    c2.metric("Daily Change", f"{summary['daily_change']:.2f}%" if summary["daily_change"] is not None else "N/A")
    c3.metric("Previous Close", f"${summary['previous_close']:.2f}" if summary["previous_close"] else "N/A")
    c4.metric("Open", f"${summary['open']:.2f}" if summary["open"] else "N/A")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Volume", f"{summary['volume']:,.0f}" if summary["volume"] else "N/A")
    c6.metric("Market Cap", f"{summary['market_cap']:,.0f}" if summary["market_cap"] else "N/A")
    c7.metric("52W High", f"${summary['52_week_high']:.2f}" if summary["52_week_high"] else "N/A")
    c8.metric("52W Low", f"${summary['52_week_low']:.2f}" if summary["52_week_low"] else "N/A")

    c9, c10 = st.columns(2)
    c9.metric("PE Ratio", f"{summary['pe_ratio']:.2f}" if summary["pe_ratio"] else "N/A")
    c10.metric("Dividend Yield", f"{summary['dividend_yield']:.2f}%" if summary["dividend_yield"] else "N/A")

    history_df = pd.DataFrame(history_resp["rows"])
    if not history_df.empty:
        history_df["Date"] = pd.to_datetime(history_df["Date"])
        st.plotly_chart(line_chart(history_df, "Date", "Close", f"{ticker} Price Trend"), use_container_width=True)

    st.subheader("Rule-Based AI Insight")
    st.info(insights["beginner_explanation"])
    if insights["supporting_signals"]:
        for insight_reason in insights["supporting_signals"]:
            st.write(f"- {insight_reason}")
    st.subheader("Buy / Hold / Sell Signal")
    st.warning(
        f"{signal['signal']} ({signal['confidence']}% confidence) - Mode: {signal['strategy_mode']}. "
        f"{signal['disclaimer']}"
    )
    for reason in signal["reasons"]:
        st.write(f"- {reason}")


if __name__ == "__main__":
    render()

