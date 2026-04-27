from __future__ import annotations

import pandas as pd
import streamlit as st

from components import api_client
from components.charts import allocation_pie


def render() -> None:
    st.header("Portfolio and Watchlist")
    tab1, tab2 = st.tabs(["Portfolio", "Watchlist"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker", value="AAPL", key="pf_ticker").upper()
        shares = c2.number_input("Shares", min_value=0.0, value=10.0, step=1.0, key="pf_shares")
        purchase_price = c3.number_input("Purchase Price", min_value=0.01, value=180.0, step=1.0, key="pf_price")
        col_a, col_b = st.columns(2)
        if col_a.button("Add Position"):
            api_client.add_portfolio_position(ticker, shares, purchase_price)
        if col_b.button("Clear Portfolio"):
            api_client.clear_portfolio()

        portfolio = api_client.get_portfolio()["positions"]
        rows = []
        total_value = 0.0
        for pos in portfolio:
            summary = api_client.get_summary(pos["ticker"])
            curr = summary["current_price"] or 0.0
            current_value = curr * pos["shares"]
            invested = pos["shares"] * pos["purchase_price"]
            pnl = current_value - invested
            pct = (pnl / invested * 100) if invested else 0.0
            total_value += current_value
            rows.append(
                {
                    "ticker": pos["ticker"],
                    "shares": pos["shares"],
                    "purchase_price": pos["purchase_price"],
                    "current_price": curr,
                    "current_value": current_value,
                    "profit_loss": pnl,
                    "return_pct": pct,
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            st.metric("Total Portfolio Value", f"${total_value:,.2f}")
            st.dataframe(df, use_container_width=True)
            st.plotly_chart(allocation_pie(df, "ticker", "current_value", "Portfolio Allocation"), use_container_width=True)
        else:
            st.info("No positions yet.")

    with tab2:
        wl_ticker = st.text_input("Add ticker to watchlist", value="NVDA", key="wl_ticker").upper()
        c1, c2 = st.columns(2)
        if c1.button("Add to Watchlist"):
            api_client.add_watchlist(wl_ticker)
        watchlist = api_client.get_watchlist()["watchlist"]
        if watchlist:
            remove_ticker = c2.selectbox("Remove ticker", options=watchlist)
            if c2.button("Remove Selected"):
                api_client.delete_watchlist(remove_ticker)
            items = []
            for t in api_client.get_watchlist()["watchlist"]:
                s = api_client.get_summary(t)
                trend = "Bullish" if (s["daily_change"] or 0) > 0 else "Bearish"
                items.append({"ticker": t, "price": s["current_price"], "daily_change_pct": s["daily_change"], "trend": trend})
            st.dataframe(pd.DataFrame(items), use_container_width=True)
        else:
            st.info("Watchlist is empty.")


if __name__ == "__main__":
    render()

