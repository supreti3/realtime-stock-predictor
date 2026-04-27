from __future__ import annotations

import pandas as pd
import streamlit as st

from components import api_client
from components.charts import heatmap, multi_line_chart


def render() -> None:
    st.header("Stock Comparison and News Sentiment")
    tab1, tab2 = st.tabs(["Compare 2-4 Stocks", "News and Sentiment"])

    with tab1:
        raw = st.text_input("Enter 2 to 4 comma-separated tickers", value="AAPL,MSFT,NVDA")
        tickers = [x.strip().upper() for x in raw.split(",") if x.strip()]
        if 2 <= len(tickers) <= 4:
            data = api_client.compare_stocks(tickers)
            prices = pd.DataFrame(data["price_rows"])
            normalized = pd.DataFrame(data["normalized_rows"])
            prices["Date"] = pd.to_datetime(prices["Date"])
            normalized["Date"] = pd.to_datetime(normalized["Date"])
            st.plotly_chart(multi_line_chart(prices, "Date", tickers, "Price Performance"), use_container_width=True)
            st.plotly_chart(multi_line_chart(normalized, "Date", tickers, "Normalized Returns"), use_container_width=True)
            stats_df = pd.DataFrame(
                {
                    "ticker": tickers,
                    "volatility": [data["volatility"].get(t) for t in tickers],
                    "average_return": [data["average_return"].get(t) for t in tickers],
                }
            )
            st.dataframe(stats_df, use_container_width=True)
            corr_df = pd.DataFrame(data["correlation"])
            st.plotly_chart(heatmap(corr_df, "Correlation Matrix"), use_container_width=True)
        else:
            st.warning("Please provide 2 to 4 tickers.")

    with tab2:
        ticker = st.text_input("Ticker for latest news", value="TSLA", key="news_ticker").upper().strip()
        if ticker:
            news = api_client.get_news(ticker)["articles"]
            if news:
                for item in news:
                    label = item["sentiment_label"]
                    emoji = "🟢" if label == "positive" else "🔴" if label == "negative" else "🟡"
                    st.markdown(f"{emoji} **{item['title']}**")
                    st.caption(f"{item.get('publisher', 'Unknown')} | Sentiment: {label}")
                    if item.get("link"):
                        st.markdown(f"[Open article]({item['link']})")
            else:
                st.info("No recent news found from free sources.")


if __name__ == "__main__":
    render()

