from __future__ import annotations

import pandas as pd
import streamlit as st

from components import api_client
from components.charts import line_chart, multi_line_chart


def render() -> None:
    st.header("Backtesting, Explainability, Scenarios, and Anomalies")
    ticker = st.text_input("Ticker for quant lab", value="AAPL").upper().strip()
    model = st.selectbox("Model", ["RandomForest", "Linear"], index=0)
    investment = st.number_input("Scenario investment amount ($)", min_value=100.0, value=5000.0, step=100.0)
    if not ticker:
        return

    with st.spinner("Running backtest and diagnostics..."):
        backtest = api_client.get_backtest(ticker, model=model)
        explain = api_client.get_explainability(ticker, model=model)["explainability"]
        anomalies = api_client.get_anomalies(ticker)["anomalies"]
        summary = api_client.get_summary(ticker)
        report = api_client.get_report(ticker)["report_html"]

    m = backtest["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Backtest Accuracy", f"{m['accuracy']:.2f}%")
    c2.metric("MAE", f"{m['mae']:.3f}")
    c3.metric("RMSE", f"{m['rmse']:.3f}")
    c4.metric("MAPE", f"{m['mape']:.2f}%")
    c5.metric("Simulated P/L", f"{backtest['profit_loss_simulation_pct']:.2f}%")

    pvsa = pd.DataFrame(backtest["predicted_vs_actual"])
    if not pvsa.empty:
        pvsa["date"] = pd.to_datetime(pvsa["date"])
        st.plotly_chart(
            multi_line_chart(pvsa, "date", ["actual_next_close", "predicted_next_close"], "Backtest: Predicted vs Actual"),
            use_container_width=True,
        )

    st.subheader("Explainable AI")
    feat_df = pd.DataFrame(explain["features"])
    st.caption(explain["explanation"])
    if not feat_df.empty:
        st.dataframe(feat_df, use_container_width=True)
        st.plotly_chart(line_chart(feat_df, "feature", "value", "Feature Impact"), use_container_width=True)

    st.subheader("Scenario Simulator")
    current_price = summary.get("current_price") or 0
    scenario_rows = []
    for move in [5, 10, 20, -5, -10, -20]:
        future_value = investment * (1 + move / 100)
        pnl = future_value - investment
        scenario_rows.append({"scenario": f"{move:+d}%", "future_value": future_value, "profit_loss": pnl})
    scenario_df = pd.DataFrame(scenario_rows)
    st.write(f"Current price snapshot for {ticker}: ${current_price:.2f}")
    st.dataframe(scenario_df, use_container_width=True)

    st.subheader("Anomaly Detection")
    anomaly_df = pd.DataFrame(anomalies)
    if not anomaly_df.empty:
        anomaly_df["Date"] = pd.to_datetime(anomaly_df["Date"])
        st.dataframe(anomaly_df, use_container_width=True)
        st.plotly_chart(line_chart(anomaly_df, "Date", "Close", "Detected Abnormal Price Days"), use_container_width=True)
    else:
        st.info("No major anomalies detected in the selected window.")

    st.subheader("Downloadable Report")
    st.download_button(
        "Download HTML Report",
        data=report,
        file_name=f"{ticker}_stock_intelligence_report.html",
        mime="text/html",
    )


if __name__ == "__main__":
    render()

