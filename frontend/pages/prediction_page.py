from __future__ import annotations

import pandas as pd
import streamlit as st

from components import api_client
from components.charts import line_chart, multi_line_chart


def render() -> None:
    st.header("Advanced Prediction Engine")
    ticker = st.text_input("Ticker for forecasting", value="NVDA").upper().strip()
    horizon = st.selectbox("Forecast horizon", options=[1, 7, 30, 90], index=1)

    if not ticker:
        return

    with st.spinner("Training and comparing models..."):
        data = api_client.get_prediction(ticker, horizon)

    st.success(f"Best model: {data['best_model']}")
    if data.get("fallback_used"):
        st.caption("Optional heavy models unavailable; fallback to sklearn models applied where needed.")

    metrics_df = pd.DataFrame(
        [
            {"Model": m["model"], "MAE": m["mae"], "RMSE": m["rmse"], "MAPE": m["mape"], "R2": m["r2"]}
            for m in data["models"]
        ]
    )
    st.dataframe(metrics_df, use_container_width=True)

    best_model = data["models"][0]
    forecast_df = pd.DataFrame(best_model["forecast"])
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    st.plotly_chart(line_chart(forecast_df, "date", "predicted_close", f"{ticker} Forecast ({best_model['model']})"), use_container_width=True)
    if {"lower_95", "upper_95"}.issubset(forecast_df.columns):
        st.caption("95% confidence interval shown as likely prediction range.")
        st.dataframe(
            forecast_df[["date", "predicted_close", "lower_95", "upper_95"]].rename(
                columns={"predicted_close": "expected_price"}
            ),
            use_container_width=True,
        )
    st.dataframe(forecast_df, use_container_width=True)

    pvsa_df = pd.DataFrame(best_model["predicted_vs_actual"])
    if not pvsa_df.empty:
        pvsa_df["idx"] = range(1, len(pvsa_df) + 1)
        st.plotly_chart(multi_line_chart(pvsa_df, "idx", ["actual", "predicted"], "Predicted vs Actual"), use_container_width=True)


if __name__ == "__main__":
    render()

