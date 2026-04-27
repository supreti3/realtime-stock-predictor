from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    import tensorflow as tf
except Exception:
    tf = None


@dataclass
class ForecastResult:
    model: str
    mae: float
    rmse: float
    mape: float
    r2: float
    predicted_values: list[float]
    forecast_dates: list[str]
    predicted_vs_actual: list[dict[str, float]]
    note: str = ""


def _create_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("Date")
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_5"] = out["Close"].pct_change(5)
    out["ma_7"] = out["Close"].rolling(7).mean()
    out["ma_30"] = out["Close"].rolling(30).mean()
    out["ema_12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["Close"].ewm(span=26, adjust=False).mean()
    out["vol_14"] = out["Close"].pct_change().rolling(14).std()
    out["target"] = out["Close"].shift(-1)
    return out.dropna()


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    features = ["ret_1", "ret_5", "ma_7", "ma_30", "ema_12", "ema_26", "vol_14"]
    X = df[features]
    y = df["target"]
    split = int(len(df) * 0.8)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:], features


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": rmse, "mape": mape, "r2": float(r2)}


def _iterative_forecast(model: Any, last_row: pd.Series, horizon: int) -> list[float]:
    predictions: list[float] = []
    current = last_row.copy()
    for _ in range(horizon):
        pred = float(model.predict(current.values.reshape(1, -1))[0])
        predictions.append(pred)
        # Keep this deterministic and lightweight for free/demo usage.
        current["ret_1"] = (pred - current["ma_7"]) / (current["ma_7"] + 1e-9)
        current["ret_5"] = current["ret_1"] * 0.8
        current["ma_7"] = (current["ma_7"] * 6 + pred) / 7
        current["ma_30"] = (current["ma_30"] * 29 + pred) / 30
        current["ema_12"] = (pred * (2 / 13)) + current["ema_12"] * (1 - (2 / 13))
        current["ema_26"] = (pred * (2 / 27)) + current["ema_26"] * (1 - (2 / 27))
        current["vol_14"] = max(current["vol_14"] * 0.95, 1e-6)
    return predictions


def _run_sklearn_models(df: pd.DataFrame, horizon: int) -> list[ForecastResult]:
    x_train, y_train, x_test, y_test, features = _split_xy(df)
    model_defs = [
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(n_estimators=250, random_state=42, max_depth=10)),
    ]
    if XGBRegressor is not None:
        model_defs.append(
            (
                "XGBoost",
                XGBRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=5,
                    objective="reg:squarederror",
                    random_state=42,
                ),
            )
        )

    results: list[ForecastResult] = []
    last_row = df[features].iloc[-1]
    future_dates = pd.bdate_range(df["Date"].iloc[-1], periods=horizon + 1)[1:]
    for name, model in model_defs:
        model.fit(x_train, y_train)
        test_preds = model.predict(x_test)
        stats = _metrics(y_test, test_preds)
        forecast = _iterative_forecast(model, last_row, horizon)
        pvsa = [
            {"actual": float(a), "predicted": float(p)}
            for a, p in zip(y_test.tail(30).values, test_preds[-30:])
        ]
        results.append(
            ForecastResult(
                model=name,
                mae=stats["mae"],
                rmse=stats["rmse"],
                mape=stats["mape"],
                r2=stats["r2"],
                predicted_values=forecast,
                forecast_dates=[d.strftime("%Y-%m-%d") for d in future_dates],
                predicted_vs_actual=pvsa,
                note="",
            )
        )
    return results


def _run_prophet(df: pd.DataFrame, horizon: int) -> ForecastResult | None:
    if Prophet is None:
        return None
    data = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    # Prophet does not accept timezone-aware datetimes.
    data["ds"] = pd.to_datetime(data["ds"]).dt.tz_localize(None)
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=horizon, freq="B")
    forecast = model.predict(future)
    pred = forecast["yhat"].tail(horizon).tolist()
    merged = forecast.merge(data, on="ds", how="inner").tail(30)
    y_true = merged["y"]
    y_pred = merged["yhat"]
    stats = _metrics(y_true, y_pred.to_numpy())
    pvsa = [{"actual": float(a), "predicted": float(p)} for a, p in zip(y_true.values, y_pred.values)]
    return ForecastResult(
        model="Prophet",
        mae=stats["mae"],
        rmse=stats["rmse"],
        mape=stats["mape"],
        r2=stats["r2"],
        predicted_values=[float(x) for x in pred],
        forecast_dates=[d.strftime("%Y-%m-%d") for d in future["ds"].tail(horizon)],
        predicted_vs_actual=pvsa,
        note="",
    )


def _run_lstm(df: pd.DataFrame, horizon: int) -> ForecastResult | None:
    if tf is None:
        return None
    prices = df["Close"].values.astype("float32")
    seq = 30
    if len(prices) <= seq + 20:
        return None
    x, y = [], []
    for i in range(seq, len(prices)):
        x.append(prices[i - seq : i])
        y.append(prices[i])
    x = np.array(x).reshape(-1, seq, 1)
    y = np.array(y)
    split = int(len(x) * 0.8)
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(seq, 1)),
            tf.keras.layers.LSTM(24, return_sequences=False),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=8, batch_size=16, verbose=0)
    test_pred = model.predict(x_test, verbose=0).flatten()
    stats = _metrics(pd.Series(y_test), test_pred)

    seq_window = prices[-seq:].copy()
    future = []
    for _ in range(horizon):
        next_val = float(model.predict(seq_window.reshape(1, seq, 1), verbose=0)[0][0])
        future.append(next_val)
        seq_window = np.append(seq_window[1:], next_val)
    future_dates = pd.bdate_range(df["Date"].iloc[-1], periods=horizon + 1)[1:]
    pvsa = [{"actual": float(a), "predicted": float(p)} for a, p in zip(y_test[-30:], test_pred[-30:])]
    return ForecastResult(
        model="LSTM",
        mae=stats["mae"],
        rmse=stats["rmse"],
        mape=stats["mape"],
        r2=stats["r2"],
        predicted_values=[float(x) for x in future],
        forecast_dates=[d.strftime("%Y-%m-%d") for d in future_dates],
        predicted_vs_actual=pvsa,
        note="Tiny LSTM for free-tier friendly demos.",
    )


def run_predictions(history: pd.DataFrame, horizon: int) -> dict[str, Any]:
    df = _create_features(history)
    if len(df) < 80:
        raise ValueError("Not enough historical data to train models.")

    results = _run_sklearn_models(df, horizon)
    enable_heavy = os.getenv("ENABLE_HEAVY_MODELS", "0") == "1"
    if enable_heavy:
        try:
            prophet_result = _run_prophet(df, horizon)
        except Exception:
            prophet_result = None
        if prophet_result:
            results.append(prophet_result)
        try:
            lstm_result = _run_lstm(df, horizon)
        except Exception:
            lstm_result = None
        if lstm_result:
            results.append(lstm_result)

    ranked = sorted(results, key=lambda r: r.rmse)
    best = ranked[0]
    best_residual_std = float(np.std([x["actual"] - x["predicted"] for x in best.predicted_vs_actual])) if best.predicted_vs_actual else 0.0
    return {
        "best_model": best.model,
        "horizon_days": horizon,
        "models": [
            {
                "model": r.model,
                "mae": r.mae,
                "rmse": r.rmse,
                "mape": r.mape,
                "r2": r.r2,
                "forecast": [
                    {
                        "date": d,
                        "predicted_close": v,
                        "lower_95": float(v - 1.96 * best_residual_std),
                        "upper_95": float(v + 1.96 * best_residual_std),
                    }
                    for d, v in zip(r.forecast_dates, r.predicted_values)
                ],
                "predicted_vs_actual": r.predicted_vs_actual,
                "note": r.note,
            }
            for r in ranked
        ],
        "fallback_used": (not enable_heavy) or XGBRegressor is None or Prophet is None or tf is None,
    }

