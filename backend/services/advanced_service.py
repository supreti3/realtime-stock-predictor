from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = ["ret_1", "ret_5", "ma_7", "ma_30", "ema_12", "ema_26", "vol_14"]


def build_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy().sort_values("Date")
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ma_7"] = df["Close"].rolling(7).mean()
    df["ma_30"] = df["Close"].rolling(30).mean()
    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["vol_14"] = df["Close"].pct_change().rolling(14).std()
    df["target"] = df["Close"].shift(-1)
    return df.dropna()


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    r2 = float(r2_score(y_true, y_pred))
    accuracy = float(max(0.0, 100 - mape))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "accuracy": accuracy}


def run_backtest(history: pd.DataFrame, model_name: str = "RandomForest") -> dict[str, Any]:
    df = build_feature_frame(history)
    if len(df) < 120:
        raise ValueError("Need more historical data for backtesting.")

    x = df[FEATURES]
    y = df["target"]
    split = int(len(df) * 0.8)
    x_train, y_train = x.iloc[:split], y.iloc[:split]
    x_test, y_test = x.iloc[split:], y.iloc[split:]
    close_today = df["Close"].iloc[split:]
    dates = df["Date"].iloc[split:]

    if model_name.lower() == "linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=12)

    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    stats = _metrics(y_test, pred)

    predicted_return = (pred - close_today.values) / (close_today.values + 1e-9)
    strategy_ret = np.where(predicted_return > 0, y_test.values / close_today.values - 1, 0)
    cumulative = np.cumprod(1 + strategy_ret)
    pnl_pct = float((cumulative[-1] - 1) * 100) if len(cumulative) else 0.0

    rows = []
    for d, actual, p, c in zip(dates, y_test.values, pred, close_today.values):
        rows.append(
            {
                "date": pd.to_datetime(d).strftime("%Y-%m-%d"),
                "close_today": float(c),
                "actual_next_close": float(actual),
                "predicted_next_close": float(p),
            }
        )

    return {
        "model": model_name,
        "metrics": stats,
        "profit_loss_simulation_pct": pnl_pct,
        "predicted_vs_actual": rows[-180:],
    }


def detect_anomalies(history: pd.DataFrame) -> dict[str, Any]:
    df = history.copy().sort_values("Date")
    df["return"] = df["Close"].pct_change()
    df["vol_z"] = (df["Volume"] - df["Volume"].rolling(30).mean()) / (df["Volume"].rolling(30).std() + 1e-9)
    df["ret_z"] = (df["return"] - df["return"].rolling(30).mean()) / (df["return"].rolling(30).std() + 1e-9)
    anomalies = df[(df["vol_z"].abs() > 2.5) | (df["ret_z"].abs() > 2.5)].copy()
    anomalies["Date"] = pd.to_datetime(anomalies["Date"]).dt.strftime("%Y-%m-%d")
    return {
        "anomalies": anomalies[["Date", "Close", "Volume", "return", "vol_z", "ret_z"]].tail(60).to_dict(orient="records")
    }


def generate_signal(indicator_latest: dict[str, Any], trend: str, strategy_mode: str = "Balanced") -> dict[str, Any]:
    rsi = indicator_latest.get("RSI")
    macd = indicator_latest.get("MACD")
    macd_signal = indicator_latest.get("MACD_SIGNAL")
    sma_7 = indicator_latest.get("SMA_7")
    sma_30 = indicator_latest.get("SMA_30")
    vol = indicator_latest.get("VOLATILITY_30D")

    score = 0
    reasons: list[str] = []
    if rsi is not None:
        if rsi < 35:
            score += 1
            reasons.append("RSI indicates potentially oversold conditions.")
        elif rsi > 70:
            score -= 1
            reasons.append("RSI indicates potentially overbought conditions.")
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            score += 1
            reasons.append("MACD is above signal line (bullish momentum).")
        else:
            score -= 1
            reasons.append("MACD is below signal line (bearish momentum).")
    if sma_7 is not None and sma_30 is not None:
        if sma_7 > sma_30:
            score += 1
            reasons.append("Short-term SMA is above long-term SMA.")
        else:
            score -= 1
            reasons.append("Short-term SMA is below long-term SMA.")
    if vol is not None and vol > 0.45:
        score -= 1
        reasons.append("High volatility reduces confidence.")
    if trend == "bullish":
        score += 1
    elif trend == "bearish":
        score -= 1

    mode = strategy_mode.lower()
    buy_cutoff, sell_cutoff = (1, -1)
    if mode == "conservative":
        buy_cutoff, sell_cutoff = (2, -2)
    elif mode == "aggressive":
        buy_cutoff, sell_cutoff = (0, 0)

    if score >= buy_cutoff:
        signal = "BUY"
    elif score <= sell_cutoff:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = int(np.clip(50 + abs(score) * 12, 50, 95))
    return {
        "signal": signal,
        "confidence": confidence,
        "strategy_mode": strategy_mode,
        "reasons": reasons,
        "disclaimer": "This signal is educational and not financial advice.",
    }


def explain_model(history: pd.DataFrame, model_name: str = "RandomForest") -> dict[str, Any]:
    df = build_feature_frame(history)
    x = df[FEATURES]
    y = df["target"]
    split = int(len(df) * 0.8)
    x_train, y_train = x.iloc[:split], y.iloc[:split]

    if model_name.lower() == "linear":
        model = LinearRegression()
        model.fit(x_train, y_train)
        coeffs = [{"feature": f, "value": float(v)} for f, v in zip(FEATURES, model.coef_)]
        coeffs = sorted(coeffs, key=lambda z: abs(z["value"]), reverse=True)
        return {
            "model": "LinearRegression",
            "type": "coefficients",
            "explanation": "Positive coefficients push predictions higher; negative push lower.",
            "features": coeffs,
        }

    model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=12)
    model.fit(x_train, y_train)
    imp = [{"feature": f, "value": float(v)} for f, v in zip(FEATURES, model.feature_importances_)]
    imp = sorted(imp, key=lambda z: z["value"], reverse=True)
    return {
        "model": "RandomForest",
        "type": "feature_importance",
        "explanation": "Higher importance means a bigger impact on model decisions.",
        "features": imp,
    }


def build_html_report(
    ticker: str,
    summary: dict[str, Any],
    insights: dict[str, Any],
    risk: dict[str, Any],
    prediction: dict[str, Any],
) -> str:
    best_model = prediction.get("best_model", "N/A")
    model_rows = "".join(
        [
            f"<tr><td>{m['model']}</td><td>{m['mae']:.3f}</td><td>{m['rmse']:.3f}</td><td>{m['mape']:.2f}%</td><td>{m['r2']:.3f}</td></tr>"
            for m in prediction.get("models", [])
        ]
    )
    return f"""
<html>
<head>
<title>{ticker} Stock Intelligence Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
h1, h2 {{ color: #0b3a75; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
</style>
</head>
<body>
<h1>{ticker} Stock Intelligence Report</h1>
<p>Educational analytics only. Not financial advice.</p>

<h2>Stock Summary</h2>
<p>Current Price: {summary.get('current_price')}</p>
<p>Daily Change (%): {summary.get('daily_change')}</p>
<p>Market Cap: {summary.get('market_cap')}</p>

<h2>AI Insights</h2>
<p>{insights.get('beginner_explanation')}</p>

<h2>Risk Analysis</h2>
<p>Volatility: {risk.get('volatility')}</p>
<p>Max Drawdown: {risk.get('max_drawdown')}</p>
<p>Sharpe Ratio: {risk.get('sharpe_ratio')}</p>
<p>Risk Score: {risk.get('risk_score')}/100</p>

<h2>Prediction Results</h2>
<p>Best Model: {best_model}</p>
<table>
<tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>R2</th></tr>
{model_rows}
</table>

<h2>Disclaimer</h2>
<p>This report is generated for educational demonstration purposes only and should not be used as investment advice.</p>
</body>
</html>
"""

