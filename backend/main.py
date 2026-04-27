from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.services.indicator_service import compute_indicators
from backend.services.insight_service import generate_insights
from backend.services.advanced_service import (
    build_html_report,
    detect_anomalies,
    explain_model,
    generate_signal,
    run_backtest,
)
from backend.services.portfolio_service import (
    add_position,
    add_watchlist_ticker,
    clear_portfolio,
    list_portfolio,
    list_watchlist,
    remove_watchlist_ticker,
)
from backend.services.prediction_service import run_predictions
from backend.services.stock_service import get_history, get_stock_summary

app = FastAPI(title="AI Stock Intelligence Platform", version="2.0.0")


class PredictRequest(BaseModel):
    horizon_days: int = Field(default=7, ge=1, le=90)


class PositionRequest(BaseModel):
    ticker: str
    shares: float = Field(gt=0)
    purchase_price: float = Field(gt=0)


class WatchlistRequest(BaseModel):
    ticker: str


class BacktestRequest(BaseModel):
    model: str = "RandomForest"


class ExplainRequest(BaseModel):
    model: str = "RandomForest"


@app.get("/stock/{ticker}/summary")
def stock_summary(ticker: str) -> dict[str, Any]:
    data = get_stock_summary(ticker)
    if data["current_price"] is None:
        raise HTTPException(status_code=404, detail="Ticker not found or unavailable.")
    return data


@app.get("/stock/{ticker}/history")
def stock_history(ticker: str, period: str = "2y", interval: str = "1d") -> dict[str, Any]:
    history = get_history(ticker, period=period, interval=interval)
    if history.empty:
        raise HTTPException(status_code=404, detail="No historical data available.")
    history["Date"] = pd.to_datetime(history["Date"]).dt.strftime("%Y-%m-%d")
    return {"ticker": ticker.upper(), "rows": history.to_dict(orient="records")}


@app.post("/stock/{ticker}/predict")
def predict_stock(ticker: str, payload: PredictRequest) -> dict[str, Any]:
    history = get_history(ticker, period="3y", interval="1d")
    if history.empty:
        raise HTTPException(status_code=404, detail="No historical data available for prediction.")
    try:
        result = run_predictions(history, payload.horizon_days)
        return {"ticker": ticker.upper(), **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/stock/{ticker}/indicators")
def stock_indicators(ticker: str) -> dict[str, Any]:
    history = get_history(ticker, period="2y", interval="1d")
    if history.empty:
        raise HTTPException(status_code=404, detail="No historical data available.")
    ind = compute_indicators(history)
    latest = ind.iloc[-1].to_dict()
    safe_latest = {k: (None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in latest.items()}
    ind = ind.copy()
    ind["Date"] = pd.to_datetime(ind["Date"]).dt.strftime("%Y-%m-%d")
    return {
        "ticker": ticker.upper(),
        "latest": safe_latest,
        "series": ind.tail(120).to_dict(orient="records"),
    }


@app.get("/stock/{ticker}/insights")
def stock_insights(ticker: str, strategy_mode: str = "Balanced") -> dict[str, Any]:
    history = get_history(ticker, period="2y", interval="1d")
    if history.empty:
        raise HTTPException(status_code=404, detail="No historical data available.")
    ind = compute_indicators(history)
    insights = generate_insights(ind)
    latest = ind.iloc[-1].to_dict()
    signal = generate_signal(latest, insights["trend_direction"], strategy_mode=strategy_mode)
    return {"ticker": ticker.upper(), "insights": insights, "signal": signal}


@app.get("/stock/{ticker}/risk")
def stock_risk(ticker: str) -> dict[str, Any]:
    history = get_history(ticker, period="3y", interval="1d")
    if history.empty:
        raise HTTPException(status_code=404, detail="No historical data available.")
    close = history["Close"].astype(float)
    ret = close.pct_change().dropna()
    if ret.empty:
        raise HTTPException(status_code=400, detail="Not enough data for risk metrics.")

    volatility = float(ret.std() * np.sqrt(252))
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_drawdown = float(drawdown.min())
    sharpe = float((ret.mean() * 252) / ((ret.std() * np.sqrt(252)) + 1e-9))
    var_95 = float(np.percentile(ret, 5))
    beta = get_stock_summary(ticker).get("beta")

    risk_score = int(np.clip((volatility * 100) + (abs(max_drawdown) * 80) + (max(0, -sharpe) * 10), 0, 100))
    return {
        "ticker": ticker.upper(),
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "value_at_risk_95": var_95,
        "beta": beta,
        "risk_score": risk_score,
    }


@app.get("/stock/{ticker}/news")
def stock_news(ticker: str) -> dict[str, Any]:
    from yfinance import Ticker

    positive = {"growth", "beat", "strong", "profit", "upgrade"}
    negative = {"loss", "lawsuit", "decline", "weak", "downgrade"}
    articles = Ticker(ticker.upper()).news or []
    scored = []
    for item in articles[:20]:
        title = (item.get("title") or "").lower()
        pos = sum(1 for w in positive if w in title)
        neg = sum(1 for w in negative if w in title)
        score = pos - neg
        label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        scored.append(
            {
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "link": item.get("link"),
                "published": item.get("providerPublishTime"),
                "sentiment_score": score,
                "sentiment_label": label,
            }
        )
    return {"ticker": ticker.upper(), "articles": scored}


@app.post("/stock/{ticker}/backtest")
def stock_backtest(ticker: str, payload: BacktestRequest) -> dict[str, Any]:
    history = get_history(ticker, period="3y", interval="1d")
    if history.empty:
        raise HTTPException(status_code=404, detail="No historical data for backtest.")
    try:
        return {"ticker": ticker.upper(), **run_backtest(history, model_name=payload.model)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/stock/{ticker}/explain")
def stock_explain(ticker: str, payload: ExplainRequest) -> dict[str, Any]:
    history = get_history(ticker, period="3y", interval="1d")
    if history.empty:
        raise HTTPException(status_code=404, detail="No historical data for explainability.")
    return {"ticker": ticker.upper(), "explainability": explain_model(history, model_name=payload.model)}


@app.get("/stock/{ticker}/anomalies")
def stock_anomalies(ticker: str) -> dict[str, Any]:
    history = get_history(ticker, period="3y", interval="1d")
    if history.empty:
        raise HTTPException(status_code=404, detail="No historical data for anomalies.")
    return {"ticker": ticker.upper(), **detect_anomalies(history)}


@app.get("/stock/{ticker}/report")
def stock_report(ticker: str) -> dict[str, Any]:
    summary = stock_summary(ticker)
    insights_resp = stock_insights(ticker)
    risk = stock_risk(ticker)
    prediction = predict_stock(ticker, PredictRequest(horizon_days=30))
    report_html = build_html_report(ticker.upper(), summary, insights_resp["insights"], risk, prediction)
    return {"ticker": ticker.upper(), "report_html": report_html}


@app.get("/portfolio")
def get_portfolio() -> dict[str, Any]:
    return {"positions": list_portfolio()}


@app.post("/portfolio")
def post_portfolio(payload: PositionRequest) -> dict[str, Any]:
    positions = add_position(payload.ticker, payload.shares, payload.purchase_price)
    return {"positions": positions}


@app.delete("/portfolio")
def delete_portfolio() -> dict[str, Any]:
    return {"positions": clear_portfolio()}


@app.get("/watchlist")
def get_watchlist() -> dict[str, Any]:
    return {"watchlist": list_watchlist()}


@app.post("/watchlist")
def post_watchlist(payload: WatchlistRequest) -> dict[str, Any]:
    return {"watchlist": add_watchlist_ticker(payload.ticker)}


@app.delete("/watchlist/{ticker}")
def delete_watchlist(ticker: str) -> dict[str, Any]:
    return {"watchlist": remove_watchlist_ticker(ticker)}


@app.get("/compare")
def compare_stocks(tickers: str) -> dict[str, Any]:
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not (2 <= len(ticker_list) <= 4):
        raise HTTPException(status_code=400, detail="Provide 2 to 4 comma-separated tickers.")
    frames = []
    for t in ticker_list:
        h = get_history(t, period="1y", interval="1d")
        if h.empty:
            continue
        frames.append(h[["Date", "Close"]].rename(columns={"Close": t}))
    if not frames:
        raise HTTPException(status_code=404, detail="No data for provided tickers.")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="Date", how="inner")
    merged = merged.sort_values("Date")
    prices = merged.set_index("Date")
    returns = prices.pct_change().dropna()
    normalized = prices / prices.iloc[0]

    return {
        "tickers": ticker_list,
        "price_rows": (
            prices.reset_index().assign(Date=lambda x: pd.to_datetime(x["Date"]).dt.strftime("%Y-%m-%d")).to_dict(orient="records")
        ),
        "normalized_rows": (
            normalized.reset_index().assign(Date=lambda x: pd.to_datetime(x["Date"]).dt.strftime("%Y-%m-%d")).to_dict(orient="records")
        ),
        "volatility": (returns.std() * np.sqrt(252)).to_dict(),
        "average_return": (returns.mean() * 252).to_dict(),
        "correlation": returns.corr().to_dict(),
    }

