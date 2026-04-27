from __future__ import annotations

from typing import Any

import requests


BASE_URL = "http://127.0.0.1:8000"


def _req(method: str, endpoint: str, **kwargs: Any) -> Any:
    url = f"{BASE_URL}{endpoint}"
    response = requests.request(method, url, timeout=120, **kwargs)
    response.raise_for_status()
    return response.json()


def get_summary(ticker: str) -> dict[str, Any]:
    return _req("GET", f"/stock/{ticker}/summary")


def get_history(ticker: str, period: str = "2y") -> dict[str, Any]:
    return _req("GET", f"/stock/{ticker}/history", params={"period": period, "interval": "1d"})


def get_prediction(ticker: str, horizon_days: int) -> dict[str, Any]:
    return _req("POST", f"/stock/{ticker}/predict", json={"horizon_days": horizon_days})


def get_indicators(ticker: str) -> dict[str, Any]:
    return _req("GET", f"/stock/{ticker}/indicators")


def get_insights(ticker: str) -> dict[str, Any]:
    return _req("GET", f"/stock/{ticker}/insights")


def get_insights_with_mode(ticker: str, strategy_mode: str) -> dict[str, Any]:
    return _req("GET", f"/stock/{ticker}/insights", params={"strategy_mode": strategy_mode})


def get_risk(ticker: str) -> dict[str, Any]:
    return _req("GET", f"/stock/{ticker}/risk")


def get_news(ticker: str) -> dict[str, Any]:
    return _req("GET", f"/stock/{ticker}/news")


def get_backtest(ticker: str, model: str = "RandomForest") -> dict[str, Any]:
    return _req("POST", f"/stock/{ticker}/backtest", json={"model": model})


def get_explainability(ticker: str, model: str = "RandomForest") -> dict[str, Any]:
    return _req("POST", f"/stock/{ticker}/explain", json={"model": model})


def get_anomalies(ticker: str) -> dict[str, Any]:
    return _req("GET", f"/stock/{ticker}/anomalies")


def get_report(ticker: str) -> dict[str, Any]:
    return _req("GET", f"/stock/{ticker}/report")


def get_portfolio() -> dict[str, Any]:
    return _req("GET", "/portfolio")


def add_portfolio_position(ticker: str, shares: float, purchase_price: float) -> dict[str, Any]:
    return _req(
        "POST",
        "/portfolio",
        json={"ticker": ticker, "shares": shares, "purchase_price": purchase_price},
    )


def clear_portfolio() -> dict[str, Any]:
    return _req("DELETE", "/portfolio")


def get_watchlist() -> dict[str, Any]:
    return _req("GET", "/watchlist")


def add_watchlist(ticker: str) -> dict[str, Any]:
    return _req("POST", "/watchlist", json={"ticker": ticker})


def delete_watchlist(ticker: str) -> dict[str, Any]:
    return _req("DELETE", f"/watchlist/{ticker}")


def compare_stocks(tickers: list[str]) -> dict[str, Any]:
    return _req("GET", "/compare", params={"tickers": ",".join(tickers)})

