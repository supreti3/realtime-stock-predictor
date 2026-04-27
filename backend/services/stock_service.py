from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_PERIOD = "2y"


@dataclass
class StockSummary:
    ticker: str
    current_price: float | None
    daily_change: float | None
    previous_close: float | None
    open_price: float | None
    volume: float | None
    market_cap: float | None
    week_52_high: float | None
    week_52_low: float | None
    pe_ratio: float | None
    dividend_yield: float | None
    beta: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "current_price": self.current_price,
            "daily_change": self.daily_change,
            "previous_close": self.previous_close,
            "open": self.open_price,
            "volume": self.volume,
            "market_cap": self.market_cap,
            "52_week_high": self.week_52_high,
            "52_week_low": self.week_52_low,
            "pe_ratio": self.pe_ratio,
            "dividend_yield": self.dividend_yield,
            "beta": self.beta,
        }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
        if np.isnan(num):
            return None
        return num
    except Exception:
        return None


def get_history(ticker: str, period: str = DEFAULT_PERIOD, interval: str = "1d") -> pd.DataFrame:
    stock = yf.Ticker(ticker.upper())
    history = stock.history(period=period, interval=interval, auto_adjust=False)
    if history.empty:
        return history
    history = history.reset_index()
    history.columns = [str(col) for col in history.columns]
    return history


def get_stock_summary(ticker: str) -> dict[str, Any]:
    stock = yf.Ticker(ticker.upper())
    info = stock.info or {}
    fast_info = getattr(stock, "fast_info", {}) or {}
    hist_5d = stock.history(period="5d", interval="1d")

    current_price = _safe_float(
        fast_info.get("last_price")
        or info.get("currentPrice")
        or info.get("regularMarketPrice")
    )
    previous_close = _safe_float(
        fast_info.get("previous_close")
        or info.get("previousClose")
        or info.get("regularMarketPreviousClose")
    )
    open_price = _safe_float(fast_info.get("open") or info.get("open"))
    volume = _safe_float(fast_info.get("last_volume") or info.get("volume"))
    market_cap = _safe_float(fast_info.get("market_cap") or info.get("marketCap"))
    week_52_high = _safe_float(fast_info.get("year_high") or info.get("fiftyTwoWeekHigh"))
    week_52_low = _safe_float(fast_info.get("year_low") or info.get("fiftyTwoWeekLow"))
    pe_ratio = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
    beta = _safe_float(info.get("beta"))

    dividend_yield_raw = _safe_float(info.get("dividendYield"))
    dividend_rate = _safe_float(info.get("dividendRate"))
    dividend_yield = None
    if dividend_yield_raw is not None:
        # yFinance can return yield as a fraction (0.004) or already-percent-like value (0.38).
        inferred_percent = None
        if dividend_rate is not None and current_price not in (None, 0):
            inferred_percent = (dividend_rate / current_price) * 100
        if inferred_percent is not None:
            as_percent = dividend_yield_raw
            as_fraction_percent = dividend_yield_raw * 100
            dividend_yield = (
                as_percent
                if abs(as_percent - inferred_percent) <= abs(as_fraction_percent - inferred_percent)
                else as_fraction_percent
            )
        else:
            dividend_yield = dividend_yield_raw * 100 if dividend_yield_raw < 1 else dividend_yield_raw

    daily_change = None
    if current_price is not None and previous_close is not None and previous_close != 0:
        daily_change = ((current_price - previous_close) / previous_close) * 100
    elif len(hist_5d) >= 2:
        p0 = _safe_float(hist_5d["Close"].iloc[-2])
        p1 = _safe_float(hist_5d["Close"].iloc[-1])
        if p0 and p1:
            daily_change = ((p1 - p0) / p0) * 100

    summary = StockSummary(
        ticker=ticker.upper(),
        current_price=current_price,
        daily_change=daily_change,
        previous_close=previous_close,
        open_price=open_price,
        volume=volume,
        market_cap=market_cap,
        week_52_high=week_52_high,
        week_52_low=week_52_low,
        pe_ratio=pe_ratio,
        dividend_yield=dividend_yield,
        beta=beta,
    )
    return summary.to_dict()
