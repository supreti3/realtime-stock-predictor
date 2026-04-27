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


def _safe_get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    try:
        if hasattr(obj, "get"):
            return obj.get(key)  # type: ignore[call-arg]
    except Exception:
        pass
    try:
        return obj[key]  # type: ignore[index]
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
    try:
        info = stock.info or {}
    except Exception:
        info = {}
    try:
        fast_info = getattr(stock, "fast_info", {}) or {}
    except Exception:
        fast_info = {}
    try:
        hist_5d = stock.history(period="5d", interval="1d")
    except Exception:
        hist_5d = pd.DataFrame()
    if hist_5d.empty:
        try:
            hist_5d = yf.download(ticker.upper(), period="5d", interval="1d", progress=False, auto_adjust=False)
        except Exception:
            hist_5d = pd.DataFrame()

    current_price = _safe_float(
        _safe_get(fast_info, "last_price")
        or _safe_get(info, "currentPrice")
        or _safe_get(info, "regularMarketPrice")
    )
    previous_close = _safe_float(
        _safe_get(fast_info, "previous_close")
        or _safe_get(info, "previousClose")
        or _safe_get(info, "regularMarketPreviousClose")
    )
    open_price = _safe_float(_safe_get(fast_info, "open") or _safe_get(info, "open"))
    volume = _safe_float(_safe_get(fast_info, "last_volume") or _safe_get(info, "volume"))
    market_cap = _safe_float(_safe_get(fast_info, "market_cap") or _safe_get(info, "marketCap"))
    week_52_high = _safe_float(_safe_get(fast_info, "year_high") or _safe_get(info, "fiftyTwoWeekHigh"))
    week_52_low = _safe_float(_safe_get(fast_info, "year_low") or _safe_get(info, "fiftyTwoWeekLow"))
    pe_ratio = _safe_float(_safe_get(info, "trailingPE") or _safe_get(info, "forwardPE"))
    beta = _safe_float(_safe_get(info, "beta"))

    dividend_yield_raw = _safe_float(_safe_get(info, "dividendYield"))
    dividend_rate = _safe_float(_safe_get(info, "dividendRate"))
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
    if current_price is None and not hist_5d.empty:
        current_price = _safe_float(hist_5d["Close"].iloc[-1])
    if previous_close is None and len(hist_5d) >= 2:
        previous_close = _safe_float(hist_5d["Close"].iloc[-2])

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
