from __future__ import annotations

from typing import Any

from backend.utils.storage import PORTFOLIO_FILE, WATCHLIST_FILE, read_json, write_json


def list_portfolio() -> list[dict[str, Any]]:
    return read_json(PORTFOLIO_FILE, [])


def add_position(ticker: str, shares: float, purchase_price: float) -> list[dict[str, Any]]:
    positions = list_portfolio()
    positions.append(
        {
            "ticker": ticker.upper(),
            "shares": float(shares),
            "purchase_price": float(purchase_price),
        }
    )
    write_json(PORTFOLIO_FILE, positions)
    return positions


def clear_portfolio() -> list[dict[str, Any]]:
    write_json(PORTFOLIO_FILE, [])
    return []


def list_watchlist() -> list[str]:
    return read_json(WATCHLIST_FILE, [])


def add_watchlist_ticker(ticker: str) -> list[str]:
    watchlist = [x.upper() for x in list_watchlist()]
    t = ticker.upper()
    if t not in watchlist:
        watchlist.append(t)
    write_json(WATCHLIST_FILE, watchlist)
    return watchlist


def remove_watchlist_ticker(ticker: str) -> list[str]:
    t = ticker.upper()
    watchlist = [x for x in list_watchlist() if x.upper() != t]
    write_json(WATCHLIST_FILE, watchlist)
    return watchlist

