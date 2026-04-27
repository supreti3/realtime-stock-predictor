from __future__ import annotations

import numpy as np
import pandas as pd


def compute_indicators(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    close = df["Close"]

    df["SMA_7"] = close.rolling(7).mean()
    df["SMA_30"] = close.rolling(30).mean()
    df["SMA_90"] = close.rolling(90).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    rolling_std = close.rolling(20).std()
    df["BB_MID"] = close.rolling(20).mean()
    df["BB_UPPER"] = df["BB_MID"] + (2 * rolling_std)
    df["BB_LOWER"] = df["BB_MID"] - (2 * rolling_std)

    df["DAILY_RETURN"] = close.pct_change()
    df["VOLATILITY_30D"] = df["DAILY_RETURN"].rolling(30).std() * np.sqrt(252)

    window = 20
    df["SUPPORT"] = df["Low"].rolling(window).min()
    df["RESISTANCE"] = df["High"].rolling(window).max()
    return df

