from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def generate_insights(ind_df: pd.DataFrame) -> dict[str, Any]:
    latest = ind_df.iloc[-1]
    sma_7 = latest.get("SMA_7")
    sma_30 = latest.get("SMA_30")
    rsi = latest.get("RSI")
    macd = latest.get("MACD")
    macd_signal = latest.get("MACD_SIGNAL")
    vol = latest.get("VOLATILITY_30D")
    daily_ret = latest.get("DAILY_RETURN")
    price = latest.get("Close")

    trend = "neutral"
    if pd.notna(sma_7) and pd.notna(sma_30):
        if sma_7 > sma_30:
            trend = "bullish"
        elif sma_7 < sma_30:
            trend = "bearish"

    momentum = "moderate"
    if pd.notna(macd) and pd.notna(macd_signal):
        if macd > macd_signal * 1.1:
            momentum = "strong"
        elif macd < macd_signal * 0.9:
            momentum = "weak"

    volatility_warning = bool(pd.notna(vol) and vol > 0.40)

    over_state = "normal"
    if pd.notna(rsi):
        if rsi >= 70:
            over_state = "overbought"
        elif rsi <= 30:
            over_state = "oversold"

    risk_points = 0
    if volatility_warning:
        risk_points += 40
    if trend == "bearish":
        risk_points += 20
    if over_state == "overbought":
        risk_points += 20
    if pd.notna(daily_ret) and abs(daily_ret) > 0.04:
        risk_points += 20
    risk_score = int(np.clip(risk_points, 0, 100))
    risk = "low" if risk_score < 35 else "medium" if risk_score < 70 else "high"

    beginner_explanation = (
        f"This stock looks {trend}. Short-term momentum is {momentum}. "
        f"RSI suggests it is {over_state}. "
        f"{'Volatility is elevated, so risk is higher than usual. ' if volatility_warning else ''}"
        f"Overall risk level is {risk} ({risk_score}/100)."
    )

    supporting = []
    if trend == "bullish":
        supporting.append("Short-term moving average is above the long-term moving average.")
    elif trend == "bearish":
        supporting.append("Short-term moving average is below the long-term moving average.")
    if over_state == "overbought":
        supporting.append("RSI is above 70, which may indicate overbought conditions.")
    if over_state == "oversold":
        supporting.append("RSI is below 30, which may indicate oversold conditions.")
    if volatility_warning:
        supporting.append("30-day annualized volatility is high, implying larger price swings.")

    return {
        "trend_direction": trend,
        "risk_level": risk,
        "risk_score": risk_score,
        "momentum_strength": momentum,
        "volatility_warning": volatility_warning,
        "overbought_oversold": over_state,
        "supporting_signals": supporting,
        "beginner_explanation": beginner_explanation,
        "snapshot": {
            "close": float(price) if pd.notna(price) else None,
            "sma_7": float(sma_7) if pd.notna(sma_7) else None,
            "sma_30": float(sma_30) if pd.notna(sma_30) else None,
            "rsi": float(rsi) if pd.notna(rsi) else None,
            "volatility_30d": float(vol) if pd.notna(vol) else None,
        },
    }

