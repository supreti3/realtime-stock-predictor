import yfinance as yf
import pandas as pd


def fetch_data(ticker: str = "AAPL", period: str = "60d", interval: str = "1m") -> pd.DataFrame:
    """
    Fetch historical stock data for the given ticker.
    period examples: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    interval examples: '1m', '2m', '5m', '15m', '1h', '1d'
    """
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
    df = df.dropna()
    df.reset_index(inplace=True)
    return df


# Quick test when you run this file directly
if __name__ == "__main__":
    data = fetch_data("AAPL", "5d", "1m")
    print(data.head())
