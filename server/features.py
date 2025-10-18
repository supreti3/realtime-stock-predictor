import pandas as pd

def compute_rsi(series, period=14):
    """Compute Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def create_features(df):
    """
    Add technical indicators as features for the prediction model.
    """
    df['return'] = df['Close'].pct_change()
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma20'] = df['Close'].rolling(window=20).mean()
    df['std20'] = df['Close'].rolling(window=20).std()
    df['rsi14'] = compute_rsi(df['Close'], 14)
    df['target'] = df['Close'].shift(-1)  # next-minute closing price (prediction target)
    df.dropna(inplace=True)
    return df


# Test when run directly
if __name__ == "__main__":
    from data_loader import fetch_data

    df = fetch_data("AAPL", "5d", "1m")
    df = create_features(df)
    print(df.head(10))
