from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from data_loader import fetch_data
from features import create_features

def train_model(ticker="AAPL"):
    print(f"Fetching data for {ticker} ...")
    df = fetch_data(ticker, period="60d", interval="5m")
    df = create_features(df)

    features = ['return', 'ma5', 'ma20', 'std20', 'rsi14']
    X = df[features]
    y = df['target']

    # Split 80% train, 20% test
    split = int(0.8 * len(df))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Validation Mean Absolute Error: {mae:.4f}")

    # Save the trained model
    joblib.dump(model, f"{ticker}_model.joblib")
    print(f"✅ Model saved successfully as {ticker}_model.joblib")

if __name__ == "__main__":
    train_model("AAPL")
