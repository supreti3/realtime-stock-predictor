from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import yfinance as yf
from server.features import create_features  # ✅ correct import


app = FastAPI(title="Real-Time Stock Predictor")
model = joblib.load("AAPL_model.joblib")


class StockRequest(BaseModel):
    ticker: str = "AAPL"


@app.post("/predict")
def predict(req: StockRequest):
    try:
        # Fetch recent stock data
        df = yf.download(req.ticker, period="1d", interval="5m", progress=False)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found for this ticker.")

        # Create technical features
        df = create_features(df)

        # Prepare latest input row for prediction
        latest = df.iloc[-1]
        features = ['return', 'ma5', 'ma20', 'std20', 'rsi14']
        X = latest[features].values.reshape(1, -1)
        prediction = model.predict(X)[0]

        # Current price extraction
        current_price = df['Close'].iloc[-1]

        # Handle pandas Series or dict cases
        if isinstance(current_price, dict):
            current_price = list(current_price.values())[0]
        elif hasattr(current_price, 'item'):
            try:
                current_price = current_price.item()
            except:
                pass

        # Convert to float
        pred_price = float(prediction)
        curr_price = float(current_price)
        change = pred_price - curr_price

        # ✅ Return cleaned JSON
        return {
            "ticker": req.ticker,
            "current_price": curr_price,
            "predicted_next_close": pred_price,
            "expected_change": change
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
