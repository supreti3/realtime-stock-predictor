# AI Stock Intelligence Platform

Advanced, free-tool stock analytics platform built for portfolio demos and educational use.  
This project combines a modular FastAPI backend, an interactive Streamlit frontend, multi-model forecasting, technical/risk analytics, watchlist and portfolio management, and rule-based AI explanations.

## Overview

The platform lets users analyze any stock ticker from Yahoo Finance, compare multiple forecasting models, view technical indicators and risk scores, track a custom portfolio/watchlist, compare multiple equities, and inspect simple news sentiment without paid APIs.

## Features

- Multi-stock dashboard with live summary stats and trend visualization
- Advanced prediction engine with:
  - Linear Regression
  - Random Forest
  - XGBoost (if installed)
  - Prophet (if installed)
  - LSTM/TensorFlow (if installed)
  - Automatic fallback to sklearn models if heavy libs are unavailable
- Forecast horizon control: 1, 7, 30, or 90 days
- Model comparison table with MAE, RMSE, MAPE, and R2
- Predicted vs actual charting for validation insight
- Technical indicators:
  - SMA(7/30/90), EMA(12/26), MACD, RSI, Bollinger Bands
  - Daily returns, rolling volatility, support/resistance
- Rule-based AI insight engine for beginner-friendly explanations
- Portfolio tracker with local persistence, P/L, returns, and allocation chart
- Watchlist with local persistence and quick trend snapshots
- 2-4 stock comparison: performance, normalized returns, volatility, average return, and correlation
- News + keyword sentiment scoring from free yFinance news feed
- Risk analysis: volatility, max drawdown, Sharpe, VaR, beta, risk score (0-100)

## Tech Stack

- **Backend:** FastAPI
- **Frontend:** Streamlit + Plotly
- **Data:** yFinance
- **ML:** scikit-learn (+ optional xgboost, prophet, tensorflow)
- **Core libs:** pandas, numpy, requests

## Architecture

```text
backend/
  main.py
  services/
    stock_service.py
    prediction_service.py
    indicator_service.py
    insight_service.py
    portfolio_service.py
  models/
  utils/

frontend/
  streamlit_app.py
  pages/
  components/
```

## Screenshots

Add screenshots here for:
- Dashboard
- Prediction Engine
- Technicals and Risk
- Portfolio and Watchlist
- Comparison and News

## Setup Instructions

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Backend

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### Run Frontend

```bash
streamlit run frontend/streamlit_app.py
```

## API Endpoints

- `GET /stock/{ticker}/summary`
- `GET /stock/{ticker}/history?period=2y&interval=1d`
- `POST /stock/{ticker}/predict`
- `GET /stock/{ticker}/indicators`
- `GET /stock/{ticker}/insights`
- `GET /stock/{ticker}/risk`
- `GET /stock/{ticker}/news`
- `GET /portfolio`
- `POST /portfolio`
- `DELETE /portfolio`
- `GET /watchlist`
- `POST /watchlist`
- `DELETE /watchlist/{ticker}`
- `GET /compare?tickers=AAPL,MSFT,NVDA`

## Future Improvements

- Add model caching and background jobs for faster repeated predictions
- Add macro indicators (rates, CPI, sector ETFs) for richer context
- Add backtesting engine and strategy simulator
- Add downloadable PDF reports
- Add user authentication + cloud persistence

## Disclaimer

This project is for educational purposes only and does **not** provide financial advice.
Always do your own research before making investment decisions.
