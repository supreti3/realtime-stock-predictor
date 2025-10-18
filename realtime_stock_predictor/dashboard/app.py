import streamlit as st
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="💹 Real-Time Stock Market Predictor", page_icon="💰", layout="wide")

# ---------- STYLING ----------
st.markdown("""
<style>
body {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}
div.block-container {
    padding-top: 1rem;
    max-width: 1150px;
}

/* 🔄 ROTATING STOCK TICKER */
.ticker-wrapper {
    overflow: hidden;
    white-space: nowrap;
    box-shadow: 0 0 12px rgba(0, 255, 150, 0.15);
    background: rgba(18, 22, 27, 0.9);
    border: 1px solid #1f242c;
    border-radius: 14px;
    margin-bottom: 2rem;
}
.ticker {
    display: inline-block;
    animation: scroll-left 35s linear infinite;
    padding: 0.8rem 0;
}
@keyframes scroll-left {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}
.stock-item {
    display: inline-block;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0 1.5rem;
    padding: 0.4rem 0.7rem;
    border-radius: 8px;
    color: white;
}
.gain {
    background: linear-gradient(90deg, #00e676, #00c853);
    box-shadow: 0 0 8px rgba(0,230,118,0.4);
}
.loss {
    background: linear-gradient(90deg, #ff5252, #e53935);
    box-shadow: 0 0 8px rgba(229,57,53,0.4);
}

/* HEADER */
h1 {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00e676, #1de9b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 20px rgba(0,255,150,0.25);
    margin-bottom: 0.5rem;
}
h3 {
    text-align: center;
    color: #9aa0a6;
    font-size: 1.2rem;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* Input */
.stTextInput > div > div > input {
    background-color: #161b22;
    color: #e6edf3;
    border-radius: 12px;
    border: 1px solid #2d333b;
    height: 3rem;
    font-size: 1rem;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00e676, #1de9b6);
    color: #0d1117;
    font-weight: bold;
    font-size: 17px;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    transition: all 0.25s ease-in-out;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #1de9b6, #00e676);
}

/* Metrics */
div[data-testid="metric-container"] {
    background: rgba(28,33,40,0.9);
    border-radius: 14px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 0 16px rgba(0,0,0,0.3);
    color: #e0e0e0;
    text-align: center;
}

/* Remove footer */
footer, .stCaption { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ---------- FETCH TOP GAINERS / LOSERS ----------
def get_top_movers():
    tickers = ["AAPL","MSFT","AMZN","GOOG","TSLA","META","NVDA","NFLX","AMD","BABA",
               "INTC","ORCL","PYPL","SHOP","DIS","PEP","KO","V","MA","CSCO"]
    df = yf.download(tickers, period="1d", interval="1h", group_by='ticker', progress=False)
    movers = []
    for t in tickers:
        try:
            data = df[t]
            change = ((data["Close"].iloc[-1] - data["Open"].iloc[0]) / data["Open"].iloc[0]) * 100
            movers.append((t, data["Close"].iloc[-1], change))
        except:
            pass
    movers_df = pd.DataFrame(movers, columns=["Ticker","Price","Change"])
    top_gainers = movers_df.sort_values("Change", ascending=False).head(5)
    top_losers = movers_df.sort_values("Change", ascending=True).head(5)
    return top_gainers, top_losers

# ---------- BUILD ROTATING TICKER ----------
gainers, losers = get_top_movers()
ticker_html = "<div class='ticker-wrapper'><div class='ticker'>"
for _, row in gainers.iterrows():
    ticker_html += f"<span class='stock-item gain'>🟢 {row['Ticker']} ${row['Price']:.2f} (+{row['Change']:.2f}%)</span>"
for _, row in losers.iterrows():
    ticker_html += f"<span class='stock-item loss'>🔻 {row['Ticker']} ${row['Price']:.2f} ({row['Change']:.2f}%)</span>"
ticker_html += "</div></div>"
st.markdown(ticker_html, unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1>Real-Time Stock Market Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>⚡ Predict future trends like a pro trader — powered by AI</h3>", unsafe_allow_html=True)

# ---------- INPUT ----------
ticker = st.text_input("🔍 Enter Stock Ticker (e.g., AAPL, TSLA, NVDA, META, AMZN):", "AAPL").upper()
API_URL = "http://127.0.0.1:8000/predict"

# ---------- PREDICT ----------
if st.button("📈 Predict Next Move"):
    with st.spinner("Fetching market data..."):
        try:
            response = requests.post(API_URL, json={"ticker": ticker})
            if response.status_code == 200:
                data = response.json()
                curr = data["current_price"]
                pred = data["predicted_next_close"]
                change = data["expected_change"]
                delta = (change / curr) * 100

                st.success(f"✅ Prediction for {ticker} as of {datetime.now().strftime('%I:%M %p')}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${curr:.2f}")
                col2.metric("Predicted Close", f"${pred:.2f}")
                col3.metric("Change (%)", f"{delta:.2f}%", delta_color="inverse")

                if delta > 0:
                    st.balloons()
                    st.markdown(f"<p style='text-align:center;color:#00e676;font-size:1.1rem;'>💹 {ticker} may rise by {delta:.2f}% — bullish signal!</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='text-align:center;color:#e53935;font-size:1.1rem;'>📉 {ticker} may drop by {abs(delta):.2f}% — watch carefully.</p>", unsafe_allow_html=True)
            else:
                st.error(f"⚠️ Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"💥 Connection error: {e}")
