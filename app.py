import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf

# Load Model
model = joblib.load("model.pkl")

st.title("📈 Stock Price Prediction App")

# User Input
ticker = st.text_input("Enter Stock Ticker (Example: AAPL)", "AAPL")

if st.button("Predict"):

    data = yf.download(ticker, period="1y")

    st.subheader("Raw Data")
    st.write(data.tail())

    # Example feature (Modify according to your model)
    data["MA_10"] = data["Close"].rolling(10).mean()
    data = data.dropna()

    X = data[["MA_10"]].tail(1)

    prediction = model.predict(X)

    st.subheader("Predicted Price")
    st.success(f"Predicted Next Price: {prediction[0]}")
