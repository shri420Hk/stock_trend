# Stock Trend Prediction using LSTM

This repository contains Python code for predicting stock trends using Long Short-Term Memory (LSTM) neural networks. We’ll use historical stock price data to train the model and make predictions.

## Prerequisites

Make sure you have the following libraries installed:
- numpy
- pandas_datareader
- matplotlib
- scikit-learn
- yfinance
- keras

You can install them using pip:
```bash
pip install numpy pandas_datareader matplotlib scikit-learn yfinance keras
# Usage
Clone this repository:
git clone https://github.com/your-username/stock-trend-prediction.git
cd stock-trend-prediction

python stock_trend_prediction.py
Enter the stock symbol (e.g., AAPL for Apple Inc.) when prompted.
The script will download historical stock data from Yahoo Finance, preprocess it, and train an LSTM model.
The trained model will predict future stock prices based on the input data.
python stock_trend_prediction.py
# Results
The script will display a plot showing historical stock prices and the predicted trend.
This Markdown syntax will display properly when viewed on GitHub as a README file.
