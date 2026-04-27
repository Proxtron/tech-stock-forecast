# tech-stock-forecast

A machine learning project for predicting major tech stock prices using a combined regression + classification approach. Built on daily OHLCV data (2019–2024) sourced from Yahoo Finance via `yfinance`.

---

## What it does

- **Regressor** — predicts next-day `log_return` (magnitude)
- **Classifier** — predicts price direction (up/down) and/or binned return category
- Both models share the same feature set and are evaluated independently, then combined at inference time

---

## Dataset

**Source:** [Major Tech Stocks Time Series (2019–2024)](https://www.kaggle.com/datasets/alfredkondoro/major-tech-stocks-time-series-2019-2024) — `major-tech-stock-2019-2024.csv` (~720 kB)

**Columns:** `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`, `Ticker`

---

## Setup

```bash
git clone https://github.com/Proxtron/tech-stock-forecast.git
cd tech-stock-forecast
pip install -r requirements.txt
```

## Run Web Dashboard

```bash
streamlit run app.py
```

## Requirements

```
pandas
numpy
scikit-learn
imblearn
matplotlib
seaborn
streamlit
plotly
```