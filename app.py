"""
Tech Stock Forecast — Streamlit dashboard.

Reproduces the pipeline from notebook.ipynb and renders the model's
train/test predictions and metrics interactively.

Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


FEATURE_COLS = [
    "MA_7", "Volume", "Ticker", "RSI_14",
    "Momentum_7d", "Volatility_7d", "Daily_Return",
]


@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv(
        "./datasets/major-tech-stock-2019-2024.csv",
        header=0, index_col=0,
    )
    df.index = pd.to_datetime(df.index)

    enc = OrdinalEncoder()
    df["Ticker"] = enc.fit_transform(df[["Ticker"]])
    ticker_map = {i: name for i, name in enumerate(enc.categories_[0])}

    g = df.groupby("Ticker")["Adj Close"]
    df["MA_7"] = g.transform(lambda x: x.rolling(7).mean())
    df["MA_30"] = g.transform(lambda x: x.rolling(30).mean())
    df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Momentum_7d"] = g.transform(lambda x: x.pct_change(7) * 100)
    df["Volatility_7d"] = g.transform(
        lambda x: x.pct_change().rolling(7).std() * (252 ** 0.5)
    )

    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        return 100 - (100 / (1 + gain / loss))

    df["RSI_14"] = df.groupby("Ticker")["Adj Close"].transform(rsi)
    df.dropna(inplace=True)

    df["Target"] = df.groupby("Ticker")["Adj Close"].transform(lambda x: x.shift(-1))
    df["Direction"] = (df["Target"] > df["Adj Close"]).astype(int)
    df.dropna(inplace=True)

    return df, ticker_map


@st.cache_resource
def train_models(df):
    train_list, test_list = [], []
    for ticker in df["Ticker"].unique():
        td = df[df["Ticker"] == ticker].sort_index()
        split_idx = int(len(td) * 0.8)
        train_list.append(td.iloc[:split_idx])
        test_list.append(td.iloc[split_idx:])

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list).copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, train_df["Direction"])
    test_df["Pred_Direction"] = clf.predict(X_test)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, train_df["Target"])
    test_df["Pred_Price"] = reg.predict(X_test)

    return train_df, test_df, clf, reg


# --- UI ---

st.set_page_config(page_title="Tech Stock Forecast", layout="wide")
st.title("Tech Stock Forecast — Model Results")
st.caption(
    "Random Forest baseline on AAPL, MSFT, AMZN, GOOGL, TSLA (2019–2023). "
    "80/20 chronological split per ticker."
)

df, ticker_map = load_and_prepare_data()
train_df, test_df, clf, reg = train_models(df)

# Sidebar
st.sidebar.header("Controls")
ticker_name = st.sidebar.selectbox("Ticker", list(ticker_map.values()))
ticker_id = {v: k for k, v in ticker_map.items()}[ticker_name]

# --- Per-ticker price chart ---
st.header(f"{ticker_name}: Actual vs Predicted Price")

train_t = train_df[train_df["Ticker"] == ticker_id]
test_t = test_df[test_df["Ticker"] == ticker_id]

split_date = test_t.index.min()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=train_t.index, y=train_t["Adj Close"],
    name="Train (actual)", line=dict(color="#1f77b4"),
))
fig.add_trace(go.Scatter(
    x=test_t.index, y=test_t["Adj Close"],
    name="Test (actual)", line=dict(color="#2ca02c"),
))
fig.add_trace(go.Scatter(
    x=test_t.index, y=test_t["Pred_Price"],
    name="Model prediction", line=dict(color="#d62728", dash="dash"),
))
split_date_str = split_date.strftime("%Y-%m-%d")
fig.add_vline(x=split_date_str, line_width=1, line_dash="dot", line_color="gray")
fig.add_annotation(
    x=split_date_str, y=1, yref="paper",
    text="train/test split", showarrow=False,
    yanchor="bottom", font=dict(color="gray"),
)
fig.update_layout(
    xaxis_title="Date", yaxis_title="Adj Close ($)",
    hovermode="x unified", height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# --- Metrics ---
st.header("Model Performance vs Baselines")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Regression (next-day price)")
    mae = mean_absolute_error(test_df["Target"], test_df["Pred_Price"])
    rmse = np.sqrt(mean_squared_error(test_df["Target"], test_df["Pred_Price"]))
    r2 = r2_score(test_df["Target"], test_df["Pred_Price"])

    # Persistence baseline: predict today's Adj Close for tomorrow.
    p_mae = mean_absolute_error(test_df["Target"], test_df["Adj Close"])
    p_rmse = np.sqrt(mean_squared_error(test_df["Target"], test_df["Adj Close"]))
    p_r2 = r2_score(test_df["Target"], test_df["Adj Close"])

    metrics_df = pd.DataFrame({
        "Model (RF)": [f"${mae:.2f}", f"${rmse:.2f}", f"{r2:.4f}"],
        "Persistence baseline": [f"${p_mae:.2f}", f"${p_rmse:.2f}", f"{p_r2:.4f}"],
    }, index=["MAE", "RMSE", "R²"])
    st.dataframe(metrics_df, use_container_width=True)
    st.caption(
        "Persistence = predict today's price for tomorrow. If the model "
        "doesn't beat this, it has no real predictive power on price level."
    )

with col2:
    st.subheader("Classification (direction)")
    acc = accuracy_score(test_df["Direction"], test_df["Pred_Direction"])
    majority = test_df["Direction"].value_counts(normalize=True).max()

    clf_df = pd.DataFrame({
        "Score": [f"{acc:.4f}", f"{majority:.4f}"],
    }, index=["Model accuracy", "Majority-class baseline"])
    st.dataframe(clf_df, use_container_width=True)

    cm = confusion_matrix(test_df["Direction"], test_df["Pred_Direction"])
    cm_df = pd.DataFrame(
        cm, index=["Actual Down", "Actual Up"],
        columns=["Pred Down", "Pred Up"],
    )
    st.write("**Confusion matrix**")
    st.dataframe(cm_df, use_container_width=True)

# --- Per-ticker breakdown ---
st.header("Per-ticker breakdown")

rows = []
for tid, name in ticker_map.items():
    sub = test_df[test_df["Ticker"] == tid]
    if len(sub) == 0:
        continue
    rows.append({
        "Ticker": name,
        "MAE ($)": mean_absolute_error(sub["Target"], sub["Pred_Price"]),
        "R²": r2_score(sub["Target"], sub["Pred_Price"]),
        "Accuracy": accuracy_score(sub["Direction"], sub["Pred_Direction"]),
        "Majority baseline": sub["Direction"].value_counts(normalize=True).max(),
    })
breakdown = pd.DataFrame(rows).set_index("Ticker").round(4)
st.dataframe(breakdown, use_container_width=True)

# --- Feature importance ---
st.header("Feature Importance (Regressor)")
imp = pd.Series(reg.feature_importances_, index=FEATURE_COLS).sort_values()
fig_imp = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h"))
fig_imp.update_layout(xaxis_title="Importance", height=350, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_imp, use_container_width=True)
