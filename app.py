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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from pipeline import FEATURE_COLS, load_and_prepare_data as load_and_prepare_data_core, train_models as train_models_core


@st.cache_data
def load_and_prepare_data():
    return load_and_prepare_data_core()


@st.cache_resource
def train_models(df):
    return train_models_core(df)


def run_dashboard():
    st.set_page_config(page_title="Tech Stock Forecast", layout="wide")
    st.title("Tech Stock Forecast — Model Results")
    st.caption(
        "Random Forest baseline on AAPL, MSFT, AMZN, GOOGL, TSLA (2019–2023). "
        "80/20 chronological split per ticker."
    )

    df, ticker_map = load_and_prepare_data()
    train_df, test_df, clf, reg = train_models(df)

    st.sidebar.header("Controls")
    ticker_name = st.sidebar.selectbox("Ticker", list(ticker_map.values()))
    ticker_id = {v: k for k, v in ticker_map.items()}[ticker_name]

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

    st.header("Model Performance vs Baselines")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Regression (next-day price)")
        mae = mean_absolute_error(test_df["Target"], test_df["Pred_Price"])
        rmse = np.sqrt(mean_squared_error(test_df["Target"], test_df["Pred_Price"]))
        r2 = r2_score(test_df["Target"], test_df["Pred_Price"])

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

    st.header("Feature Importance (Regressor)")
    imp = pd.Series(reg.feature_importances_, index=FEATURE_COLS).sort_values()
    fig_imp = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h"))
    fig_imp.update_layout(xaxis_title="Importance", height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_imp, use_container_width=True)


if __name__ == "__main__":
    run_dashboard()
