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
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt

from pipeline import (
    ALL_NOTEBOOK_FEATURES,
    FEATURE_COLS,
    load_and_prepare_data,
    train_models,
)

# --- UI ---

st.set_page_config(page_title="Tech Stock Forecast", layout="wide")
st.title("Tech Stock Forecast — Model Results")
st.caption(
    "XGBoost model on AAPL, MSFT, AMZN, GOOGL, TSLA (2019–2023). "
    "80/20 chronological split per ticker."
)

df, model_data, ticker_map = load_and_prepare_data()
train_df, test_df, clf, reg, clf_wf_df, reg_wf_df = train_models(model_data)

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

    p_mae = mean_absolute_error(test_df["Target"], test_df["Adj Close"])
    p_rmse = np.sqrt(mean_squared_error(test_df["Target"], test_df["Adj Close"]))
    p_r2 = r2_score(test_df["Target"], test_df["Adj Close"])

    metrics_df = {
        "Model (XGB)": [f"${mae:.2f}", f"${rmse:.2f}", f"{r2:.4f}"],
        "Persistence baseline": [f"${p_mae:.2f}", f"${p_rmse:.2f}", f"{p_r2:.4f}"],
    }
    st.dataframe(metrics_df, use_container_width=True)
    st.caption(
        "Persistence = predict today's price for tomorrow. If the model "
        "doesn't beat this, it has no real predictive power on price level."
    )

with col2:
    st.subheader("Classification (direction)")
    acc = accuracy_score(test_df["Direction"], test_df["Pred_Direction"])
    majority = test_df["Direction"].value_counts(normalize=True).max()

    clf_df = {
        "Score": [f"{acc:.4f}", f"{majority:.4f}"],
    }
    st.dataframe(
        clf_df,
        use_container_width=True,
    )

    cm = confusion_matrix(test_df["Direction"], test_df["Pred_Direction"])
    st.write("**Confusion matrix**")
    st.dataframe(
        cm,
        use_container_width=True,
    )

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

st.divider()
st.header("Additional Diagnostics")

# classifier importance
st.subheader("Feature Importance (Classifier)")
clf_imp = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values()
fig_clf_imp = go.Figure(go.Bar(x=clf_imp.values, y=clf_imp.index, orientation="h"))
fig_clf_imp.update_layout(xaxis_title="Importance", height=350, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_clf_imp, use_container_width=True)

# confusion matrix heatmap + classification report
st.subheader("Classification Model Evaluation")
col3, col4 = st.columns(2)

with col3:
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix(test_df["Direction"], test_df["Pred_Direction"]),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_cm,
    )
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

with col4:
    report_df = pd.DataFrame(
        classification_report(
            test_df["Direction"],
            test_df["Pred_Direction"],
            output_dict=True,
        )
    ).transpose()
    st.dataframe(report_df, use_container_width=True)

# correlation heatmaps
st.subheader("Feature Correlation Matrices")

fig_corr1, ax_corr1 = plt.subplots(figsize=(14, 10))
sns.heatmap(
    df[ALL_NOTEBOOK_FEATURES].corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    ax=ax_corr1,
)
st.pyplot(fig_corr1)
plt.close(fig_corr1)

fig_corr2, ax_corr2 = plt.subplots(figsize=(8, 6))
sns.heatmap(
    model_data[FEATURE_COLS].corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    ax=ax_corr2,
)
st.pyplot(fig_corr2)
plt.close(fig_corr2)

# histograms
st.subheader("Feature Distributions")
model_data[FEATURE_COLS].hist(bins=40, figsize=(14, 12))
plt.tight_layout()
st.pyplot(plt.gcf())
plt.close(plt.gcf())

# regression diagnostics
st.subheader("Regression Model Evaluation")
col5, col6 = st.columns(2)

with col5:
    fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))
    ax_scatter.scatter(test_df["Target"], test_df["Pred_Price"], alpha=0.5)
    ax_scatter.set_xlabel("Actual Price")
    ax_scatter.set_ylabel("Predicted Price")
    st.pyplot(fig_scatter)
    plt.close(fig_scatter)

with col6:
    residuals = test_df["Target"] - test_df["Pred_Price"]
    fig_res, ax_res = plt.subplots(figsize=(7, 5))
    ax_res.scatter(test_df["Pred_Price"], residuals, alpha=0.5)
    ax_res.axhline(0)
    ax_res.set_xlabel("Predicted Price")
    ax_res.set_ylabel("Residuals")
    st.pyplot(fig_res)
    plt.close(fig_res)

# walk-forward validation
st.subheader("Walk-Forward Validation")
col7, col8 = st.columns(2)

with col7:
    st.write("**Classification folds**")
    st.dataframe(clf_wf_df, use_container_width=True)

with col8:
    st.write("**Regression folds**")
    st.dataframe(reg_wf_df, use_container_width=True)