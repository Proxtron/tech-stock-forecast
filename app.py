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
    classification_report
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt


FEATURE_COLS = [
    "Volume",
    "Ticker",
    "MA_7",
    "Daily_Return",
    "Momentum_7d",
    "Volatility_7d",
    "RSI_14",
    "MACD",
    "Volume_Change",
]

ALL_NOTEBOOK_FEATURES = [
    "Open",
    "High",
    "Low",
    "Volume",
    "Ticker",
    "MA_7",
    "MA_30",
    "Daily_Return",
    "Momentum_7d",
    "Volatility_7d",
    "RSI_14",
    "MACD",
    "Volume_Change",
    "Lag_1",
    "Lag_2",
    "Lag_3",
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
    df["MA_7"] = g.transform(lambda x: x.shift(1).rolling(7).mean())
    df["MA_30"] = g.transform(lambda x: x.shift(1).rolling(30).mean())
    df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Momentum_7d"] = g.transform(lambda x: x.shift(1).pct_change(7) * 100)
    df["Volatility_7d"] = g.transform(
        lambda x: x.shift(1).pct_change().rolling(7).std() * (252 ** 0.5)
    )

    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        return 100 - (100 / (1 + gain / loss))

    df["RSI_14"] = g.transform(lambda x: rsi(x.shift(1)))

    ema_10 = g.transform(lambda x: x.shift(1).ewm(span=10).mean())
    ema_25 = g.transform(lambda x: x.shift(1).ewm(span=25).mean())
    df["MACD"] = ema_10 - ema_25

    df["Volume_Change"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.pct_change())

    for lag in [1, 2, 3]:
        df[f"Lag_{lag}"] = g.transform(lambda x, lag=lag: x.shift(lag))

    df["Target"] = g.transform(lambda x: x.shift(-1))
    df["Direction"] = (df["Target"] > df["Adj Close"]).astype(int)

    useful_feats_X = df[FEATURE_COLS].copy()
    trgt_y = df[["Target", "Direction"]].copy()

    data = pd.concat([useful_feats_X, trgt_y, df[["Adj Close"]]], axis=1)
    data = data.dropna().copy()

    df = df.loc[data.index].copy()

    return df, data, ticker_map


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

    y_train_clf = train_df["Direction"]
    y_train_reg = train_df["Target"]

    clf = XGBClassifier(
        subsample=0.8,
        n_estimators=300,
        learning_rate=0.01,
        max_depth=4,
        random_state=42,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train_clf)
    test_df["Pred_Direction"] = clf.predict(X_test)

    reg = XGBRegressor(
        subsample=0.8,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.01,
        random_state=42,
    )
    reg.fit(X_train, y_train_reg)
    test_df["Pred_Price"] = reg.predict(X_test)

    # walk-forward validation summaries
    X_all = df[FEATURE_COLS].values
    y_all_clf = df["Direction"].values
    y_all_reg = df["Target"].values

    tscv = TimeSeriesSplit(n_splits=5)
    clf_wf_rows = []
    reg_wf_rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all), start=1):
        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(X_all[train_idx])
        X_te = fold_scaler.transform(X_all[test_idx])

        clf_wf = XGBClassifier(
            subsample=0.8,
            n_estimators=300,
            learning_rate=0.01,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
        )
        clf_wf.fit(X_tr, y_all_clf[train_idx])
        clf_wf_rows.append({
            "Fold": fold,
            "Accuracy": accuracy_score(y_all_clf[test_idx], clf_wf.predict(X_te)),
        })

        reg_wf = XGBRegressor(
            subsample=0.8,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.01,
            random_state=42,
        )
        reg_wf.fit(X_tr, y_all_reg[train_idx])
        y_pred_fold = reg_wf.predict(X_te)

        reg_wf_rows.append({
            "Fold": fold,
            "MAE": mean_absolute_error(y_all_reg[test_idx], y_pred_fold),
            "RMSE": np.sqrt(mean_squared_error(y_all_reg[test_idx], y_pred_fold)),
            "R²": r2_score(y_all_reg[test_idx], y_pred_fold),
        })

    return train_df, test_df, clf, reg, pd.DataFrame(clf_wf_rows), pd.DataFrame(reg_wf_rows)


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

    # Persistence baseline: predict today's Adj Close for tomorrow.
    p_mae = mean_absolute_error(test_df["Target"], test_df["Adj Close"])
    p_rmse = np.sqrt(mean_squared_error(test_df["Target"], test_df["Adj Close"]))
    p_r2 = r2_score(test_df["Target"], test_df["Adj Close"])

    metrics_df = pd.DataFrame({
        "Model (XGB)": [f"${mae:.2f}", f"${rmse:.2f}", f"{r2:.4f}"],
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
df_hist = model_data[FEATURE_COLS].hist(bins=40, figsize=(14, 12))
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
