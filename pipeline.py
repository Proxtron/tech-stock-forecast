"""
Shared data loading and model training for the tech stock forecast app.

Kept free of Streamlit so the pipeline can be unit-tested without running the UI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "datasets" / "major-tech-stock-2019-2024.csv"

FEATURE_COLS = [
    "MA_7", "Volume", "Ticker", "RSI_14",
    "Momentum_7d", "Volatility_7d", "Daily_Return",
]


def load_and_prepare_data(csv_path: str | Path | None = None) -> tuple[pd.DataFrame, dict[int, str]]:
    path = Path(csv_path) if csv_path is not None else DEFAULT_DATA_PATH
    df = pd.read_csv(path, header=0, index_col=0)
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


def train_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, RandomForestClassifier, RandomForestRegressor]:
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
