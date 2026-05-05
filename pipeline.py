from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


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

DEFAULT_DATA_PATH = "./datasets/major-tech-stock-2019-2024.csv"


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss))


def load_and_prepare_data(csv_path: str = DEFAULT_DATA_PATH):
    df = pd.read_csv(csv_path, header=0, index_col=0)
    df.index = pd.to_datetime(df.index)

    enc = OrdinalEncoder()
    df["Ticker"] = enc.fit_transform(df[["Ticker"]])
    ticker_map = {i: name for i, name in enumerate(enc.categories_[0])}

    g = df.groupby("Ticker")["Adj Close"]

    df["MA_7"] = g.transform(lambda x: x.shift(1).rolling(window=7).mean())
    df["MA_30"] = g.transform(lambda x: x.shift(1).rolling(window=30).mean())
    df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Momentum_7d"] = g.transform(lambda x: x.shift(1).pct_change(periods=7) * 100)
    df["Volatility_7d"] = g.transform(
        lambda x: x.shift(1).pct_change().rolling(7).std() * (252 ** 0.5)
    )
    df["RSI_14"] = g.transform(lambda x: _rsi(x.shift(1)))

    ema_10 = g.transform(lambda x: x.shift(1).ewm(span=10).mean())
    ema_25 = g.transform(lambda x: x.shift(1).ewm(span=25).mean())
    df["MACD"] = ema_10 - ema_25

    df["Volume_Change"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.pct_change())

    for lag in [1, 2, 3]:
        df[f"Lag_{lag}"] = g.transform(lambda x, lag=lag: x.shift(lag))

    df["Target"] = g.transform(lambda x: x.shift(-1))
    df["Direction"] = (df["Target"] > df["Adj Close"]).astype(int)

    required_cols = FEATURE_COLS + ["Target", "Direction", "Adj Close"]
    mask = df[required_cols].notna().all(axis=1)

    df = df.loc[mask].copy()
    data = df[FEATURE_COLS + ["Target", "Direction", "Adj Close"]].copy()

    return df, data, ticker_map


def train_models(data: pd.DataFrame):
    train_list, test_list = [], []

    for ticker in data["Ticker"].unique():
        td = data[data["Ticker"] == ticker].sort_index()
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

    X_all = data[FEATURE_COLS].values
    y_all_clf = data["Direction"].values
    y_all_reg = data["Target"].values

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
        clf_wf_rows.append(
            {
                "Fold": fold,
                "Accuracy": accuracy_score(y_all_clf[test_idx], clf_wf.predict(X_te)),
            }
        )

        reg_wf = XGBRegressor(
            subsample=0.8,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.01,
            random_state=42,
        )
        reg_wf.fit(X_tr, y_all_reg[train_idx])
        y_pred_fold = reg_wf.predict(X_te)

        reg_wf_rows.append(
            {
                "Fold": fold,
                "MAE": mean_absolute_error(y_all_reg[test_idx], y_pred_fold),
                "RMSE": np.sqrt(mean_squared_error(y_all_reg[test_idx], y_pred_fold)),
                "R²": r2_score(y_all_reg[test_idx], y_pred_fold),
            }
        )

    return train_df, test_df, clf, reg, pd.DataFrame(clf_wf_rows), pd.DataFrame(reg_wf_rows)