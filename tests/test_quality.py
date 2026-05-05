import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import (
    DEFAULT_DATA_PATH,
    FEATURE_COLS,
    ALL_NOTEBOOK_FEATURES,
    load_and_prepare_data,
    train_models,
)


def test_default_data_path_exists():
    assert Path(DEFAULT_DATA_PATH).exists(), f"Missing dataset: {DEFAULT_DATA_PATH}"


def test_load_and_prepare_data_returns_expected_types():
    df, data, ticker_map = load_and_prepare_data()

    assert isinstance(df, pd.DataFrame)
    assert isinstance(data, pd.DataFrame)
    assert isinstance(ticker_map, dict)
    assert len(df) > 0
    assert len(data) > 0
    assert len(ticker_map) == 5


def test_required_columns_exist():
    df, data, _ = load_and_prepare_data()

    required_df_cols = set(ALL_NOTEBOOK_FEATURES + ["Adj Close", "Target", "Direction"])
    required_data_cols = set(FEATURE_COLS + ["Adj Close", "Target", "Direction"])

    assert required_df_cols.issubset(df.columns), required_df_cols - set(df.columns)
    assert required_data_cols.issubset(data.columns), required_data_cols - set(data.columns)


def test_modeling_data_has_no_missing_values():
    _, data, _ = load_and_prepare_data()
    assert data[FEATURE_COLS + ["Target", "Direction", "Adj Close"]].isnull().sum().sum() == 0


def test_ticker_is_numeric_and_direction_is_binary():
    _, data, _ = load_and_prepare_data()

    assert pd.api.types.is_numeric_dtype(data["Ticker"])
    assert set(data["Direction"].unique()).issubset({0, 1})


def test_filtered_df_matches_modeling_rows():
    df, data, _ = load_and_prepare_data()

    assert len(df) == len(data)
    assert df.index.equals(data.index)


def test_chronological_split_per_ticker():
    _, data, _ = load_and_prepare_data()
    train_df, test_df, _, _, _, _ = train_models(data)

    for ticker in data["Ticker"].unique():
        sub_train = train_df[train_df["Ticker"] == ticker]
        sub_test = test_df[test_df["Ticker"] == ticker]

        assert len(sub_train) > 0
        assert len(sub_test) > 0
        assert sub_train.index.max() < sub_test.index.min()


def test_prediction_lengths_and_columns():
    _, data, _ = load_and_prepare_data()
    train_df, test_df, clf, reg, clf_wf_df, reg_wf_df = train_models(data)

    assert len(train_df) + len(test_df) == len(data)
    assert "Pred_Direction" in test_df.columns
    assert "Pred_Price" in test_df.columns
    assert len(test_df["Pred_Direction"]) == len(test_df)
    assert len(test_df["Pred_Price"]) == len(test_df)

    assert clf is not None
    assert reg is not None
    assert len(clf_wf_df) == 5
    assert len(reg_wf_df) == 5


def test_prediction_values_are_valid():
    _, data, _ = load_and_prepare_data()
    _, test_df, _, _, _, _ = train_models(data)

    assert set(pd.Series(test_df["Pred_Direction"]).unique()).issubset({0, 1})
    assert np.isfinite(test_df["Pred_Price"]).all()


def test_metrics_are_finite():
    _, data, _ = load_and_prepare_data()
    _, test_df, _, _, clf_wf_df, reg_wf_df = train_models(data)

    mae = float(np.mean(np.abs(test_df["Target"] - test_df["Pred_Price"])))
    rmse = float(np.sqrt(np.mean((test_df["Target"] - test_df["Pred_Price"]) ** 2)))
    acc = float((test_df["Direction"] == test_df["Pred_Direction"]).mean())

    assert math.isfinite(mae)
    assert math.isfinite(rmse)
    assert math.isfinite(acc)
    assert 0.0 <= acc <= 1.0

    assert np.isfinite(clf_wf_df["Accuracy"]).all()
    assert np.isfinite(reg_wf_df["MAE"]).all()
    assert np.isfinite(reg_wf_df["RMSE"]).all()
    assert np.isfinite(reg_wf_df["R²"]).all()


def test_expected_tickers_present():
    _, data, ticker_map = load_and_prepare_data()
    ticker_names = set(ticker_map.values())

    assert ticker_names == {"AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"}
    assert data["Ticker"].nunique() == 5


def test_leakage_safe_ma7_matches_manual_calculation():
    df, _, _ = load_and_prepare_data()

    ticker_value = df["Ticker"].iloc[0]
    sub = df[df["Ticker"] == ticker_value].sort_index()

    idx = sub.index[40]
    manual = sub["Adj Close"].shift(1).rolling(window=7).mean().loc[idx]
    stored = sub.loc[idx, "MA_7"]

    assert np.isclose(manual, stored, equal_nan=True)


def test_leakage_safe_momentum_matches_manual_calculation():
    df, _, _ = load_and_prepare_data()

    ticker_value = df["Ticker"].iloc[0]
    sub = df[df["Ticker"] == ticker_value].sort_index()

    idx = sub.index[40]
    manual = sub["Adj Close"].shift(1).pct_change(periods=7).loc[idx] * 100
    stored = sub.loc[idx, "Momentum_7d"]

    assert np.isclose(manual, stored, equal_nan=True)

def test_leakage_safe_ma7_matches_manual_calculation():
    df, _, _ = load_and_prepare_data()

    ticker_value = df["Ticker"].iloc[0]
    sub = df[df["Ticker"] == ticker_value].sort_index().copy().reset_index(drop=True)

    pos = 40
    manual = sub["Adj Close"].shift(1).rolling(window=7).mean().iloc[pos]
    stored = sub["MA_7"].iloc[pos]

    assert np.isclose(manual, stored, equal_nan=True)


def test_leakage_safe_momentum_matches_manual_calculation():
    df, _, _ = load_and_prepare_data()

    ticker_value = df["Ticker"].iloc[0]
    sub = df[df["Ticker"] == ticker_value].sort_index().copy().reset_index(drop=True)

    pos = 40
    manual = sub["Adj Close"].shift(1).pct_change(periods=7).iloc[pos] * 100
    stored = sub["Momentum_7d"].iloc[pos]

    assert np.isclose(manual, stored, equal_nan=True)

def test_importing_app_is_safe():
    import app  # noqa: F401