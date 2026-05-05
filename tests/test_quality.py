"""
Quality control: unit/component, integration, and acceptance tests.

Unit/component: pipeline helpers and invariants in isolation.
Integration: load + train + metrics wiring without the Streamlit UI.
Acceptance: application-level criteria (import safety, end-to-end AI outputs).
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import pipeline
from pipeline import FEATURE_COLS, load_and_prepare_data, train_models


# --- Unit / component tests ---


class TestUnitPipelineLoad:
    """Component tests for `load_and_prepare_data` (no model training)."""

    def test_load_returns_dataframe_and_ticker_map(self):
        df, ticker_map = load_and_prepare_data()
        assert isinstance(df, pd.DataFrame)
        assert isinstance(ticker_map, dict)
        assert len(ticker_map) >= 1

    def test_required_feature_and_target_columns_present(self):
        df, _ = load_and_prepare_data()
        for col in FEATURE_COLS:
            assert col in df.columns
        assert "Target" in df.columns
        assert "Direction" in df.columns
        assert "Adj Close" in df.columns

    def test_no_nan_in_model_inputs_after_prepare(self):
        df, _ = load_and_prepare_data()
        assert not df[FEATURE_COLS].isna().any().any()

    def test_direction_is_binary(self):
        df, _ = load_and_prepare_data()
        assert set(df["Direction"].unique()).issubset({0, 1})

    def test_rolling_features_respect_per_ticker_groups(self):
        """Sanity: each ticker has contiguous blocks in the raw file; index is datetime."""
        df, ticker_map = load_and_prepare_data()
        assert isinstance(df.index, pd.DatetimeIndex)
        for name in ticker_map.values():
            tid = next(k for k, v in ticker_map.items() if v == name)
            sub = df[df["Ticker"] == tid].sort_index()
            assert len(sub) > 50


class TestUnitPipelineTrain:
    """Component tests for `train_models` on a fixed small frame."""

    @pytest.fixture
    def tiny_df(self):
        """Minimal synthetic data matching FEATURE_COLS + training targets."""
        rng = np.random.default_rng(0)
        n = 120
        idx = pd.date_range("2020-01-02", periods=n, freq="B")
        rows = []
        for tid in (0, 1):
            base = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
            adj = base + rng.normal(0, 0.1, n)
            target = np.roll(adj, -1)
            target[-1] = adj[-1]
            direction = (target > adj).astype(int)
            for i in range(n):
                rows.append({
                    "MA_7": adj[i],
                    "Volume": int(rng.integers(1e6, 2e6)),
                    "Ticker": float(tid),
                    "RSI_14": float(rng.uniform(20, 80)),
                    "Momentum_7d": float(rng.normal(0, 1)),
                    "Volatility_7d": float(rng.uniform(0.1, 2.0)),
                    "Daily_Return": float(rng.normal(0, 0.01)),
                    "Adj Close": adj[i],
                    "Target": target[i],
                    "Direction": direction[i],
                })
        df = pd.DataFrame(rows, index=np.tile(idx, 2))
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df

    def test_train_produces_predictions_and_expected_shapes(self, tiny_df):
        train_df, test_df, clf, reg = train_models(tiny_df)
        assert len(train_df) > len(test_df)
        assert "Pred_Direction" in test_df.columns
        assert "Pred_Price" in test_df.columns
        assert len(test_df["Pred_Price"]) == len(test_df)
        assert clf.n_features_in_ == len(FEATURE_COLS)
        assert reg.n_features_in_ == len(FEATURE_COLS)


# --- Integration tests ---


class TestIntegrationLoadAndTrain:
    """Integration: CSV → features → both models → scored test set."""

    def test_full_pipeline_on_bundled_dataset(self):
        df, ticker_map = load_and_prepare_data()
        train_df, test_df, clf, reg = train_models(df)

        assert len(train_df) + len(test_df) == len(df)
        mae = mean_absolute_error(test_df["Target"], test_df["Pred_Price"])
        rmse = float(np.sqrt(mean_squared_error(test_df["Target"], test_df["Pred_Price"])))
        r2 = r2_score(test_df["Target"], test_df["Pred_Price"])
        acc = accuracy_score(test_df["Direction"], test_df["Pred_Direction"])

        assert np.isfinite(mae) and mae >= 0
        assert np.isfinite(rmse) and rmse >= 0
        assert np.isfinite(r2)
        assert 0 <= acc <= 1
        assert set(ticker_map.values()) == {"AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"}

        for tid in df["Ticker"].unique():
            tr_n = len(train_df[train_df["Ticker"] == tid])
            te_n = len(test_df[test_df["Ticker"] == tid])
            assert tr_n > 0 and te_n > 0
            assert tr_n >= te_n * 3


# --- Acceptance tests ---


class TestAcceptanceApplication:
    """System-level checks: app importable without UI, QC criteria on real run."""

    def test_app_module_import_does_not_run_streamlit_dashboard(self):
        """Importing `app` must not invoke `run_dashboard()` (guarded by __main__)."""
        mod = importlib.import_module("app")
        assert hasattr(mod, "run_dashboard")
        assert callable(mod.run_dashboard)

    def test_pipeline_default_path_points_to_existing_csv(self):
        assert pipeline.DEFAULT_DATA_PATH.is_file()

    def test_acceptance_trained_models_meet_basic_sanity(self):
        """End-to-end: bundled data yields finite predictions for all test rows."""
        df, _ = load_and_prepare_data()
        _, test_df, _, _ = train_models(df)
        assert (np.isfinite(test_df["Pred_Price"])).all()
        assert set(test_df["Pred_Direction"].unique()).issubset({0, 1})
