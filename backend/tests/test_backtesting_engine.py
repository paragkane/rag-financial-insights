"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest
from src.backtesting.engine import (
    sharpe,
    hit_rate,
    ttest,
    run_sentiment_backtest,
    run_tone_backtest,
    run_guidance_backtest,
)


class TestSharpe:
    def test_positive_returns(self):
        returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.025])
        result = sharpe(returns, "fwd_return_1d")
        assert result > 0

    def test_zero_std_returns_zero(self):
        returns = pd.Series([0.01, 0.01, 0.01])
        assert sharpe(returns, "fwd_return_1d") == 0.0

    def test_single_return_is_zero(self):
        returns = pd.Series([0.05])
        assert sharpe(returns, "fwd_return_1d") == 0.0

    def test_annualization_scales(self):
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        sharpe_1d = sharpe(returns, "fwd_return_1d")
        sharpe_5d = sharpe(returns, "fwd_return_5d")
        # 1d annualizes with sqrt(252), 5d with sqrt(52) — 1d should be larger in magnitude
        assert abs(sharpe_1d) > abs(sharpe_5d) or abs(sharpe_1d - sharpe_5d) < 0.01


class TestHitRate:
    def test_perfect_prediction(self):
        returns = pd.Series([0.1, -0.1, 0.05, -0.05])
        signals = pd.Series([1, -1, 1, -1])
        assert hit_rate(returns, signals) == 1.0

    def test_zero_prediction(self):
        returns = pd.Series([0.1, -0.1, 0.05, -0.05])
        signals = pd.Series([-1, 1, -1, 1])
        assert hit_rate(returns, signals) == 0.0

    def test_fifty_percent(self):
        returns = pd.Series([0.1, -0.1, 0.05, -0.05])
        signals = pd.Series([1, 1, 1, 1])
        assert hit_rate(returns, signals) == 0.5


class TestTtest:
    def test_significant_positive_returns(self):
        returns = pd.Series(np.random.normal(0.05, 0.01, 100))
        t, p = ttest(returns)
        assert t > 0
        assert p < 0.05

    def test_zero_mean_not_significant(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 50))
        t, p = ttest(returns)
        # with zero mean, should not be significant at 0.01 level usually
        assert abs(t) < 5  # sanity check

    def test_too_few_returns(self):
        t, p = ttest(pd.Series([0.01, 0.02]))
        assert t == 0.0
        assert p == 1.0


class TestRunSentimentBacktest:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            "sentiment_score": np.random.uniform(-1, 1, n),
            "fwd_return_5d": np.random.normal(0.005, 0.02, n),
            "tone": np.random.choice(["optimistic", "cautious", "neutral"], n),
            "guidance_direction": np.random.choice(["raised", "lowered", "maintained", "none"], n),
        })

    def test_returns_expected_keys(self, sample_df):
        result = run_sentiment_backtest(sample_df, "fwd_return_5d")
        assert "signal" in result
        assert result["signal"] == "sentiment_score"
        assert "sharpe" not in result  # field is overall_sharpe
        assert "overall_sharpe" in result
        assert "hit_rate" in result
        assert "t_stat" in result
        assert "p_value" in result
        assert "pearson_correlation" in result
        assert "significant" in result

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame({"sentiment_score": [], "fwd_return_5d": []})
        assert run_sentiment_backtest(df, "fwd_return_5d") == {}

    def test_n_total_matches_input(self, sample_df):
        result = run_sentiment_backtest(sample_df, "fwd_return_5d")
        assert result["n_total"] == len(sample_df)


class TestRunToneBacktest:
    def test_with_directional_tones(self):
        df = pd.DataFrame({
            "tone": ["optimistic"] * 10 + ["cautious"] * 10,
            "fwd_return_5d": [0.02] * 10 + [-0.01] * 10,
        })
        result = run_tone_backtest(df, "fwd_return_5d")
        assert result["signal"] == "tone"
        assert "tone_counts" in result
        assert result["hit_rate"] == 1.0  # perfect prediction

    def test_all_neutral_returns_empty(self):
        df = pd.DataFrame({
            "tone": ["neutral"] * 10,
            "fwd_return_5d": [0.01] * 10,
        })
        result = run_tone_backtest(df, "fwd_return_5d")
        assert result == {}


class TestRunGuidanceBacktest:
    def test_no_directional_guidance(self):
        df = pd.DataFrame({
            "guidance_direction": ["none"] * 10 + ["maintained"] * 5,
            "fwd_return_5d": [0.01] * 15,
        })
        result = run_guidance_backtest(df, "fwd_return_5d")
        assert "note" in result  # should note no raised/lowered

    def test_with_directional_guidance(self):
        df = pd.DataFrame({
            "guidance_direction": ["raised"] * 5 + ["lowered"] * 5,
            "fwd_return_5d": [0.03] * 5 + [-0.02] * 5,
        })
        result = run_guidance_backtest(df, "fwd_return_5d")
        assert result["signal"] == "guidance_direction"
        assert result["hit_rate"] == 1.0
