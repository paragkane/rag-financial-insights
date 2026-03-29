"""Tests for the signal aligner module."""

import pandas as pd
import pytest
from src.backtesting.signal_aligner import next_trading_day


class TestNextTradingDay:
    @pytest.fixture
    def price_df(self):
        dates = pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"])
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)
        df.index.name = "date"
        return df

    def test_exact_match(self, price_df):
        result = next_trading_day("2025-01-06", price_df)
        assert result == "2025-01-06"

    def test_weekend_rolls_forward(self, price_df):
        # Jan 4 2025 is a Saturday, should roll to Jan 6 (Monday)
        result = next_trading_day("2025-01-04", price_df)
        assert result == "2025-01-06"

    def test_no_future_date_returns_none(self, price_df):
        result = next_trading_day("2025-12-31", price_df)
        assert result is None

    def test_past_date_returns_first_available(self, price_df):
        result = next_trading_day("2025-01-01", price_df)
        assert result == "2025-01-06"
