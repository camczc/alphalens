"""
tests/test_signals.py

Smoke test for the signal engine using live yfinance data.
Does NOT require a running database — tests the computation logic directly.

Run:
    pytest tests/test_signals.py -v
"""

import pytest
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def aapl_prices() -> pd.DataFrame:
    """Pull 2 years of AAPL data directly from yfinance."""
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="2y", interval="1d", auto_adjust=False)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# -----------------------------------------------------------------------
# Import signal computation logic directly (bypasses DB)
# -----------------------------------------------------------------------

def compute_signals_standalone(df: pd.DataFrame) -> pd.DataFrame:
    """Standalone version of SignalEngine._compute_all_signals for testing."""
    import ta

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    signals = pd.DataFrame(index=df.index)
    signals["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd_ind = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    signals["macd"] = macd_ind.macd()
    signals["macd_signal"] = macd_ind.macd_signal()
    signals["macd_hist"] = macd_ind.macd_diff()

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    signals["bb_upper"] = bb.bollinger_hband()
    signals["bb_lower"] = bb.bollinger_lband()
    signals["bb_pct"] = bb.bollinger_pband()

    signals["sma_20"] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    signals["sma_50"] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    signals["sma_200"] = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    signals["ema_12"] = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
    signals["ema_26"] = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()

    signals["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=volume
    ).on_balance_volume()

    typical_price = (high + low + close) / 3
    signals["vwap"] = (
        (typical_price * volume).rolling(20).sum() /
        volume.rolling(20).sum()
    )

    return signals


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestSignalComputation:

    def test_price_data_loads(self, aapl_prices):
        assert not aapl_prices.empty
        assert "close" in aapl_prices.columns
        assert len(aapl_prices) > 200, "Need 200+ days for SMA-200"

    def test_signals_compute_without_error(self, aapl_prices):
        signals = compute_signals_standalone(aapl_prices)
        assert not signals.empty

    def test_rsi_range(self, aapl_prices):
        signals = compute_signals_standalone(aapl_prices)
        rsi = signals["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI must be in [0, 100]"

    def test_bollinger_band_ordering(self, aapl_prices):
        signals = compute_signals_standalone(aapl_prices)
        valid = signals.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all(), "Upper band must >= lower"

    def test_bb_pct_range(self, aapl_prices):
        """bb_pct should mostly be in [0,1] — extreme moves can exceed it."""
        signals = compute_signals_standalone(aapl_prices)
        pct = signals["bb_pct"].dropna()
        in_range = ((pct >= -0.5) & (pct <= 1.5)).mean()
        assert in_range > 0.95, f"Most bb_pct values should be near [0,1], got {in_range:.1%} in range"

    def test_sma_alignment(self, aapl_prices):
        """SMA-200 should have more NaNs than SMA-50 than SMA-20."""
        signals = compute_signals_standalone(aapl_prices)
        nan_20 = signals["sma_20"].isna().sum()
        nan_50 = signals["sma_50"].isna().sum()
        nan_200 = signals["sma_200"].isna().sum()
        assert nan_20 < nan_50 < nan_200

    def test_macd_histogram_equals_diff(self, aapl_prices):
        """MACD histogram = MACD line - signal line."""
        signals = compute_signals_standalone(aapl_prices)
        valid = signals.dropna(subset=["macd", "macd_signal", "macd_hist"])
        expected = valid["macd"] - valid["macd_signal"]
        diff = (valid["macd_hist"] - expected).abs()
        assert (diff < 1e-6).all(), "MACD histogram should equal MACD - signal"

    def test_obv_is_cumulative(self, aapl_prices):
        """OBV should change every day (not be constant)."""
        signals = compute_signals_standalone(aapl_prices)
        obv = signals["obv"].dropna()
        unique_vals = obv.nunique()
        assert unique_vals > len(obv) * 0.5, "OBV should have many unique values"

    def test_no_all_nan_columns(self, aapl_prices):
        signals = compute_signals_standalone(aapl_prices)
        for col in signals.columns:
            non_null = signals[col].notna().sum()
            assert non_null > 0, f"Column {col} is entirely NaN"


class TestCompositeScore:

    def test_composite_score_range(self, aapl_prices):
        """Composite score must be in [-1, 1]."""
        from app.services.signals import SignalEngine

        # Use the private method directly without DB
        engine = SignalEngine.__new__(SignalEngine)
        signals = compute_signals_standalone(aapl_prices)
        scores = engine._compute_composite_score(signals, aapl_prices["close"])

        valid = scores.dropna()
        assert (valid >= -1).all() and (valid <= 1).all()

    def test_composite_score_not_constant(self, aapl_prices):
        from app.services.signals import SignalEngine
        engine = SignalEngine.__new__(SignalEngine)
        signals = compute_signals_standalone(aapl_prices)
        scores = engine._compute_composite_score(signals, aapl_prices["close"])

        std = scores.dropna().std()
        assert std > 0.05, f"Score should vary over time, std={std:.4f}"
