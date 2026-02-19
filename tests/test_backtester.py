"""
tests/test_backtester.py

Tests for the backtesting engine.
Uses synthetic price data to make tests deterministic and fast.

Run:
    pytest tests/test_backtester.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from app.services.backtester import (
    BacktestEngine,
    BacktestResult,
    CompositeScoreStrategy,
    RSIMeanReversionStrategy,
    GoldenCrossStrategy,
    MACDCrossoverStrategy,
)


# ------------------------------------------------------------------
# Synthetic data fixtures
# ------------------------------------------------------------------

def make_price_df(n=500, seed=42, trend=0.0003) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a slight upward trend."""
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")
    returns = np.random.normal(trend, 0.015, n)
    close = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "open":      close * (1 + np.random.normal(0, 0.003, n)),
        "high":      close * (1 + np.abs(np.random.normal(0, 0.005, n))),
        "low":       close * (1 - np.abs(np.random.normal(0, 0.005, n))),
        "close":     close,
        "adj_close": close,
        "volume":    np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)
    return df


def make_signals_df(prices: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic signals using real computation."""
    import ta
    close = prices["close"]
    high = prices["high"]
    low = prices["low"]
    volume = prices["volume"]

    signals = pd.DataFrame(index=prices.index)
    signals["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd = ta.trend.MACD(close=close)
    signals["macd"] = macd.macd()
    signals["macd_signal"] = macd.macd_signal()
    signals["macd_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close=close)
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

    # Composite score (simplified)
    rsi_score = (50 - signals["rsi_14"]) / 50
    signals["composite_score"] = rsi_score.clip(-1, 1).fillna(0)

    return signals


@pytest.fixture(scope="module")
def prices():
    return make_price_df()


@pytest.fixture(scope="module")
def signals(prices):
    return make_signals_df(prices)


# ------------------------------------------------------------------
# Mock BacktestEngine (bypasses DB)
# ------------------------------------------------------------------

def make_engine(prices_df, signals_df):
    """Create a BacktestEngine with mocked DB dependencies."""
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.ingestion = MagicMock()
    engine.signal_engine = MagicMock()
    engine.ingestion.get_price_dataframe.return_value = prices_df
    engine.signal_engine._compute_all_signals.return_value = signals_df
    engine.db = MagicMock()
    return engine


# ------------------------------------------------------------------
# Tests: Signal generation
# ------------------------------------------------------------------

class TestStrategySignals:

    def test_composite_strategy_returns_binary_positions(self, prices, signals):
        strategy = CompositeScoreStrategy()
        pos = strategy.generate_signals(prices, signals)
        assert set(pos.unique()).issubset({0.0, 1.0})

    def test_rsi_strategy_returns_binary_positions(self, prices, signals):
        strategy = RSIMeanReversionStrategy()
        pos = strategy.generate_signals(prices, signals)
        assert set(pos.unique()).issubset({0.0, 1.0})

    def test_golden_cross_returns_binary_positions(self, prices, signals):
        strategy = GoldenCrossStrategy()
        pos = strategy.generate_signals(prices, signals)
        assert set(pos.unique()).issubset({0.0, 1.0})

    def test_macd_crossover_returns_binary_positions(self, prices, signals):
        strategy = MACDCrossoverStrategy()
        pos = strategy.generate_signals(prices, signals)
        assert set(pos.unique()).issubset({0.0, 1.0})

    def test_signals_index_matches_prices(self, prices, signals):
        strategy = CompositeScoreStrategy()
        pos = strategy.generate_signals(prices, signals)
        assert pos.index.equals(prices.index)


# ------------------------------------------------------------------
# Tests: Portfolio simulation
# ------------------------------------------------------------------

class TestPortfolioSimulation:

    def test_portfolio_never_goes_negative(self, prices, signals):
        engine = make_engine(prices, signals)
        strategy = CompositeScoreStrategy()
        pos = strategy.generate_signals(prices, signals)
        portfolio = engine._simulate_portfolio(prices, pos, 10_000, 0.001, 0.0005)
        assert (portfolio["portfolio_value"] >= 0).all()

    def test_always_invested_equals_buy_and_hold(self, prices, signals):
        """If we're always in position, returns should approximate buy & hold (minus costs)."""
        engine = make_engine(prices, signals)
        always_in = pd.Series(1.0, index=prices.index)
        portfolio = engine._simulate_portfolio(prices, always_in, 10_000, 0.0, 0.0)

        expected = 10_000 * (prices["adj_close"].iloc[-1] / prices["adj_close"].iloc[0])
        actual = portfolio["portfolio_value"].iloc[-1]
        # Should be within 1% (tiny floating point differences)
        assert abs(actual - expected) / expected < 0.01

    def test_never_invested_stays_flat(self, prices, signals):
        """If we never enter a position, portfolio should stay at initial capital."""
        engine = make_engine(prices, signals)
        never_in = pd.Series(0.0, index=prices.index)
        portfolio = engine._simulate_portfolio(prices, never_in, 10_000, 0.001, 0.0005)
        assert abs(portfolio["portfolio_value"].iloc[-1] - 10_000) < 1.0

    def test_transaction_costs_reduce_returns(self, prices, signals):
        engine = make_engine(prices, signals)
        strategy = CompositeScoreStrategy()
        pos = strategy.generate_signals(prices, signals)

        no_cost = engine._simulate_portfolio(prices, pos, 10_000, 0.0, 0.0)
        with_cost = engine._simulate_portfolio(prices, pos, 10_000, 0.001, 0.0005)

        # Returns with costs should be <= returns without costs
        assert with_cost["portfolio_value"].iloc[-1] <= no_cost["portfolio_value"].iloc[-1]


# ------------------------------------------------------------------
# Tests: Metrics
# ------------------------------------------------------------------

class TestBacktestMetrics:

    def _run_backtest(self, prices, signals, strategy):
        engine = make_engine(prices, signals)
        pos = strategy.generate_signals(prices, signals)
        portfolio = engine._simulate_portfolio(prices, pos, 10_000, 0.001, 0.0005)

        bench = pd.Series(0.0, index=prices.index)  # flat benchmark for deterministic tests

        return engine._compute_metrics(
            symbol="TEST",
            strategy=strategy,
            portfolio=portfolio,
            benchmark_returns=bench,
            initial_capital=10_000,
            start_date=prices.index[0].date(),
            end_date=prices.index[-1].date(),
        )

    def test_total_return_calculation(self, prices, signals):
        result = self._run_backtest(prices, signals, CompositeScoreStrategy())
        assert isinstance(result.total_return, float)
        assert -1.0 <= result.total_return <= 10.0  # sanity bounds

    def test_sharpe_ratio_is_finite(self, prices, signals):
        result = self._run_backtest(prices, signals, CompositeScoreStrategy())
        assert np.isfinite(result.sharpe_ratio)

    def test_max_drawdown_is_negative_or_zero(self, prices, signals):
        result = self._run_backtest(prices, signals, CompositeScoreStrategy())
        assert result.max_drawdown <= 0

    def test_win_rate_in_valid_range(self, prices, signals):
        result = self._run_backtest(prices, signals, CompositeScoreStrategy())
        assert 0.0 <= result.win_rate <= 1.0

    def test_equity_curve_starts_at_initial_capital(self, prices, signals):
        result = self._run_backtest(prices, signals, CompositeScoreStrategy())
        assert abs(result.equity_curve[0]["value"] - 10_000) < 1.0

    def test_equity_curve_length_matches_price_data(self, prices, signals):
        result = self._run_backtest(prices, signals, CompositeScoreStrategy())
        assert len(result.equity_curve) == len(prices)

    def test_trade_log_entries_have_required_fields(self, prices, signals):
        result = self._run_backtest(prices, signals, CompositeScoreStrategy())
        for trade in result.trade_log:
            assert "entry_date" in trade
            assert "exit_date" in trade
            assert "return" in trade
            assert "duration_days" in trade
            assert "profitable" in trade

    def test_all_strategies_produce_results(self, prices, signals):
        strategies = [
            CompositeScoreStrategy(),
            RSIMeanReversionStrategy(),
            GoldenCrossStrategy(),
            MACDCrossoverStrategy(),
        ]
        for strategy in strategies:
            result = self._run_backtest(prices, signals, strategy)
            assert isinstance(result, BacktestResult)
            assert result.strategy_name == strategy.name
