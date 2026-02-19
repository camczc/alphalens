"""
Backtesting Engine

Vectorized backtester for signal-driven strategies.

Design philosophy:
  - Vectorized (pandas-based) for speed — no row-by-row loops
  - Realistic: includes transaction costs, slippage, and position sizing
  - Strategy-agnostic: any function that returns a pd.Series of signals works
  - Benchmarked: always compares against SPY buy-and-hold

Usage:
    engine = BacktestEngine(db)

    result = engine.run(
        symbol="AAPL",
        strategy=CompositeScoreStrategy(buy_threshold=0.3, sell_threshold=-0.1),
        start_date=date(2020, 1, 1),
        end_date=date(2024, 12, 31),
        initial_capital=10_000,
    )

    print(result.summary())
    engine.save(result)
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Optional, Protocol
import json

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from app.db.models import BacktestRun
from app.services.ingestion import IngestionService
from app.services.signals import SignalEngine

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05  # annualized, update as needed


# ------------------------------------------------------------------
# Strategy Protocol — any strategy must implement this interface
# ------------------------------------------------------------------

class Strategy(Protocol):
    name: str
    params: dict

    def generate_signals(self, prices: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """
        Returns a pd.Series with values:
          +1  = enter long
           0  = hold / no position
          -1  = exit (go to cash)
        Index must match prices.index.
        """
        ...


# ------------------------------------------------------------------
# Built-in Strategies
# ------------------------------------------------------------------

@dataclass
class CompositeScoreStrategy:
    """
    Trade based on the composite signal score.
    Buy when score crosses above buy_threshold, sell when it drops below sell_threshold.
    """
    buy_threshold: float = 0.25
    sell_threshold: float = -0.10
    name: str = "CompositeScore"

    @property
    def params(self) -> dict:
        return {"buy_threshold": self.buy_threshold, "sell_threshold": self.sell_threshold}

    def generate_signals(self, prices: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        score = signals["composite_score"]
        position = pd.Series(0, index=prices.index, dtype=float)

        in_position = False
        for i in range(len(prices)):
            if pd.isna(score.iloc[i]):
                position.iloc[i] = 0
                continue
            if not in_position and score.iloc[i] > self.buy_threshold:
                in_position = True
            elif in_position and score.iloc[i] < self.sell_threshold:
                in_position = False
            position.iloc[i] = 1 if in_position else 0

        return position


@dataclass
class RSIMeanReversionStrategy:
    """
    Classic mean reversion: buy oversold, sell overbought.
    """
    oversold: float = 30.0
    overbought: float = 70.0
    name: str = "RSIMeanReversion"

    @property
    def params(self) -> dict:
        return {"oversold": self.oversold, "overbought": self.overbought}

    def generate_signals(self, prices: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        rsi = signals["rsi_14"]
        position = pd.Series(0, index=prices.index, dtype=float)

        in_position = False
        for i in range(len(prices)):
            if pd.isna(rsi.iloc[i]):
                continue
            if not in_position and rsi.iloc[i] < self.oversold:
                in_position = True
            elif in_position and rsi.iloc[i] > self.overbought:
                in_position = False
            position.iloc[i] = 1 if in_position else 0

        return position


@dataclass
class GoldenCrossStrategy:
    """
    Classic trend-following: buy on SMA50 > SMA200 (golden cross), sell on death cross.
    """
    name: str = "GoldenCross"

    @property
    def params(self) -> dict:
        return {}

    def generate_signals(self, prices: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        sma50 = signals["sma_50"]
        sma200 = signals["sma_200"]
        position = np.where(sma50 > sma200, 1, 0)
        return pd.Series(position, index=prices.index, dtype=float)


@dataclass
class MACDCrossoverStrategy:
    """
    Buy when MACD histogram turns positive, sell when it turns negative.
    """
    name: str = "MACDCrossover"

    @property
    def params(self) -> dict:
        return {}

    def generate_signals(self, prices: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        hist = signals["macd_hist"]
        position = np.where(hist > 0, 1, 0)
        return pd.Series(position, index=prices.index, dtype=float)


# ------------------------------------------------------------------
# Backtest Result
# ------------------------------------------------------------------

@dataclass
class BacktestResult:
    symbol: str
    strategy_name: str
    strategy_params: dict
    start_date: date
    end_date: date
    initial_capital: float

    # Core performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    benchmark_return: float = 0.0       # SPY buy-and-hold
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    avg_trade_duration_days: float = 0.0
    calmar_ratio: float = 0.0           # CAGR / |max drawdown|
    volatility_annualized: float = 0.0

    # Series data (stored as lists for JSON serialization)
    equity_curve: list = field(default_factory=list)   # [{date, value, benchmark_value}]
    trade_log: list = field(default_factory=list)      # [{entry_date, exit_date, return, ...}]

    def summary(self) -> str:
        alpha = self.total_return - self.benchmark_return
        return (
            f"\n{'='*55}\n"
            f"  Backtest: {self.symbol} | {self.strategy_name}\n"
            f"  Period:   {self.start_date} → {self.end_date}\n"
            f"{'='*55}\n"
            f"  Total Return:       {self.total_return:+.2%}\n"
            f"  Benchmark (SPY):    {self.benchmark_return:+.2%}\n"
            f"  Alpha:              {alpha:+.2%}\n"
            f"  Annualized Return:  {self.annualized_return:+.2%}\n"
            f"  Annualized Vol:     {self.volatility_annualized:.2%}\n"
            f"  Sharpe Ratio:       {self.sharpe_ratio:.3f}\n"
            f"  Sortino Ratio:      {self.sortino_ratio:.3f}\n"
            f"  Calmar Ratio:       {self.calmar_ratio:.3f}\n"
            f"  Max Drawdown:       {self.max_drawdown:.2%}\n"
            f"  Win Rate:           {self.win_rate:.1%}\n"
            f"  Total Trades:       {self.total_trades}\n"
            f"  Avg Trade Duration: {self.avg_trade_duration_days:.1f} days\n"
            f"{'='*55}\n"
        )


# ------------------------------------------------------------------
# Backtesting Engine
# ------------------------------------------------------------------

class BacktestEngine:

    def __init__(self, db: Session):
        self.db = db
        self.ingestion = IngestionService(db)
        self.signal_engine = SignalEngine(db)

    def run(
        self,
        symbol: str,
        strategy,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        initial_capital: float = 10_000.0,
        commission: float = 0.001,      # 0.1% per trade (realistic for retail)
        slippage: float = 0.0005,       # 0.05% slippage on fills
    ) -> BacktestResult:
        """
        Run a backtest for a given strategy and return results.
        """
        logger.info(f"Running backtest: {symbol} | {strategy.name}")

        # --- Load price and signal data ---
        prices = self.ingestion.get_price_dataframe(symbol, start_date, end_date)
        signals_df = self.signal_engine._compute_all_signals(prices)

        # Align indices
        common_idx = prices.index.intersection(signals_df.index)
        prices = prices.loc[common_idx]
        signals_df = signals_df.loc[common_idx]

        if start_date:
            prices = prices[prices.index.date >= start_date]  # type: ignore
            signals_df = signals_df[signals_df.index >= pd.Timestamp(start_date)]
        if end_date:
            prices = prices[prices.index.date <= end_date]  # type: ignore
            signals_df = signals_df[signals_df.index <= pd.Timestamp(end_date)]

        if len(prices) < 30:
            raise ValueError(f"Not enough data for backtest: {len(prices)} rows")

        # --- Generate position signals ---
        raw_signals = strategy.generate_signals(prices, signals_df)

        # --- Simulate portfolio ---
        portfolio = self._simulate_portfolio(
            prices=prices,
            signals=raw_signals,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
        )

        # --- Fetch benchmark (SPY) ---
        benchmark_returns = self._get_benchmark_returns(
            start=prices.index[0].date(),
            end=prices.index[-1].date(),
        )

        # --- Compute metrics ---
        result = self._compute_metrics(
            symbol=symbol,
            strategy=strategy,
            portfolio=portfolio,
            benchmark_returns=benchmark_returns,
            initial_capital=initial_capital,
            start_date=prices.index[0].date(),
            end_date=prices.index[-1].date(),
        )

        logger.info(f"Backtest complete. Sharpe: {result.sharpe_ratio:.3f}, Return: {result.total_return:.2%}")
        return result

    def save(self, result: BacktestResult) -> BacktestRun:
        """Persist a backtest result to the database."""
        run = BacktestRun(
            ticker_symbol=result.symbol,
            strategy_name=result.strategy_name,
            strategy_params=result.strategy_params,
            start_date=result.start_date,
            end_date=result.end_date,
            total_return=result.total_return,
            annualized_return=result.annualized_return,
            benchmark_return=result.benchmark_return,
            sharpe_ratio=result.sharpe_ratio,
            sortino_ratio=result.sortino_ratio,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            total_trades=result.total_trades,
            avg_trade_duration_days=result.avg_trade_duration_days,
            equity_curve=result.equity_curve,
            trade_log=result.trade_log,
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        logger.info(f"Saved backtest run id={run.id}")
        return run

    # ------------------------------------------------------------------
    # Core simulation (vectorized)
    # ------------------------------------------------------------------

    def _simulate_portfolio(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float,
        commission: float,
        slippage: float,
    ) -> pd.DataFrame:
        """
        Vectorized portfolio simulation.
        Returns a DataFrame with daily portfolio value, positions, and trades.
        """
        close = prices["adj_close"].fillna(prices["close"])
        signals = signals.reindex(close.index).fillna(0)

        # Detect position changes (entry/exit)
        position_changes = signals.diff().fillna(signals)
        entries = position_changes > 0    # 0→1
        exits = position_changes < 0     # 1→0

        # Daily returns of the asset
        asset_returns = close.pct_change().fillna(0)

        # Strategy returns = position * asset_returns - costs on trade days
        cost = (commission + slippage) * (entries | exits).astype(float)
        strategy_returns = signals.shift(1).fillna(0) * asset_returns - cost

        # Cumulative portfolio value
        portfolio_value = initial_capital * (1 + strategy_returns).cumprod()

        portfolio = pd.DataFrame({
            "close": close,
            "position": signals,
            "asset_return": asset_returns,
            "strategy_return": strategy_returns,
            "portfolio_value": portfolio_value,
            "is_entry": entries,
            "is_exit": exits,
        })

        return portfolio

    def _build_trade_log(self, portfolio: pd.DataFrame) -> list:
        trades = []
        entry_date = None
        entry_price = None

        for idx, row in portfolio.iterrows():
            if row["is_entry"] and entry_date is None:
                entry_date = idx
                entry_price = row["close"]
            elif row["is_exit"] and entry_date is not None:
                exit_price = row["close"]
                trade_return = (exit_price - entry_price) / entry_price
                duration = (idx - entry_date).days
                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(idx.date()),
                    "entry_price": round(float(entry_price), 4),
                    "exit_price": round(float(exit_price), 4),
                    "return": round(float(trade_return), 6),
                    "duration_days": duration,
                    "profitable": trade_return > 0,
                })
                entry_date = None
                entry_price = None

        return trades

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        symbol: str,
        strategy,
        portfolio: pd.DataFrame,
        benchmark_returns: pd.Series,
        initial_capital: float,
        start_date: date,
        end_date: date,
    ) -> BacktestResult:

        pv = portfolio["portfolio_value"]
        daily_returns = portfolio["strategy_return"]
        trades = self._build_trade_log(portfolio)

        # Total return
        total_return = float((pv.iloc[-1] / initial_capital) - 1)

        # Annualized return (CAGR)
        n_years = len(pv) / TRADING_DAYS_PER_YEAR
        annualized_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

        # Volatility
        vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

        # Sharpe ratio
        excess_daily = daily_returns - (RISK_FREE_RATE / TRADING_DAYS_PER_YEAR)
        sharpe = float(
            excess_daily.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        ) if daily_returns.std() > 0 else 0.0

        # Sortino ratio (only penalizes downside volatility)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = float(downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        sortino = float(
            (annualized_return - RISK_FREE_RATE) / downside_std
        ) if downside_std > 0 else 0.0

        # Max drawdown
        rolling_max = pv.cummax()
        drawdown = (pv - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        # Calmar ratio
        calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

        # Benchmark return (align dates)
        aligned_bench = benchmark_returns.reindex(pv.index).fillna(0)
        benchmark_total = float((1 + aligned_bench).prod() - 1)

        # Trade stats
        win_rate = float(
            sum(1 for t in trades if t["profitable"]) / len(trades)
        ) if trades else 0.0
        avg_duration = float(
            np.mean([t["duration_days"] for t in trades])
        ) if trades else 0.0

        # Equity curve (for charting)
        bench_cumulative = initial_capital * (1 + aligned_bench).cumprod()
        equity_curve = [
            {
                "date": str(idx.date()),
                "value": round(float(v), 2),
                "benchmark_value": round(float(bench_cumulative.get(idx, initial_capital)), 2),
            }
            for idx, v in pv.items()
        ]

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy.name,
            strategy_params=strategy.params,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            benchmark_return=benchmark_total,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            volatility_annualized=vol,
            win_rate=win_rate,
            total_trades=len(trades),
            avg_trade_duration_days=avg_duration,
            equity_curve=equity_curve,
            trade_log=trades,
        )

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def _get_benchmark_returns(self, start: date, end: date) -> pd.Series:
        """Fetch SPY daily returns for the benchmark comparison."""
        try:
            spy = yf.Ticker("SPY")
            df = spy.history(
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                auto_adjust=True,
            )
            returns = df["Close"].pct_change().fillna(0)
            returns.index = pd.to_datetime(returns.index).tz_localize(None)
            return returns
        except Exception as e:
            logger.warning(f"Could not fetch SPY benchmark: {e}")
            return pd.Series(dtype=float)
