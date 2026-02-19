"""
Database models for AlphaLens.

Tables:
  - tickers:        Tracked stocks with metadata
  - price_data:     OHLCV daily price history
  - signals:        Computed quant signals per ticker/date
  - backtest_runs:  Saved backtest results
  - news_items:     Raw news articles for RAG
"""

from datetime import datetime, date
from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, Date, Text, JSON, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Ticker(Base):
    __tablename__ = "tickers"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    price_data = relationship("PriceData", back_populates="ticker", cascade="all, delete-orphan")
    signals = relationship("Signal", back_populates="ticker", cascade="all, delete-orphan")
    news_items = relationship("NewsItem", back_populates="ticker", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Ticker {self.symbol}>"


class PriceData(Base):
    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    adj_close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    interval = Column(String(5), default="1d")  # 1d, 1wk, 1mo

    __table_args__ = (
        UniqueConstraint("ticker_id", "date", "interval", name="uq_price_ticker_date_interval"),
    )

    ticker = relationship("Ticker", back_populates="price_data")

    def __repr__(self):
        return f"<PriceData {self.ticker_id} {self.date}>"


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)

    # Momentum signals
    rsi_14 = Column(Float)          # RSI 14-period
    macd = Column(Float)            # MACD line
    macd_signal = Column(Float)     # MACD signal line
    macd_hist = Column(Float)       # MACD histogram
    bb_upper = Column(Float)        # Bollinger Band upper
    bb_lower = Column(Float)        # Bollinger Band lower
    bb_pct = Column(Float)          # % position within bands

    # Trend signals
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)

    # Volume signals
    obv = Column(Float)             # On-balance volume
    vwap = Column(Float)            # Volume weighted avg price

    # Composite score (-1 to 1, bearish to bullish)
    composite_score = Column(Float)

    computed_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker_id", "date", name="uq_signal_ticker_date"),
    )

    ticker = relationship("Ticker", back_populates="signals")

    def __repr__(self):
        return f"<Signal {self.ticker_id} {self.date} score={self.composite_score}>"


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True)
    ticker_symbol = Column(String(10), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    strategy_params = Column(JSON)          # Strategy config dict
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)

    # Performance metrics
    total_return = Column(Float)            # e.g. 0.42 = 42%
    annualized_return = Column(Float)
    benchmark_return = Column(Float)        # SPY return same period
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)            # e.g. -0.25 = -25%
    win_rate = Column(Float)
    total_trades = Column(Integer)
    avg_trade_duration_days = Column(Float)

    # Full results for charting
    equity_curve = Column(JSON)             # List of {date, value}
    trade_log = Column(JSON)                # List of trade dicts

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<BacktestRun {self.ticker_symbol} {self.strategy_name} sharpe={self.sharpe_ratio}>"


class NewsItem(Base):
    __tablename__ = "news_items"

    id = Column(Integer, primary_key=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    headline = Column(String(500), nullable=False)
    summary = Column(Text)
    url = Column(String(1000))
    source = Column(String(100))
    published_at = Column(DateTime, nullable=False, index=True)
    sentiment_score = Column(Float)         # -1 to 1
    embedding = Column(JSON)                # Vector for RAG (stored as list)

    created_at = Column(DateTime, default=datetime.utcnow)

    ticker = relationship("Ticker", back_populates="news_items")

    def __repr__(self):
        return f"<NewsItem {self.ticker_id} {self.published_at}>"
