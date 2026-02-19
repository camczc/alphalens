"""
app/models/schemas.py

Pydantic schemas for all API request and response bodies.
Keeps FastAPI route handlers clean and provides automatic validation + docs.
"""

from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Tickers
# ------------------------------------------------------------------

class TickerCreate(BaseModel):
    symbol: str = Field(..., example="AAPL", description="Stock ticker symbol")


class TickerResponse(BaseModel):
    id: int
    symbol: str
    name: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    market_cap: Optional[float]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TickerListResponse(BaseModel):
    tickers: list[TickerResponse]
    total: int


# ------------------------------------------------------------------
# Price Data
# ------------------------------------------------------------------

class PriceDataResponse(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float

    class Config:
        from_attributes = True


# ------------------------------------------------------------------
# Signals
# ------------------------------------------------------------------

class IndicatorDetail(BaseModel):
    value: Optional[float]
    interpretation: str


class BollingerDetail(BaseModel):
    pct_b: Optional[float]
    interpretation: str


class TrendDetail(BaseModel):
    above_sma_20: Optional[bool]
    above_sma_50: Optional[bool]
    above_sma_200: Optional[bool]
    golden_cross: Optional[bool]
    interpretation: str


class SignalIndicators(BaseModel):
    rsi: IndicatorDetail
    macd: dict
    bollinger: BollingerDetail
    trend: TrendDetail


class SignalSummaryResponse(BaseModel):
    symbol: str
    as_of: str
    current_price: float
    composite_score: float = Field(
        ..., ge=-1, le=1, description="Composite signal score from -1 (bearish) to +1 (bullish)"
    )
    overall_signal: str = Field(
        ..., description="One of: STRONG BUY, BUY, NEUTRAL, SELL, STRONG SELL"
    )
    indicators: SignalIndicators


class RawSignalRow(BaseModel):
    date: date
    rsi_14: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_hist: Optional[float]
    bb_upper: Optional[float]
    bb_lower: Optional[float]
    bb_pct: Optional[float]
    sma_20: Optional[float]
    sma_50: Optional[float]
    sma_200: Optional[float]
    ema_12: Optional[float]
    ema_26: Optional[float]
    obv: Optional[float]
    vwap: Optional[float]
    composite_score: Optional[float]


class RawSignalsResponse(BaseModel):
    symbol: str
    rows: list[RawSignalRow]


# ------------------------------------------------------------------
# Backtest
# ------------------------------------------------------------------

class BacktestRequest(BaseModel):
    symbol: str = Field(..., example="AAPL")
    strategy: str = Field(
        ...,
        example="composite",
        description="One of: composite, rsi, golden_cross, macd"
    )
    start_date: date = Field(default=date(2020, 1, 1))
    end_date: date = Field(default_factory=date.today)
    initial_capital: float = Field(default=10_000.0, gt=0)
    commission: float = Field(default=0.001, ge=0, le=0.05)
    slippage: float = Field(default=0.0005, ge=0, le=0.05)
    save: bool = Field(default=False, description="Persist results to database")


class TradeEntry(BaseModel):
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float = Field(alias="return")
    duration_days: int
    profitable: bool

    class Config:
        populate_by_name = True


class EquityCurvePoint(BaseModel):
    date: str
    value: float
    benchmark_value: float


class BacktestResponse(BaseModel):
    symbol: str
    strategy_name: str
    strategy_params: dict
    start_date: date
    end_date: date
    initial_capital: float

    # Metrics
    total_return: float
    annualized_return: float
    benchmark_return: float
    alpha: float                        # total_return - benchmark_return
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    volatility_annualized: float
    win_rate: float
    total_trades: int
    avg_trade_duration_days: float

    # Series
    equity_curve: list[EquityCurvePoint]
    trade_log: list[dict]

    # Optional DB id if saved
    backtest_id: Optional[int] = None


# ------------------------------------------------------------------
# Research / Analysis
# ------------------------------------------------------------------

class ResearchRequest(BaseModel):
    symbol: str = Field(..., example="NVDA")
    question: Optional[str] = Field(
        default=None,
        example="Is NVDA overvalued right now?",
        description="Optional natural language question to answer in the brief"
    )


class ResearchResponse(BaseModel):
    symbol: str
    generated_at: datetime
    question: Optional[str]
    brief: str                          # Full markdown research brief from LLM
    signal_summary: SignalSummaryResponse
    sources_used: list[str]


# ------------------------------------------------------------------
# Health / Meta
# ------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    env: str
    version: str = "0.1.0"
