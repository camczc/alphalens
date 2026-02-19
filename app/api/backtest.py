"""
app/api/backtest.py

Endpoints:
  POST /backtest              - run a backtest
  GET  /backtest              - list saved backtest runs
  GET  /backtest/{id}         - get a specific backtest result
  POST /backtest/compare      - run all strategies and return comparison
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from app.db.session import get_db
from app.db.models import BacktestRun
from app.models.schemas import BacktestRequest, BacktestResponse, EquityCurvePoint
from app.services.backtester import (
    BacktestEngine,
    CompositeScoreStrategy,
    RSIMeanReversionStrategy,
    GoldenCrossStrategy,
    MACDCrossoverStrategy,
)

router = APIRouter()

STRATEGY_MAP = {
    "composite":    CompositeScoreStrategy,
    "rsi":          RSIMeanReversionStrategy,
    "golden_cross": GoldenCrossStrategy,
    "macd":         MACDCrossoverStrategy,
}


def result_to_response(result, backtest_id: Optional[int] = None) -> BacktestResponse:
    return BacktestResponse(
        symbol=result.symbol,
        strategy_name=result.strategy_name,
        strategy_params=result.strategy_params,
        start_date=result.start_date,
        end_date=result.end_date,
        initial_capital=result.initial_capital,
        total_return=result.total_return,
        annualized_return=result.annualized_return,
        benchmark_return=result.benchmark_return,
        alpha=result.total_return - result.benchmark_return,
        sharpe_ratio=result.sharpe_ratio,
        sortino_ratio=result.sortino_ratio,
        calmar_ratio=result.calmar_ratio,
        max_drawdown=result.max_drawdown,
        volatility_annualized=result.volatility_annualized,
        win_rate=result.win_rate,
        total_trades=result.total_trades,
        avg_trade_duration_days=result.avg_trade_duration_days,
        equity_curve=[EquityCurvePoint(**p) for p in result.equity_curve],
        trade_log=result.trade_log,
        backtest_id=backtest_id,
    )


@router.post("", response_model=BacktestResponse)
def run_backtest(body: BacktestRequest, db: Session = Depends(get_db)):
    """
    Run a backtest for a given ticker and strategy.

    Returns full performance metrics including Sharpe ratio, max drawdown,
    win rate, equity curve, and trade log — all benchmarked against SPY.
    """
    if body.strategy not in STRATEGY_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{body.strategy}'. Choose from: {list(STRATEGY_MAP.keys())}"
        )

    strategy = STRATEGY_MAP[body.strategy]()
    engine = BacktestEngine(db)

    try:
        result = engine.run(
            symbol=body.symbol.upper(),
            strategy=strategy,
            start_date=body.start_date,
            end_date=body.end_date,
            initial_capital=body.initial_capital,
            commission=body.commission,
            slippage=body.slippage,
        )

        backtest_id = None
        if body.save:
            saved = engine.save(result)
            backtest_id = saved.id

        return result_to_response(result, backtest_id)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")


@router.post("/compare", response_model=list[BacktestResponse])
def compare_strategies(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 10_000.0,
    db: Session = Depends(get_db),
):
    """
    Run all available strategies on a ticker and return results side by side.
    Great for finding which strategy has historically worked best for a stock.
    """
    from datetime import date as date_type
    start = date_type.fromisoformat(start_date) if start_date else date_type(2020, 1, 1)
    end = date_type.fromisoformat(end_date) if end_date else date_type.today()

    engine = BacktestEngine(db)
    results = []

    for strategy_cls in STRATEGY_MAP.values():
        strategy = strategy_cls()
        try:
            result = engine.run(
                symbol=symbol.upper(),
                strategy=strategy,
                start_date=start,
                end_date=end,
                initial_capital=initial_capital,
            )
            results.append(result_to_response(result))
        except Exception as e:
            # Don't fail the whole comparison if one strategy errors
            continue

    if not results:
        raise HTTPException(status_code=500, detail="All strategies failed")

    # Sort by Sharpe ratio descending
    results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
    return results


@router.get("", response_model=list[dict])
def list_backtests(
    symbol: Optional[str] = Query(default=None),
    limit: int = Query(default=20, le=100),
    db: Session = Depends(get_db),
):
    """List saved backtest runs, optionally filtered by symbol."""
    query = db.query(BacktestRun)
    if symbol:
        query = query.filter(BacktestRun.ticker_symbol == symbol.upper())
    runs = query.order_by(BacktestRun.created_at.desc()).limit(limit).all()

    return [{
        "id": r.id,
        "symbol": r.ticker_symbol,
        "strategy": r.strategy_name,
        "start_date": str(r.start_date),
        "end_date": str(r.end_date),
        "total_return": r.total_return,
        "sharpe_ratio": r.sharpe_ratio,
        "max_drawdown": r.max_drawdown,
        "created_at": str(r.created_at),
    } for r in runs]


@router.get("/{backtest_id}", response_model=BacktestResponse)
def get_backtest(backtest_id: int, db: Session = Depends(get_db)):
    """Retrieve a previously saved backtest by ID."""
    run = db.query(BacktestRun).filter(BacktestRun.id == backtest_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

    from app.services.backtester import BacktestResult
    from datetime import date as dt

    result = BacktestResult(
        symbol=run.ticker_symbol,
        strategy_name=run.strategy_name,
        strategy_params=run.strategy_params or {},
        start_date=run.start_date,
        end_date=run.end_date,
        initial_capital=10_000,
        total_return=run.total_return,
        annualized_return=run.annualized_return,
        benchmark_return=run.benchmark_return,
        sharpe_ratio=run.sharpe_ratio,
        sortino_ratio=run.sortino_ratio,
        max_drawdown=run.max_drawdown,
        win_rate=run.win_rate,
        total_trades=run.total_trades,
        avg_trade_duration_days=run.avg_trade_duration_days,
        equity_curve=run.equity_curve or [],
        trade_log=run.trade_log or [],
    )
    return result_to_response(result, backtest_id=run.id)
