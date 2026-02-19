"""
scripts/run_backtest.py

Run a backtest from the command line and print results.

Usage:
    python scripts/run_backtest.py --ticker AAPL --strategy composite
    python scripts/run_backtest.py --ticker MSFT --strategy rsi --start 2020-01-01 --end 2024-01-01
    python scripts/run_backtest.py --ticker NVDA --strategy all   # compare all strategies
    python scripts/run_backtest.py --ticker AAPL --strategy composite --save  # save to DB
"""

import sys
import os
import argparse
from datetime import date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import SessionLocal, init_db
from app.services.ingestion import IngestionService
from app.services.backtester import (
    BacktestEngine,
    CompositeScoreStrategy,
    RSIMeanReversionStrategy,
    GoldenCrossStrategy,
    MACDCrossoverStrategy,
)

STRATEGIES = {
    "composite": CompositeScoreStrategy(),
    "rsi": RSIMeanReversionStrategy(),
    "golden_cross": GoldenCrossStrategy(),
    "macd": MACDCrossoverStrategy(),
}


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main():
    parser = argparse.ArgumentParser(description="Run a backtest")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol")
    parser.add_argument(
        "--strategy",
        default="composite",
        choices=list(STRATEGIES.keys()) + ["all"],
        help="Strategy to use"
    )
    parser.add_argument("--start", type=parse_date, default=date(2020, 1, 1))
    parser.add_argument("--end", type=parse_date, default=date.today())
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--save", action="store_true", help="Save results to DB")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    ingestion = IngestionService(db)
    engine = BacktestEngine(db)

    # Ensure we have price data
    try:
        ingestion.get_price_dataframe(args.ticker, args.start, args.end)
    except ValueError:
        print(f"No data found for {args.ticker}, fetching...")
        ingestion.fetch_price_history(args.ticker, args.start, args.end)

    strategies_to_run = (
        list(STRATEGIES.values()) if args.strategy == "all"
        else [STRATEGIES[args.strategy]]
    )

    results = []
    for strategy in strategies_to_run:
        print(f"\nRunning {strategy.name} on {args.ticker}...")
        try:
            result = engine.run(
                symbol=args.ticker,
                strategy=strategy,
                start_date=args.start,
                end_date=args.end,
                initial_capital=args.capital,
            )
            print(result.summary())
            results.append(result)

            if args.save:
                saved = engine.save(result)
                print(f"  Saved to DB with id={saved.id}")

        except Exception as e:
            print(f"  Error: {e}")

    # If comparing all strategies, print a comparison table
    if args.strategy == "all" and len(results) > 1:
        print("\n" + "="*70)
        print(f"  STRATEGY COMPARISON — {args.ticker}")
        print("="*70)
        print(f"  {'Strategy':<22} {'Return':>10} {'Sharpe':>8} {'Drawdown':>10} {'Win Rate':>10}")
        print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
        for r in results:
            print(
                f"  {r.strategy_name:<22} "
                f"{r.total_return:>+10.2%} "
                f"{r.sharpe_ratio:>8.3f} "
                f"{r.max_drawdown:>10.2%} "
                f"{r.win_rate:>10.1%}"
            )
        print(f"\n  Benchmark (SPY buy & hold): {results[0].benchmark_return:+.2%}")
        print("="*70)

    db.close()


if __name__ == "__main__":
    main()
