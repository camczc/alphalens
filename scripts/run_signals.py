"""
scripts/run_signals.py

Compute signals for a ticker and print a summary to the console.
Great for sanity-checking before plugging into the API.

Usage:
    python scripts/run_signals.py --ticker AAPL
    python scripts/run_signals.py --ticker AAPL MSFT NVDA
"""

import sys
import os
import argparse
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import SessionLocal, init_db
from app.services.ingestion import IngestionService
from app.services.signals import SignalEngine


def print_summary(summary: dict):
    symbol = summary["symbol"]
    score = summary["composite_score"]
    signal = summary["overall_signal"]
    price = summary["current_price"]
    ind = summary["indicators"]

    bar_len = int((score + 1) / 2 * 20)  # map [-1,1] to [0,20]
    bar = "█" * bar_len + "░" * (20 - bar_len)

    print(f"\n{'='*50}")
    print(f"  {symbol}  |  ${price:.2f}  |  {signal}")
    print(f"  Composite Score: {score:+.3f}")
    print(f"  [{bar}]  -1 (bearish) ←→ +1 (bullish)")
    print(f"{'='*50}")
    print(f"  RSI-14:       {ind['rsi']['value']:.1f}  →  {ind['rsi']['interpretation']}")
    print(f"  MACD Hist:    {ind['macd']['histogram']:.4f}  →  {ind['macd']['interpretation']}")
    print(f"  Bollinger %B: {ind['bollinger']['pct_b']:.3f}  →  {ind['bollinger']['interpretation']}")
    trend = ind["trend"]
    print(f"  Trend:        {trend['interpretation']}")
    print(f"    Above SMA20: {trend['above_sma_20']}  |  SMA50: {trend['above_sma_50']}  |  SMA200: {trend['above_sma_200']}")
    print(f"    Golden Cross: {trend['golden_cross']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compute and print signals for tickers")
    parser.add_argument("--ticker", nargs="+", required=True)
    parser.add_argument("--store", action="store_true", help="Also store signals to DB")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    ingestion = IngestionService(db)
    engine = SignalEngine(db)

    for symbol in args.ticker:
        try:
            # Make sure we have price data
            try:
                ingestion.get_price_dataframe(symbol)
            except ValueError:
                print(f"No price data for {symbol}, fetching now...")
                ingestion.fetch_price_history(symbol)

            if args.store:
                rows = engine.compute_and_store(symbol)
                print(f"Stored {rows} signal rows for {symbol}")
                summary = engine.get_signal_summary(symbol)
            else:
                # Compute in memory without storing
                price_df = ingestion.get_price_dataframe(symbol)
                signals_df = engine._compute_all_signals(price_df)
                current_price = float(price_df["close"].iloc[-1])
                last = signals_df.iloc[-1]
                summary = engine._build_summary_from_row(symbol, last, current_price)

            print_summary(summary)

        except Exception as e:
            print(f"\nError for {symbol}: {e}")

    db.close()


if __name__ == "__main__":
    main()
