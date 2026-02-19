"""
scripts/seed_data.py

Seed historical price data for one or more tickers:
    python scripts/seed_data.py --ticker AAPL
    python scripts/seed_data.py --ticker AAPL MSFT NVDA --years 3
"""

import sys
import os
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import SessionLocal, init_db
from app.services.ingestion import IngestionService


def main():
    parser = argparse.ArgumentParser(description="Seed historical price data")
    parser.add_argument("--ticker", nargs="+", required=True, help="Ticker symbol(s)")
    parser.add_argument("--years", type=int, default=5, help="Years of history to fetch")
    args = parser.parse_args()

    init_db()

    db = SessionLocal()
    service = IngestionService(db)

    for i, symbol in enumerate(args.ticker):
        if i > 0:
            print("  Waiting 5s to avoid rate limiting...")
            time.sleep(5)

        print(f"\nSeeding {symbol}...")
        try:
            rows = service.fetch_price_history(symbol)
            print(f"  ✓ {rows} rows inserted for {symbol}")
        except Exception as e:
            print(f"  ✗ Error for {symbol}: {e}")
            print("  Waiting 15s before retrying...")
            time.sleep(15)
            try:
                rows = service.fetch_price_history(symbol)
                print(f"  ✓ {rows} rows inserted for {symbol} (retry succeeded)")
            except Exception as e2:
                print(f"  ✗ Failed after retry: {e2}")

    db.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
