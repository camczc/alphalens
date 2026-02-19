"""
Data Ingestion Service

Responsibilities:
- Fetch OHLCV price history from yfinance
- Fetch ticker metadata (name, sector, market cap)
- Upsert data into PostgreSQL
- Scheduled refresh via APScheduler
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional
import time

import yfinance as yf
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from app.db.models import Ticker, PriceData
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class IngestionService:

    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------
    # Ticker management
    # ------------------------------------------------------------------

    def add_ticker(self, symbol: str) -> Ticker:
        """Add a new ticker to track. Creates DB record without fetching metadata."""
        symbol = symbol.upper().strip()

        # Check if already exists
        existing = self.db.query(Ticker).filter(Ticker.symbol == symbol).first()
        if existing:
            logger.info(f"Ticker {symbol} already tracked")
            return existing

        # Create ticker record with just the symbol — skip .info to avoid rate limits
        # Metadata (name, sector, etc.) can be enriched later
        ticker = Ticker(symbol=symbol)
        self.db.add(ticker)
        self.db.commit()
        self.db.refresh(ticker)

        logger.info(f"Added ticker {symbol}")
        return ticker

    def enrich_metadata(self, symbol: str) -> Ticker:
        """
        Optionally fetch and store metadata from yfinance.
        Call this separately from add_ticker to avoid rate limits during bulk seeding.
        """
        ticker = self.db.query(Ticker).filter(Ticker.symbol == symbol.upper()).first()
        if not ticker:
            raise ValueError(f"Ticker {symbol} not found")

        try:
            yf_ticker = yf.Ticker(symbol)
            info = yf_ticker.info or {}
            ticker.name = info.get("longName") or info.get("shortName")
            ticker.sector = info.get("sector")
            ticker.industry = info.get("industry")
            ticker.market_cap = info.get("marketCap")
            self.db.commit()
            self.db.refresh(ticker)
            logger.info(f"Enriched metadata for {symbol}")
        except Exception as e:
            logger.warning(f"Could not enrich metadata for {symbol}: {e}")

        return ticker

    def get_or_create_ticker(self, symbol: str) -> Ticker:
        """Get existing ticker or create it."""
        ticker = self.db.query(Ticker).filter(Ticker.symbol == symbol.upper()).first()
        if not ticker:
            ticker = self.add_ticker(symbol)
        return ticker

    # ------------------------------------------------------------------
    # Price data ingestion
    # ------------------------------------------------------------------

    def fetch_price_history(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        interval: str = "1d",
    ) -> int:
        """
        Fetch and store price history for a ticker.
        Returns number of rows inserted/updated.
        """
        ticker = self.get_or_create_ticker(symbol)

        # Default: fetch last N years
        if start_date is None:
            start_date = date.today() - timedelta(days=365 * settings.default_lookback_years)
        if end_date is None:
            end_date = date.today()

        logger.info(f"Fetching {symbol} prices from {start_date} to {end_date} [{interval}]")

        yf_ticker = yf.Ticker(symbol)
        df = yf_ticker.history(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval=interval,
            auto_adjust=False,
        )

        if df.empty:
            logger.warning(f"No price data returned for {symbol}")
            return 0

        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Normalize date column (could be 'date' or 'datetime')
        date_col = "date" if "date" in df.columns else "datetime"
        df["date"] = pd.to_datetime(df[date_col]).dt.date

        rows_upserted = 0
        for _, row in df.iterrows():
            stmt = insert(PriceData).values(
                ticker_id=ticker.id,
                date=row["date"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                adj_close=float(row.get("adj_close", row["close"])),
                volume=float(row["volume"]),
                interval=interval,
            ).on_conflict_do_update(
                constraint="uq_price_ticker_date_interval",
                set_={
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "adj_close": float(row.get("adj_close", row["close"])),
                    "volume": float(row["volume"]),
                }
            )
            self.db.execute(stmt)
            rows_upserted += 1

        self.db.commit()
        logger.info(f"Upserted {rows_upserted} rows for {symbol}")
        return rows_upserted

    def get_price_dataframe(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Load price data from DB into a pandas DataFrame.
        Used by the signal engine and backtester.
        """
        ticker = self.db.query(Ticker).filter(Ticker.symbol == symbol.upper()).first()
        if not ticker:
            raise ValueError(f"Ticker {symbol} not found in DB. Add it first.")

        query = self.db.query(PriceData).filter(
            PriceData.ticker_id == ticker.id,
            PriceData.interval == interval,
        )

        if start_date:
            query = query.filter(PriceData.date >= start_date)
        if end_date:
            query = query.filter(PriceData.date <= end_date)

        rows = query.order_by(PriceData.date.asc()).all()

        if not rows:
            raise ValueError(
                f"No price data in DB for {symbol}. "
                f"Run fetch_price_history() first."
            )

        df = pd.DataFrame([{
            "date": r.date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "adj_close": r.adj_close,
            "volume": r.volume,
        } for r in rows])

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df

    # ------------------------------------------------------------------
    # Refresh (called by scheduler)
    # ------------------------------------------------------------------

    def refresh_all_tickers(self, interval: str = "1d"):
        """
        Incremental refresh — only fetches data since last stored date.
        Called daily by APScheduler.
        """
        tickers = self.db.query(Ticker).filter(Ticker.is_active == True).all()
        logger.info(f"Refreshing {len(tickers)} tickers...")

        for ticker in tickers:
            try:
                last_price = (
                    self.db.query(PriceData)
                    .filter(PriceData.ticker_id == ticker.id, PriceData.interval == interval)
                    .order_by(PriceData.date.desc())
                    .first()
                )
                start = (last_price.date + timedelta(days=1)) if last_price else None
                self.fetch_price_history(ticker.symbol, start_date=start, interval=interval)
                time.sleep(2)  # be polite to Yahoo
            except Exception as e:
                logger.error(f"Failed to refresh {ticker.symbol}: {e}")

        logger.info("Refresh complete")
