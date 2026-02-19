"""
app/api/signals.py

Endpoints:
  GET /signals/{symbol}        - get signal summary (human-readable)
  GET /signals/{symbol}/raw    - get raw signal rows as time series
  POST /signals/{symbol}/compute - recompute and store signals
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import date
from typing import Optional

from app.db.session import get_db
from app.db.models import Ticker, Signal
from app.models.schemas import SignalSummaryResponse, RawSignalsResponse, RawSignalRow
from app.services.signals import SignalEngine
from app.services.ingestion import IngestionService

router = APIRouter()


@router.get("/{symbol}", response_model=SignalSummaryResponse)
def get_signal_summary(symbol: str, db: Session = Depends(get_db)):
    """
    Get a human-readable signal summary for a ticker including the composite score,
    individual indicator interpretations, and overall buy/sell signal.
    """
    engine = SignalEngine(db)
    try:
        summary = engine.get_signal_summary(symbol.upper())
        return SignalSummaryResponse(**summary)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing signals: {e}")


@router.get("/{symbol}/raw", response_model=RawSignalsResponse)
def get_raw_signals(
    symbol: str,
    n: int = Query(default=30, ge=1, le=500, description="Number of rows to return"),
    db: Session = Depends(get_db),
):
    """
    Get raw signal time series for a ticker.
    Useful for charting indicator history.
    """
    engine = SignalEngine(db)
    try:
        df = engine.get_latest_signals(symbol.upper(), n=n)
        rows = [
            RawSignalRow(date=idx, **{k: v for k, v in row.items()})
            for idx, row in df.iterrows()
        ]
        return RawSignalsResponse(symbol=symbol.upper(), rows=rows)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{symbol}/compute", status_code=202)
def compute_signals(symbol: str, db: Session = Depends(get_db)):
    """
    Recompute and store all signals for a ticker.
    Call this after refreshing price data.
    """
    ticker = db.query(Ticker).filter(Ticker.symbol == symbol.upper()).first()
    if not ticker:
        raise HTTPException(status_code=404, detail=f"Ticker {symbol} not found")

    engine = SignalEngine(db)
    try:
        rows = engine.compute_and_store(symbol.upper())
        return {"message": f"Computed {rows} signal rows for {symbol.upper()}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing signals: {e}")
