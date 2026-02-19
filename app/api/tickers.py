"""
app/api/tickers.py

Endpoints:
  GET  /tickers          - list all tracked tickers
  POST /tickers          - add a new ticker to track
  GET  /tickers/{symbol} - get metadata for a single ticker
  DELETE /tickers/{symbol} - stop tracking a ticker
  POST /tickers/{symbol}/refresh - trigger a price data refresh
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Ticker
from app.models.schemas import TickerCreate, TickerResponse, TickerListResponse
from app.services.ingestion import IngestionService

router = APIRouter()


@router.get("", response_model=TickerListResponse)
def list_tickers(db: Session = Depends(get_db)):
    """List all tickers currently being tracked."""
    tickers = db.query(Ticker).filter(Ticker.is_active == True).order_by(Ticker.symbol).all()
    return TickerListResponse(
        tickers=[TickerResponse.model_validate(t) for t in tickers],
        total=len(tickers),
    )


@router.post("", response_model=TickerResponse, status_code=201)
def add_ticker(body: TickerCreate, db: Session = Depends(get_db)):
    """
    Add a new ticker to track and immediately fetch its price history.
    """
    service = IngestionService(db)
    try:
        ticker = service.add_ticker(body.symbol)
        # Kick off price history fetch
        service.fetch_price_history(body.symbol)
        return TickerResponse.model_validate(ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add ticker: {e}")


@router.get("/{symbol}", response_model=TickerResponse)
def get_ticker(symbol: str, db: Session = Depends(get_db)):
    """Get metadata for a single ticker."""
    ticker = db.query(Ticker).filter(Ticker.symbol == symbol.upper()).first()
    if not ticker:
        raise HTTPException(status_code=404, detail=f"Ticker {symbol} not found")
    return TickerResponse.model_validate(ticker)


@router.delete("/{symbol}", status_code=204)
def remove_ticker(symbol: str, db: Session = Depends(get_db)):
    """Stop tracking a ticker (soft delete)."""
    ticker = db.query(Ticker).filter(Ticker.symbol == symbol.upper()).first()
    if not ticker:
        raise HTTPException(status_code=404, detail=f"Ticker {symbol} not found")
    ticker.is_active = False
    db.commit()


@router.post("/{symbol}/refresh", status_code=202)
def refresh_ticker(
    symbol: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Trigger an incremental price data refresh for a ticker.
    Runs in the background and returns immediately.
    """
    ticker = db.query(Ticker).filter(Ticker.symbol == symbol.upper()).first()
    if not ticker:
        raise HTTPException(status_code=404, detail=f"Ticker {symbol} not found")

    service = IngestionService(db)
    background_tasks.add_task(service.fetch_price_history, symbol)

    return {"message": f"Refresh for {symbol} started in background"}
