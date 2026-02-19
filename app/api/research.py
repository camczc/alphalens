"""
app/api/research.py

Endpoints:
  POST /analyze          - generate a research brief for a ticker
  GET  /analyze/{symbol} - get the latest cached brief for a ticker
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.schemas import ResearchRequest, ResearchResponse, SignalSummaryResponse
from app.services.research import ResearchService

router = APIRouter()


@router.post("", response_model=ResearchResponse)
def generate_research(body: ResearchRequest, db: Session = Depends(get_db)):
    """
    Generate an AI-powered research brief for a stock.

    Combines quantitative signal data (RSI, MACD, composite score),
    recent price performance, and news headlines into a structured
    analyst-style report written by Claude.

    Results are cached for 4 hours to avoid redundant LLM calls.
    """
    service = ResearchService(db)
    try:
        result = service.generate_brief(
            symbol=body.symbol.upper(),
            question=body.question,
        )

        signal = result["signal_summary"]

        return ResearchResponse(
            symbol=result["symbol"],
            generated_at=datetime.fromisoformat(result["generated_at"]),
            question=result.get("question"),
            brief=result["brief"],
            signal_summary=SignalSummaryResponse(**signal),
            sources_used=result.get("sources_used", []),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research generation failed: {e}")


@router.get("/{symbol}", response_model=ResearchResponse)
def get_cached_research(symbol: str, db: Session = Depends(get_db)):
    """
    Get the most recently cached research brief for a ticker.
    If no cached brief exists, generates a fresh one.
    """
    service = ResearchService(db)
    try:
        result = service.generate_brief(symbol=symbol.upper(), use_cache=True)

        signal = result["signal_summary"]
        return ResearchResponse(
            symbol=result["symbol"],
            generated_at=datetime.fromisoformat(result["generated_at"]),
            question=result.get("question"),
            brief=result["brief"],
            signal_summary=SignalSummaryResponse(**signal),
            sources_used=result.get("sources_used", []),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research generation failed: {e}")
