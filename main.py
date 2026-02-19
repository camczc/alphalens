from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="AlphaLens",
    description="AI-powered stock research assistant with quantitative edge layer",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from app.api import tickers, signals, backtest, research
from app.models.schemas import HealthResponse

app.include_router(tickers.router, prefix="/tickers", tags=["Tickers"])
app.include_router(signals.router, prefix="/signals", tags=["Signals"])
app.include_router(backtest.router, prefix="/backtest", tags=["Backtest"])
app.include_router(research.router, prefix="/analyze", tags=["Research"])


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", env=settings.env)
