# AlphaLens 📈

An AI-powered stock research assistant with a quantitative edge layer.

## Architecture

```
alphalens/
├── app/
│   ├── api/          # FastAPI route handlers
│   ├── core/         # Config, settings, constants
│   ├── db/           # Database models, migrations, session
│   ├── models/       # Pydantic schemas (request/response)
│   ├── services/     # Business logic
│   │   ├── ingestion.py      # Data pipeline (yfinance + polygon)
│   │   ├── signals.py        # Quant signal engine
│   │   ├── backtester.py     # Backtesting engine
│   │   └── research.py       # LLM research layer
├── scripts/          # One-off scripts (seed DB, backfill data)
├── tests/            # Pytest test suite
├── .env.example
├── requirements.txt
└── main.py
```

## Stack
- **Backend:** Python 3.11, FastAPI, SQLAlchemy, PostgreSQL
- **Data:** yfinance, polygon.io, pandas, numpy
- **Quant:** vectorbt, scipy
- **AI:** OpenAI / Anthropic API + LangChain for RAG
- **Infra:** Docker, APScheduler for cron jobs

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Fill in your API keys and DB URL

# 3. Initialize the database
python scripts/init_db.py

# 4. Seed historical data for a ticker
python scripts/seed_data.py --ticker AAPL --years 5

# 5. Run the API
uvicorn main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/analyze/{ticker}` | Full research brief for a ticker |
| GET | `/signals/{ticker}` | Raw quant signals |
| POST | `/backtest` | Run a strategy backtest |
| GET | `/tickers` | List tracked tickers |
| POST | `/tickers` | Add a ticker to track |

## Running the App

```bash
# Terminal 1 — start the API
uvicorn main:app --reload

# Terminal 2 — start the dashboard
streamlit run frontend/app.py
```

Then open:
- **Dashboard** → http://localhost:8501
- **API Docs (Swagger)** → http://localhost:8000/docs

## Quick Start Demo

```bash
# Seed some data
python scripts/seed_data.py --ticker AAPL NVDA MSFT TSLA

# Compute signals
python scripts/run_signals.py --ticker AAPL NVDA --store

# Run a backtest comparison
python scripts/run_backtest.py --ticker NVDA --strategy all --start 2020-01-01
```

## Build Roadmap
- [x] Step 1: Data ingestion pipeline + DB schema
- [x] Step 2: Quant signal engine (RSI, MACD, Bollinger, composite score)
- [x] Step 3: Backtesting engine (Sharpe, Sortino, drawdown, equity curve)
- [x] Step 4: FastAPI endpoints with Swagger docs
- [x] Step 5: LLM research layer (Claude-powered analyst briefs)
- [x] Step 6: Streamlit dashboard (Research, Backtest, Compare pages)
