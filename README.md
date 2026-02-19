# AlphaLens 📈

An AI-powered stock research platform combining quantitative signal analysis with Claude-generated analyst briefs.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase-blue)

## What it does

- **Signal Scorecard** — computes RSI, MACD, Bollinger Bands, moving averages, OBV, and a composite score (-1 to +1) for any tracked ticker
- **AI Research Briefs** — Claude reads the signal data and writes a structured analyst brief with thesis, risks, and outlook
- **Backtesting Engine** — simulate 4 trading strategies (Composite, RSI Mean Reversion, Golden Cross, MACD Crossover) with full metrics: Sharpe, Sortino, Calmar, max drawdown, win rate, equity curve
- **Strategy Comparison** — run all 4 strategies on the same ticker and see which historically performs best, with a leaderboard and combined equity curves

## Stack

- **Backend:** FastAPI + SQLAlchemy + PostgreSQL (Supabase)
- **Data:** yfinance, pandas, numpy, ta (technical analysis)
- **AI:** Anthropic Claude API
- **Frontend:** Streamlit + Plotly
- **Scheduler:** APScheduler (daily data refresh)

## Architecture

```
alphalens/
├── app/
│   ├── api/              # FastAPI route handlers
│   │   ├── tickers.py    # Ticker management
│   │   ├── signals.py    # Signal computation endpoints
│   │   ├── backtest.py   # Backtesting endpoints
│   │   └── research.py   # AI research brief endpoints
│   ├── core/             # Config + settings
│   ├── db/               # SQLAlchemy models + session
│   ├── models/           # Pydantic request/response schemas
│   └── services/
│       ├── ingestion.py  # yfinance data pipeline
│       ├── signals.py    # Quant signal engine
│       ├── backtester.py # Strategy backtesting engine
│       └── research.py   # Claude-powered research layer
├── frontend/
│   ├── app.py            # Streamlit entry point
│   └── views/            # Research, Backtest, Compare pages
├── scripts/              # CLI tools for seeding and running
└── tests/                # Pytest test suite
```

## Setup

**Prerequisites:** Python 3.11, PostgreSQL (or Supabase free tier)

```bash
# 1. Clone
git clone https://github.com/camczc/alphalens.git
cd alphalens

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp env.example .env
# Edit .env with your DATABASE_URL and ANTHROPIC_API_KEY

# 5. Initialize database
python scripts/init_db.py

# 6. Seed price history
python scripts/seed_data.py --ticker AAPL NVDA MSFT TSLA

# 7. Compute signals
python scripts/run_signals.py --ticker AAPL NVDA MSFT TSLA --store
```

## Running

```bash
# Terminal 1 — API server
uvicorn main:app --reload

# Terminal 2 — Streamlit dashboard
python -m streamlit run frontend/app.py
```

- **Dashboard** → http://localhost:8501
- **API Docs (Swagger)** → http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/signals/{ticker}` | Latest signal scorecard |
| GET | `/analyze/{ticker}` | Claude-generated research brief |
| POST | `/backtest` | Run a strategy backtest |
| POST | `/backtest/compare` | Compare all strategies |
| GET | `/tickers` | List tracked tickers |
| POST | `/tickers` | Add a ticker |

## Strategies

| Strategy | Logic | Best for |
|----------|-------|----------|
| Composite | Multi-signal aggregate score | General use |
| RSI Mean Reversion | Buy oversold, sell overbought | Sideways markets |
| Golden Cross | SMA50 vs SMA200 crossover | Trending markets |
| MACD Crossover | MACD histogram sign change | Momentum plays |

## Environment Variables

```
DATABASE_URL=postgresql://...     # PostgreSQL connection string
ANTHROPIC_API_KEY=sk-ant-...      # Required for AI research briefs
ENV=development
LOG_LEVEL=INFO
```
