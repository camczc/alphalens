"""
frontend/api_client.py

Thin client that wraps all calls to the AlphaLens FastAPI backend.
Centralizes the base URL and error handling so pages stay clean.
"""

import requests
import streamlit as st
from typing import Optional

BASE_URL = "http://localhost:8000"


def _get(endpoint: str, params: dict = None) -> dict | list | None:
    try:
        resp = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to AlphaLens API. Make sure the server is running: `uvicorn main:app --reload`")
        return None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e))
        st.error(f"API error: {detail}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def _post(endpoint: str, body: dict) -> dict | list | None:
    try:
        resp = requests.post(f"{BASE_URL}{endpoint}", json=body, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to AlphaLens API. Make sure the server is running.")
        return None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e))
        st.error(f"API error: {detail}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# ------------------------------------------------------------------
# API methods
# ------------------------------------------------------------------

def get_signal_summary(symbol: str) -> dict | None:
    return _get(f"/signals/{symbol}")


def get_raw_signals(symbol: str, n: int = 90) -> dict | None:
    return _get(f"/signals/{symbol}/raw", params={"n": n})


def generate_research(symbol: str, question: Optional[str] = None) -> dict | None:
    body = {"symbol": symbol}
    if question:
        body["question"] = question
    return _post("/analyze", body)


def run_backtest(
    symbol: str,
    strategy: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 10_000,
) -> dict | None:
    return _post("/backtest", {
        "symbol": symbol,
        "strategy": strategy,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
    })


def compare_strategies(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 10_000,
) -> list | None:
    return _post("/backtest/compare", params={
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
    })


def add_ticker(symbol: str) -> dict | None:
    return _post("/tickers", {"symbol": symbol})


def list_tickers() -> list | None:
    data = _get("/tickers")
    return data["tickers"] if data else None
