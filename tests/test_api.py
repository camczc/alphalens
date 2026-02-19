"""
tests/test_api.py

Integration tests for the FastAPI layer.
Uses TestClient with a mocked database — no real DB or network needed.

Run:
    pytest tests/test_api.py -v
"""

import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from main import app
from app.db.session import get_db
from app.db.models import Ticker, Signal, BacktestRun


# ------------------------------------------------------------------
# Mock DB setup
# ------------------------------------------------------------------

def make_mock_ticker(symbol="AAPL"):
    t = MagicMock(spec=Ticker)
    t.id = 1
    t.symbol = symbol
    t.name = f"{symbol} Inc."
    t.sector = "Technology"
    t.industry = "Consumer Electronics"
    t.market_cap = 3_000_000_000_000.0
    t.is_active = True
    t.created_at = datetime(2024, 1, 1)
    return t


def make_mock_db():
    db = MagicMock()
    ticker = make_mock_ticker()

    # Mock query chain for Ticker
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.all.return_value = [ticker]
    mock_query.first.return_value = ticker
    mock_query.limit.return_value = mock_query
    db.query.return_value = mock_query

    return db, ticker


@pytest.fixture
def client():
    mock_db, mock_ticker = make_mock_db()

    def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c, mock_db, mock_ticker
    app.dependency_overrides.clear()


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        c, _, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ------------------------------------------------------------------
# Tickers endpoints
# ------------------------------------------------------------------

class TestTickersAPI:

    def test_list_tickers_returns_200(self, client):
        c, db, ticker = client
        resp = c.get("/tickers")
        assert resp.status_code == 200
        data = resp.json()
        assert "tickers" in data
        assert "total" in data

    def test_get_ticker_returns_200(self, client):
        c, db, ticker = client
        resp = c.get("/tickers/AAPL")
        assert resp.status_code == 200
        assert resp.json()["symbol"] == "AAPL"

    def test_get_nonexistent_ticker_returns_404(self, client):
        c, db, _ = client
        db.query.return_value.filter.return_value.first.return_value = None
        resp = c.get("/tickers/FAKE")
        assert resp.status_code == 404

    def test_add_ticker_calls_ingestion_service(self, client):
        c, db, ticker = client
        with patch("app.api.tickers.IngestionService") as mock_svc:
            instance = mock_svc.return_value
            instance.add_ticker.return_value = ticker
            instance.fetch_price_history.return_value = 100
            resp = c.post("/tickers", json={"symbol": "AAPL"})
            assert resp.status_code == 201

    def test_delete_ticker_soft_deletes(self, client):
        c, db, ticker = client
        resp = c.delete("/tickers/AAPL")
        assert resp.status_code == 204
        assert ticker.is_active == False
        db.commit.assert_called()


# ------------------------------------------------------------------
# Signals endpoints
# ------------------------------------------------------------------

class TestSignalsAPI:

    def test_get_signals_calls_engine(self, client):
        c, db, ticker = client
        mock_summary = {
            "symbol": "AAPL",
            "as_of": "2024-01-15",
            "current_price": 185.50,
            "composite_score": 0.42,
            "overall_signal": "BUY",
            "indicators": {
                "rsi": {"value": 45.2, "interpretation": "neutral"},
                "macd": {"histogram": 0.25, "interpretation": "bullish (positive histogram)"},
                "bollinger": {"pct_b": 0.55, "interpretation": "mid-range"},
                "trend": {
                    "above_sma_20": True,
                    "above_sma_50": True,
                    "above_sma_200": True,
                    "golden_cross": True,
                    "interpretation": "Strong uptrend (golden cross)"
                }
            }
        }
        with patch("app.api.signals.SignalEngine") as mock_eng:
            mock_eng.return_value.get_signal_summary.return_value = mock_summary
            resp = c.get("/signals/AAPL")
            assert resp.status_code == 200
            data = resp.json()
            assert data["symbol"] == "AAPL"
            assert data["overall_signal"] == "BUY"
            assert -1 <= data["composite_score"] <= 1

    def test_compute_signals_returns_202(self, client):
        c, db, ticker = client
        with patch("app.api.signals.SignalEngine") as mock_eng:
            mock_eng.return_value.compute_and_store.return_value = 500
            resp = c.post("/signals/AAPL/compute")
            assert resp.status_code == 202


# ------------------------------------------------------------------
# Backtest endpoints
# ------------------------------------------------------------------

class TestBacktestAPI:

    def _mock_backtest_result(self):
        from app.services.backtester import BacktestResult
        return BacktestResult(
            symbol="AAPL",
            strategy_name="CompositeScore",
            strategy_params={"buy_threshold": 0.25},
            start_date=date(2020, 1, 1),
            end_date=date(2024, 1, 1),
            initial_capital=10_000,
            total_return=0.85,
            annualized_return=0.165,
            benchmark_return=0.62,
            sharpe_ratio=1.32,
            sortino_ratio=1.87,
            max_drawdown=-0.22,
            calmar_ratio=0.75,
            volatility_annualized=0.18,
            win_rate=0.58,
            total_trades=24,
            avg_trade_duration_days=61.5,
            equity_curve=[{"date": "2020-01-02", "value": 10000.0, "benchmark_value": 10000.0}],
            trade_log=[],
        )

    def test_run_backtest_returns_200(self, client):
        c, db, ticker = client
        with patch("app.api.backtest.BacktestEngine") as mock_eng:
            mock_eng.return_value.run.return_value = self._mock_backtest_result()
            resp = c.post("/backtest", json={
                "symbol": "AAPL",
                "strategy": "composite",
                "start_date": "2020-01-01",
                "end_date": "2024-01-01",
                "initial_capital": 10000,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["symbol"] == "AAPL"
            assert "sharpe_ratio" in data
            assert "equity_curve" in data
            assert "alpha" in data

    def test_invalid_strategy_returns_400(self, client):
        c, _, _ = client
        resp = c.post("/backtest", json={
            "symbol": "AAPL",
            "strategy": "fake_strategy",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
        })
        assert resp.status_code == 400

    def test_list_backtests_returns_200(self, client):
        c, db, _ = client
        mock_run = MagicMock()
        mock_run.id = 1
        mock_run.ticker_symbol = "AAPL"
        mock_run.strategy_name = "CompositeScore"
        mock_run.start_date = date(2020, 1, 1)
        mock_run.end_date = date(2024, 1, 1)
        mock_run.total_return = 0.85
        mock_run.sharpe_ratio = 1.32
        mock_run.max_drawdown = -0.22
        mock_run.created_at = datetime(2024, 6, 1)
        db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_run]
        db.query.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_run]

        resp = c.get("/backtest")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
