"""
tests/test_research.py

Tests for the LLM research layer.
Mocks the Anthropic client so no real API calls are made.

Run:
    pytest tests/test_research.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from app.services.research import ResearchService, _fmt_pct


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

MOCK_SIGNAL_SUMMARY = {
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

MOCK_PRICE_CONTEXT = {
    "current_price": 185.50,
    "prev_close": 183.20,
    "day_change_pct": 1.25,
    "change_1w": 0.032,
    "change_1m": 0.087,
    "change_3m": 0.154,
    "change_1y": 0.421,
    "high_52w": 199.62,
    "low_52w": 124.17,
    "pct_from_52w_high": -7.07,
}

MOCK_NEWS = [
    {
        "title": "Apple Reports Record Q4 Earnings",
        "summary": "Apple beat analyst expectations with strong iPhone sales.",
        "source": "Reuters",
    },
    {
        "title": "Vision Pro Launch Drives Developer Interest",
        "summary": "Apple's spatial computing headset sees strong developer adoption.",
        "source": "Bloomberg",
    },
]

MOCK_BRIEF = """## AAPL Research Brief
*Generated: January 15, 2024*

### Executive Summary
Apple shows a bullish technical setup with a composite score of 0.42...

### Technical Analysis
RSI at 45.2 is neutral territory...

### Price Momentum
AAPL is up 42.1% over the past year...

### News & Sentiment
Recent earnings beat and Vision Pro launch are positive catalysts...

### Risk Factors
- Valuation remains stretched relative to peers
- China market headwinds

### Analyst Verdict
**Signal:** BUY
**Confidence:** High
The technical setup is constructive across multiple timeframes...
"""


@pytest.fixture
def mock_service():
    db = MagicMock()
    service = ResearchService.__new__(ResearchService)
    service.db = db
    service.signal_engine = MagicMock()
    service.ingestion = MagicMock()
    service.client = MagicMock()

    service.signal_engine.get_signal_summary.return_value = MOCK_SIGNAL_SUMMARY
    service._get_price_context = MagicMock(return_value=MOCK_PRICE_CONTEXT)
    service._fetch_news = MagicMock(return_value=(MOCK_NEWS, ["https://reuters.com/aapl"]))
    service._call_llm = MagicMock(return_value=MOCK_BRIEF)

    return service


# ------------------------------------------------------------------
# Tests: Brief generation
# ------------------------------------------------------------------

class TestResearchService:

    def test_generate_brief_returns_required_keys(self, mock_service):
        result = mock_service.generate_brief("AAPL")
        assert "brief" in result
        assert "signal_summary" in result
        assert "sources_used" in result
        assert "generated_at" in result
        assert "symbol" in result

    def test_generate_brief_symbol_is_uppercased(self, mock_service):
        result = mock_service.generate_brief("aapl")
        assert result["symbol"] == "AAPL"

    def test_generate_brief_calls_signal_engine(self, mock_service):
        mock_service.generate_brief("AAPL")
        mock_service.signal_engine.get_signal_summary.assert_called_once_with("AAPL")

    def test_generate_brief_calls_llm(self, mock_service):
        mock_service.generate_brief("AAPL")
        mock_service._call_llm.assert_called_once()

    def test_generate_brief_with_question(self, mock_service):
        result = mock_service.generate_brief("AAPL", question="Is AAPL overvalued?")
        assert result["question"] == "Is AAPL overvalued?"
        # Question should appear in the prompt sent to LLM
        prompt_arg = mock_service._call_llm.call_args[0][0]
        assert "Is AAPL overvalued?" in prompt_arg

    def test_generate_brief_without_question(self, mock_service):
        result = mock_service.generate_brief("AAPL")
        assert result["question"] is None

    def test_brief_text_is_returned_from_llm(self, mock_service):
        result = mock_service.generate_brief("AAPL")
        assert result["brief"] == MOCK_BRIEF

    def test_sources_are_included(self, mock_service):
        result = mock_service.generate_brief("AAPL")
        assert len(result["sources_used"]) > 0

    def test_caching_prevents_duplicate_llm_calls(self, mock_service):
        # Patch cache to be empty, then call twice
        import app.services.research as research_module
        research_module._brief_cache.clear()

        mock_service.generate_brief("MSFT", use_cache=True)
        mock_service.generate_brief("MSFT", use_cache=True)

        # LLM should only be called once (second call hits cache)
        assert mock_service._call_llm.call_count == 1

    def test_cache_bypass_forces_fresh_generation(self, mock_service):
        import app.services.research as research_module
        research_module._brief_cache.clear()

        mock_service.generate_brief("TSLA", use_cache=False)
        mock_service.generate_brief("TSLA", use_cache=False)

        assert mock_service._call_llm.call_count == 2

    def test_llm_failure_raises_runtime_error(self, mock_service):
        import app.services.research as research_module
        research_module._brief_cache.clear()
        mock_service._call_llm.side_effect = RuntimeError("API quota exceeded")

        with pytest.raises(RuntimeError):
            mock_service.generate_brief("NVDA", use_cache=False)


# ------------------------------------------------------------------
# Tests: Prompt construction
# ------------------------------------------------------------------

class TestPromptConstruction:

    def test_prompt_contains_symbol(self, mock_service):
        prompt = mock_service._build_prompt(
            "NVDA", MOCK_SIGNAL_SUMMARY, MOCK_PRICE_CONTEXT, MOCK_NEWS, None
        )
        assert "NVDA" in prompt

    def test_prompt_contains_composite_score(self, mock_service):
        prompt = mock_service._build_prompt(
            "AAPL", MOCK_SIGNAL_SUMMARY, MOCK_PRICE_CONTEXT, MOCK_NEWS, None
        )
        assert "0.42" in prompt

    def test_prompt_contains_rsi_value(self, mock_service):
        prompt = mock_service._build_prompt(
            "AAPL", MOCK_SIGNAL_SUMMARY, MOCK_PRICE_CONTEXT, MOCK_NEWS, None
        )
        assert "45.2" in prompt

    def test_prompt_contains_news_headlines(self, mock_service):
        prompt = mock_service._build_prompt(
            "AAPL", MOCK_SIGNAL_SUMMARY, MOCK_PRICE_CONTEXT, MOCK_NEWS, None
        )
        assert "Apple Reports Record Q4 Earnings" in prompt

    def test_prompt_contains_question_when_provided(self, mock_service):
        prompt = mock_service._build_prompt(
            "AAPL", MOCK_SIGNAL_SUMMARY, MOCK_PRICE_CONTEXT, MOCK_NEWS,
            "Should I buy AAPL today?"
        )
        assert "Should I buy AAPL today?" in prompt

    def test_prompt_contains_price_performance(self, mock_service):
        prompt = mock_service._build_prompt(
            "AAPL", MOCK_SIGNAL_SUMMARY, MOCK_PRICE_CONTEXT, MOCK_NEWS, None
        )
        assert "185.50" in prompt

    def test_prompt_contains_disclaimer(self, mock_service):
        prompt = mock_service._build_prompt(
            "AAPL", MOCK_SIGNAL_SUMMARY, MOCK_PRICE_CONTEXT, MOCK_NEWS, None
        )
        assert "Not financial advice" in prompt


# ------------------------------------------------------------------
# Tests: Helpers
# ------------------------------------------------------------------

class TestHelpers:

    def test_fmt_pct_positive(self):
        assert _fmt_pct(0.154) == "+15.40%"

    def test_fmt_pct_negative(self):
        assert _fmt_pct(-0.087) == "-8.70%"

    def test_fmt_pct_none(self):
        assert _fmt_pct(None) == "N/A"

    def test_fmt_pct_zero(self):
        assert _fmt_pct(0.0) == "+0.00%"
