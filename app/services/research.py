"""
app/services/research.py

LLM Research Layer

Generates a structured analyst-style research brief for a given ticker by:
  1. Pulling the quant signal summary (from SignalEngine)
  2. Fetching recent news headlines (from yfinance)
  3. Building a rich context prompt
  4. Calling the LLM (Anthropic Claude or OpenAI GPT-4) to generate the brief
  5. Caching results to avoid redundant API calls

The brief includes:
  - Executive summary
  - Technical analysis narrative
  - Recent news sentiment
  - Risk factors
  - Analyst verdict (with confidence level)

Usage:
    service = ResearchService(db)
    brief = await service.generate_brief("NVDA", question="Is NVDA overvalued?")
"""

import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf
from anthropic import Anthropic
from sqlalchemy.orm import Session

from app.services.signals import SignalEngine
from app.services.ingestion import IngestionService
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Simple in-memory cache: {cache_key: (generated_at, brief)}
_brief_cache: dict[str, tuple[datetime, dict]] = {}
CACHE_TTL_HOURS = 4


class ResearchService:

    def __init__(self, db: Session):
        self.db = db
        self.signal_engine = SignalEngine(db)
        self.ingestion = IngestionService(db)
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_brief(
        self,
        symbol: str,
        question: Optional[str] = None,
        use_cache: bool = True,
    ) -> dict:
        """
        Generate a full research brief for a ticker.
        Returns a dict with keys: brief, signal_summary, sources_used, generated_at
        """
        symbol = symbol.upper()
        cache_key = self._cache_key(symbol, question)

        # Check cache
        if use_cache and cache_key in _brief_cache:
            cached_at, cached_result = _brief_cache[cache_key]
            if datetime.utcnow() - cached_at < timedelta(hours=CACHE_TTL_HOURS):
                logger.info(f"Cache hit for {symbol}")
                return cached_result

        logger.info(f"Generating research brief for {symbol}")

        # 1. Get quant signal summary
        try:
            signal_summary = self.signal_engine.get_signal_summary(symbol)
        except ValueError:
            # No signals in DB yet — compute on the fly
            price_df = self.ingestion.get_price_dataframe(symbol)
            signals_df = self.signal_engine._compute_all_signals(price_df)
            current_price = float(price_df["close"].iloc[-1])
            last_row = signals_df.iloc[-1]
            signal_summary = self.signal_engine._build_summary_dict(
                symbol, last_row, current_price
            )

        # 2. Fetch recent price context
        price_context = self._get_price_context(symbol)

        # 3. Fetch recent news
        news_items, sources = self._fetch_news(symbol)

        # 4. Build prompt and call LLM
        prompt = self._build_prompt(symbol, signal_summary, price_context, news_items, question)
        brief_text = self._call_llm(prompt)

        result = {
            "symbol": symbol,
            "generated_at": datetime.utcnow().isoformat(),
            "question": question,
            "brief": brief_text,
            "signal_summary": signal_summary,
            "sources_used": sources,
        }

        # Cache it
        _brief_cache[cache_key] = (datetime.utcnow(), result)
        return result

    # ------------------------------------------------------------------
    # Context gathering
    # ------------------------------------------------------------------

    def _get_price_context(self, symbol: str) -> dict:
        """Gather recent price performance metrics for context."""
        try:
            df = self.ingestion.get_price_dataframe(symbol)
            close = df["close"]

            current = float(close.iloc[-1])
            prev_close = float(close.iloc[-2]) if len(close) > 1 else current

            def pct_change_over(days: int) -> Optional[float]:
                if len(close) > days:
                    return float((current - close.iloc[-(days + 1)]) / close.iloc[-(days + 1)])
                return None

            # 52-week high/low
            year_data = close.iloc[-252:] if len(close) >= 252 else close
            high_52w = float(year_data.max())
            low_52w = float(year_data.min())
            pct_from_high = (current - high_52w) / high_52w

            return {
                "current_price": round(current, 2),
                "prev_close": round(prev_close, 2),
                "day_change_pct": round((current - prev_close) / prev_close * 100, 2),
                "change_1w": pct_change_over(5),
                "change_1m": pct_change_over(21),
                "change_3m": pct_change_over(63),
                "change_1y": pct_change_over(252),
                "high_52w": round(high_52w, 2),
                "low_52w": round(low_52w, 2),
                "pct_from_52w_high": round(pct_from_high * 100, 2),
            }
        except Exception as e:
            logger.warning(f"Could not get price context for {symbol}: {e}")
            return {}

    def _fetch_news(self, symbol: str) -> tuple[list[dict], list[str]]:
        """
        Fetch recent news headlines via yfinance.
        Returns (news_items, source_urls)
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            items = []
            sources = []

            for article in news[:10]:  # cap at 10 articles
                content = article.get("content", {})
                title = content.get("title", "")
                summary = content.get("summary", "")
                provider = content.get("provider", {}).get("displayName", "")
                url = content.get("canonicalUrl", {}).get("url", "")

                if title:
                    items.append({
                        "title": title,
                        "summary": summary[:300] if summary else "",
                        "source": provider,
                    })
                    if url:
                        sources.append(url)

            return items, sources
        except Exception as e:
            logger.warning(f"Could not fetch news for {symbol}: {e}")
            return [], []

    # ------------------------------------------------------------------
    # Prompt engineering
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        symbol: str,
        signal_summary: dict,
        price_context: dict,
        news_items: list[dict],
        question: Optional[str],
    ) -> str:
        """
        Build the LLM prompt. Structured to produce a consistent,
        well-formatted research brief every time.
        """
        ind = signal_summary.get("indicators", {})
        score = signal_summary.get("composite_score", 0)
        overall = signal_summary.get("overall_signal", "NEUTRAL")

        # Format news block
        news_block = ""
        if news_items:
            news_block = "\n".join([
                f"- [{item['source']}] {item['title']}"
                + (f"\n  Summary: {item['summary']}" if item.get("summary") else "")
                for item in news_items
            ])
        else:
            news_block = "No recent news available."

        # Format price performance
        price_block = ""
        if price_context:
            price_block = f"""
Current Price: ${price_context.get('current_price', 'N/A')}
Day Change: {price_context.get('day_change_pct', 'N/A')}%
1-Week Return: {_fmt_pct(price_context.get('change_1w'))}
1-Month Return: {_fmt_pct(price_context.get('change_1m'))}
3-Month Return: {_fmt_pct(price_context.get('change_3m'))}
1-Year Return: {_fmt_pct(price_context.get('change_1y'))}
52-Week High: ${price_context.get('high_52w', 'N/A')} ({price_context.get('pct_from_52w_high', 'N/A')}% from high)
52-Week Low: ${price_context.get('low_52w', 'N/A')}
""".strip()

        question_block = (
            f"\n\nSPECIFIC QUESTION TO ADDRESS: {question}"
            if question else ""
        )

        prompt = f"""You are a quantitative equity research analyst. Generate a structured research brief for {symbol}.

=== QUANTITATIVE SIGNAL DATA ===
Composite Score: {score:.3f} (range: -1.0 bearish to +1.0 bullish)
Overall Signal: {overall}

Technical Indicators:
- RSI-14: {ind.get('rsi', {}).get('value', 'N/A')} → {ind.get('rsi', {}).get('interpretation', '')}
- MACD Histogram: {ind.get('macd', {}).get('histogram', 'N/A')} → {ind.get('macd', {}).get('interpretation', '')}
- Bollinger %B: {ind.get('bollinger', {}).get('pct_b', 'N/A')} → {ind.get('bollinger', {}).get('interpretation', '')}
- Trend: {ind.get('trend', {}).get('interpretation', 'N/A')}
- Above SMA-20/50/200: {ind.get('trend', {}).get('above_sma_20')}/{ind.get('trend', {}).get('above_sma_50')}/{ind.get('trend', {}).get('above_sma_200')}
- Golden Cross: {ind.get('trend', {}).get('golden_cross')}

=== PRICE PERFORMANCE ===
{price_block}

=== RECENT NEWS ===
{news_block}
{question_block}

=== INSTRUCTIONS ===
Write a concise, professional research brief using the exact markdown structure below.
Base your analysis strictly on the data provided. Do not fabricate earnings estimates,
price targets, or analyst ratings. Be specific — cite the actual indicator values.

---

## {symbol} Research Brief
*Generated: {datetime.utcnow().strftime('%B %d, %Y')}*

### Executive Summary
[2-3 sentence overview: current signal, price momentum, and key thesis]

### Technical Analysis
[3-4 sentences interpreting RSI, MACD, Bollinger Bands, and moving average trend.
Cite specific values. Explain what they mean together, not just individually.]

### Price Momentum
[2-3 sentences on recent price performance across timeframes. Note proximity to 52w high/low.]

### News & Sentiment
[2-3 sentences synthesizing the news headlines. Note dominant themes and any catalysts.]

### Risk Factors
[2-3 bullet points of key risks given current technical setup]

### Analyst Verdict
**Signal:** {overall}
**Confidence:** [Low / Medium / High — based on signal agreement across indicators]
[2-3 sentences with the bottom line and what to watch for confirmation or invalidation]

---
*This brief is generated by AlphaLens and is for informational purposes only. Not financial advice.*
"""
        return prompt

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Call Claude to generate the research brief."""
        try:
            message = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"Failed to generate research brief: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(symbol: str, question: Optional[str]) -> str:
        raw = f"{symbol}:{question or ''}:{datetime.utcnow().strftime('%Y-%m-%d-%H')}"
        return hashlib.md5(raw.encode()).hexdigest()


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:+.2f}%"
