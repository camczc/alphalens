"""
Signal Engine

Computes technical indicators and a composite score for a given ticker.

Signals computed:
  Momentum:  RSI-14, MACD, Bollinger Band %
  Trend:     SMA20/50/200, EMA12/26, crossover flags
  Volume:    OBV, VWAP approximation
  Composite: Normalized score from -1 (bearish) to +1 (bullish)

Usage:
    engine = SignalEngine(db)
    engine.compute_and_store(symbol="AAPL")
    df = engine.get_latest_signals(symbol="AAPL", n=30)
"""

import logging
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
import ta  # technical analysis library

from app.db.models import Ticker, Signal
from app.services.ingestion import IngestionService

logger = logging.getLogger(__name__)


class SignalEngine:

    def __init__(self, db: Session):
        self.db = db
        self.ingestion = IngestionService(db)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_and_store(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> int:
        """
        Compute all signals for a ticker and upsert into DB.
        Returns number of rows written.
        """
        df = self.ingestion.get_price_dataframe(symbol, start_date, end_date)

        if len(df) < 30:
            raise ValueError(
                f"Need at least 30 days of price data to compute signals. "
                f"Got {len(df)} rows for {symbol}."
            )

        ticker = self.db.query(Ticker).filter(Ticker.symbol == symbol.upper()).first()
        signals_df = self._compute_all_signals(df)

        rows_written = 0
        for idx, row in signals_df.iterrows():
            stmt = insert(Signal).values(
                ticker_id=ticker.id,
                date=idx.date(),
                rsi_14=self._safe_float(row.get("rsi_14")),
                macd=self._safe_float(row.get("macd")),
                macd_signal=self._safe_float(row.get("macd_signal")),
                macd_hist=self._safe_float(row.get("macd_hist")),
                bb_upper=self._safe_float(row.get("bb_upper")),
                bb_lower=self._safe_float(row.get("bb_lower")),
                bb_pct=self._safe_float(row.get("bb_pct")),
                sma_20=self._safe_float(row.get("sma_20")),
                sma_50=self._safe_float(row.get("sma_50")),
                sma_200=self._safe_float(row.get("sma_200")),
                ema_12=self._safe_float(row.get("ema_12")),
                ema_26=self._safe_float(row.get("ema_26")),
                obv=self._safe_float(row.get("obv")),
                vwap=self._safe_float(row.get("vwap")),
                composite_score=self._safe_float(row.get("composite_score")),
                computed_at=datetime.utcnow(),
            ).on_conflict_do_update(
                constraint="uq_signal_ticker_date",
                set_={
                    "rsi_14": self._safe_float(row.get("rsi_14")),
                    "macd": self._safe_float(row.get("macd")),
                    "macd_signal": self._safe_float(row.get("macd_signal")),
                    "macd_hist": self._safe_float(row.get("macd_hist")),
                    "bb_upper": self._safe_float(row.get("bb_upper")),
                    "bb_lower": self._safe_float(row.get("bb_lower")),
                    "bb_pct": self._safe_float(row.get("bb_pct")),
                    "sma_20": self._safe_float(row.get("sma_20")),
                    "sma_50": self._safe_float(row.get("sma_50")),
                    "sma_200": self._safe_float(row.get("sma_200")),
                    "ema_12": self._safe_float(row.get("ema_12")),
                    "ema_26": self._safe_float(row.get("ema_26")),
                    "obv": self._safe_float(row.get("obv")),
                    "vwap": self._safe_float(row.get("vwap")),
                    "composite_score": self._safe_float(row.get("composite_score")),
                    "computed_at": datetime.utcnow(),
                }
            )
            self.db.execute(stmt)
            rows_written += 1

        self.db.commit()
        logger.info(f"Stored {rows_written} signal rows for {symbol}")
        return rows_written

    def get_latest_signals(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Load the most recent N signal rows from DB as a DataFrame."""
        ticker = self.db.query(Ticker).filter(Ticker.symbol == symbol.upper()).first()
        if not ticker:
            raise ValueError(f"Ticker {symbol} not found")

        rows = (
            self.db.query(Signal)
            .filter(Signal.ticker_id == ticker.id)
            .order_by(Signal.date.desc())
            .limit(n)
            .all()
        )

        records = [{
            "date": r.date,
            "rsi_14": r.rsi_14,
            "macd": r.macd,
            "macd_signal": r.macd_signal,
            "macd_hist": r.macd_hist,
            "bb_upper": r.bb_upper,
            "bb_lower": r.bb_lower,
            "bb_pct": r.bb_pct,
            "sma_20": r.sma_20,
            "sma_50": r.sma_50,
            "sma_200": r.sma_200,
            "ema_12": r.ema_12,
            "ema_26": r.ema_26,
            "obv": r.obv,
            "vwap": r.vwap,
            "composite_score": r.composite_score,
        } for r in rows]

        df = pd.DataFrame(records).set_index("date").sort_index()
        return df

    def get_signal_summary(self, symbol: str) -> dict:
        """
        Returns a human-readable signal summary for a ticker.
        Used by the LLM research layer.
        """
        df = self.get_latest_signals(symbol, n=1)
        if df.empty:
            raise ValueError(f"No signals found for {symbol}")

        row = df.iloc[-1]
        price_df = self.ingestion.get_price_dataframe(symbol)
        current_price = float(price_df["close"].iloc[-1])

        rsi = row["rsi_14"]
        macd_hist = row["macd_hist"]
        bb_pct = row["bb_pct"]
        score = row["composite_score"]

        # Derive human-readable interpretations
        rsi_interp = (
            "oversold (bullish signal)" if rsi < 30
            else "overbought (bearish signal)" if rsi > 70
            else "neutral"
        )
        macd_interp = "bullish (positive histogram)" if macd_hist > 0 else "bearish (negative histogram)"
        bb_interp = (
            "near lower band (potential bounce)" if bb_pct < 0.2
            else "near upper band (potential reversal)" if bb_pct > 0.8
            else "mid-range"
        )
        trend_interp = self._trend_interpretation(row, current_price)

        overall = (
            "STRONG BUY" if score > 0.6
            else "BUY" if score > 0.2
            else "NEUTRAL" if score > -0.2
            else "SELL" if score > -0.6
            else "STRONG SELL"
        )

        return {
            "symbol": symbol,
            "as_of": str(df.index[-1]),
            "current_price": current_price,
            "composite_score": round(score, 3),
            "overall_signal": overall,
            "indicators": {
                "rsi": {"value": round(rsi, 2), "interpretation": rsi_interp},
                "macd": {"histogram": round(macd_hist, 4), "interpretation": macd_interp},
                "bollinger": {"pct_b": round(bb_pct, 3), "interpretation": bb_interp},
                "trend": trend_interp,
            }
        }

    # ------------------------------------------------------------------
    # Signal computation (pure pandas/numpy, no DB)
    # ------------------------------------------------------------------

    def _compute_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Takes a price DataFrame, returns a signals DataFrame."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        signals = pd.DataFrame(index=df.index)

        # --- RSI ---
        signals["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # --- MACD ---
        macd_indicator = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        signals["macd"] = macd_indicator.macd()
        signals["macd_signal"] = macd_indicator.macd_signal()
        signals["macd_hist"] = macd_indicator.macd_diff()

        # --- Bollinger Bands ---
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        signals["bb_upper"] = bb.bollinger_hband()
        signals["bb_lower"] = bb.bollinger_lband()
        signals["bb_pct"] = bb.bollinger_pband()  # 0=lower band, 1=upper band

        # --- Moving Averages ---
        signals["sma_20"] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        signals["sma_50"] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
        signals["sma_200"] = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
        signals["ema_12"] = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
        signals["ema_26"] = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()

        # --- Volume ---
        signals["obv"] = ta.volume.OnBalanceVolumeIndicator(
            close=close, volume=volume
        ).on_balance_volume()

        # VWAP approximation (daily): (H+L+C)/3 * Volume / cumulative volume
        # True VWAP resets intraday; for daily bars we use a 20-day rolling version
        typical_price = (high + low + close) / 3
        signals["vwap"] = (
            (typical_price * volume).rolling(20).sum() /
            volume.rolling(20).sum()
        )

        # --- Composite Score ---
        signals["composite_score"] = self._compute_composite_score(signals, close)

        return signals

    def _compute_composite_score(self, signals: pd.DataFrame, close: pd.Series) -> pd.Series:
        """
        Combines individual signals into a single score from -1 to +1.

        Scoring logic:
          RSI:      <30 = +1, >70 = -1, linear interpolation between
          MACD:     sign of histogram, scaled by magnitude
          BB %:     <0.2 = +0.5, >0.8 = -0.5, else 0
          Trend:    +1 if close > SMA50 > SMA200, -1 if below both
          EMA cross: +0.5 if EMA12 > EMA26, -0.5 otherwise

        Weights: RSI 30%, MACD 25%, BB 15%, Trend 20%, EMA 10%
        """
        scores = pd.DataFrame(index=signals.index)

        # RSI score: map [0,100] → [+1,-1] with oversold=bullish
        scores["rsi_score"] = (50 - signals["rsi_14"]) / 50
        scores["rsi_score"] = scores["rsi_score"].clip(-1, 1)

        # MACD score: normalize histogram by a rolling std
        macd_std = signals["macd_hist"].rolling(60).std()
        scores["macd_score"] = (signals["macd_hist"] / macd_std).clip(-1, 1).fillna(0)

        # Bollinger score
        scores["bb_score"] = 0.0
        scores.loc[signals["bb_pct"] < 0.2, "bb_score"] = 0.5
        scores.loc[signals["bb_pct"] > 0.8, "bb_score"] = -0.5

        # Trend score: price position relative to SMAs
        trend_score = pd.Series(0.0, index=signals.index)
        above_50 = close > signals["sma_50"]
        above_200 = close > signals["sma_200"]
        sma50_above_200 = signals["sma_50"] > signals["sma_200"]

        trend_score[above_50 & above_200 & sma50_above_200] = 1.0   # golden cross zone
        trend_score[above_50 & above_200] = 0.5
        trend_score[~above_50 & ~above_200 & ~sma50_above_200] = -1.0  # death cross zone
        trend_score[~above_50 & ~above_200] = -0.5
        scores["trend_score"] = trend_score

        # EMA crossover score
        scores["ema_score"] = np.where(
            signals["ema_12"] > signals["ema_26"], 0.5, -0.5
        )

        # Weighted composite
        composite = (
            0.30 * scores["rsi_score"] +
            0.25 * scores["macd_score"] +
            0.15 * scores["bb_score"] +
            0.20 * scores["trend_score"] +
            0.10 * scores["ema_score"]
        )

        return composite.clip(-1, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_summary_dict(self, symbol: str, row: pd.Series, current_price: float) -> dict:
        """Build a signal summary dict from a signals row. Used by ResearchService."""
        rsi = row.get("rsi_14")
        macd_hist = row.get("macd_hist")
        bb_pct = row.get("bb_pct")
        score = row.get("composite_score", 0) or 0

        rsi_interp = (
            "oversold (bullish signal)" if rsi and rsi < 30
            else "overbought (bearish signal)" if rsi and rsi > 70
            else "neutral"
        )
        macd_interp = "bullish (positive histogram)" if macd_hist and macd_hist > 0 else "bearish (negative histogram)"
        bb_interp = (
            "near lower band (potential bounce)" if bb_pct and bb_pct < 0.2
            else "near upper band (potential reversal)" if bb_pct and bb_pct > 0.8
            else "mid-range"
        )
        trend_interp = self._trend_interpretation(row, current_price)
        overall = (
            "STRONG BUY" if score > 0.6 else "BUY" if score > 0.2
            else "NEUTRAL" if score > -0.2 else "SELL" if score > -0.6
            else "STRONG SELL"
        )

        return {
            "symbol": symbol,
            "as_of": str(row.name.date() if hasattr(row.name, 'date') else ""),
            "current_price": current_price,
            "composite_score": round(float(score), 3),
            "overall_signal": overall,
            "indicators": {
                "rsi": {"value": round(float(rsi), 2) if rsi else None, "interpretation": rsi_interp},
                "macd": {"histogram": round(float(macd_hist), 4) if macd_hist else None, "interpretation": macd_interp},
                "bollinger": {"pct_b": round(float(bb_pct), 3) if bb_pct else None, "interpretation": bb_interp},
                "trend": trend_interp,
            }
        }

    def _trend_interpretation(self, row: pd.Series, current_price: float) -> dict:
        above_sma20 = current_price > row["sma_20"] if row["sma_20"] else None
        above_sma50 = current_price > row["sma_50"] if row["sma_50"] else None
        above_sma200 = current_price > row["sma_200"] if row["sma_200"] else None

        golden_cross = (row["sma_50"] > row["sma_200"]) if (row["sma_50"] and row["sma_200"]) else None

        return {
            "above_sma_20": above_sma20,
            "above_sma_50": above_sma50,
            "above_sma_200": above_sma200,
            "golden_cross": golden_cross,
            "interpretation": (
                "Strong uptrend (golden cross)" if golden_cross and above_sma50
                else "Uptrend" if above_sma50
                else "Death cross (downtrend)" if golden_cross is False and not above_sma50
                else "Downtrend"
            )
        }

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            f = float(val)
            return None if np.isnan(f) or np.isinf(f) else f
        except (TypeError, ValueError):
            return None
