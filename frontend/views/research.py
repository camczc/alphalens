"""
frontend/pages/research.py

Research page — search a ticker, get AI brief + signal scorecard.
"""

import streamlit as st
import pandas as pd
from frontend.api_client import get_signal_summary, get_raw_signals, generate_research
import plotly.graph_objects as go


def render():
    st.title("🔍 Stock Research")
    st.caption("AI-generated analyst briefs powered by quantitative signals + Claude")

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    col1, col2 = st.columns([2, 3])
    with col1:
        symbol = st.text_input(
            "Ticker Symbol",
            placeholder="AAPL, NVDA, MSFT...",
            help="Enter any US stock ticker"
        ).upper().strip()

    with col2:
        question = st.text_input(
            "Optional: Ask a specific question",
            placeholder="Is NVDA overextended after its recent run?",
            help="Leave blank for a general research brief"
        )

    generate_btn = st.button("Generate Research Brief", type="primary", disabled=not symbol)

    if not symbol:
        _render_landing()
        return

    # ------------------------------------------------------------------
    # Signal Scorecard (always show, fast)
    # ------------------------------------------------------------------
    with st.spinner(f"Loading signals for {symbol}..."):
        signals = get_signal_summary(symbol)

    if signals:
        _render_signal_scorecard(signals)
        _render_signal_chart(symbol)

    # ------------------------------------------------------------------
    # AI Research Brief (on button click)
    # ------------------------------------------------------------------
    if generate_btn:
        with st.spinner(f"Generating research brief for {symbol}... (may take 10-15s)"):
            result = generate_research(symbol, question if question else None)

        if result:
            st.divider()
            st.subheader("📄 Research Brief")
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            col_meta1.metric("Generated", result.get("generated_at", "")[:10])
            col_meta2.metric("Sources Used", len(result.get("sources_used", [])))
            col_meta3.metric(
                "Signal",
                result.get("signal_summary", {}).get("overall_signal", "—")
            )

            st.markdown(result["brief"])

            if result.get("sources_used"):
                with st.expander("📎 Sources"):
                    for url in result["sources_used"]:
                        st.markdown(f"- {url}")


# ------------------------------------------------------------------
# Components
# ------------------------------------------------------------------

def _render_signal_scorecard(signals: dict):
    score = signals.get("composite_score", 0)
    overall = signals.get("overall_signal", "NEUTRAL")
    price = signals.get("current_price", 0)
    ind = signals.get("indicators", {})

    # Color coding
    signal_colors = {
        "STRONG BUY": "🟢",
        "BUY": "🟩",
        "NEUTRAL": "🟡",
        "SELL": "🟠",
        "STRONG SELL": "🔴",
    }
    emoji = signal_colors.get(overall, "⚪")

    st.divider()
    st.subheader(f"📊 Signal Scorecard — {signals['symbol']}")

    # Top metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${price:,.2f}")
    c2.metric("Composite Score", f"{score:+.3f}", help="-1 = bearish, +1 = bullish")
    c3.metric("Overall Signal", f"{emoji} {overall}")
    c4.metric("As Of", signals.get("as_of", "—"))

    # Score gauge
    fig = _make_gauge(score)
    st.plotly_chart(fig, use_container_width=True)

    # Indicator breakdown
    st.markdown("**Indicator Breakdown**")
    col_a, col_b = st.columns(2)

    with col_a:
        rsi = ind.get("rsi", {})
        _signal_card("RSI-14", rsi.get("value"), rsi.get("interpretation"), suffix="")

        macd = ind.get("macd", {})
        _signal_card("MACD Histogram", macd.get("histogram"), macd.get("interpretation"), fmt=".4f")

    with col_b:
        bb = ind.get("bollinger", {})
        _signal_card("Bollinger %B", bb.get("pct_b"), bb.get("interpretation"), fmt=".3f")

        trend = ind.get("trend", {})
        st.markdown(f"""
**Trend**
- Above SMA-20: {'✅' if trend.get('above_sma_20') else '❌'}  
- Above SMA-50: {'✅' if trend.get('above_sma_50') else '❌'}  
- Above SMA-200: {'✅' if trend.get('above_sma_200') else '❌'}  
- Golden Cross: {'✅' if trend.get('golden_cross') else '❌ Death Cross'}  
- **{trend.get('interpretation', '—')}**
""")


def _render_signal_chart(symbol: str):
    raw = get_raw_signals(symbol, n=90)
    if not raw or not raw.get("rows"):
        return

    df = pd.DataFrame(raw["rows"])
    df["date"] = pd.to_datetime(df["date"])

    with st.expander("📈 Signal History (90 days)", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Composite Score", "RSI", "MACD"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["composite_score"],
                mode="lines", name="Composite Score",
                line=dict(color="#00b4d8", width=2)
            ))
            fig.add_hline(y=0.25, line_dash="dash", line_color="green", annotation_text="Buy threshold")
            fig.add_hline(y=-0.10, line_dash="dash", line_color="red", annotation_text="Sell threshold")
            fig.add_hline(y=0, line_color="gray", line_width=0.5)
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(range=[-1, 1]), showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["rsi_14"],
                mode="lines", name="RSI-14",
                line=dict(color="#f77f00", width=2)
            ))
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(range=[0, 100]), showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            fig = go.Figure()
            colors = ["green" if v >= 0 else "red" for v in df["macd_hist"].fillna(0)]
            fig.add_trace(go.Bar(
                x=df["date"], y=df["macd_hist"],
                name="MACD Histogram", marker_color=colors
            ))
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["macd"],
                mode="lines", name="MACD", line=dict(color="#00b4d8")
            ))
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["macd_signal"],
                mode="lines", name="Signal", line=dict(color="#ff6b6b", dash="dash")
            ))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)


def _signal_card(label: str, value, interpretation: str, suffix: str = "", fmt: str = ".2f"):
    val_str = f"{value:{fmt}}{suffix}" if value is not None else "N/A"
    st.markdown(f"**{label}:** `{val_str}`  \n_{interpretation}_")


def _make_gauge(score: float) -> go.Figure:
    color = (
        "#2dc653" if score > 0.4
        else "#57cc99" if score > 0.1
        else "#ffd166" if score > -0.1
        else "#ef476f" if score > -0.4
        else "#d62828"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "", "font": {"size": 28}},
        gauge={
            "axis": {"range": [-1, 1], "tickvals": [-1, -0.5, 0, 0.5, 1]},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "steps": [
                {"range": [-1, -0.6], "color": "#d62828"},
                {"range": [-0.6, -0.2], "color": "#ef476f"},
                {"range": [-0.2, 0.2], "color": "#ffd166"},
                {"range": [0.2, 0.6], "color": "#57cc99"},
                {"range": [0.6, 1], "color": "#2dc653"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
        domain={"x": [0.2, 0.8], "y": [0, 1]},
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def _render_landing():
    st.divider()
    st.markdown("""
    ### How it works
    1. **Enter a ticker** to instantly see the signal scorecard
    2. **Click Generate** to get a full AI research brief
    3. **Ask a question** to get a targeted analysis
    
    #### What's in the signal scorecard?
    | Indicator | What it measures |
    |---|---|
    | RSI-14 | Momentum — overbought or oversold |
    | MACD | Trend direction and momentum shifts |
    | Bollinger %B | Price position relative to volatility bands |
    | Composite Score | Weighted aggregate of all signals (-1 to +1) |
    """)
