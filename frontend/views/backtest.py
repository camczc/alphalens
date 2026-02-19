"""
frontend/pages/backtest.py

Backtest page — configure and visualize a strategy backtest.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
from frontend.api_client import run_backtest


STRATEGY_DESCRIPTIONS = {
    "composite": "Uses the AlphaLens composite score (RSI + MACD + Bollinger + Trend + EMA). The most sophisticated strategy.",
    "rsi": "Classic mean reversion: buy when RSI < 30 (oversold), sell when RSI > 70 (overbought).",
    "golden_cross": "Trend following: buy when SMA-50 crosses above SMA-200 (golden cross), sell on death cross.",
    "macd": "Momentum: buy when MACD histogram turns positive, sell when it turns negative.",
}


def render():
    st.title("📊 Backtest")
    st.caption("Simulate a trading strategy on historical data and measure performance")

    # ------------------------------------------------------------------
    # Config sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.subheader("Backtest Config")
        symbol = st.text_input("Ticker", value="AAPL").upper().strip()
        strategy = st.selectbox(
            "Strategy",
            options=list(STRATEGY_DESCRIPTIONS.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
        )
        st.caption(STRATEGY_DESCRIPTIONS[strategy])

        col1, col2 = st.columns(2)
        start = col1.date_input("Start", value=date(2020, 1, 1))
        end = col2.date_input("End", value=date.today())
        capital = st.number_input("Initial Capital ($)", value=10_000, step=1_000, min_value=1_000)
        run_btn = st.button("Run Backtest", type="primary", disabled=not symbol)

    # ------------------------------------------------------------------
    # Run backtest
    # ------------------------------------------------------------------
    if run_btn:
        with st.spinner(f"Backtesting {strategy} on {symbol}..."):
            result = run_backtest(
                symbol=symbol,
                strategy=strategy,
                start_date=str(start),
                end_date=str(end),
                initial_capital=capital,
            )

        if result:
            st.session_state["backtest_result"] = result

    result = st.session_state.get("backtest_result")

    if not result:
        _render_landing()
        return

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    _render_metrics(result)
    _render_equity_curve(result)
    _render_trade_log(result)


def _render_metrics(r: dict):
    alpha = r["total_return"] - r["benchmark_return"]

    st.subheader(f"Results — {r['symbol']} · {r['strategy_name']}")
    st.caption(f"{r['start_date']} → {r['end_date']} · ${r['initial_capital']:,.0f} initial capital")

    # Top row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total Return",
        f"{r['total_return']:+.1%}",
        delta=f"{alpha:+.1%} vs SPY",
        delta_color="normal",
    )
    c2.metric("Sharpe Ratio", f"{r['sharpe_ratio']:.3f}", help="Risk-adjusted return. >1 is good, >2 is excellent.")
    c3.metric("Max Drawdown", f"{r['max_drawdown']:.1%}", delta_color="inverse")
    c4.metric("Win Rate", f"{r['win_rate']:.1%}")

    # Second row
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("CAGR", f"{r['annualized_return']:+.1%}")
    c6.metric("Sortino Ratio", f"{r['sortino_ratio']:.3f}", help="Like Sharpe but only penalizes downside volatility.")
    c7.metric("Calmar Ratio", f"{r['calmar_ratio']:.3f}", help="CAGR / |Max Drawdown|. Higher is better.")
    c8.metric("Total Trades", r["total_trades"])

    # Benchmark comparison
    st.caption(f"**SPY Benchmark Return (same period):** {r['benchmark_return']:+.1%} · **Alpha:** {alpha:+.1%}")


def _render_equity_curve(r: dict):
    st.divider()
    st.subheader("Equity Curve")

    df = pd.DataFrame(r["equity_curve"])
    if df.empty:
        st.warning("No equity curve data.")
        return

    df["date"] = pd.to_datetime(df["date"])
    final_val = df["value"].iloc[-1]
    final_bench = df["benchmark_value"].iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["value"],
        mode="lines", name=f"{r['strategy_name']} (${final_val:,.0f})",
        line=dict(color="#00b4d8", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,180,216,0.07)"
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["benchmark_value"],
        mode="lines", name=f"SPY Buy & Hold (${final_bench:,.0f})",
        line=dict(color="#adb5bd", width=1.5, dash="dash"),
    ))
    fig.add_hline(
        y=r["initial_capital"],
        line_dash="dot", line_color="#6c757d", line_width=1,
        annotation_text="Initial Capital"
    )
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_tickprefix="$",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown chart
    rolling_max = df["value"].cummax()
    drawdown = (df["value"] - rolling_max) / rolling_max * 100

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["date"], y=drawdown,
        mode="lines", name="Drawdown",
        line=dict(color="#ef476f", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,71,111,0.15)"
    ))
    fig2.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_ticksuffix="%",
        showlegend=False,
        title=dict(text="Drawdown (%)", font=dict(size=13)),
    )
    st.plotly_chart(fig2, use_container_width=True)


def _render_trade_log(r: dict):
    trades = r.get("trade_log", [])
    if not trades:
        return

    st.divider()
    st.subheader(f"Trade Log ({len(trades)} trades)")

    df = pd.DataFrame(trades)
    df["return"] = df["return"].apply(lambda x: f"{x:+.2%}")
    df["profitable"] = df["profitable"].apply(lambda x: "✅" if x else "❌")
    df = df.rename(columns={
        "entry_date": "Entry",
        "exit_date": "Exit",
        "entry_price": "Entry $",
        "exit_price": "Exit $",
        "return": "Return",
        "duration_days": "Days",
        "profitable": "Win",
    })
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_landing():
    st.divider()
    st.markdown("""
    ### Configure a backtest in the sidebar to get started.

    **Available strategies:**
    | Strategy | Logic | Best for |
    |---|---|---|
    | Composite | Multi-signal aggregate score | General use |
    | RSI Mean Reversion | Buy oversold, sell overbought | Sideways/volatile markets |
    | Golden Cross | SMA50 vs SMA200 crossover | Trending markets |
    | MACD Crossover | MACD histogram sign change | Momentum plays |

    **Metrics explained:**
    - **Sharpe Ratio** — return per unit of risk. >1 is solid, >2 is excellent
    - **Max Drawdown** — worst peak-to-trough loss during the period
    - **Calmar Ratio** — CAGR divided by max drawdown (reward/risk)
    - **Win Rate** — % of individual trades that were profitable
    """)
