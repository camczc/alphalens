"""
frontend/pages/compare.py

Compare all 4 strategies on a single ticker side by side.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import date
from frontend.api_client import BASE_URL


STRATEGY_COLORS = {
    "CompositeScore":    "#00b4d8",
    "RSIMeanReversion":  "#f77f00",
    "GoldenCross":       "#2dc653",
    "MACDCrossover":     "#9b5de5",
}


def render():
    st.title("⚖️ Strategy Comparison")
    st.caption("Run all 4 strategies on the same ticker and see which performs best")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    symbol = col1.text_input("Ticker", value="AAPL").upper().strip()
    start = col2.date_input("Start", value=date(2020, 1, 1))
    end = col3.date_input("End", value=date.today())
    capital = col4.number_input("Capital ($)", value=10_000, step=1_000, min_value=1_000)

    run_btn = st.button("Compare All Strategies", type="primary", disabled=not symbol)

    if run_btn:
        with st.spinner(f"Running all strategies on {symbol}... (this takes ~15s)"):
            try:
                resp = requests.post(
                    f"{BASE_URL}/backtest/compare",
                    params={
                        "symbol": symbol,
                        "start_date": str(start),
                        "end_date": str(end),
                        "initial_capital": capital,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                results = resp.json()
                st.session_state["compare_results"] = results
                st.session_state["compare_symbol"] = symbol
            except Exception as e:
                st.error(f"Comparison failed: {e}")
                return

    results = st.session_state.get("compare_results")
    if not results:
        _render_landing()
        return

    _render_comparison_table(results)
    _render_combined_equity_curve(results)
    _render_radar_chart(results)


def _render_comparison_table(results: list):
    st.divider()
    st.subheader("Strategy Leaderboard")
    st.caption("Sorted by Sharpe Ratio (risk-adjusted return)")

    rows = []
    benchmark = results[0]["benchmark_return"] if results else 0

    for r in results:
        rows.append({
            "Strategy": r["strategy_name"],
            "Total Return": f"{r['total_return']:+.1%}",
            "Alpha vs SPY": f"{r['total_return'] - benchmark:+.1%}",
            "CAGR": f"{r['annualized_return']:+.1%}",
            "Sharpe": f"{r['sharpe_ratio']:.3f}",
            "Sortino": f"{r['sortino_ratio']:.3f}",
            "Max Drawdown": f"{r['max_drawdown']:.1%}",
            "Win Rate": f"{r['win_rate']:.1%}",
            "Trades": r["total_trades"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    spy_return = results[0]["benchmark_return"]
    st.caption(f"📌 SPY Buy & Hold (same period): {spy_return:+.1%}")


def _render_combined_equity_curve(results: list):
    st.divider()
    st.subheader("Equity Curves — All Strategies vs SPY")

    fig = go.Figure()
    benchmark_added = False

    for r in results:
        if not r.get("equity_curve"):
            continue

        df = pd.DataFrame(r["equity_curve"])
        df["date"] = pd.to_datetime(df["date"])
        color = STRATEGY_COLORS.get(r["strategy_name"], "#888")
        final = df["value"].iloc[-1]

        fig.add_trace(go.Scatter(
            x=df["date"], y=df["value"],
            mode="lines",
            name=f"{r['strategy_name']} (${final:,.0f})",
            line=dict(color=color, width=2.5),
        ))

        if not benchmark_added:
            bench_final = df["benchmark_value"].iloc[-1]
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["benchmark_value"],
                mode="lines",
                name=f"SPY Buy & Hold (${bench_final:,.0f})",
                line=dict(color="#adb5bd", width=1.5, dash="dash"),
            ))
            benchmark_added = True

    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_tickprefix="$",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_radar_chart(results: list):
    st.divider()
    st.subheader("Risk/Reward Profile")
    st.caption("Normalized across strategies — larger area = better overall profile")

    categories = ["Return", "Sharpe", "Sortino", "Low Drawdown", "Win Rate"]
    fig = go.Figure()

    for r in results:
        # Normalize each metric to 0-1 range for radar
        max_return = max(abs(x["total_return"]) for x in results) or 1
        max_sharpe = max(abs(x["sharpe_ratio"]) for x in results) or 1
        max_sortino = max(abs(x["sortino_ratio"]) for x in results) or 1

        values = [
            max(0, r["total_return"]) / max_return,
            max(0, r["sharpe_ratio"]) / max_sharpe,
            max(0, r["sortino_ratio"]) / max_sortino,
            1 - abs(r["max_drawdown"]),     # invert drawdown (lower = better)
            r["win_rate"],
        ]
        values.append(values[0])  # close the polygon

        color = STRATEGY_COLORS.get(r["strategy_name"], "#888")
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=r["strategy_name"],
            line=dict(color=color),
            fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in color else color,
            opacity=0.8,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_landing():
    st.divider()
    st.markdown("""
    ### Enter a ticker and click **Compare All Strategies** to get started.

    This runs all 4 strategies on the same ticker over the same period and shows:
    - **Leaderboard table** — key metrics side by side
    - **Combined equity curves** — visualize how each strategy performed vs SPY
    - **Radar chart** — risk/reward profile at a glance

    Great for finding which strategy historically fits a stock's behavior best.
    """)
