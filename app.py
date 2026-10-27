import gc
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import threading

from engine.data_loader import fetch_equity_data
from engine.monte_carlo import run_gbm_simulation
from engine.risk_metrics import compute_risk_metrics
from engine.llm_analyst import generate_risk_memo

# 1. Page Configuration
st.set_page_config(
    page_title="StochastiQ | Quantitative Risk Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Initialize Lightweight Session State
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "sim_data" not in st.session_state:
    st.session_state.sim_data = None
if "ai_memo" not in st.session_state:
    st.session_state.ai_memo = None

@st.cache_resource
def get_locks() -> dict:
    return {"sim": threading.Semaphore(2), "ai": threading.Semaphore(2)}


@st.cache_data(ttl=900, max_entries=8, show_spinner=False)
def compute_bundle(
    ticker_input: str,
    sim_days: int,
    num_simulations: int,
    stress_mode: bool,
    vol_model: str = "standard"
) -> dict:
    """Fetch data, simulate, and return ONLY lightweight results (cached, no big matrices)."""
    with get_locks()["sim"]:
        data = fetch_equity_data(ticker_input, period="2y", vol_model=vol_model)

        effective_drift = data["annualized_drift"] - 0.15 if stress_mode else data["annualized_drift"]
        effective_vol = data["annualized_volatility"] * 1.50 if stress_mode else data["annualized_volatility"]

        sim_raw = run_gbm_simulation(
            current_price=data["current_price"],
            drift=effective_drift,
            volatility=effective_vol,
            days=sim_days,
            simulations=num_simulations
        )

        risk_results = compute_risk_metrics(data["simple_returns"], data["history"])

        paths = sim_raw["price_paths"]
        # One percentile pass instead of five separate ones
        p5, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=1).astype(float)
        bands = {
            "days_axis": np.arange(paths.shape[0]),
            "p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95,
            "mean": paths.mean(axis=1, dtype=np.float64)
        }

        counts, bin_edges = np.histogram(sim_raw["terminal_prices"], bins=50)
        hist_data = {"counts": counts, "bin_centers": 0.5 * (bin_edges[:-1] + bin_edges[1:])}

        sim_summary = {k: sim_raw[k] for k in (
            "mean_target", "median_target", "percentile_5",
            "percentile_25", "percentile_75", "percentile_95"
        )}

        del paths, sim_raw
        gc.collect()

        return {
            "ticker": data["ticker"],
            "current_price": data["current_price"],
            "currency": data["currency"],
            "base_drift": data["annualized_drift"],
            "base_vol": data["annualized_volatility"],
            "vol_standard": data["vol_standard"],
            "vol_ewma": data["vol_ewma"],
            "vol_model": data["vol_model"],
            "effective_drift": effective_drift,
            "effective_vol": effective_vol,
            "bands": bands,
            "hist_data": hist_data,
            "sim_summary": sim_summary,
            "risk_results": risk_results,
            "sim_days": sim_days,
            "num_simulations": num_simulations,
            "stress_mode": stress_mode
        }
# 3. Memory-Safe Plotly Chart Builders (Uses pre-aggregated arrays only)
# 3. Memory-Safe Plotly Chart Builders (Uses pre-aggregated arrays only)
def build_fan_chart(bands: dict, ticker: str, currency: str, num_simulations: int) -> go.Figure:
    days_axis = bands["days_axis"]

    fig = go.Figure()

    # 5th - 95th Percentile Outer Confidence Band
    fig.add_trace(go.Scatter(
        x=days_axis, y=bands["p95"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=days_axis, y=bands["p5"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(56, 189, 248, 0.14)",
        name="5% – 95% Confidence Band"
    ))

    # 25th - 75th Percentile Interquartile Band
    fig.add_trace(go.Scatter(
        x=days_axis, y=bands["p75"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=days_axis, y=bands["p25"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(56, 189, 248, 0.30)",
        name="25% – 75% Interquartile Range"
    ))

    # Median (P50) & Expected Mean Trajectories
    fig.add_trace(go.Scatter(
        x=days_axis, y=bands["p50"],
        mode="lines",
        line=dict(color="#38bdf8", width=2.5),
        name="Median Trajectory (P50)"
    ))
    fig.add_trace(go.Scatter(
        x=days_axis, y=bands["mean"],
        mode="lines",
        line=dict(color="#f59e0b", width=2, dash="dash"),
        name="Expected Mean Trajectory"
    ))

    fig.update_layout(
        title=dict(
            text=f"{ticker} — Monte Carlo GBM Price Cone ({num_simulations:,} Paths)",
            y=0.96, x=0.0
        ),
        xaxis_title="Trading Days Forward",
        yaxis_title=f"Projected Price ({currency})",
        template="plotly_dark",
        hovermode="x unified",
        height=460,
        # Move legend below the chart so it never collides with the title
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        margin=dict(l=20, r=20, t=55, b=85)
    )
    return fig


def build_terminal_histogram(hist_data: dict, sim_summary: dict, current_price: float, currency: str) -> go.Figure:
    max_count = max(hist_data["counts"]) if len(hist_data["counts"]) > 0 else 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hist_data["bin_centers"],
        y=hist_data["counts"],
        marker_color="rgba(56, 189, 248, 0.65)",
        marker_line=dict(color="rgba(255, 255, 255, 0.15)", width=1),
        name="Frequency",
        showlegend=False
    ))

    # Draw vertical reference lines without overlapping top text; display values cleanly in bottom legend
    fig.add_vline(x= sim_summary["percentile_5"], line_dash="dash", line_color="#ef4444", line_width=2)
    fig.add_vline(x=current_price, line_dash="dot", line_color="#e2e8f0", line_width=2)
    fig.add_vline(x=sim_summary["median_target"], line_dash="solid", line_color="#38bdf8", line_width=2.5)

    # Dummy traces strictly for a clean, collision-free bottom legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="#ef4444", width=2, dash="dash"),
        name=f"P5 Tail: {sim_summary['percentile_5']:.1f}{currency}"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="#e2e8f0", width=2, dash="dot"),
        name=f"Spot: {current_price:.1f}{currency}"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="#38bdf8", width=2.5, dash="solid"),
        name=f"Median: {sim_summary['median_target']:.1f}{currency}"
    ))

    fig.update_layout(
        title=dict(
            text="Terminal Price Distribution at Horizon",
            y=0.96, x=0.0
        ),
        xaxis_title=f"Terminal Price ({currency})",
        yaxis_title="Simulation Frequency",
        template="plotly_dark",
        height=460,
        bargap=0.05,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        margin=dict(l=20, r=20, t=55, b=85)
    )
    return fig


# 4. Isolated AI Report Fragment (Prevents full-page chart re-rendering when clicked)
@st.fragment
@st.fragment
def render_ai_report_section(bundle: dict):
    st.subheader("🤖 AI Executive Quantitative Risk Memo")
    st.caption(
        "Generates an institutional risk analysis alongside a plain-English summary "
        "for non-technical readers using Google Gemini."
    )

    if st.button("✨ Generate AI Quantitative Risk Report", type="secondary"):
        used = st.session_state.get("ai_calls", 0)
        if used >= 5:
            st.warning("Demo limit reached for this session (5 reports).")
        else:
            st.session_state["ai_calls"] = used + 1
        ai_lock = get_locks()["ai"]
        if ai_lock.acquire(blocking=False):
            try:
                with st.spinner(
                    "🧠 Generating your risk report… This usually takes 10–20 seconds. "
                    "Take a look at the data above while your report is being prepared."
                ):
                    report_data = {
                        "current_price": bundle["current_price"],
                        "currency": bundle["currency"],
                        "annualized_drift": bundle["effective_drift"],
                        "annualized_volatility": bundle["effective_vol"]
                    }
                    st.session_state.ai_memo = generate_risk_memo(
                        ticker=bundle["ticker"],
                        data=report_data,
                        sim_results=bundle["sim_summary"],
                        risk_results=bundle["risk_results"],
                        sim_days=bundle["sim_days"],
                        stress_mode=bundle["stress_mode"]
                    )
            finally:
                ai_lock.release()
        else:
            st.info("A report is already being generated. Please wait a moment.")

    if st.session_state.ai_memo:
        with st.container(border=True):
            st.markdown(st.session_state.ai_memo)


# 5. Sidebar Controls & Macro Stress Test Module
with st.sidebar:
    st.title("⚡ StochastiQ")
    st.caption("Ito-Adjusted GBM & Tail-Risk Engine")
    st.divider()

    ticker_input = st.text_input(
        "Equity Ticker Symbol",
        value="NVDA",
        help="Examples: NVDA, LLY, AAPL, MSFT, THYAO.IS, GARAN.IS"
    )

    sim_days = st.slider(
        "Simulation Horizon (Trading Days)",
        min_value=30,
        max_value=252,
        value=252,
        step=10,
        help="252 trading days = 1 full market year"
    )

    num_simulations = st.selectbox(
        "Monte Carlo Paths",
        options=[1000, 5000, 10000],
        index=2
    )

    vol_model_choice = st.radio(
        "Volatility Estimation Model",
        options=["Standard (Equal-Weighted)", "EWMA (RiskMetrics λ=0.94)"],
        index=0,
        help="Standard weights all 2-year historical days equally. EWMA applies exponential decay (λ=0.94) to weight recent market volatility more heavily."
    )
    vol_model_key = "ewma" if "EWMA" in vol_model_choice else "standard"

    st.divider()
    st.subheader("⚠️ Macro Stress Test")
    stress_mode = st.toggle(
        "Apply Crisis Regime Shock",
        value=False,
        help="Simulates a market crash regime: Volatility is amplified by 1.5x (+50%) and Annualized Drift is shocked by -15%."
    )

    run_btn = st.button("🚀 Run Quantitative Engine", type="primary", use_container_width=True)


# 6. Main Header
st.title("📊 StochastiQ: Stochastic Scenario & Tail-Risk Analytics")
st.markdown(
    "Institutional-grade equity risk engine powered by **Ito-adjusted Geometric Brownian Motion**, "
    "unbiased sample tail-risk metrics (**ddof=1**), and **Gemini AI** executive reporting."
)

# 7. Execution Pipeline (Pre-computes chart vectors and frees heavy matrices from RAM)
# 7. Execution Pipeline (cached + serialized; only lightweight results are kept)
if run_btn:
    with st.spinner(f"Calibrating stochastic parameters and simulating {num_simulations:,} paths for {ticker_input.strip().upper()}..."):
        try:
            st.session_state.sim_data = compute_bundle(
                ticker_input.strip().upper(), sim_days, num_simulations, stress_mode, vol_model_key
            )
            st.session_state.analysis_done = True
            st.session_state.ai_memo = None
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
            st.session_state.analysis_done = False


# 8. Render Dashboard

if st.session_state.analysis_done and st.session_state.sim_data:
    bundle = st.session_state.sim_data
    sim = bundle["sim_summary"]
    risk = bundle["risk_results"]
    cur = bundle["currency"]
    spot = bundle["current_price"]

    if bundle.get("vol_model") == "ewma":
        st.info(
            f"📊 **EWMA Volatility Regime Active (RiskMetrics λ=0.94):** Recent market weighting adjusted "
            f"base annualized volatility from **{bundle['vol_standard']*100:.2f}%** (2Y Equal-Weighted) "
            f"to **{bundle['vol_ewma']*100:.2f}%** (EWMA)."
        )

    if bundle["stress_mode"]:
        st.warning(
            f"**STRESS TEST REGIME ACTIVE:** Annualized Volatility shocked from "
            f"**{bundle['base_vol']*100:.1f}%** to **{bundle['effective_vol']*100:.1f}%** | "
            f"Annualized Drift adjusted from **{bundle['base_drift']*100:.1f}%** to **{bundle['effective_drift']*100:.1f}%**."
        )

    # --- Section A: Top KPI Cards ---
    st.subheader(f"📌 Key Quantitative Metrics — {bundle['ticker']}")
    col1, col2, col3, col4, col5 = st.columns(5)

    expected_return_pct = ((sim["median_target"] - spot) / spot) * 100

    col1.metric(
        label="Current Spot Price",
        value=f"{spot:.2f} {cur}",
        delta=f"Drift: {bundle['effective_drift']*100:.1f}% / yr"
    )
    col2.metric(
        label=f"Median Target ({bundle['sim_days']}d)",
        value=f"{sim['median_target']:.2f} {cur}",
        delta=f"{expected_return_pct:+.1f}% vs Spot"
    )
    col3.metric(
        label="Annualized Volatility (σ)",
        value=f"{bundle['effective_vol']*100:.2f}%",
        delta="Stress Shock (1.5x)" if bundle["stress_mode"] else "Historical (2Y)",
        delta_color="inverse" if bundle["stress_mode"] else "off"
    )
    col4.metric(
        label="95% Daily CVaR (Tail Loss)",
        value=f"-{risk['cvar_95']*100:.2f}%",
        delta=f"95% Hist VaR: -{risk['var_hist_95']*100:.2f}%",
        delta_color="inverse"
    )
    col5.metric(
        label="Max Historical Drawdown",
        value=f"{risk['max_drawdown']*100:.2f}%",
        delta="2-Year Peak-to-Trough",
        delta_color="inverse"
    )

    st.divider()

    # --- Section B: Interactive Stochastic Charts ---
    st.subheader("📈 Stochastic Price Projection & Terminal Distribution")
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        fan_fig = build_fan_chart(bundle["bands"], bundle["ticker"], cur, bundle["num_simulations"])
        st.plotly_chart(fan_fig, use_container_width=True)

    with chart_col2:
        hist_fig = build_terminal_histogram(bundle["hist_data"], sim, spot, cur)
        st.plotly_chart(hist_fig, use_container_width=True)

    # --- Section C: Scenario & Tail-Risk Breakdown Tables ---
    tab1, tab2 = st.tabs(["🎯 Monte Carlo Price Percentiles", "🛡️ Daily Tail-Risk Matrix (VaR & CVaR)"])

    with tab1:
        perc_df = pd.DataFrame({
            "Scenario Tier": [
                "Bearish Tail (5th Percentile - P5)",
                "Lower Quartile (25th Percentile - P25)",
                "Median Trajectory (50th Percentile - P50)",
                "Expected Mean Target",
                "Upper Quartile (75th Percentile - P75)",
                "Bullish Tail (95th Percentile - P95)"
            ],
            f"Projected Price ({cur})": [
                f"{sim['percentile_5']:.2f} {cur}",
                f"{sim['percentile_25']:.2f} {cur}",
                f"{sim['median_target']:.2f} {cur}",
                f"{sim['mean_target']:.2f} {cur}",
                f"{sim['percentile_75']:.2f} {cur}",
                f"{sim['percentile_95']:.2f} {cur}"
            ],
            "Implied Return (%)": [
                f"{((sim['percentile_5'] - spot) / spot) * 100:+.2f}%",
                f"{((sim['percentile_25'] - spot) / spot) * 100:+.2f}%",
                f"{((sim['median_target'] - spot) / spot) * 100:+.2f}%",
                f"{((sim['mean_target'] - spot) / spot) * 100:+.2f}%",
                f"{((sim['percentile_75'] - spot) / spot) * 100:+.2f}%",
                f"{((sim['percentile_95'] - spot) / spot) * 100:+.2f}%"
            ]
        })
        st.dataframe(perc_df, use_container_width=True, hide_index=True)

    with tab2:
        risk_df = pd.DataFrame({
            "Confidence Level": ["95% Confidence (α = 0.05)", "99% Confidence (α = 0.01)"],
            "Parametric VaR (Gaussian)": [
                f"-{risk['var_param_95']*100:.2f}%",
                f"-{risk['var_param_99']*100:.2f}%"
            ],
            "Historical VaR (Empirical)": [
                f"-{risk['var_hist_95']*100:.2f}%",
                f"-{risk['var_hist_99']*100:.2f}%"
            ],
            "Conditional VaR / Expected Shortfall (CVaR)": [
                f"-{risk['cvar_95']*100:.2f}%",
                f"-{risk['cvar_99']*100:.2f}%"
            ]
        })
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

    st.divider()

    # --- Section D: Isolated Gemini AI Report Fragment ---
    render_ai_report_section(bundle)

else:
    st.info("👈 Select an equity ticker and parameters in the sidebar, then click **Run Quantitative Engine** to start.")