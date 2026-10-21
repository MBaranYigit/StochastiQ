import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

def generate_risk_memo(
    ticker: str,
    data: dict,
    sim_results: dict,
    risk_results: dict,
    sim_days: int,
    stress_mode: bool = False
) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "ERROR: GEMINI_API_KEY environment variable not found. Please check your .env file."

    client = genai.Client(api_key=api_key)
    currency = data.get("currency", "$")

    # System instruction defining both the institutional quant persona and the plain-English translator role
    system_instruction = (
        "You are a Senior Quantitative Risk Analyst at an institutional investment fund. "
        "Your task is to write a two-part risk report in English based strictly on the provided "
        "stochastic simulation (Geometric Brownian Motion) and tail-risk metrics (VaR, CVaR, Max Drawdown). "
        "Do not invent or extrapolate numbers outside the provided payload.\n\n"
        "Structure your response into two distinct sections:\n"
        "1. '### Part 1: Institutional Quantitative Risk Memo': A rigorous technical analysis for quant/risk teams. "
        "Compare Parametric VaR vs. Historical VaR to assess fat-tail (leptokurtic) risk, interpret CVaR (Expected Shortfall) "
        "under extreme market regimes, and evaluate the skewness of the Monte Carlo terminal price distribution (Mean vs. Median).\n"
        "2. '### Part 2: Plain-English Summary (For Non-Technical Readers)': A jargon-free, intuitive translation of the numbers "
        "for everyday investors. Explain clearly what the simulation predicts for the future price range, what a 'bad day' looks like "
        "in simple terms (without using terms like leptokurtic or stochastic), and the overall risk vs. reward balance."
    )

    regime_context = (
        "WARNING: This simulation was executed under a STRESS TEST regime (elevated volatility / market shock). "
        "Prioritize capital preservation, tail-risk exposure, and downside liquidity constraints in your analysis."
        if stress_mode else
        "Standard Historical Market Regime (calibrated on 2-year realized log returns)."
    )

    # Structured quantitative payload combining outputs from Day 1, Day 2, and Day 3
    user_prompt = f"""
    Generate the two-part Risk Report for **{ticker}** based on the following mathematical outputs:

    ### 1. Underlying Asset & Stochastic Process Parameters (Ito-Adjusted)
    - Current Spot Price: {data['current_price']:.2f} {currency}
    - Annualized Drift (Mu): {data['annualized_drift'] * 100:.2f}%
    - Annualized Volatility (Sigma): {data['annualized_volatility'] * 100:.2f}%
    - Simulation Horizon: {sim_days} Trading Days (10,000 Monte Carlo Paths)
    - Regime Context: {regime_context}

    ### 2. Geometric Brownian Motion (GBM) Terminal Distribution
    - Expected Mean Target Price: {sim_results['mean_target']:.2f} {currency}
    - Median Target Price (P50): {sim_results['median_target']:.2f} {currency}
    - Bullish Tail Scenario (P95 - 95th Percentile): {sim_results['percentile_95']:.2f} {currency}
    - Bearish Tail Scenario (P5 - 5th Percentile): {sim_results['percentile_5']:.2f} {currency}
    - Interquartile Range (P25 - P75): {sim_results['percentile_25']:.2f} {currency} to {sim_results['percentile_75']:.2f} {currency}

    ### 3. Daily Tail-Risk & Historical Drawdown Metrics (Sample Unbiased, ddof=1)
    - 95% Parametric VaR (Daily): {risk_results['var_param_95'] * 100:.2f}%
    - 95% Historical VaR (Daily): {risk_results['var_hist_95'] * 100:.2f}%
    - 95% CVaR / Expected Shortfall (Daily): {risk_results['cvar_95'] * 100:.2f}%
    - 99% Parametric VaR (Daily): {risk_results['var_param_99'] * 100:.2f}%
    - 99% Historical VaR (Daily): {risk_results['var_hist_99'] * 100:.2f}%
    - 99% CVaR / Expected Shortfall (Daily): {risk_results['cvar_99'] * 100:.2f}%
    - Historical Maximum Drawdown (Peak-to-Trough): {risk_results['max_drawdown'] * 100:.2f}%

    Please ensure both 'Part 1: Institutional Quantitative Risk Memo' and 'Part 2: Plain-English Summary (For Non-Technical Readers)' are clearly separated with Markdown headings and bullet points.
    """

    # Use Gemini 3.8 Flash directly with a lightweight retry for temporary 503 spikes
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                )
            )
            return response.text
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(2)  # Wait 2 seconds and retry if server is busy
                continue
            return f"LLM Report Generation Error: {str(e)}"

        
# --- END-TO-END VERIFICATION BLOCK ---
if __name__ == "__main__":
    # Mock data payload to test API connection and English report formatting
    mock_data = {
        "current_price": 135.50,
        "currency": "$",
        "annualized_drift": 0.28,
        "annualized_volatility": 0.38
    }
    mock_sim = {
        "mean_target": 179.20,
        "median_target": 166.80,
        "percentile_5": 89.40,
        "percentile_25": 128.10,
        "percentile_75": 215.60,
        "percentile_95": 310.50
    }
    mock_risk = {
        "var_param_95": 0.038,
        "var_hist_95": 0.042,
        "cvar_95": 0.059,
        "var_param_99": 0.054,
        "var_hist_99": 0.065,
        "cvar_99": 0.081,
        "max_drawdown": -0.295
    }

    print("Generating Executive Risk Memo & Plain-English Summary via Gemini API...\n")
    memo = generate_risk_memo("NVDA", mock_data, mock_sim, mock_risk, sim_days=252, stress_mode=False)
    print(memo)