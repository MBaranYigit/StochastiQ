import os
import re
import ssl
import json
import time
import math
import urllib.request
import urllib.error
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Settings ----------------------------------------------------------------
TOTAL_TIME_BUDGET = 45      # total time budget across all model attempts (seconds)
PER_CALL_TIMEOUT = 25       # upper limit for a single API call (seconds)
MAX_MODELS_TO_TRY = 4       # maximum number of models to try
MAX_OUTPUT_TOKENS = 8192    # kept high so thinking models don't return empty text

_EXCLUDE = ("tts", "image", "live", "audio", "embedding", "robotics",
            "computer-use", "learnlm", "gemma")


def _format_currency(amount: float, currency: str) -> str:
    """Formats currency safely so Streamlit Markdown never confuses '$' with LaTeX math."""
    if currency == "$":
        return f"USD {amount:.2f}"
    return f"{amount:.2f} {currency}"




def _rank(name: str) -> int:
    """Lower score = tried first. Lite models have the most generous free quotas."""
    score = 0
    if "flash" not in name:
        score += 10
    if "lite" not in name:
        score += 2
    if "preview" in name or "exp" in name:
        score += 3
    return score


def _get_available_gemini_models(api_key: str, ssl_ctx: ssl.SSLContext) -> list:
    """
    Queries the Google API dynamically to discover which Gemini models are active
    for this user's API key, prioritizing stable Flash models for speed and availability.
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
    try:
        req = urllib.request.Request(url, headers={"x-goog-api-key": api_key})
        with urllib.request.urlopen(req, timeout=10, context=ssl_ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        models = []
        for m in data.get("models", []):
            name = m.get("name", "").replace("models/", "")
            if ("generateContent" in m.get("supportedGenerationMethods", [])
                    and name.startswith("gemini")
                    and not any(x in name for x in _EXCLUDE)):
                models.append(name)

        # Newer versions first (reverse alphabetical), then a stable sort by reliability rank
        return sorted(sorted(models, reverse=True), key=_rank)
    except Exception:
        # Safe fallback list if the model listing call fails
        return ["gemini-flash-latest", "gemini-flash-lite-latest"]


def _call_gemini(model: str, key: str, prompt: str, ctx, timeout: int = PER_CALL_TIMEOUT) -> str:
    """Sends a request to a single Gemini model. Raises an exception on failure."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.35, "maxOutputTokens": MAX_OUTPUT_TOKENS},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    candidates = result.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"no candidates ({result.get('promptFeedback')})")

    cand = candidates[0]
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise RuntimeError(f"empty text (finishReason={cand.get('finishReason')})")
    return text


def _build_deterministic_quant_report(
    ticker: str,
    data: dict,
    sim_results: dict,
    risk_results: dict,
    sim_days: int,
    stress_mode: bool = False,
    fallback_reason: str = "High Server Demand"
) -> str:
    """
    Dynamic Rule-Based Quant Reporting Engine (Zero-Downtime Fallback).
    """
    cur = data.get("currency", "$")
    spot_str = _format_currency(data["current_price"], cur)
    mean_str = _format_currency(sim_results["mean_target"], cur)
    med_str = _format_currency(sim_results["median_target"], cur)
    p5_str = _format_currency(sim_results["percentile_5"], cur)
    p25_str = _format_currency(sim_results["percentile_25"], cur)
    p75_str = _format_currency(sim_results["percentile_75"], cur)
    p95_str = _format_currency(sim_results["percentile_95"], cur)

    spot = data["current_price"]
    drift_pct = data["annualized_drift"] * 100
    vol_pct = data["annualized_volatility"] * 100

    exp_ret_med = ((sim_results["median_target"] - spot) / spot) * 100
    exp_ret_mean = ((sim_results["mean_target"] - spot) / spot) * 100
    downside_p5_pct = ((sim_results["percentile_5"] - spot) / spot) * 100
    upside_p95_pct = ((sim_results["percentile_95"] - spot) / spot) * 100

    vp95 = risk_results["var_param_95"] * 100
    vh95 = risk_results["var_hist_95"] * 100
    cvar95 = risk_results["cvar_95"] * 100
    vp99 = risk_results["var_param_99"] * 100
    vh99 = risk_results["var_hist_99"] * 100
    cvar99 = risk_results["cvar_99"] * 100
    mdd = risk_results["max_drawdown"] * 100

    if vol_pct > 45.0:
        vol_profile = f"high-beta, aggressive volatility profile ({vol_pct:.2f}% annualized), where variance drag significantly widens the terminal price cone"
    elif vol_pct > 25.0:
        vol_profile = f"moderate growth volatility regime ({vol_pct:.2f}% annualized), balancing directional drift against stochastic dispersion"
    else:
        vol_profile = f"low-volatility, defensive equity profile ({vol_pct:.2f}% annualized), exhibiting tight clustering around the central trajectory"

    fat_tail_spread_99 = vh99 - vp99
    if fat_tail_spread_99 > 0.25:
        kurtosis_diagnosis = (
            f"Pronounced leptokurtic (fat-tail) risk: 99% Historical VaR (-{vh99:.2f}%) exceeds Gaussian Parametric VaR (-{vp99:.2f}%) "
            f"by {fat_tail_spread_99:.2f}%. Historical sell-offs in {ticker} have been sharper than normal distribution models predict."
        )
    elif fat_tail_spread_99 < -0.15:
        kurtosis_diagnosis = (
            f"Thin-tailed empirical downside: 99% Historical VaR (-{vh99:.2f}%) is milder than Parametric VaR (-{vp99:.2f}%), "
            f"suggesting realized extreme daily losses have been more contained than theoretical Gaussian volatility implies."
        )
    else:
        kurtosis_diagnosis = (
            f"Near-Gaussian tail alignment: 99% Historical VaR (-{vh99:.2f}%) tracks closely with Parametric VaR (-{vp99:.2f}%), "
            f"confirming that empirical tail shocks conform closely to theoretical normal bounds."
        )

    regime_text = (
        "MACRO STRESS TEST REGIME (1.5x Volatility Shock & -15% Drift Compression)"
        if stress_mode else
        "Standard Historical Market Regime (2-Year Realized Calibration)"
    )

    # Analytically calculates what percentage of the simulations end above the current price
    t = sim_days / 252.0
    sigma = data["annualized_volatility"]
    mu = data["annualized_drift"]
    d2 = ((mu - 0.5 * (sigma ** 2)) * t) / (sigma * math.sqrt(t)) if (sigma > 0 and t > 0) else 0.0
    prob_above = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0))) * 100.0

    stress_note = " (with the macro stress adjustment applied: volatility ×1.5 and reduced drift)" if stress_mode else ""

    return f"""Under the selected model assumptions{stress_note} over a **{sim_days}-trading-day** horizon for **{ticker}** (current price: **{spot_str}**), the median simulated price is **{med_str}** and the mean simulated price is **{mean_str}**. Approximately **{prob_above:.1f}%** of simulated outcomes end above the current price.

The 5th percentile (**{p5_str}**) represents a downside scenario that is exceeded on the lower side in roughly 5% of simulations, while the 95th percentile (**{p95_str}**) marks the upper bound exceeded in roughly 5% of simulations. Looking at actual historical daily returns over the past two years, the 95% daily Historical Value at Risk (VaR) is **-{vh95:.2f}%**, meaning daily losses exceeded this threshold on roughly 5% of trading days, with an average loss (CVaR) of **-{cvar95:.2f}%** on those worst days and a historical maximum peak-to-trough drawdown of **{mdd:.2f}%**.

*Note: All simulation outputs and risk figures are derived strictly from historical price data and statistical model assumptions. They represent mathematical model outputs rather than market predictions and do not constitute financial advice.*"""

def generate_risk_memo(
    ticker: str,
    data: dict,
    sim_results: dict,
    risk_results: dict,
    sim_days: int,
    stress_mode: bool = False
) -> str:
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return _build_deterministic_quant_report(
            ticker, data, sim_results, risk_results, sim_days, stress_mode,
            fallback_reason="Missing GEMINI_API_KEY in .env"
        )

    clean_key = api_key.strip()
    cur = data.get("currency", "$")
    cur_code = "USD" if cur == "$" else "TRY"
        # Pre-compute the tail diagnosis here so the model never has to infer it
    tail_gap = (risk_results['var_hist_99'] - risk_results['var_param_99']) * 100
    if tail_gap > 0.25:
        tail_note = "Historical extreme losses were WORSE than a normal distribution predicts (fatter tails)."
    elif tail_gap < -0.15:
        tail_note = "Historical extreme losses were MILDER than a normal distribution predicts."
    else:
        tail_note = "Historical extreme losses were CLOSE to what a normal distribution predicts."

    # Pre-compute % changes vs spot
    spot_p = data['current_price']
    med_ret = (sim_results['median_target'] / spot_p - 1) * 100
    p5_ret = (sim_results['percentile_5'] / spot_p - 1) * 100
    p95_ret = (sim_results['percentile_95'] / spot_p - 1) * 100

    stress_note = ""
    if stress_mode:
        stress_note = (
            f"NOTE: A crisis stress test is ON. Drift and volatility are hypothetical shocked inputs "
            f"(volatility x1.5, drift -15 points), not {ticker}'s real figures, so describe the price range "
            f"as a crisis scenario. VaR, CVaR and drawdown are normal historical figures.\n"
        )

    t = sim_days / 252.0
    sigma = data["annualized_volatility"]
    mu = data["annualized_drift"]
    d2 = ((mu - 0.5 * (sigma ** 2)) * t) / (sigma * math.sqrt(t)) if (sigma > 0 and t > 0) else 0.0
    prob_above = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0))) * 100.0

    prompt_text = (
        f"Write a clean, objective, single-section statistical summary in English for {ticker}. "
        f"Do NOT split the text into multiple parts or headings. Do NOT use exaggerated hedge-fund jargon. "
        f"Never use the '$' symbol (always write '{cur_code}' instead, e.g., '{cur_code} 150.00').\n\n"
        f"Follow this exact tone and style:\n"
        f"'Under the selected model assumptions, the median simulated price is {cur_code} X. Approximately X% of simulated outcomes end above the current price. "
        f"The 5th percentile represents a downside scenario that is exceeded on the lower side in roughly 5% of simulations.'\n\n"
        f"Summarize these metrics in 2 short paragraphs:\n"
        f"- Current Price: {cur_code} {data['current_price']:.2f} | Horizon: {sim_days} trading days | Stress Mode Active: {stress_mode}\n"
        f"- Median Simulated Price: {cur_code} {sim_results['median_target']:.2f} | Mean Simulated Price: {cur_code} {sim_results['mean_target']:.2f}\n"
        f"- Percentage of simulations ending above current price: {prob_above:.1f}%\n"
        f"- 5th Percentile: {cur_code} {sim_results['percentile_5']:.2f} | 95th Percentile: {cur_code} {sim_results['percentile_95']:.2f}\n"
        f"- 2-Year Historical Daily Risk: 95% Historical VaR = -{risk_results['var_hist_95']*100:.2f}%, 95% CVaR = -{risk_results['cvar_95']*100:.2f}%, Max Drawdown = {risk_results['max_drawdown']*100:.2f}%\n\n"
        f"End with a short sentence stating that these figures are derived from historical data and statistical assumptions, are model outputs rather than predictions, and are not financial advice."
    )
    ssl_ctx = ssl.create_default_context()

    available_models = _get_available_gemini_models(clean_key, ssl_ctx)[:MAX_MODELS_TO_TRY]
    diagnostics = []
    deadline = time.time() + TOTAL_TIME_BUDGET

    for model_name in available_models:
        if time.time() > deadline:
            diagnostics.append("time budget exceeded")
            break

        for attempt in range(2):  # retry once on 429/503
            try:
                remaining = max(5, int(deadline - time.time()))
                text = _call_gemini(
                    model_name, clean_key, prompt_text, ssl_ctx,
                    timeout=min(PER_CALL_TIMEOUT, remaining)
                )
                clean_text = text.replace("$", "USD ")
                return f"{clean_text}\n\n---\n*🟢 Generated live via **Google Gemini (`{model_name}`)** API*"

            except urllib.error.HTTPError as e:
                raw = ""
                msg = ""
                try:
                    raw = e.read().decode("utf-8")
                    msg = json.loads(raw).get("error", {}).get("message", "")
                except Exception:
                    pass
                diagnostics.append(f"{model_name}: HTTP {e.code} {msg[:100]}")

                # 429/503: if Google's suggested delay is short, wait and retry the same model
                if e.code in (429, 503) and attempt == 0:
                    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', raw)
                    wait = float(m.group(1)) + 1 if m else 5
                    if wait <= 25 and time.time() + wait < deadline:
                        time.sleep(wait)
                        continue
                break  # 404, 400, 403 or a long wait: move on to the next model

            except Exception as e:
                diagnostics.append(f"{model_name}: {type(e).__name__} {str(e)[:100]}")
                break

    diag_str = " | ".join(diagnostics) if diagnostics else "No active models responded"
    return _build_deterministic_quant_report(
        ticker,
        data,
        sim_results,
        risk_results,
        sim_days,
        stress_mode,
        fallback_reason=diag_str
    )
def _build_deterministic_quant_report(
    ticker: str,
    data: dict,
    sim_results: dict,
    risk_results: dict,
    sim_days: int,
    stress_mode: bool = False,
    fallback_reason: str = "High Server Demand"
) -> str:
    """
    Short rule-based report (zero-downtime fallback). Same format as the AI report.
    """
    cur = data.get("currency", "$")
    spot = data["current_price"]

    spot_str = _format_currency(spot, cur)
    med_str = _format_currency(sim_results["median_target"], cur)
    p5_str = _format_currency(sim_results["percentile_5"], cur)
    p95_str = _format_currency(sim_results["percentile_95"], cur)

    med_ret = ((sim_results["median_target"] - spot) / spot) * 100
    p5_ret = ((sim_results["percentile_5"] - spot) / spot) * 100
    p95_ret = ((sim_results["percentile_95"] - spot) / spot) * 100

    vh95 = risk_results["var_hist_95"] * 100
    cvar95 = risk_results["cvar_95"] * 100
    vh99 = risk_results["var_hist_99"] * 100
    vp99 = risk_results["var_param_99"] * 100
    mdd = risk_results["max_drawdown"] * 100

    # Tail check: same thresholds as before
    tail_gap = vh99 - vp99
    if tail_gap > 0.25:
        tail_note = "Extreme historical losses were worse than a normal distribution predicts (fatter tails)."
    elif tail_gap < -0.15:
        tail_note = "Extreme historical losses were milder than a normal distribution predicts."
    else:
        tail_note = "Extreme historical losses were close to what a normal distribution predicts."

    if stress_mode:
        scenario_note = (
            " A crisis stress test is on (volatility x1.5, drift -15 points), "
            "so this range is a hypothetical crisis scenario."
        )
    else:
        scenario_note = ""

    return f"""### Technical Summary

* **Price outlook:** Over {sim_days} trading days, the median simulated price for **{ticker}** is **{med_str}** (**{med_ret:+.1f}%** vs spot **{spot_str}**), with a bad case (P5) of **{p5_str}** (**{p5_ret:+.1f}%**) and a good case (P95) of **{p95_str}** (**{p95_ret:+.1f}%**). The drift is a past-based assumption, not a forecast.{scenario_note}
* **Daily risk:** On 95% of days, losses have historically stayed within **-{vh95:.2f}%** (95% VaR); on the worst 5% of days the average loss was **-{cvar95:.2f}%** (CVaR). {tail_note}
* **Drawdown:** The steepest peak-to-trough drop over the last two years was **{mdd:.2f}%**.

### Plain-English Summary

* **Realistic range:** The typical outcome is around **{med_str}**, but it could reasonably land anywhere from about **{p5_str}** to **{p95_str}**.
* **Bad days:** A rough day can cost around **{vh95:.1f}%** or more, and in the last two years the stock once fell **{abs(mdd):.1f}%** from a peak.

---
*⚡ Generated via **StochastiQ Algorithmic Quant Engine** (Fallback active — Diagnostic: `{fallback_reason}`)*
"""