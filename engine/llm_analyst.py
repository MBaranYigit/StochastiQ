import os
import re
import ssl
import json
import time
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

    return f"""### Part 1: Institutional Quantitative Risk Memo

* **Asset Profile & Stochastic Calibration:** **{ticker}** exhibits a {vol_profile}. Under the **{regime_text}**, the Ito-adjusted annualized drift is calibrated at **{drift_pct:.2f}%** over a **{sim_days}-trading-day** horizon (10,000 Monte Carlo paths).
* **Terminal Distribution & Asymmetry:** From the current spot price of **{spot_str}**, the median (P50) trajectory reaches **{med_str}** (**{exp_ret_med:+.2f}%**), while the expected mean reaches **{mean_str}** (**{exp_ret_mean:+.2f}%**). The divergence between mean and median reflects right-skewed log-normal compounding.
* **Scenario Dispersion (Quartile & Tail Cones):** The core 50% interquartile band (P25–P75) spans **{p25_str}** to **{p75_str}**. Under extreme 90% confidence bounds, the bearish 5th percentile (P5) tests **{p5_str}** (**{downside_p5_pct:+.2f}%**), whereas the bullish 95th percentile (P95) extends to **{p95_str}** (**{upside_p95_pct:+.2f}%**).
* **Parametric vs. Empirical VaR (Kurtosis Check):** {kurtosis_diagnosis}
* **Expected Shortfall (CVaR) & Drawdown Liquidity:** Daily 95% Historical VaR stands at **-{vh95:.2f}%**, with Conditional VaR (Expected Shortfall) averaging **-{cvar95:.2f}%** during the worst 5% of trading sessions (**-{cvar99:.2f}%** at the 99% level). Combined with a 2-year Maximum Drawdown of **{mdd:.2f}%**, institutional position sizing should enforce strict stop-loss or hedge overlays.

---

### Part 2: Plain-English Summary (For Non-Technical Readers)

* **Expected Price Range:** Over the next **{sim_days} trading days**, computer simulations show the most typical middle outcome for **{ticker}** is around **{med_str}** (**{exp_ret_med:+.1f}%** from today's **{spot_str}**). If market momentum is strong (top 5% of scenarios), the stock could reach **{p95_str}**; if a severe slump occurs (bottom 5% of scenarios), it could drop toward **{p5_str}**.
* **Day-to-Day Risk (What a Bad Day Looks Like):** On 19 out of 20 normal trading days (95% of the time), single-day drops should not exceed **-{vh95:.2f}%**. However, on the rare 1-in-20 turbulent market days when losses break past that level, the average daily drop has historically been **-{cvar95:.2f}%**.
* **Historical Worst-Case Pullback:** Looking back over the last two years, the steepest drop from a peak to a bottom was **{mdd:.2f}%**. Investors holding **{ticker}** should be comfortable riding out temporary swings of this magnitude.

---
*⚡ Generated via **StochastiQ Algorithmic Quant Engine** (Fallback active — Diagnostic: `{fallback_reason}`)*
"""


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

    prompt_text = (
        f"You are a risk analyst. Write a SHORT, simple risk summary for {ticker} in English.\n"
        f"Rules: never use the '$' symbol (write '{cur_code}' instead) and no LaTeX; use only the numbers below; "
        f"no investment advice or buy/sell wording; no options terms (gamma, delta, implied volatility); "
        f"every bullet is 1-2 short sentences; the whole answer stays under 200 words.\n\n"
        f"Data for {ticker}:\n"
        f"- Spot price: {cur_code} {spot_p:.2f}\n"
        f"- Assumed yearly growth (drift): {data['annualized_drift']*100:.2f}% | Yearly volatility: {data['annualized_volatility']*100:.2f}%\n"
        f"- Simulation: {sim_days} trading days, 10,000 paths\n"
        f"- Simulated end price: Median {cur_code} {sim_results['median_target']:.2f} ({med_ret:+.1f}% vs spot), "
        f"Mean {cur_code} {sim_results['mean_target']:.2f}, "
        f"Bad case (P5) {cur_code} {sim_results['percentile_5']:.2f} ({p5_ret:+.1f}% vs spot), "
        f"Good case (P95) {cur_code} {sim_results['percentile_95']:.2f} ({p95_ret:+.1f}% vs spot)\n"
        f"- Daily risk (historical): 95% VaR -{risk_results['var_hist_95']*100:.2f}%, "
        f"95% CVaR -{risk_results['cvar_95']*100:.2f}%, 99% VaR -{risk_results['var_hist_99']*100:.2f}%, "
        f"99% CVaR -{risk_results['cvar_99']*100:.2f}%\n"
        f"- Worst 2-year drop from a peak (max drawdown): {risk_results['max_drawdown']*100:.2f}%\n"
        f"- Tail check: {tail_note}\n"
        f"{stress_note}\n"
        f"Use exactly this format:\n"
        f"### Technical Summary\n"
        f"- **Price outlook:** median, bad case and good case, each with its % change vs spot. Explain that the "
        f"bad case means only about 1 in 20 simulated outcomes ended lower, and the good case means only about "
        f"1 in 20 ended higher (so they are not the absolute worst or best). Add one clause saying the drift "
        f"is a past-based assumption, not a forecast.\n"
        f"- **Daily risk:** the 95% VaR and CVaR, plus the tail check in one sentence.\n"
        f"- **Drawdown:** the max drawdown and what it means.\n\n"
        f"### Plain-English Summary\n"
        f"- Two short bullets for a non-technical reader: first, say that about 9 out of 10 simulated outcomes "
        f"ended between the bad case and the good case (with % changes vs spot) and that the upside mostly "
        f"reflects past strong growth continuing, not a prediction; second, the 95% VaR explained as "
        f"'on roughly 1 in 20 days the loss can be bigger than X%', plus the worst 2-year drop."
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