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
    Dynamic Rule-Based Quant Reporting Engine (Zero-Downtime Fallback) — Draft 1 Style.
    """
    cur = data.get("currency", "$")
    spot_str = _format_currency(data["current_price"], cur)
    mean_str = _format_currency(sim_results["mean_target"], cur)
    med_str = _format_currency(sim_results["median_target"], cur)
    p5_str = _format_currency(sim_results["percentile_5"], cur)
    p95_str = _format_currency(sim_results["percentile_95"], cur)

    t = sim_days / 252.0
    sigma = data["annualized_volatility"]
    mu = data["annualized_drift"]
    d2 = ((mu - 0.5 * (sigma ** 2)) * t) / (sigma * math.sqrt(t)) if (sigma > 0 and t > 0) else 0.0
    prob_above = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0))) * 100.0
    paths_out_of_10 = round(prob_above / 10.0)

    vp95 = risk_results["var_param_95"] * 100
    vh95 = risk_results["var_hist_95"] * 100
    cvar95 = risk_results["cvar_95"] * 100
    mdd = risk_results["max_drawdown"] * 100

    stress_note = " (with the macro stress adjustment applied: volatility ×1.5 and reduced drift)" if stress_mode else ""

    if prob_above >= 55:
        momentum_comment = (
            f"its historical upward trend is strong enough that roughly **{paths_out_of_10} out of 10** "
            f"simulated paths (**{prob_above:.1f}%**) still finish in profit."
        )
    elif prob_above >= 45:
        momentum_comment = (
            f"its historical trend and daily volatility largely balance out, leaving roughly **{paths_out_of_10} out of 10** "
            f"simulated paths (**{prob_above:.1f}%**) above the starting price."
        )
    else:
        momentum_comment = (
            f"high volatility and weak historical trend pull the majority of paths downward, with only about "
            f"**{paths_out_of_10} out of 10** simulated paths (**{prob_above:.1f}%**) finishing above today's price."
        )

    if vh95 > vp95:
        tail_comment = (
            f"Comparing actual historical drops (**-{vh95:.2f}%** Historical VaR) against standard bell-curve estimates "
            f"(**-{vp95:.2f}%** Parametric VaR) shows that real-world market sell-offs in this stock have been sharper "
            f"than a smooth, textbook distribution would suggest."
        )
    else:
        tail_comment = (
            f"Comparing actual historical drops (**-{vh95:.2f}%** Historical VaR) against standard bell-curve estimates "
            f"(**-{vp95:.2f}%** Parametric VaR) shows that historical daily sell-offs have generally stayed in line with "
            f"(or slightly milder than) a standard textbook distribution."
        )

    return f"""Under the selected model assumptions{stress_note} over a **{sim_days}-trading-day** horizon, the median simulated price for **{ticker}** is **{med_str}** (compared to the current price of **{spot_str}**), with approximately **{prob_above:.1f}%** of simulated outcomes ending above today's price.

* **Upward Momentum vs. Daily Swings:** Even though the stock experiences wide price swings—creating a broad 90% range between **{p5_str}** (5th percentile) and **{p95_str}** (95th percentile)—{momentum_comment}
* **Why the Average is Higher than the Median:** The average simulated price (**{mean_str}**) sits noticeably higher than the typical middle outcome (**{med_str}**). This happens because a small number of strong rally scenarios pull the average up; therefore, the **median ({med_str})** is a more realistic benchmark for a typical scenario than the overall average.
* **Theory vs. Real-World Sell-Offs:** {tail_comment}
* **What Happens on the Worst Days:** On 95% of historical trading days, single-day drops stayed milder than **-{vh95:.2f}%** (VaR). However, during the worst 5% of market sell-offs, losses did not just stop there—they deepened to an average drop of **-{cvar95:.2f}%** (CVaR), with a worst two-year peak-to-bottom decline of **{mdd:.2f}%**.

*Note: These figures and insights are based strictly on historical data and model assumptions. They represent mathematical model outputs rather than predictions and are not financial advice.*"""


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

    t = sim_days / 252.0
    sigma = data["annualized_volatility"]
    mu = data["annualized_drift"]
    d2 = ((mu - 0.5 * (sigma ** 2)) * t) / (sigma * math.sqrt(t)) if (sigma > 0 and t > 0) else 0.0
    prob_above = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0))) * 100.0
    paths_out_of_10 = round(prob_above / 10.0)

    stress_context = (
        " (noting that the macro stress adjustment—volatility ×1.5 and reduced drift—is active)"
        if stress_mode else ""
    )

    prompt_text = (
        f"Write a clear, easy-to-understand analytical summary in English for {ticker}. "
        f"Use zero academic jargon (do not use words like leptokurtic, Gaussian, or variance drag). "
        f"Never use the '$' symbol (always write '{cur_code}' instead, e.g., '{cur_code} 216.00'). "
        f"Use Markdown bolding (**...**) for key numbers and phrases so the text is easy to scan.\n\n"
        f"Follow this EXACT layout (1 intro paragraph, 4 bullet points with these exact bold titles, and 1 italic disclaimer at the end):\n\n"
        f"Under the selected model assumptions{stress_context} over a **{sim_days}-trading-day** horizon, the median simulated price for **{ticker}** is **{cur_code} {sim_results['median_target']:.2f}** (compared to the current price of **{cur_code} {data['current_price']:.2f}**), with approximately **{prob_above:.1f}%** of simulated outcomes ending above today's price.\n\n"
        f"* **Upward Momentum vs. Daily Swings:** Explain the 90% range between **{cur_code} {sim_results['percentile_5']:.2f}** (5th percentile) and **{cur_code} {sim_results['percentile_95']:.2f}** (95th percentile), and interpret what roughly **{paths_out_of_10} out of 10** simulated paths (**{prob_above:.1f}%**) finishing above today's price means for trend strength versus daily swings.\n"
        f"* **Why the Average is Higher than the Median:** Explain why the average simulated price (**{cur_code} {sim_results['mean_target']:.2f}**) sits higher than the median (**{cur_code} {sim_results['median_target']:.2f}**) due to a small number of strong rally scenarios pulling the average up, making the **median ({cur_code} {sim_results['median_target']:.2f})** a more realistic benchmark for a typical scenario.\n"
        f"* **Theory vs. Real-World Sell-Offs:** Compare actual historical drops (**-{risk_results['var_hist_95']*100:.2f}%** Historical VaR) against standard bell-curve estimates (**-{risk_results['var_param_95']*100:.2f}%** Parametric VaR) and state in plain English whether real-world sell-offs in this stock have been sharper or milder than a smooth textbook distribution suggests.\n"
        f"* **What Happens on the Worst Days:** Explain that on 95% of trading days single-day drops stayed milder than **-{risk_results['var_hist_95']*100:.2f}%** (VaR), but during the worst 5% of market sell-offs losses deepened to an average drop of **-{risk_results['cvar_95']*100:.2f}%** (CVaR), with a worst two-year peak-to-bottom decline of **{risk_results['max_drawdown']*100:.2f}%**.\n\n"
        f"*Note: These figures and insights are based strictly on historical data and model assumptions. They represent mathematical model outputs rather than predictions and are not financial advice.*"
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