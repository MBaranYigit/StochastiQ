# StochastiQ: Equity Scenario & Tail-Risk Analytics

**Live Demo:** https://stochastiqv1.streamlit.app

**StochastiQ** is an interactive equity risk analytics app built with Python, Streamlit, and Plotly. It combines Geometric Brownian Motion (GBM) Monte Carlo simulations with historical and parametric risk metrics (VaR, CVaR, and Maximum Drawdown) to explore forward-looking price scenarios and historical downside risk.

---

## 🎯 Features

* **Monte Carlo Price Simulation:** Generates up to 10,000 price paths using Geometric Brownian Motion calibrated from historical daily returns.
* **Probability Fan Charts:** Displays 5th–95th percentile bands, 25th–75th interquartile ranges, and mean versus median trajectories over the chosen horizon.
* **Tail-Risk Metrics:** Calculates daily Parametric VaR, Historical VaR, Conditional VaR (Expected Shortfall) at 95% and 99% confidence levels, and 2-year Maximum Drawdown.
* **Macro Stress Scenario:** Applies an optional shock (volatility ×1.5 and reduced drift) to test how wider uncertainty affects the projected price cone.
* **AI & Rule-Based Risk Summary:** Uses the Google Gemini API gemini-flash-lite-latest with a rule-based fallback template to generate a structured technical summary and a plain-English explanation of the outputs.

---

## 🧮 Methodology

### 1. Geometric Brownian Motion (Exact Discretization)
Asset prices are modeled under the stochastic differential equation:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Applying **Itô's Lemma** to $\ln(S_t)$ yields the exact discrete-time evolution step over $\Delta t = \frac{1}{252}$:

$$S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z_t \right), \quad Z_t \sim \mathcal{N}(0, 1)$$

Annualized drift ($\mu = \bar{r}_{log} \times 252 + 0.5\sigma^2$) and sample volatility ($\sigma$, with `ddof=1`) are estimated from a 2-year historical lookback window.

### 2. Value at Risk (VaR), CVaR & Stress Testing
* **VaR & CVaR Calculation:** Compares Gaussian Parametric VaR against non-parametric Historical VaR and Expected Shortfall (CVaR) at $\alpha = 0.05$ and $\alpha = 0.01$. VaR is computed from actual historical returns.
* **Stress Test Adjustments:** Enabling the stress toggle scales input volatility by ×1.5 and adjusts annualized drift downward by %15.

---

## ⚠️ Model Limitations

* **Constant Volatility & Volatility Clustering:** Standard GBM assumes constant annualized volatility ($\sigma$) across the entire simulation horizon. Real equity markets exhibit volatility clustering, where turbulent periods follow one another.
* **Stress Test Scope:** The built-in stress test is a static scenario analysis; it raises the constant volatility level but does not model time-varying volatility.
* **Fat Tails vs. Log-Normal Simulation:** Because simulated GBM log-returns are normally distributed (producing a log-normal terminal price distribution), the Monte Carlo paths themselves do not generate fat tails. Fat-tail behavior is only captured when evaluating empirical historical returns rather than simulated GBM returns.
* **Interpretation of Outputs:** VaR figures and the 5th–95th percentile cone are model outputs, not predictions.

---

## 🗺️ Roadmap

* **Volatility Modeling:** Implement EWMA ($\lambda = 0.94$) to update the starting volatility estimate; GARCH(1,1) to model forward-looking time-varying volatility in the simulation.
* **Multi-Tier Stress Presets:** Add selectable historical crisis scenarios alongside custom user-defined shock sliders.
* **Out-of-Sample Backtesting:** Evaluate historical cone coverage and VaR breach frequency using the Kupiec POF (Proportion of Failures) test.

---

## 🛠️ Tech Stack

* **Core & Math:** Python, NumPy, Pandas, SciPy
* **Market Data:** Historical daily equity price feed (Mostly yfinance)
* **Frontend & Visualization:** Streamlit, Plotly
* **Reporting:** Google Gemini REST API gemini-flash-lite-latest with a deterministic Python fallback formatter

---

## 📌 Disclaimer

This is an educational project, not financial advice.