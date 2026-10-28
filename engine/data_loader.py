import yfinance as yf
import numpy as np
import pandas as pd

def fetch_equity_data(ticker: str, period: str = "2y", vol_model: str = "standard") -> dict:
    clean_ticker = ticker.strip().upper()
    stock = yf.Ticker(clean_ticker)
    df = stock.history(period=period)

    if df.empty or len(df) < 30:
        raise ValueError(f"Insufficient historical price data found for '{clean_ticker}'.")

    # Ensure prices is a clean 1D Series with no missing or non-positive values
    prices = df['Close'].squeeze().dropna()
    prices = prices[prices > 0]

    if len(prices) < 30:
        raise ValueError(f"Insufficient valid closing prices found for '{clean_ticker}'.")

    # Simple returns (for monetary loss metrics such as VaR and CVaR)
    simple_returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    # Logarithmic returns (for GBM stochastic modeling)
    log_returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()

    daily_log_mean = float(log_returns.mean())
    
    # 1. Standard Equal-Weighted Volatility
    daily_std_equal = float(log_returns.std(ddof=1))
    vol_standard = float(daily_std_equal * np.sqrt(252))

    # 2. RiskMetrics EWMA Volatility (lambda = 0.94 -> alpha = 1 - 0.94 = 0.06)
    ewma_series = log_returns.ewm(alpha=0.06, adjust=False).std().dropna()
    daily_std_ewma = float(ewma_series.iloc[-1]) if not ewma_series.empty else daily_std_equal
    
    # Fallback guard in case of any numerical edge case
    if not np.isfinite(daily_std_ewma) or daily_std_ewma <= 0:
        daily_std_ewma = daily_std_equal

    vol_ewma = float(daily_std_ewma * np.sqrt(252))

    # Select active volatility based on user choice
    if vol_model.lower() == "ewma":
        annualized_volatility = vol_ewma
    else:
        annualized_volatility = vol_standard
    
    # By Itô's Lemma, E[ln(S_t/S_0)] = (mu - 0.5 * sigma^2) * t
    # True annual drift (mu) uses the 2-year sample variance to remain consistent with the 2-year sample mean
    annualized_drift = float((daily_log_mean * 252) + 0.5 * (vol_standard ** 2))

    currency = "₺" if clean_ticker.endswith(".IS") else "$"

    return {
        "ticker": clean_ticker,
        "currency": currency,
        "current_price": float(prices.iloc[-1]),
        "history": prices,
        "simple_returns": simple_returns,
        "log_returns": log_returns,
        "annualized_drift": annualized_drift,
        "annualized_volatility": annualized_volatility,
        "vol_standard": vol_standard,
        "vol_ewma": vol_ewma,
        "vol_model": vol_model.lower()
    }

# --- FOR TESTING ONLY (runs when this file is executed directly) ---
if __name__ == "__main__":
    test_data = fetch_equity_data("THYAO.IS", vol_model="ewma")
    print(f"Ticker: {test_data['ticker']}")
    print(f"Latest Price: {test_data['current_price']:.2f} {test_data['currency']}")
    print(f"Annualized Drift (Mu): {test_data['annualized_drift']*100:.2f}%")
    print(f"Standard Volatility: {test_data['vol_standard']*100:.2f}%")
    print(f"EWMA Volatility (λ=0.94): {test_data['vol_ewma']*100:.2f}%")