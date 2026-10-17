import yfinance as yf
import numpy as np
import pandas as pd

def fetch_equity_data(ticker: str, period: str = "2y") -> dict:
    clean_ticker = ticker.strip().upper()
    stock = yf.Ticker(clean_ticker)
    df = stock.history(period=period)

    if df.empty or len(df) < 30:
        raise ValueError(f"'{clean_ticker}' için yeterli fiyat verisi bulunamadı.")

    prices = df['Close']

    # Simple returns (for monetary loss metrics such as VaR and CVaR)
    simple_returns = prices.pct_change().dropna()

    # Logarithmic returns (for GBM stochastic modeling)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    daily_log_mean = float(log_returns.mean())
    daily_log_std = float(log_returns.std(ddof=1))

    annualized_volatility = float(daily_log_std * np.sqrt(252))
    
    # By Itô's Lemma, E[ln(S_t/S_0)] = (mu - 0.5 * sigma^2) * t
    # True annual drift (mu) = annualized log mean + 0.5 * sigma^2
    annualized_drift = float((daily_log_mean * 252) + 0.5 * (annualized_volatility ** 2))

    currency = "₺" if clean_ticker.endswith(".IS") else "$"

    return {
        "ticker": clean_ticker,
        "currency": currency,
        "current_price": float(prices.iloc[-1]),
        "history": prices,
        "simple_returns": simple_returns,
        "log_returns": log_returns,
        "annualized_drift": annualized_drift,
        "annualized_volatility": annualized_volatility
    }

# --- FOR TESTING ONLY (runs when this file is executed directly) ---
if __name__ == "__main__":
    test_data = fetch_equity_data("THYAO.IS")
    print(f"Hisse: {test_data['ticker']}")
    print(f"Son Fiyat: {test_data['current_price']:.2f} {test_data['currency']}")
    print(f"Yıllık Drift (Mu): %{test_data['annualized_drift']*100:.2f}")
    print(f"Yıllık Volatilite (Sigma): %{test_data['annualized_volatility']*100:.2f}")