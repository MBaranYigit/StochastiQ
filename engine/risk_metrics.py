import numpy as np
import pandas as pd
from scipy.stats import norm

def compute_risk_metrics(simple_returns: pd.Series, prices: pd.Series) -> dict:
    returns = simple_returns.values
    results = {}
    
    # Sample mean and unbiased sample standard deviation (ddof=1)
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    
    for level in [0.95, 0.99]:
        alpha = 1.0 - level
        tag = int(level * 100)
        
        # Parametric VaR (norm.ppf(alpha) is negative, so mu + z*sigma gives the left-tail return)
        param_var = -(mu + norm.ppf(alpha) * sigma)
        
        # Historical VaR (empirical quantile of historical simple returns)
        q_alpha = np.percentile(returns, alpha * 100)
        hist_var = -q_alpha
        
        # Conditional VaR / Expected Shortfall (mean loss beyond the Historical VaR threshold)
        tail_losses = returns[returns <= q_alpha]
        cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else hist_var
        
        results[f"var_param_{tag}"] = float(param_var)
        results[f"var_hist_{tag}"] = float(hist_var)
        results[f"cvar_{tag}"] = float(cvar)
        
    # Maximum Drawdown (peak-to-trough decline over the historical period)
    cumulative_max = np.maximum.accumulate(prices.values)
    drawdowns = (prices.values - cumulative_max) / cumulative_max
    results["max_drawdown"] = float(np.min(drawdowns))
    
    return results

# --- MATHEMATICAL VERIFICATION BLOCK ---
if __name__ == "__main__":
    # Test 1: Controlled Price Series for Exact Max Drawdown Verification
    # Peak is 200, Trough is 120 -> Exact Drawdown MUST be (120 - 200) / 200 = -0.40 (-40.00%)
    dummy_prices = pd.Series([100.0, 150.0, 200.0, 160.0, 120.0, 180.0])
    dummy_returns = dummy_prices.pct_change().dropna()
    
    dummy_res = compute_risk_metrics(dummy_returns, dummy_prices)
    print("=== SANITY CHECK 1: MAX DRAWDOWN ===")
    print(f"Expected Max Drawdown: -40.00% | Computed: {dummy_res['max_drawdown']*100:.2f}%\n")
    
    # Test 2: Statistical Gaussian Verification (100,000 days with mu=0, sigma=2%)
    # Theoretical 95% Parametric VaR = 1.64485 * 2% = 3.29%
    rng = np.random.default_rng(42)
    sim_returns = pd.Series(rng.normal(0.0, 0.02, 100000))
    sim_prices = pd.Series(100.0 * np.cumprod(1.0 + sim_returns))
    
    sim_res = compute_risk_metrics(sim_returns, sim_prices)
    print("=== SANITY CHECK 2: VaR & CVaR COHERENCE ===")
    print(f"Theoretical 95% VaR : 3.29%")
    print(f"Computed Param VaR  : {sim_res['var_param_95']*100:.2f}%")
    print(f"Computed Hist VaR   : {sim_res['var_hist_95']*100:.2f}%")
    print(f"Computed 95% CVaR   : {sim_res['cvar_95']*100:.2f}% (Must be > VaR)")
    print(f"99% VaR vs 95% VaR  : {sim_res['var_param_99']*100:.2f}% > {sim_res['var_param_95']*100:.2f}%")