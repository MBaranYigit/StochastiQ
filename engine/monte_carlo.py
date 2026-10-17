import numpy as np

def run_gbm_simulation(
    current_price: float,
    drift: float,
    volatility: float,
    days: int = 252,
    simulations: int = 10000,
    seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    
    # Shape: (days, simulations) -> Random shock (Z_t) for each day and each scenario
    z = rng.standard_normal((days, simulations))
    
    # Since the real-world drift is applied from day 1, subtracting 0.5 * sigma^2 here is correct
    drift_component = (drift - 0.5 * (volatility ** 2)) * dt
    diffusion_component = volatility * np.sqrt(dt) * z
    
    daily_multipliers = np.exp(drift_component + diffusion_component)
    
    price_paths = np.empty((days + 1, simulations))
    price_paths[0] = current_price
    price_paths[1:] = current_price * np.cumprod(daily_multipliers, axis=0)
    
    terminal_prices = price_paths[-1]
    
    return {
        "price_paths": price_paths,
        "terminal_prices": terminal_prices,
        "mean_target": float(np.mean(terminal_prices)),
        "median_target": float(np.median(terminal_prices)),
        "percentile_5": float(np.percentile(terminal_prices, 5)),
        "percentile_25": float(np.percentile(terminal_prices, 25)),
        "percentile_75": float(np.percentile(terminal_prices, 75)),
        "percentile_95": float(np.percentile(terminal_prices, 95)),
    }

# --- MATHEMATICAL VALIDATION TEST ---
if __name__ == "__main__":
    # Example parameters: Starting price 100 TL, annual drift 30% (0.30), volatility 40% (0.40)
    S0 = 100.0
    mu = 0.30
    sigma = 0.40
    
    res = run_gbm_simulation(current_price=S0, drift=mu, volatility=sigma, simulations=10000)
    
   # Theoretical expected values (stochastic calculus formulas)
    teorik_mean = S0 * np.exp(mu * 1.0)
    teorik_median = S0 * np.exp((mu - 0.5 * sigma**2) * 1.0)
    
    print("=== MONTE CARLO GBM DOĞRULAMA TESTİ ===")
    print(f"Matris Boyutu (Gün+1, Senaryo): {res['price_paths'].shape}")
    print(f"Teorik Ortalama: {teorik_mean:.2f} | Simülasyon Ortalaması: {res['mean_target']:.2f}")
    print(f"Teorik Medyan  : {teorik_median:.2f} | Simülasyon Medyanı   : {res['median_target']:.2f}")
    print(f"%5 Kötü Senaryo: {res['percentile_5']:.2f} | %95 İyi Senaryo: {res['percentile_95']:.2f}")