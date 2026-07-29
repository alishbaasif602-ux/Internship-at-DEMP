"""
generate_data.py
----------------
Generates a synthetic daily "stock price" dataset that mimics the statistical
behaviour of a real equity (Tesla/Apple/Google-style): an upward trend,
weekly + yearly seasonality, a longer business cycle, and stochastic noise
(via a small random walk / GBM-like component).

Why synthetic data?
This sandbox's outbound network is restricted to package registries
(PyPI, npm, GitHub, etc.) and cannot reach Yahoo Finance / Kaggle to
download a real CSV. The generator below produces a dataset with the
*same shape and columns* (Date, Open, High, Low, Close, Volume) as a
real Yahoo Finance export, so every downstream step (preprocessing,
sliding windows, RNN/LSTM/GRU training) works unchanged.

>>> To use REAL data instead <<<
Simply download e.g. TSLA.csv from Yahoo Finance / Kaggle and place it
at data/stock_prices.csv with columns: Date, Open, High, Low, Close, Volume.
The notebook and src/ code do not need to change.
"""

import numpy as np
import pandas as pd


def generate_stock_data(
    n_days: int = 1500,
    start_price: float = 150.0,
    trend_per_day: float = 0.05,
    seasonal_amplitude: float = 8.0,
    cycle_amplitude: float = 15.0,
    noise_std: float = 2.5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)

    # 1. Trend component (long-term upward drift)
    trend = trend_per_day * t

    # 2. Seasonality component (weekly + yearly periodicity)
    weekly_season = seasonal_amplitude * 0.3 * np.sin(2 * np.pi * t / 5)
    yearly_season = seasonal_amplitude * np.sin(2 * np.pi * t / 252)

    # 3. Cyclic component (longer irregular business cycle, ~2 years)
    cycle = cycle_amplitude * np.sin(2 * np.pi * t / 500 + 0.5)

    # 4. Noise component (random walk increments -> realistic stock "jitter")
    noise = np.cumsum(rng.normal(0, noise_std, size=n_days)) * 0.15
    noise += rng.normal(0, noise_std, size=n_days)  # + pure white noise

    close = start_price + trend + weekly_season + yearly_season + cycle + noise
    close = np.maximum(close, 1.0)  # prices can't go negative

    daily_range = np.abs(rng.normal(1.5, 0.5, size=n_days))
    open_ = close + rng.normal(0, 1.0, size=n_days)
    high = np.maximum(open_, close) + daily_range
    low = np.minimum(open_, close) - daily_range
    volume = rng.integers(1_000_000, 8_000_000, size=n_days)

    dates = pd.bdate_range(start="2019-01-01", periods=n_days)  # business days

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Close": close.round(2),
            "Volume": volume,
        }
    )

    # Inject a handful of missing values to make preprocessing realistic
    missing_idx = rng.choice(n_days, size=int(n_days * 0.01), replace=False)
    df.loc[missing_idx, "Close"] = np.nan

    return df


if __name__ == "__main__":
    df = generate_stock_data()
    df.to_csv("data/stock_prices.csv", index=False)
    print(f"Saved data/stock_prices.csv with {len(df)} rows")
    print(df.head())
