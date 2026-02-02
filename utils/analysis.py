"""
Analysis utilities for ABM results.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def estimate_sigma(paths, dt):
    """Estimate volatility from ABM price paths."""
    paths = np.asarray(paths, dtype=float)
    dt = float(dt)

    if paths.ndim != 2:
        raise ValueError("ABM paths must be a 2D array of shape (n_paths, n_steps+1).")
    if not np.all(np.isfinite(paths)) or np.min(paths) <= 0:
        raise ValueError("ABM paths contain non-positive or non-finite prices.")
    
    lr = np.diff(np.log(paths), axis=1)
    if lr.shape[1] < 2:
        raise ValueError("Not enough time steps to estimate sigma.")

    start = int(0.1 * lr.shape[1])
    lr_tail = lr[:, start:] if start < lr.shape[1] else lr
    sigmas = np.std(lr_tail, axis=1) / np.sqrt(dt)
    sigmas = sigmas[np.isfinite(sigmas) & (sigmas > 0)]

    if sigmas.size == 0:
        raise ValueError("Estimated sigma is non-positive/non-finite.")

    s = float(np.median(sigmas))

    if not np.isfinite(s) or s <= 0:
        raise ValueError("Estimated sigma is non-positive/non-finite.")

    return s


def calculate_stylized_facts(prices, returns=None):
    """Calculate stylized facts from price/return series."""
    if returns is None:
        returns = np.diff(np.log(prices))
    
    returns = np.asarray(returns)
    
    facts = {
        "mean_return": float(np.mean(returns)),
        "volatility": float(np.std(returns)),
        "skewness": float(_skewness(returns)),
        "kurtosis": float(_kurtosis(returns)),
        "autocorr_abs": float(_autocorr_abs(returns)),
        "volatility_clustering": float(_volatility_clustering(returns))
    }
    
    return facts


def _skewness(x):
    """Calculate skewness."""
    x = x - np.mean(x)
    return np.mean(x**3) / (np.std(x)**3 + 1e-10)


def _kurtosis(x):
    """Calculate excess kurtosis."""
    x = x - np.mean(x)
    return np.mean(x**4) / (np.std(x)**4 + 1e-10) - 3.0


def _autocorr_abs(x, lag=1):
    """Autocorrelation of absolute returns."""
    abs_x = np.abs(x)
    if len(abs_x) <= lag:
        return 0.0
    return float(np.corrcoef(abs_x[:-lag], abs_x[lag:])[0, 1])


def _volatility_clustering(x, window=20):
    """Measure volatility clustering."""
    abs_x = np.abs(x)
    if len(abs_x) < 2 * window:
        return 0.0
    
    rolling_vol = pd.Series(abs_x).rolling(window=window).std()
    return float(rolling_vol.autocorr(lag=1) if len(rolling_vol) > 1 else 0.0)

