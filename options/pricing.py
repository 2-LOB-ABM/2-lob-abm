"""
Option pricing models: Black-Scholes, Time-Fractional Black-Scholes, and Heston.
"""
import numpy as np
from scipy.stats import norm
from enum import Enum


class PricingModel(Enum):
    """Enumeration of available pricing models."""
    BS = "BlackScholes"
    TFBS = "TimeFractionalBS"
    HESTON = "Heston"


def sample_positive_stable(alpha, size, rng):
    """Sample positive stable random variables for TFBS."""
    alpha = float(alpha)
    U = rng.uniform(0.0, np.pi, size=size)
    W = rng.exponential(1.0, size=size)
    part1 = np.sin(alpha * U) / (np.sin(U) ** alpha)
    part2 = (np.sin((1.0 - alpha) * U) / W) ** (1.0 - alpha)
    return part1 * part2


class BlackScholesPricer:
    """Black-Scholes option pricing."""
    
    @staticmethod
    def price_call(S0, K, r, sigma, T):
        """Price a call option."""
        S0, K, r, sigma, T = float(S0), float(K), float(r), float(sigma), float(T)
        
        if T <= 0:
            return max(S0 - K, 0.0)
        if sigma <= 0:
            return max(S0 - K * np.exp(-r * T), 0.0)
        
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    @staticmethod
    def price_put(S0, K, r, sigma, T):
        """Price a put option."""
        S0, K, r, sigma, T = float(S0), float(K), float(r), float(sigma), float(T)
        
        if T <= 0:
            return max(K - S0, 0.0)
        if sigma <= 0:
            return max(K * np.exp(-r * T) - S0, 0.0)
        
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))

    @staticmethod
    def delta_call(S, K, r, sigma, T):
        """Delta of a call option."""
        S, K, r, sigma, T = float(S), float(K), float(r), float(sigma), float(T)
        
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
        return float(norm.cdf(d1))

    @staticmethod
    def delta_put(S, K, r, sigma, T):
        """Delta of a put option."""
        return float(BlackScholesPricer.delta_call(S, K, r, sigma, T) - 1.0)

    @staticmethod
    def gamma(S, K, r, sigma, T):
        """Gamma (same for call and put)."""
        S, K, r, sigma, T = float(S), float(K), float(r), float(sigma), float(T)
        
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
        return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))

    @staticmethod
    def vega(S, K, r, sigma, T):
        """Vega (same for call and put)."""
        S, K, r, sigma, T = float(S), float(K), float(r), float(sigma), float(T)
        
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
        return float(S * norm.pdf(d1) * np.sqrt(T))


class TimeFractionalBSPricer:
    """Time-Fractional Black-Scholes option pricing via Monte Carlo."""
    
    def __init__(self, alpha=0.85, n_mc=10000):
        """
        Args:
            alpha: Memory parameter (0 < alpha <= 1), where alpha=1 recovers BS
            n_mc: Number of Monte Carlo simulations
        """
        self.alpha = float(alpha)
        self.n_mc = int(n_mc)

    def price_call(self, S0, K, r, sigma, T, seed=None):
        """Price a call option using TFBS."""
        rng = np.random.default_rng(seed)
        S = sample_positive_stable(self.alpha, self.n_mc, rng)
        E = (T / np.maximum(S, 1e-16)) ** self.alpha
        
        Z = rng.normal(0.0, 1.0, size=self.n_mc)
        ST = S0 * np.exp((r - 0.5 * sigma * sigma) * E + sigma * np.sqrt(E) * Z)
        payoff = np.maximum(ST - K, 0.0)
        return float(np.exp(-r * T) * np.mean(payoff))

    def price_put(self, S0, K, r, sigma, T, seed=None):
        """Price a put option using TFBS."""
        rng = np.random.default_rng(seed)
        S = sample_positive_stable(self.alpha, self.n_mc, rng)
        E = (T / np.maximum(S, 1e-16)) ** self.alpha
        
        Z = rng.normal(0.0, 1.0, size=self.n_mc)
        ST = S0 * np.exp((r - 0.5 * sigma * sigma) * E + sigma * np.sqrt(E) * Z)
        payoff = np.maximum(K - ST, 0.0)
        return float(np.exp(-r * T) * np.mean(payoff))

    def delta_call(self, S, K, r, sigma, T, seed=None, eps=0.01):
        """Delta via finite difference."""
        if S <= 0 or abs(eps * S) < 1e-10:
            return 0.0
        p_up = self.price_call(S * (1 + eps), K, r, sigma, T, seed)
        p_down = self.price_call(S * (1 - eps), K, r, sigma, T, seed)
        return float((p_up - p_down) / (2 * eps * S))

    def delta_put(self, S, K, r, sigma, T, seed=None, eps=0.01):
        """Delta via finite difference."""
        if S <= 0 or abs(eps * S) < 1e-10:
            return 0.0
        p_up = self.price_put(S * (1 + eps), K, r, sigma, T, seed)
        p_down = self.price_put(S * (1 - eps), K, r, sigma, T, seed)
        return float((p_up - p_down) / (2 * eps * S))

    def gamma(self, S, K, r, sigma, T, seed=None, eps=0.01):
        """Gamma via finite difference."""
        if S <= 0 or abs(eps * S) < 1e-10:
            return 0.0
        p_up = self.price_call(S * (1 + eps), K, r, sigma, T, seed)
        p_mid = self.price_call(S, K, r, sigma, T, seed)
        p_down = self.price_call(S * (1 - eps), K, r, sigma, T, seed)
        denom = (eps * S) ** 2
        return float((p_up - 2 * p_mid + p_down) / denom) if denom > 1e-10 else 0.0

    def vega(self, S, K, r, sigma, T, seed=None, eps=0.01):
        """Vega via finite difference."""
        if sigma <= 0 or abs(eps * sigma) < 1e-10:
            return 0.0
        p_up = self.price_call(S, K, r, sigma * (1 + eps), T, seed)
        p_down = self.price_call(S, K, r, sigma * (1 - eps), T, seed)
        return float((p_up - p_down) / (2 * eps * sigma))


class HestonPricer:
    """Heston stochastic volatility model option pricing."""
    
    def __init__(self, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=None, n_mc=10000, n_steps=100):
        """
        Args:
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma_v: Volatility of volatility
            rho: Correlation between asset and variance
            v0: Initial variance (defaults to theta)
            n_mc: Number of Monte Carlo paths
            n_steps: Number of time steps per path
        """
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)
        self.v0 = float(v0 if v0 is not None else theta)
        self.n_mc = int(n_mc)
        self.n_steps = int(n_steps)

    def price_call(self, S0, K, r, sigma, T, seed=None):
        """Price a call option using Heston model."""
        # Use sigma as initial volatility estimate, but evolve variance stochastically
        rng = np.random.default_rng(seed)
        dt = T / self.n_steps
        
        S = np.full(self.n_mc, float(S0))
        v = np.full(self.n_mc, self.v0)
        
        for _ in range(self.n_steps):
            Z1 = rng.normal(0, 1, self.n_mc)
            Z2 = rng.normal(0, 1, self.n_mc)
            Z_v = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            # Update variance (Feller condition: 2*kappa*theta > sigma_v^2)
            v = np.maximum(v + self.kappa * (self.theta - v) * dt + self.sigma_v * np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * Z_v, 0.0)
            
            # Update asset price
            S = S * np.exp((r - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * Z1)
        
        payoff = np.maximum(S - K, 0.0)
        return float(np.exp(-r * T) * np.mean(payoff))

    def price_put(self, S0, K, r, sigma, T, seed=None):
        """Price a put option using Heston model."""
        rng = np.random.default_rng(seed)
        dt = T / self.n_steps
        
        S = np.full(self.n_mc, float(S0))
        v = np.full(self.n_mc, self.v0)
        
        for _ in range(self.n_steps):
            Z1 = rng.normal(0, 1, self.n_mc)
            Z2 = rng.normal(0, 1, self.n_mc)
            Z_v = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            v = np.maximum(v + self.kappa * (self.theta - v) * dt + self.sigma_v * np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * Z_v, 0.0)
            S = S * np.exp((r - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * Z1)
        
        payoff = np.maximum(K - S, 0.0)
        return float(np.exp(-r * T) * np.mean(payoff))

    def delta_call(self, S, K, r, sigma, T, seed=None, eps=0.01):
        """Delta via finite difference."""
        if S <= 0 or abs(eps * S) < 1e-10:
            return 0.0
        p_up = self.price_call(S * (1 + eps), K, r, sigma, T, seed)
        p_down = self.price_call(S * (1 - eps), K, r, sigma, T, seed)
        return float((p_up - p_down) / (2 * eps * S))

    def delta_put(self, S, K, r, sigma, T, seed=None, eps=0.01):
        """Delta via finite difference."""
        if S <= 0 or abs(eps * S) < 1e-10:
            return 0.0
        p_up = self.price_put(S * (1 + eps), K, r, sigma, T, seed)
        p_down = self.price_put(S * (1 - eps), K, r, sigma, T, seed)
        return float((p_up - p_down) / (2 * eps * S))

    def gamma(self, S, K, r, sigma, T, seed=None, eps=0.01):
        """Gamma via finite difference."""
        if S <= 0 or abs(eps * S) < 1e-10:
            return 0.0
        p_up = self.price_call(S * (1 + eps), K, r, sigma, T, seed)
        p_mid = self.price_call(S, K, r, sigma, T, seed)
        p_down = self.price_call(S * (1 - eps), K, r, sigma, T, seed)
        denom = (eps * S) ** 2
        return float((p_up - 2 * p_mid + p_down) / denom) if denom > 1e-10 else 0.0

    def vega(self, S, K, r, sigma, T, seed=None, eps=0.01):
        """Vega via finite difference."""
        if sigma <= 0 or abs(eps * sigma) < 1e-10:
            return 0.0
        p_up = self.price_call(S, K, r, sigma * (1 + eps), T, seed)
        p_down = self.price_call(S, K, r, sigma * (1 - eps), T, seed)
        return float((p_up - p_down) / (2 * eps * sigma))


def get_pricer(model_type: PricingModel, **kwargs):
    """Factory function to get a pricer instance."""
    if model_type == PricingModel.BS:
        return BlackScholesPricer()
    elif model_type == PricingModel.TFBS:
        return TimeFractionalBSPricer(**kwargs)
    elif model_type == PricingModel.HESTON:
        return HestonPricer(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

