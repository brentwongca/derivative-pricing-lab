"""Monte Carlo pricing under geometric Brownian motion, with variance reduction.

Every pricer returns ``(estimate, standard_error)``. The standard error is the
point: a Monte Carlo price without one cannot be compared with anything.
"""

from __future__ import annotations

from math import exp, log, sqrt

import numpy as np

from .black_scholes import norm_cdf


def _estimate(discounted_payoffs: np.ndarray) -> tuple[float, float]:
    return float(discounted_payoffs.mean()), float(discounted_payoffs.std(ddof=1) / sqrt(len(discounted_payoffs)))


def _payoff(level: np.ndarray, strike: float, kind: str) -> np.ndarray:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return np.maximum(level - strike, 0.0) if kind == "call" else np.maximum(strike - level, 0.0)


def european(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    expiry: float,
    paths: int,
    kind: str = "call",
    div: float = 0.0,
    antithetic: bool = True,
    seed: int = 0,
) -> tuple[float, float]:
    """European option from terminal prices sampled exactly.

    With ``antithetic`` each normal draw ``z`` is paired with ``-z`` and the
    two payoffs are averaged. The pair is one observation, so the standard
    error stays honest: ``paths`` is the number of pairs.
    """
    draws = np.random.default_rng(seed).standard_normal(paths)
    drift = (rate - div - 0.5 * vol * vol) * expiry
    diffusion = vol * sqrt(expiry)
    payoffs = _payoff(spot * np.exp(drift + diffusion * draws), strike, kind)
    if antithetic:
        payoffs = 0.5 * (payoffs + _payoff(spot * np.exp(drift - diffusion * draws), strike, kind))
    return _estimate(exp(-rate * expiry) * payoffs)


def geometric_asian_exact(
    spot: float, strike: float, rate: float, vol: float, expiry: float, fixings: int, kind: str = "call", div: float = 0.0
) -> float:
    """Closed-form price of an Asian option on the geometric average of ``fixings`` equally spaced prices.

    The geometric average of lognormal prices is itself lognormal, so the
    Black-Scholes argument applies with an adjusted mean and variance.
    """
    dt = expiry / fixings
    mean = log(spot) + (rate - div - 0.5 * vol * vol) * dt * (fixings + 1) / 2.0
    variance = vol * vol * dt * (fixings + 1) * (2 * fixings + 1) / (6.0 * fixings)
    deviation = sqrt(variance)
    d1 = (mean - log(strike) + variance) / deviation
    d2 = d1 - deviation
    forward = exp(mean + 0.5 * variance)
    discount = exp(-rate * expiry)
    if kind == "call":
        return discount * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
    return discount * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))


def arithmetic_asian(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    expiry: float,
    fixings: int,
    paths: int,
    kind: str = "call",
    div: float = 0.0,
    control_variate: bool = True,
    seed: int = 0,
) -> tuple[float, float]:
    """Asian option on the arithmetic average, which has no closed form.

    With ``control_variate`` the geometric-average option, priced on the same
    paths, corrects the estimate: its simulated error is known exactly because
    its true price is known, and the two payoffs are almost perfectly
    correlated.
    """
    dt = expiry / fixings
    draws = np.random.default_rng(seed).standard_normal((paths, fixings))
    log_paths = log(spot) + np.cumsum((rate - div - 0.5 * vol * vol) * dt + vol * sqrt(dt) * draws, axis=1)
    discount = exp(-rate * expiry)
    arithmetic = discount * _payoff(np.exp(log_paths).mean(axis=1), strike, kind)
    if not control_variate:
        return _estimate(arithmetic)

    geometric = discount * _payoff(np.exp(log_paths.mean(axis=1)), strike, kind)
    covariance = np.cov(arithmetic, geometric)
    beta = covariance[0, 1] / covariance[1, 1]
    exact = geometric_asian_exact(spot, strike, rate, vol, expiry, fixings, kind, div)
    return _estimate(arithmetic - beta * (geometric - exact))
