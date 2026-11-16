"""Black-Scholes-Merton prices, Greeks and implied volatility for European options.

Notation used across the package:

    spot    current price of the underlying
    strike  exercise price
    rate    continuously compounded risk-free rate
    vol     annualised volatility
    expiry  time to expiry in years
    div     continuous dividend yield
"""

from __future__ import annotations

from math import erf, exp, log, pi, sqrt


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _check(spot: float, strike: float, vol: float, expiry: float, kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if spot <= 0 or strike <= 0 or vol <= 0 or expiry <= 0:
        raise ValueError("spot, strike, vol and expiry must be positive")


def _d1_d2(spot: float, strike: float, rate: float, vol: float, expiry: float, div: float) -> tuple[float, float]:
    d1 = (log(spot / strike) + (rate - div + 0.5 * vol * vol) * expiry) / (vol * sqrt(expiry))
    return d1, d1 - vol * sqrt(expiry)


def price(
    spot: float, strike: float, rate: float, vol: float, expiry: float, kind: str = "call", div: float = 0.0
) -> float:
    _check(spot, strike, vol, expiry, kind)
    d1, d2 = _d1_d2(spot, strike, rate, vol, expiry, div)
    forward_leg = spot * exp(-div * expiry)
    strike_leg = strike * exp(-rate * expiry)
    if kind == "call":
        return forward_leg * norm_cdf(d1) - strike_leg * norm_cdf(d2)
    return strike_leg * norm_cdf(-d2) - forward_leg * norm_cdf(-d1)


def greeks(
    spot: float, strike: float, rate: float, vol: float, expiry: float, kind: str = "call", div: float = 0.0
) -> dict[str, float]:
    """Sensitivities per unit change: vega per 1.00 of vol, theta per year, rho per 1.00 of rate."""
    _check(spot, strike, vol, expiry, kind)
    d1, d2 = _d1_d2(spot, strike, rate, vol, expiry, div)
    sign = 1.0 if kind == "call" else -1.0
    carry = exp(-div * expiry)
    discount = exp(-rate * expiry)
    return {
        "delta": sign * carry * norm_cdf(sign * d1),
        "gamma": carry * norm_pdf(d1) / (spot * vol * sqrt(expiry)),
        "vega": spot * carry * norm_pdf(d1) * sqrt(expiry),
        "theta": (
            -spot * carry * norm_pdf(d1) * vol / (2.0 * sqrt(expiry))
            - sign * rate * strike * discount * norm_cdf(sign * d2)
            + sign * div * spot * carry * norm_cdf(sign * d1)
        ),
        "rho": sign * strike * expiry * discount * norm_cdf(sign * d2),
    }


def implied_vol(
    target: float,
    spot: float,
    strike: float,
    rate: float,
    expiry: float,
    kind: str = "call",
    div: float = 0.0,
    tolerance: float = 1e-8,
) -> float:
    """Volatility that reproduces ``target``, by Newton steps kept inside a bisection bracket.

    Newton converges in a few steps near the money but can overshoot where vega
    is tiny. Any step that leaves the bracket is replaced by its midpoint, so
    the search cannot diverge.
    """
    _check(spot, strike, 1.0, expiry, kind)
    low, high = 1e-6, 5.0
    floor = price(spot, strike, rate, low, expiry, kind, div)
    ceiling = price(spot, strike, rate, high, expiry, kind, div)
    if not floor <= target <= ceiling:
        raise ValueError(f"price {target} is outside the no-arbitrage range [{floor:.6g}, {ceiling:.6g}]")

    vol = 0.2
    for _ in range(100):
        error = price(spot, strike, rate, vol, expiry, kind, div) - target
        if abs(error) < tolerance:
            return vol
        if error > 0:
            high = vol
        else:
            low = vol
        vega = greeks(spot, strike, rate, vol, expiry, kind, div)["vega"]
        step = vol - error / vega if vega > 1e-12 else low - 1.0
        vol = step if low < step < high else 0.5 * (low + high)
    raise RuntimeError("implied volatility search did not converge")
