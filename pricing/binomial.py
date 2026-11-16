"""Cox-Ross-Rubinstein binomial tree for European and American options."""

from __future__ import annotations

from math import exp, sqrt

import numpy as np


def price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    expiry: float,
    steps: int,
    kind: str = "call",
    american: bool = False,
    div: float = 0.0,
) -> float:
    """Value by backward induction on a recombining tree with ``steps`` periods.

    Each period the underlying moves up by ``u = exp(vol * sqrt(dt))`` or down
    by ``1 / u``. The risk-neutral up probability makes the expected growth
    equal to ``rate - div``. An American option is worth the larger of its
    continuation value and its immediate exercise value at every node.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    dt = expiry / steps
    up = exp(vol * sqrt(dt))
    down = 1.0 / up
    prob = (exp((rate - div) * dt) - down) / (up - down)
    if not 0.0 < prob < 1.0:
        raise ValueError("time step too coarse for these inputs: risk-neutral probability outside (0, 1)")
    discount = exp(-rate * dt)
    sign = 1.0 if kind == "call" else -1.0

    def payoff(level: np.ndarray) -> np.ndarray:
        return np.maximum(sign * (level - strike), 0.0)

    ups = np.arange(steps + 1)
    values = payoff(spot * up**ups * down ** (steps - ups))
    for step in range(steps - 1, -1, -1):
        values = discount * (prob * values[1:] + (1.0 - prob) * values[:-1])
        if american:
            ups = np.arange(step + 1)
            values = np.maximum(values, payoff(spot * up**ups * down ** (step - ups)))
    return float(values[0])
