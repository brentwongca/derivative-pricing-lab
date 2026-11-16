from math import exp

import pytest

from pricing import binomial, monte_carlo
from pricing import black_scholes as bs

SPOT, STRIKE, RATE, VOL, EXPIRY = 100.0, 100.0, 0.05, 0.2, 1.0
KINDS = ["call", "put"]


# --- Black-Scholes -----------------------------------------------------------


def test_textbook_values():
    assert bs.price(SPOT, STRIKE, RATE, VOL, EXPIRY, "call") == pytest.approx(10.4506, abs=1e-4)
    assert bs.price(SPOT, STRIKE, RATE, VOL, EXPIRY, "put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
@pytest.mark.parametrize("div", [0.0, 0.03])
def test_put_call_parity(strike, div):
    call = bs.price(SPOT, strike, RATE, VOL, EXPIRY, "call", div)
    put = bs.price(SPOT, strike, RATE, VOL, EXPIRY, "put", div)
    assert call - put == pytest.approx(SPOT * exp(-div * EXPIRY) - strike * exp(-RATE * EXPIRY))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("strike", [90.0, 110.0])
def test_greeks_match_finite_differences(kind, strike):
    div, bump = 0.02, 1e-4
    greeks = bs.greeks(SPOT, strike, RATE, VOL, EXPIRY, kind, div)

    def value(spot=SPOT, rate=RATE, vol=VOL, expiry=EXPIRY):
        return bs.price(spot, strike, rate, vol, expiry, kind, div)

    def slope(name):
        base = {"spot": SPOT, "rate": RATE, "vol": VOL, "expiry": EXPIRY}[name]
        return (value(**{name: base + bump}) - value(**{name: base - bump})) / (2 * bump)

    # Central differences carry their own truncation error, so compare to six significant figures.
    assert greeks["delta"] == pytest.approx(slope("spot"), rel=1e-6)
    assert greeks["vega"] == pytest.approx(slope("vol"), rel=1e-6)
    assert greeks["rho"] == pytest.approx(slope("rate"), rel=1e-6)
    assert greeks["theta"] == pytest.approx(-slope("expiry"), rel=1e-6)  # value decays as expiry shrinks
    curvature = (value(spot=SPOT + 0.01) - 2 * value() + value(spot=SPOT - 0.01)) / 0.01**2
    assert greeks["gamma"] == pytest.approx(curvature, rel=1e-4)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("strike", [70.0, 100.0, 140.0])
@pytest.mark.parametrize("vol", [0.08, 0.25, 0.9])
def test_implied_vol_recovers_the_input(kind, strike, vol):
    target = bs.price(SPOT, strike, RATE, vol, EXPIRY, kind)
    recovered = bs.implied_vol(target, SPOT, strike, RATE, EXPIRY, kind)
    assert bs.price(SPOT, strike, RATE, recovered, EXPIRY, kind) == pytest.approx(target, abs=1e-7)


def test_implied_vol_rejects_prices_that_admit_arbitrage():
    below_intrinsic = SPOT - 80.0 * exp(-RATE * EXPIRY) - 0.5
    with pytest.raises(ValueError, match="no-arbitrage"):
        bs.implied_vol(below_intrinsic, SPOT, 80.0, RATE, EXPIRY, "call")


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        bs.price(SPOT, STRIKE, RATE, VOL, EXPIRY, "straddle")
    with pytest.raises(ValueError):
        bs.price(SPOT, STRIKE, RATE, -0.2, EXPIRY)
    with pytest.raises(ValueError):
        binomial.price(SPOT, STRIKE, RATE, VOL, EXPIRY, steps=0)


# --- Binomial tree -----------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("div", [0.0, 0.03])
def test_tree_converges_to_black_scholes(kind, div):
    exact = bs.price(SPOT, 105.0, RATE, VOL, EXPIRY, kind, div)
    coarse = abs(binomial.price(SPOT, 105.0, RATE, VOL, EXPIRY, 50, kind, div=div) - exact)
    fine = abs(binomial.price(SPOT, 105.0, RATE, VOL, EXPIRY, 2000, kind, div=div) - exact)
    assert fine < 5e-3
    assert fine < coarse


def test_single_step_tree_matches_hand_calculation():
    # u = e^0.2, d = 1/u, p = (e^0.05 - d) / (u - d); only the up node pays.
    up, down = exp(0.2), exp(-0.2)
    prob = (exp(0.05) - down) / (up - down)
    expected = exp(-0.05) * prob * (100.0 * up - 100.0)
    assert binomial.price(SPOT, STRIKE, RATE, VOL, EXPIRY, 1, "call") == pytest.approx(expected)


def test_american_call_without_dividends_is_never_exercised_early():
    european = binomial.price(SPOT, STRIKE, RATE, VOL, EXPIRY, 500, "call")
    american = binomial.price(SPOT, STRIKE, RATE, VOL, EXPIRY, 500, "call", american=True)
    assert american == pytest.approx(european, abs=1e-12)


def test_early_exercise_has_value_for_puts_and_for_calls_on_dividend_payers():
    european_put = binomial.price(SPOT, STRIKE, RATE, VOL, EXPIRY, 500, "put")
    american_put = binomial.price(SPOT, STRIKE, RATE, VOL, EXPIRY, 500, "put", american=True)
    assert american_put > european_put + 0.3

    european_call = binomial.price(SPOT, STRIKE, RATE, VOL, EXPIRY, 500, "call", div=0.08)
    american_call = binomial.price(SPOT, STRIKE, RATE, VOL, EXPIRY, 500, "call", american=True, div=0.08)
    assert american_call > european_call + 0.05


def test_american_put_is_worth_at_least_immediate_exercise():
    deep = binomial.price(60.0, STRIKE, RATE, VOL, EXPIRY, 500, "put", american=True)
    assert deep == pytest.approx(40.0)  # so far in the money that exercising now is optimal


# --- Monte Carlo -------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_european_estimate_is_within_three_standard_errors(kind):
    exact = bs.price(SPOT, 105.0, RATE, VOL, EXPIRY, kind, 0.02)
    estimate, error = monte_carlo.european(SPOT, 105.0, RATE, VOL, EXPIRY, 200_000, kind, div=0.02, seed=11)
    assert abs(estimate - exact) < 3 * error
    assert error < 0.03


def test_antithetic_draws_beat_plain_sampling_at_equal_cost():
    _, paired = monte_carlo.european(SPOT, STRIKE, RATE, VOL, EXPIRY, 50_000, antithetic=True, seed=2)
    _, plain = monte_carlo.european(SPOT, STRIKE, RATE, VOL, EXPIRY, 100_000, antithetic=False, seed=2)
    assert paired < plain  # both use 100,000 payoff evaluations


@pytest.mark.parametrize("kind", KINDS)
def test_geometric_asian_with_one_fixing_is_a_european_option(kind):
    exact = monte_carlo.geometric_asian_exact(SPOT, 95.0, RATE, VOL, EXPIRY, 1, kind, 0.01)
    assert exact == pytest.approx(bs.price(SPOT, 95.0, RATE, VOL, EXPIRY, kind, 0.01))


def test_control_variate_cuts_the_error_without_moving_the_estimate():
    args = (SPOT, STRIKE, RATE, VOL, EXPIRY, 50, 40_000)
    plain, plain_error = monte_carlo.arithmetic_asian(*args, control_variate=False, seed=5)
    controlled, controlled_error = monte_carlo.arithmetic_asian(*args, control_variate=True, seed=5)

    assert controlled_error < plain_error / 10
    assert abs(plain - controlled) < 3 * plain_error


def test_arithmetic_average_call_is_worth_more_than_the_geometric_one():
    # The arithmetic mean is never below the geometric mean, so neither is the call payoff.
    arithmetic, error = monte_carlo.arithmetic_asian(SPOT, STRIKE, RATE, VOL, EXPIRY, 50, 40_000, seed=5)
    geometric = monte_carlo.geometric_asian_exact(SPOT, STRIKE, RATE, VOL, EXPIRY, 50)
    assert arithmetic - geometric > 5 * error


def test_averaging_makes_an_asian_call_cheaper_than_the_european_one():
    asian, _ = monte_carlo.arithmetic_asian(SPOT, STRIKE, RATE, VOL, EXPIRY, 50, 40_000, seed=5)
    assert asian < bs.price(SPOT, STRIKE, RATE, VOL, EXPIRY, "call")
