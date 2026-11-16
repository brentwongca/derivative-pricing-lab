"""Measure how fast each numerical method approaches the exact price.

Run from the repository root:  python examples/convergence.py
Writes docs/convergence.png and prints the tables used in the README.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pricing import binomial, monte_carlo
from pricing import black_scholes as bs

SPOT, STRIKE, RATE, VOL, EXPIRY = 100.0, 105.0, 0.05, 0.2, 1.0
FIXINGS = 50


def main() -> None:
    exact = bs.price(SPOT, STRIKE, RATE, VOL, EXPIRY, "call")
    print(f"Black-Scholes call: {exact:.6f}\n")

    steps = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    tree_errors = [abs(binomial.price(SPOT, STRIKE, RATE, VOL, EXPIRY, n, "call") - exact) for n in steps]
    print("| Steps | Tree price error |\n|---:|---:|")
    for n, error in zip(steps, tree_errors):
        print(f"| {n} | {error:.5f} |")

    paths = [1_000, 4_000, 16_000, 64_000, 256_000]
    plain = [monte_carlo.european(SPOT, STRIKE, RATE, VOL, EXPIRY, 2 * n, antithetic=False, seed=1)[1] for n in paths]
    paired = [monte_carlo.european(SPOT, STRIKE, RATE, VOL, EXPIRY, n, antithetic=True, seed=1)[1] for n in paths]
    asian_paths = [1_000, 4_000, 16_000, 64_000]
    asian_plain = [
        monte_carlo.arithmetic_asian(SPOT, STRIKE, RATE, VOL, EXPIRY, FIXINGS, n, control_variate=False, seed=1)
        for n in asian_paths
    ]
    asian_controlled = [
        monte_carlo.arithmetic_asian(SPOT, STRIKE, RATE, VOL, EXPIRY, FIXINGS, n, control_variate=True, seed=1)
        for n in asian_paths
    ]

    print("\n| Payoff evaluations | Plain std. error | Antithetic std. error |\n|---:|---:|---:|")
    for n, a, b in zip(paths, plain, paired):
        print(f"| {2 * n:,} | {a:.5f} | {b:.5f} |")
    print("\n| Paths | Asian price, plain | Std. error | Asian price, control variate | Std. error |")
    print("|---:|---:|---:|---:|---:|")
    for n, a, b in zip(asian_paths, asian_plain, asian_controlled):
        print(f"| {n:,} | {a[0]:.4f} | {a[1]:.5f} | {b[0]:.4f} | {b[1]:.5f} |")

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.loglog(steps, tree_errors, "o-", label="CRR tree")
    left.loglog(steps, [tree_errors[0] * steps[0] / n for n in steps], "--", color="grey", label="1 / steps")
    left.set_xlabel("Tree steps")
    left.set_ylabel("Absolute error against Black-Scholes")
    left.set_title("Binomial tree, European call")
    left.legend(frameon=False)

    right.loglog([2 * n for n in paths], plain, "o-", label="European, plain")
    right.loglog([2 * n for n in paths], paired, "o-", label="European, antithetic")
    right.loglog(asian_paths, [e for _, e in asian_plain], "s-", label="Asian, plain")
    right.loglog(asian_paths, [e for _, e in asian_controlled], "s-", label="Asian, control variate")
    right.set_xlabel("Payoff evaluations")
    right.set_ylabel("Standard error")
    right.set_title("Monte Carlo")
    right.legend(frameon=False, fontsize=8)

    for axis in (left, right):
        axis.grid(alpha=0.25, which="both")
    figure.tight_layout()
    output = Path(__file__).resolve().parents[1] / "docs" / "convergence.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=140)
    print(f"\nwrote {output.name}")


if __name__ == "__main__":
    main()
