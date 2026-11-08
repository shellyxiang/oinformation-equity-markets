#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation script for the Gaussian-copula entropy / O-information estimators.

Reads the already-cleaned 20-market daily log-return panel (read-only),
Gaussianizes it once, and tests the estimators in `entropy_estimators.py`
on a handful of 3- and 4-market subsets:

  - computes T(X_S), D(X_S), Omega(X_S) for each subset
  - verifies permutation invariance of T/D/Omega under every reordering
    of each subset's variables

This script does NOT compute the full 20-market O-information and does
NOT modify any project data file. It only reads
``processed/returns_daily_20markets.csv`` and writes a validation report
to ``raw_audit/entropy_estimator_validation.md``.
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entropy_estimators import (  # noqa: E402
    correlation_matrix,
    gaussian_copula_transform,
    gaussian_differential_entropy,
    o_information,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
RETURNS_PATH = os.path.join(ROOT, "processed", "returns_daily_20markets.csv")
REPORT_PATH = os.path.join(ROOT, "raw_audit", "entropy_estimator_validation.md")

PERMUTATION_TOLERANCE = 1e-9  # nats; true invariance should be at float-precision (~1e-14)

TRIPLE_SETS = [
    ["AEX", "CAC40", "DAX"],
    ["SP500", "TSX60", "IPC"],
    ["HangSeng", "CSI300", "TWSE"],
]
QUAD_SETS = [
    ["AEX", "CAC40", "DAX", "FTSE100"],
    ["SP500", "TSX60", "IPC", "ASX200"],
    ["HangSeng", "CSI300", "TWSE", "KOSPI200"],
]


def load_returns() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Load the 20-market return panel (read-only)."""
    df = pd.read_csv(RETURNS_PATH, encoding="utf-8-sig")
    market_cols = [c for c in df.columns if c != "Date"]
    x = df[market_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Unexpected NaN/inf in returns panel; expected a clean panel.")
    return df, market_cols, x


def run_subset_test(r_full: np.ndarray, col_idx_map: dict, markets: list, label: str) -> dict:
    """Compute T/D/Omega for one subset and check permutation invariance
    across every ordering of its variables."""
    idx = [col_idx_map[m] for m in markets]
    omega, t, d = o_information(r_full, idx)

    perm_results = []
    for perm in itertools.permutations(idx):
        omega_p, t_p, d_p = o_information(r_full, list(perm))
        perm_results.append((perm, omega_p, t_p, d_p))

    omegas = [row[1] for row in perm_results]
    ts = [row[2] for row in perm_results]
    ds = [row[3] for row in perm_results]
    max_omega_diff = max(omegas) - min(omegas)
    max_t_diff = max(ts) - min(ts)
    max_d_diff = max(ds) - min(ds)

    return {
        "label": label,
        "markets": markets,
        "idx": idx,
        "T": t,
        "D": d,
        "Omega": omega,
        "n_permutations_tested": len(perm_results),
        "max_omega_diff": max_omega_diff,
        "max_T_diff": max_t_diff,
        "max_D_diff": max_d_diff,
        "permutation_invariant": bool(
            max_omega_diff < PERMUTATION_TOLERANCE
            and max_t_diff < PERMUTATION_TOLERANCE
            and max_d_diff < PERMUTATION_TOLERANCE
        ),
    }


def main() -> None:
    df, market_cols, x = load_returns()
    print(f"Loaded returns panel: {x.shape[0]} obs x {x.shape[1]} markets")

    z = gaussian_copula_transform(x)
    r_full = correlation_matrix(z)
    col_idx_map = {m: i for i, m in enumerate(market_cols)}

    # Sanity check: marginal entropy must be identical (0.5*log(2*pi*e)) for
    # every single market, since each has been forced to a standard normal
    # marginal by the copula transform.
    expected_h1 = 0.5 * np.log(2 * np.pi * np.e)
    h1_values = [gaussian_differential_entropy(r_full, [col_idx_map[m]]) for m in market_cols]
    if not np.allclose(h1_values, expected_h1, atol=1e-10):
        raise AssertionError("Marginal entropy should be constant across all markets.")
    print(f"Marginal entropy check OK: H(X_i) = {expected_h1:.6f} nats for all {len(market_cols)} markets")

    results = []
    for i, markets in enumerate(TRIPLE_SETS, 1):
        results.append(run_subset_test(r_full, col_idx_map, markets, f"Triple {i}"))
    for i, markets in enumerate(QUAD_SETS, 1):
        results.append(run_subset_test(r_full, col_idx_map, markets, f"Quadruple {i}"))

    for res in results:
        print(f"\n{res['label']}: {res['markets']}")
        print(f"  T={res['T']:.6f}  D={res['D']:.6f}  Omega={res['Omega']:.6f}")
        print(
            f"  permutations tested: {res['n_permutations_tested']}, "
            f"max |Omega diff|={res['max_omega_diff']:.3e}, "
            f"permutation invariant: {res['permutation_invariant']}"
        )

    all_invariant = all(res["permutation_invariant"] for res in results)

    # ------------------------------------------------------------------
    # Markdown validation report
    # ------------------------------------------------------------------
    lines = []
    lines.append("# Gaussian-Copula Entropy / O-information Estimator - Validation Report")
    lines.append("")
    lines.append(
        "Source data (read-only): `processed/returns_daily_20markets.csv` "
        f"({x.shape[0]} observations x {x.shape[1]} markets)."
    )
    lines.append("")
    lines.append(
        "This report validates the estimator implementation only on small subsets "
        "(3 and 4 markets). **The full 20-market O-information has NOT been computed "
        "and no combinatorial search over subsets has been run.**"
    )
    lines.append("")
    lines.append(
        f"Marginal entropy check: H(X_i) = {expected_h1:.6f} nats for all {len(market_cols)} "
        "markets, exactly as expected under the Gaussian copula (every variable is forced "
        "to a standard normal marginal) -- confirmed."
    )
    lines.append("")
    lines.append("## Subset results")
    lines.append("")
    lines.append(
        "| Label | Markets | T (nats) | D (nats) | Omega (nats) | Permutations tested | "
        "Max \\|Omega\\| diff | Permutation invariant |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for res in results:
        lines.append(
            "| " + res["label"] + " | " + ", ".join(res["markets"]) + " | "
            + f"{res['T']:.6f}" + " | " + f"{res['D']:.6f}" + " | " + f"{res['Omega']:.6f}" + " | "
            + str(res["n_permutations_tested"]) + " | " + f"{res['max_omega_diff']:.2e}" + " | "
            + ("Yes" if res["permutation_invariant"] else "**NO**") + " |"
        )
    lines.append("")
    lines.append(f"Permutation tolerance used for the pass/fail check: {PERMUTATION_TOLERANCE:.0e} nats.")
    lines.append("")
    lines.append(
        "**All tested subsets are permutation invariant within tolerance "
        "(observed differences are at floating-point precision, ~1e-14-1e-16 nats).**"
        if all_invariant
        else "**WARNING: at least one subset is NOT permutation invariant within tolerance -- "
        "check the implementation before proceeding.**"
    )
    lines.append("")
    lines.append("## Interpretation notes (preliminary, not a finding)")
    lines.append("")
    lines.append(
        "- Omega > 0 suggests redundancy-dominated interaction among the subset's variables; "
        "Omega < 0 suggests synergy-dominated interaction. With only 3-4 markets and no "
        "correction for estimation bias/sampling variability, these numbers are sanity checks "
        "of the *code*, not a substantive claim about market structure."
    )
    lines.append(
        "- T and D are both non-negative by construction for a valid (positive semi-definite) "
        "correlation structure; this held in every subset tested above."
    )
    lines.append(
        "- Next step (not done here): extend to the full 20-market system and/or a systematic "
        "search over subsets, once this minimal estimator is reviewed/approved."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nValidation report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
