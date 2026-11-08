# Gaussian-Copula Entropy / O-information Estimator - Validation Report

Source data (read-only): `processed/returns_daily_20markets.csv` (3792 observations x 20 markets).

This report validates the estimator implementation only on small subsets (3 and 4 markets). **The full 20-market O-information has NOT been computed and no combinatorial search over subsets has been run.**

Marginal entropy check: H(X_i) = 1.418939 nats for all 20 markets, exactly as expected under the Gaussian copula (every variable is forced to a standard normal marginal) -- confirmed.

## Subset results

| Label | Markets | T (nats) | D (nats) | Omega (nats) | Permutations tested | Max \|Omega\| diff | Permutation invariant |
|---|---|---|---|---|---|---|---|
| Triple 1 | AEX, CAC40, DAX | 1.801610 | 1.113536 | 0.688074 | 6 | 0.00e+00 | Yes |
| Triple 2 | SP500, TSX60, IPC | 0.657826 | 0.485028 | 0.172798 | 6 | 1.33e-15 | Yes |
| Triple 3 | HangSeng, CSI300, TWSE | 0.373907 | 0.322675 | 0.051232 | 6 | 4.44e-16 | Yes |
| Quadruple 1 | AEX, CAC40, DAX, FTSE100 | 2.532123 | 1.214379 | 1.317743 | 24 | 3.55e-15 | Yes |
| Quadruple 2 | SP500, TSX60, IPC, ASX200 | 0.712460 | 0.501260 | 0.211201 | 24 | 2.66e-15 | Yes |
| Quadruple 3 | HangSeng, CSI300, TWSE, KOSPI200 | 0.749941 | 0.526848 | 0.223093 | 24 | 0.00e+00 | Yes |

Permutation tolerance used for the pass/fail check: 1e-09 nats.

**All tested subsets are permutation invariant within tolerance (observed differences are at floating-point precision, ~1e-14-1e-16 nats).**

## Interpretation notes (preliminary, not a finding)

- Omega > 0 suggests redundancy-dominated interaction among the subset's variables; Omega < 0 suggests synergy-dominated interaction. With only 3-4 markets and no correction for estimation bias/sampling variability, these numbers are sanity checks of the *code*, not a substantive claim about market structure.
- T and D are both non-negative by construction for a valid (positive semi-definite) correlation structure; this held in every subset tested above.
- Next step (not done here): extend to the full 20-market system and/or a systematic search over subsets, once this minimal estimator is reviewed/approved.