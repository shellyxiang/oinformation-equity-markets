#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate H1 test: real 20-market triadic O-information vs. a matched
Gaussian-copula null, using the VALIDATED rank-reordered (Iman-Conover-
style) null construction.

This is the third null-construction attempt. The first two failed for
diagnosed reasons (see raw_audit/real_triplet_matched_null_H1_summary.md
-- never completed, and raw_audit/gaussianized_null_smoketest.md):
  1. Empirical-inverse-CDF reconstruction (linear interpolation over
     3,792 order statistics) distorted local kNN geometry.
  2. Rank-Gaussianizing the kNN input created a shared finite value
     grid across columns, inflating Omega by a near-constant
     +0.8-0.9 nat artifact.
This construction avoids both: every null column is an EXACT
permutation of that market's own observed (tie-safe) return values,
reordered only to match the rank pattern of an independently simulated
Gaussian-copula-consistent latent score. No interpolation, no
smoothing, no Gaussianization of the kNN input. Validated in
raw_audit/rank_reordered_null_smoketest.md (GO on all 4 criteria,
AEX-CAC40-DAX: null mean 0.7068 vs. observed 0.7386 vs. GC benchmark
0.6881; synthetic sanity check also passed).

H0: The observed triadic O-information landscape is compatible with the
    matched Gaussian-copula dependence structure.
HA: The observed landscape contains systematic deviations beyond that
    matched Gaussian-copula structure.

SCOPE: single aggregate test (one global p-value). Does NOT test
individual triplets, does NOT run quadruplet/incremental-Delta
inference, does NOT run k=3/k=10 robustness, does NOT run any further
validation. Primary specification only: k=5, alpha=1e-5 tie-safe
jitter, fixed documented seed, N=3,792, all 20 markets, all 1,140
triplets.

Reads (READ-ONLY): processed/returns_daily_20markets.csv,
processed/triplet_oinfo_20markets_tiesafe.csv (observed Omega_t^obs,
reused directly, not recomputed). Does NOT modify either file nor any
raw source CSV, nor save a jittered/null panel to disk.

Reuses entropy_estimators.py and knn_entropy_estimators.py exactly as
implemented (not modified).

Outputs (all existing files preserved; this writes NEW, clearly
labeled files only)
-----------------------------------------------------------------------
  raw_audit/real_triplet_matched_null_H1_rankreordered.csv
      (per-triplet: Omega_obs, null mean/SD, Z_obs, D_ijk)
  raw_audit/real_triplet_matched_null_H1_rankreordered_summary.md
  raw_audit/real_triplet_matched_null_H1_rankreordered_null_replicates.csv
      (full 59 x 1,140 null Omega grid, for audit)
"""

from __future__ import annotations

import hashlib
import itertools
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entropy_estimators import (  # noqa: E402
    correlation_matrix,
    gaussian_copula_transform,
)
from knn_entropy_estimators import knn_entropy_multi_k  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
RETURNS_PATH = os.path.join(ROOT, "processed", "returns_daily_20markets.csv")
TRIPLET_PATH = os.path.join(ROOT, "processed", "triplet_oinfo_20markets_tiesafe.csv")
OUT_CSV = os.path.join(ROOT, "raw_audit", "real_triplet_matched_null_H1_rankreordered.csv")
OUT_MD = os.path.join(ROOT, "raw_audit", "real_triplet_matched_null_H1_rankreordered_summary.md")
OUT_NULL_CSV = os.path.join(ROOT, "raw_audit", "real_triplet_matched_null_H1_rankreordered_null_replicates.csv")

PRIMARY_K = 5
ALPHA = 1e-5
TIE_JITTER_SEED = 20251210  # identical to the validated tie-safe triplet run
B_NULL = 59
ALPHA_SIG = 0.05
NULL_MASTER_SEED = 20260401  # new, documented seed for this full-scale run
KNN_WORKERS = -1


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def h_single(x_sub: np.ndarray) -> float:
    return knn_entropy_multi_k(x_sub, [PRIMARY_K], workers=KNN_WORKERS)[PRIMARY_K]


def compute_all_triplet_omegas_k5(x: np.ndarray, triplets_idx: List[Tuple[int, int, int]],
                                    pairs_idx: List[Tuple[int, int]], M: int) -> np.ndarray:
    h1 = {i: h_single(x[:, [i]]) for i in range(M)}
    h2 = {p: h_single(x[:, list(p)]) for p in pairs_idx}
    out = np.empty(len(triplets_idx))
    for idx, (i, j, k) in enumerate(triplets_idx):
        h3 = h_single(x[:, [i, j, k]])
        out[idx] = h1[i] + h1[j] + h1[k] - (h2[(i, j)] + h2[(i, k)] + h2[(j, k)]) + h3
    return out


def apply_tie_safe_jitter(x: np.ndarray, seed: int) -> np.ndarray:
    sds = x.std(axis=0, ddof=1)
    rng = np.random.default_rng(seed)
    jitter = rng.standard_normal(x.shape) * (ALPHA * sds)[None, :]
    return x + jitter


def rank_reorder_null(observed: np.ndarray, r_gc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Iman-Conover-style rank-reordering null: every column is an exact
    permutation of ``observed``'s own column values, reordered to match
    the rank pattern of an independently drawn latent Gaussian score
    with correlation ``r_gc``."""
    n, m = observed.shape
    try:
        L = np.linalg.cholesky(r_gc)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(r_gc + 1e-10 * np.eye(m))
    z_star = rng.standard_normal((n, m)) @ L.T
    null = np.empty_like(observed)
    for col in range(m):
        sorted_vals = np.sort(observed[:, col])
        order = np.argsort(z_star[:, col])
        null[order, col] = sorted_vals
    return null


def fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def main() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load (read-only)
    # ------------------------------------------------------------------
    hash_returns_before = file_sha256(RETURNS_PATH)
    hash_triplet_before = file_sha256(TRIPLET_PATH)

    df = pd.read_csv(RETURNS_PATH, encoding="utf-8-sig")
    market_names = [c for c in df.columns if c != "Date"]
    assert len(market_names) == 20
    x = df[market_names].to_numpy(dtype=float)
    N, M = x.shape
    assert N == 3792 and M == 20

    triplet_df = pd.read_csv(TRIPLET_PATH, encoding="utf-8-sig")
    assert len(triplet_df) == 1140

    triplets_idx = list(itertools.combinations(range(M), 3))
    pairs_idx = list(itertools.combinations(range(M), 2))

    # Observed Omega_t^obs: reuse the already-validated tie-safe file directly
    omega_obs = np.empty(len(triplets_idx))
    d_ijk = np.empty(len(triplets_idx))
    triplet_lookup = {frozenset([r["market_1"], r["market_2"], r["market_3"]]): (r["omega_knn_k5"], r["D_knn5_minus_gc"])
                      for _, r in triplet_df.iterrows()}
    for idx, (i, j, k) in enumerate(triplets_idx):
        key = frozenset([market_names[i], market_names[j], market_names[k]])
        omega_obs[idx], d_ijk[idx] = triplet_lookup[key]
    print(f"Loaded observed tie-safe Omega_t^obs for all {len(triplets_idx)} triplets (reused, not recomputed).")

    # Tie-safe jitter applied to the OBSERVED panel -- this is the exact
    # data whose values get permuted into each null replicate
    x_jittered = apply_tie_safe_jitter(x, seed=TIE_JITTER_SEED)
    print(f"Applied validated tie-safe jitter (alpha={ALPHA:.0e}, seed={TIE_JITTER_SEED}) to the "
          "observed panel (the exact values permuted into every null replicate).")

    # ------------------------------------------------------------------
    # Fit full 20x20 latent Gaussian-copula correlation matrix
    # ------------------------------------------------------------------
    z_gauss = gaussian_copula_transform(x_jittered)
    r_gc = correlation_matrix(z_gauss)
    eigvals = np.linalg.eigvalsh(r_gc)
    min_eig, max_eig = float(eigvals.min()), float(eigvals.max())
    cond_number = max_eig / min_eig
    is_pd = bool(np.all(eigvals > 0))
    print(f"Fitted R^GC_hat (20x20): min eig={min_eig:.4f}, max eig={max_eig:.4f}, "
          f"condition number={cond_number:.2f}, PD={is_pd}")
    assert is_pd

    # ------------------------------------------------------------------
    # Generate B=59 rank-reordered null panels, compute 1,140-triplet Omega each
    # ------------------------------------------------------------------
    print(f"\nGenerating {B_NULL} rank-reordered null panels (N={N}, M={M}) and computing their "
          f"exhaustive 1,140-triplet Omega (k={PRIMARY_K})...")
    null_omega = np.empty((B_NULL, len(triplets_idx)))
    marginal_check_all_ok = True
    sorted_obs_cols = [np.sort(x_jittered[:, c]) for c in range(M)]
    for b in range(B_NULL):
        rng = np.random.default_rng(NULL_MASTER_SEED + b)
        null_panel = rank_reorder_null(x_jittered, r_gc, rng)
        for c in range(M):
            if not np.array_equal(np.sort(null_panel[:, c]), sorted_obs_cols[c]):
                marginal_check_all_ok = False
        null_omega[b, :] = compute_all_triplet_omegas_k5(null_panel, triplets_idx, pairs_idx, M)
        if (b + 1) % 10 == 0 or b == 0:
            elapsed = time.time() - t0
            eta = elapsed / (b + 1) * (B_NULL - b - 1)
            print(f"  null replicate {b+1}/{B_NULL} done ({elapsed:.1f}s elapsed, ETA {eta:.1f}s)")
    print(f"All {B_NULL} null replicates complete at {time.time()-t0:.1f}s")
    print(f"Marginal preservation check (every null column an exact permutation of the observed "
          f"values, all {B_NULL} replicates x {M} markets): {marginal_check_all_ok}")
    assert marginal_check_all_ok

    # ------------------------------------------------------------------
    # Sanity: no NA/Inf, no zero-distance pathology (sample check)
    # ------------------------------------------------------------------
    assert np.all(np.isfinite(null_omega)) and np.all(np.isfinite(omega_obs))
    chk_rng = np.random.default_rng(777)
    chk_panel = rank_reorder_null(x_jittered, r_gc, chk_rng)
    min_dist = np.inf
    for col in range(M):
        tree = cKDTree(chk_panel[:, [col]])
        dist, _ = tree.query(chk_panel[:, [col]], k=PRIMARY_K + 1, p=2, workers=KNN_WORKERS)
        min_dist = min(min_dist, float(dist[:, PRIMARY_K].min()))
    no_zero_distance = min_dist > np.finfo(float).tiny * 1e6
    print(f"Sanity check (one sample null panel): no zero-distance kNN pathology: {no_zero_distance} "
          f"(min k=5-NN distance: {min_dist:.3e})")
    assert no_zero_distance

    # ------------------------------------------------------------------
    # Triplet-specific standardization, LOO inner reference, primary stat
    # ------------------------------------------------------------------
    mu_t = null_omega.mean(axis=0)
    sigma_t = np.maximum(null_omega.std(axis=0, ddof=1), 1e-12)
    z_obs = (omega_obs - mu_t) / sigma_t
    t_obs = float(np.mean(np.abs(z_obs)))

    total = null_omega.sum(axis=0)
    total_sq = (null_omega ** 2).sum(axis=0)
    mu_loo = (total[None, :] - null_omega) / (B_NULL - 1)
    sumsq_remaining = total_sq[None, :] - null_omega ** 2
    var_loo = (sumsq_remaining - (B_NULL - 1) * mu_loo ** 2) / (B_NULL - 2)
    sigma_loo = np.sqrt(np.maximum(var_loo, 1e-12))
    z_null = (null_omega - mu_loo) / sigma_loo
    t_null = np.mean(np.abs(z_null), axis=1)

    p_value = (1 + int(np.sum(t_null >= t_obs))) / (B_NULL + 1)
    print(f"\nT_obs = {t_obs:.4f}")
    print(f"Null T distribution: mean={t_null.mean():.4f}, median={np.median(t_null):.4f}, "
          f"min={t_null.min():.4f}, max={t_null.max():.4f}")
    print(f"Aggregate H1 p-value = {p_value:.4f}  (alpha={ALPHA_SIG})")

    # ------------------------------------------------------------------
    # Confirm no data files were modified
    # ------------------------------------------------------------------
    hash_returns_after = file_sha256(RETURNS_PATH)
    hash_triplet_after = file_sha256(TRIPLET_PATH)
    files_unchanged = (hash_returns_before == hash_returns_after) and (hash_triplet_before == hash_triplet_after)
    print(f"Source files unchanged: {files_unchanged}")
    assert files_unchanged

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    main_rows = []
    for idx, (i, j, k) in enumerate(triplets_idx):
        main_rows.append({
            "market_1": market_names[i], "market_2": market_names[j], "market_3": market_names[k],
            "omega_obs": omega_obs[idx], "mu_t_null": mu_t[idx], "sigma_t_null": sigma_t[idx],
            "Z_t_obs": z_obs[idx], "D_ijk": d_ijk[idx],
        })
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nWritten: {OUT_CSV}  ({len(main_df)} rows)")

    null_rows = []
    for b in range(B_NULL):
        for idx, (i, j, k) in enumerate(triplets_idx):
            null_rows.append({
                "replicate": b, "market_1": market_names[i], "market_2": market_names[j],
                "market_3": market_names[k], "omega_null": null_omega[b, idx],
            })
    pd.DataFrame(null_rows).to_csv(OUT_NULL_CSV, index=False, encoding="utf-8-sig")
    print(f"Written: {OUT_NULL_CSV}  ({len(null_rows)} rows)")

    write_report(main_df, z_obs, t_obs, t_null, p_value, min_eig, max_eig, cond_number, is_pd,
                 files_unchanged, no_zero_distance, min_dist, marginal_check_all_ok)
    print(f"Written: {OUT_MD}")
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("DONE.")


def write_report(main_df: pd.DataFrame, z_obs: np.ndarray, t_obs: float, t_null: np.ndarray,
                  p_value: float, min_eig: float, max_eig: float, cond_number: float, is_pd: bool,
                  files_unchanged: bool, no_zero_distance: bool, min_dist: float,
                  marginal_check_all_ok: bool) -> None:
    L: List[str] = []
    add = L.append

    add("# Aggregate H1 Test (Rank-Reordered Null): Real 20-Market Triadic O-information")
    add("")
    add("**Null construction: validated rank-reordered (Iman-Conover-style) null** -- the third "
        "attempt, after two prior null constructions failed for diagnosed reasons (empirical-"
        "inverse-CDF reconstruction distorted local kNN geometry; rank-Gaussianizing the kNN input "
        "created a shared-grid artifact). This construction never interpolates, smooths, or "
        "transforms the observed return values -- every null column is an exact permutation of that "
        "market's own tie-safe observed values. Validated in "
        "`raw_audit/rank_reordered_null_smoketest.md` (GO on all 4 criteria) before this full-scale "
        "run.")
    add("")
    add("## Hypotheses")
    add("")
    add("- **H0**: The observed triadic O-information landscape is compatible with the matched "
        "Gaussian-copula dependence structure.")
    add("- **HA**: The observed landscape contains systematic deviations beyond that matched "
        "Gaussian-copula structure.")
    add("")
    add("This is a SINGLE aggregate test. No individual triplet is tested or labeled significant. "
        "No quadruplet, incremental-Delta, or k=3/k=10 robustness analysis is included in this step.")
    add("")

    add("## 1. Specification")
    add("")
    add(f"- k={PRIMARY_K} (primary only); tie-safe jitter alpha={ALPHA:.0e}, seed={TIE_JITTER_SEED} "
        "(identical to the validated tie-safe triplet file)")
    add(f"- N=3,792, M=20 markets, all 1,140 triplets")
    add(f"- B={B_NULL} rank-reordered null replicates; master seed={NULL_MASTER_SEED}")
    add(f"- R^GC_hat (20x20): min eigenvalue={fmt(min_eig,4)}, max eigenvalue={fmt(max_eig,4)}, "
        f"condition number={fmt(cond_number,2)}, positive definite={is_pd}")
    add("")

    add("## 2. Construction and preservation checks")
    add("")
    add(f"- Every null column is an exact permutation of the observed tie-safe values (verified for "
        f"all {B_NULL} replicates x 20 markets): **{marginal_check_all_ok}**")
    add(f"- No zero-distance kNN pathology in a sampled null panel: **{no_zero_distance}** "
        f"(min k=5-NN distance: {min_dist:.3e})")
    add(f"- Returns panel and tie-safe triplet file byte-identical before/after: **{files_unchanged}**")
    add("")

    add("## 3. Primary aggregate result")
    add("")
    add(f"**T_obs = {fmt(t_obs,4)}**")
    add("")
    add(f"**Aggregate p-value = {fmt(p_value,4)}** (one-sided, p=(1+#{{b: T_null^(b)>=T_obs}})/60; "
        f"nominal alpha={ALPHA_SIG})")
    add("")
    verdict = "REJECT H0" if p_value <= ALPHA_SIG else "DO NOT REJECT H0"
    add(f"**Verdict: {verdict} at alpha={ALPHA_SIG}.**")
    add("")

    add("## 4. Secondary diagnostics")
    add("")
    add(f"Null T distribution: mean={fmt(t_null.mean())}, median={fmt(np.median(t_null))}, "
        f"SD={fmt(t_null.std(ddof=1))}, min={fmt(t_null.min())}, max={fmt(t_null.max())}, "
        f"P5={fmt(np.percentile(t_null,5))}, P95={fmt(np.percentile(t_null,95))}")
    percentile_rank = float(np.mean(t_null < t_obs)) * 100
    add(f"Percentile rank of T_obs within the null T distribution: **{fmt(percentile_rank,1)}%**")
    add("")
    add("Distribution of Z_t_obs (triplet-specific standardized deviations -- descriptive only, "
        "not individual test statistics):")
    add(f"- Mean={fmt(z_obs.mean())}, median={fmt(np.median(z_obs))}, SD={fmt(z_obs.std(ddof=1))}, "
        f"min={fmt(z_obs.min())}, max={fmt(z_obs.max())}")
    add(f"- Proportion \\|Z_t_obs\\|>1: {fmt(np.mean(np.abs(z_obs)>1),3)}")
    add(f"- Proportion \\|Z_t_obs\\|>2: {fmt(np.mean(np.abs(z_obs)>2),3)}")
    add(f"- Proportion \\|Z_t_obs\\|>3: {fmt(np.mean(np.abs(z_obs)>3),3)}")
    add("")
    r_d_z, p_d_z = pearsonr(main_df["D_ijk"].abs(), np.abs(z_obs))
    add(f"Pearson correlation(\\|D_ijk\\|, \\|Z_t_obs\\|) across all 1,140 triplets (screening "
        f"diagnostic only): **{fmt(r_d_z,4)}** (p={fmt(p_d_z,4)})")
    add("")

    add("## 5. Interpretation")
    add("")
    if p_value > ALPHA_SIG:
        add("**Not significant at alpha=0.05.** Per the established interpretation discipline:")
        add("")
        add("> The observed triadic O-information landscape is broadly compatible with a matched "
            "Gaussian-copula dependence structure at the aggregate level.")
    else:
        add("**Significant at alpha=0.05.**")
        add("")
        add("> The aggregate triadic O-information landscape contains systematic deviations from "
            "the matched Gaussian-copula reference that cannot be explained by the fitted "
            "Gaussian/pairwise dependence structure alone.")
    add("")
    add("**No individual triplet is identified as significant. No further step (quadruplet, "
        "incremental-Delta, k=3/k=10 robustness, or additional validation) was run.**")
    add("")
    add("---")
    add("")
    add(f"Full per-triplet table: `raw_audit/real_triplet_matched_null_H1_rankreordered.csv` "
        f"({len(main_df)} rows). Full null replicate grid: "
        f"`raw_audit/real_triplet_matched_null_H1_rankreordered_null_replicates.csv` "
        f"({B_NULL} x {len(main_df)} rows).")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
