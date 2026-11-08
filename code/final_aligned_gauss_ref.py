"""Aligned-variant Gaussian-reference calibration of the global local-synergy share (same procedure as C5).

Variants from routeC_async_check.variants: A baseline (check), B Asia t+1, C Americas t-1, D non-overlapping 2-day.
Per variant: own GC normal scores and R; empirical share(omega_t < 0); reference = 999 iid draws of (T_v x 20)
from N(0, R_v) -> normal-score transform -> R re-estimated -> local omega -> share. Seed 20261108 + variant offset.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from entropy_estimators import correlation_matrix, gaussian_copula_transform  # noqa: E402
from routeC_async_check import variants  # noqa: E402
from routeC_local_oinfo import gc, local_omega  # noqa: E402

B = 999


def main():
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    xj = h1.apply_tie_safe_jitter(df[mk].to_numpy(float), seed=h1.TIE_JITTER_SEED)
    res = {}
    for off, (nm, (X, _)) in enumerate(variants(df, mk, xj).items()):
        z, R = gc(X)
        emp = float((local_omega(z, R) < 0).mean())
        rng = np.random.default_rng(20261108 + 10 * (off + 1))
        L = np.linalg.cholesky(R)
        T = len(X)
        ref = []
        for _ in range(B):
            zg = gaussian_copula_transform(rng.standard_normal((T, 20)) @ L.T)
            ref.append(float((local_omega(zg, correlation_matrix(zg)) < 0).mean()))
        ref = np.array(ref)
        lo, hi = np.percentile(ref, [2.5, 97.5])
        iu = np.triu_indices(20, 1)
        res[nm] = {"T": T, "mean_rho": round(float(R[iu].mean()), 3), "empirical": round(emp, 4),
                   "ref_mean": round(float(ref.mean()), 4), "ref_95": [round(float(lo), 4), round(float(hi), 4)],
                   "ratio": round(emp / float(ref.mean()), 3), "above_band": bool(emp > hi),
                   "share_ref_ge_emp": round(float((ref >= emp).mean()), 4)}
        print(nm, res[nm], flush=True)
    json.dump(res, open(os.path.join(h1.ROOT, "raw_audit", "final_aligned_gauss_ref.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
