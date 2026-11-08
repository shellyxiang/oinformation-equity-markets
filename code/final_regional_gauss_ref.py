"""Regional Gaussian-reference calibration of the local synergy-day share (same procedure as C5 in final_inference.py).

Per region (EU 9, AS 7, AM 3 markets; REG from routeC_local_oinfo): empirical share(omega_t^r < 0) from the region's own
GC normal scores and correlation matrix; reference = 999 iid draws of T x n_r from N(0, R_r), passed through the same
normal-score transform, R re-estimated, local omega computed, share recorded. Seed 20261108 + region offset.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from entropy_estimators import correlation_matrix, gaussian_copula_transform  # noqa: E402
from routeC_local_oinfo import REG, gc, local_omega  # noqa: E402

B = 999


def main():
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    xj = h1.apply_tie_safe_jitter(df[mk].to_numpy(float), seed=h1.TIE_JITTER_SEED)
    T = len(xj)
    res = {}
    for off, (r, ms) in enumerate(REG.items()):
        z, R = gc(xj[:, [mk.index(m) for m in ms]])
        emp = float((local_omega(z, R) < 0).mean())
        rng = np.random.default_rng(20261108 + off + 1)
        L = np.linalg.cholesky(R)
        ref = []
        for _ in range(B):
            zg = gaussian_copula_transform(rng.standard_normal((T, len(ms))) @ L.T)
            ref.append(float((local_omega(zg, correlation_matrix(zg)) < 0).mean()))
        ref = np.array(ref)
        lo, hi = np.percentile(ref, [2.5, 97.5])
        iu = np.triu_indices(len(ms), 1)
        res[r] = {"n_markets": len(ms), "mean_rho": round(float(R[iu].mean()), 3), "sd_rho": round(float(R[iu].std()), 3),
                  "empirical": round(emp, 4), "ref_mean": round(float(ref.mean()), 4), "ref_95": [round(float(lo), 4), round(float(hi), 4)],
                  "ratio": round(emp / float(ref.mean()), 3), "excess_pp": round(100 * (emp - float(ref.mean())), 2),
                  "above_band": bool(emp > hi), "share_ref_ge_emp": round(float((ref >= emp).mean()), 4)}
        print(r, res[r], flush=True)
    json.dump(res, open(os.path.join(h1.ROOT, "raw_audit", "final_regional_gauss_ref.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
