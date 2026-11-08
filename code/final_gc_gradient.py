"""Full-system Gaussian-copula first-order O-information gradient (manuscript definition, 2026-11-05).

dOmega_j = Omega(X) - Omega(X_{-j}), j = 1..20, GC estimator on tie-safe-jittered returns (paper pipeline).
Diagnostics: ranking; Spearman/Pearson with average pairwise correlation of j and with |PC1 loading| of j;
half-sample stability (normal scores and R re-estimated per half); daily vs weekly (W-FRI sums, re-estimated).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from entropy_estimators import correlation_matrix, gaussian_copula_transform, o_information  # noqa: E402


def grad(x):
    R = correlation_matrix(gaussian_copula_transform(x))
    n = R.shape[0]
    full = o_information(R, list(range(n)))[0]
    g = np.array([full - o_information(R, [i for i in range(n) if i != j])[0] for j in range(n)])
    avg = (R.sum(1) - 1) / (n - 1)
    v = np.linalg.eigh(R)[1][:, -1]
    return g, avg, np.abs(v), full


def cmp(a, b):
    return {"spearman": round(float(spearmanr(a, b)[0]), 3), "pearson": round(float(pearsonr(a, b)[0]), 3)}


def main():
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    xj = h1.apply_tie_safe_jitter(df[mk].to_numpy(float), seed=h1.TIE_JITTER_SEED)
    g, avg, pc1, full = grad(xj)
    h = len(xj) // 2
    g1, *_ = grad(xj[:h])
    g2, *_ = grad(xj[h:])
    wk = df.set_index("Date")[mk].groupby(pd.Grouper(freq="W-FRI")).sum()
    wk = wk[df.set_index("Date")[mk].groupby(pd.Grouper(freq="W-FRI")).size() > 0]
    gw, *_ = grad(h1.apply_tie_safe_jitter(wk.to_numpy(float), seed=h1.TIE_JITTER_SEED))
    rk = lambda v: (-v).argsort().argsort() + 1
    tab = pd.DataFrame({"market": mk, "dOmega": g.round(4), "rank": rk(g), "avg_corr": avg.round(3), "abs_PC1": pc1.round(3),
                        "dOmega_H1": g1.round(4), "rank_H1": rk(g1), "dOmega_H2": g2.round(4), "rank_H2": rk(g2),
                        "dOmega_weekly": gw.round(4), "rank_weekly": rk(gw)}).sort_values("rank")
    res = {"Omega20": round(float(full), 4), "all_positive": bool((g > 0).all()), "range": [round(float(g.min()), 4), round(float(g.max()), 4)],
           "vs_avg_corr": cmp(g, avg), "vs_abs_PC1": cmp(g, pc1), "H1_vs_H2": cmp(g1, g2), "daily_vs_weekly": cmp(g, gw),
           "max_rank_shift_H1_H2": int(np.abs(rk(g1) - rk(g2)).max()), "max_rank_shift_daily_weekly": int(np.abs(rk(g) - rk(gw)).max()),
           "sign_positive_H1_H2_W": [int((g1 > 0).sum()), int((g2 > 0).sum()), int((gw > 0).sum())],
           "n_weekly": int(len(wk))}
    out = os.path.join(h1.ROOT, "raw_audit")
    tab.to_csv(os.path.join(out, "final_gc_gradient_markets.csv"), index=False)
    json.dump(res, open(os.path.join(out, "final_gc_gradient_summary.json"), "w"), indent=1)
    print(tab.to_string(index=False)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
