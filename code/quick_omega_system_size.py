"""System-level O-information across system size k = 3..20 (descriptive), with state and stability checks.

Fixed before running (2026-11-05):
- Universe: the 20 columns of processed/returns_daily_20markets.csv (verified = Layer-1 universe).
- Estimator for k = 3..20: Gaussian-copula O-information (entropy_estimators.o_information on the GC
  correlation matrix of tie-safe-jittered returns, as in the Layer-1 pipeline). kNN is NOT used above d = 4
  (Layer-1 dimensionality validation: material bias / sign reversal at d >= 5).
- Empirical (kNN, k=5) vs Gaussian decomposition only at k = 3, 4, full sample, from the existing files
  processed/triplet_oinfo_20markets_tiesafe.csv and processed/incremental_oinfo_3to4_20markets_tiesafe.csv.
- Subsets: exhaustive when C(20,k) <= 5000 (k = 3, 4, 17..20); otherwise a fixed random sample of 5000 distinct
  subsets (seed 20261105, identical subset list reused for every state / half / frequency).
- States: (i) G-based as specified (G = EW mean return; normal q30<G<q70; down G<=q20);
  (ii) exogenous: previous-day CBOE VIX close in its bottom vs top full-sample tercile.
- Benchmarks per subset: mean pairwise GC correlation rho(S); equicorrelated-Gaussian O-information
  Omega_eq(k, rho(S)); OLS Omega ~ rho + k (+ rho^2, rho*k). Normalizations Omega/k, Omega/C(k,2) (diagnostic).
- Stability: first vs second half; daily vs weekly (W-FRI sums).
"""
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from entropy_estimators import correlation_matrix, gaussian_copula_transform, o_information  # noqa: E402

ROOT = h1.ROOT
OUT = os.path.join(ROOT, "raw_audit")
KS = list(range(3, 21))
NS, SEED = 5000, 20261105


def subsets():
    rng = np.random.default_rng(SEED)
    out = {}
    for k in KS:
        if math.comb(20, k) <= NS:
            out[k] = [tuple(c) for c in itertools.combinations(range(20), k)]
        else:
            s = set()
            while len(s) < NS:
                s.add(tuple(sorted(rng.choice(20, k, replace=False).tolist())))
            out[k] = sorted(s)
    return out


def omega_eq(k, r):
    """O-information of a k-dim equicorrelated Gaussian with correlation r."""
    ld = lambda m: (m - 1) * math.log(1 - r) + math.log(1 + (m - 1) * r)
    T = -0.5 * ld(k)
    D = 0.5 * (k * ld(k - 1) - (k - 1) * ld(k))  # fixed 2026-11-05: sign was flipped (returned T + D)
    return T - D


def curve(x, SUB):
    R = correlation_matrix(gaussian_copula_transform(x))
    rows = []
    for k, ss in SUB.items():
        for s in ss:
            sub = R[np.ix_(s, s)]
            rho = float(sub[np.triu_indices(k, 1)].mean())
            rows.append((k, o_information(R, s)[0], rho, omega_eq(k, rho)))
    d = pd.DataFrame(rows, columns=["k", "omega", "rho", "omega_eq"])
    summ = d.groupby("k").agg(mean=("omega", "mean"), median=("omega", "median"), sd=("omega", "std"),
                              p5=("omega", lambda v: v.quantile(.05)), p95=("omega", lambda v: v.quantile(.95)),
                              frac_pos=("omega", lambda v: (v > 0).mean()), frac_neg=("omega", lambda v: (v < 0).mean()),
                              mean_rho=("rho", "mean"), mean_omega_eq=("omega_eq", "mean"), n_subsets=("omega", "size"))
    summ["mean_over_k"] = summ["mean"] / summ.index
    summ["mean_over_pairs"] = summ["mean"] / [math.comb(k, 2) for k in summ.index]
    return d, summ


def fits(summ):
    k, y = summ.index.to_numpy(float), summ["mean"].to_numpy()
    out = {}
    for nm, X in (("linear", np.column_stack([np.ones_like(k), k])), ("quadratic", np.column_stack([np.ones_like(k), k, k ** 2]))):
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        out[nm] = {"coef": [round(float(v), 5) for v in b], "R2": round(float(1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2)), 4)}
    m = y > 0
    b = np.polyfit(np.log(k[m]), np.log(y[m]), 1)
    out["power"] = {"exponent": round(float(b[0]), 3), "R2": round(float(np.corrcoef(np.log(k[m]), np.log(y[m]))[0, 1] ** 2), 4)}
    return out


def main():
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    xj = h1.apply_tie_safe_jitter(df[mk].to_numpy(float), seed=h1.TIE_JITTER_SEED)
    SUB = subsets()
    res = {"universe": mk, "n_subsets": {k: len(v) for k, v in SUB.items()}}
    d_full, s_full = curve(xj, SUB)
    s_full.round(5).to_csv(os.path.join(OUT, "omega_system_size_full.csv"))
    res["fits_full"] = fits(s_full)
    # regression benchmark
    X = np.column_stack([np.ones(len(d_full)), d_full.rho, d_full.k, d_full.rho ** 2, d_full.rho * d_full.k])
    for nm, cols in (("rho_only", [0, 1]), ("rho_k", [0, 1, 2]), ("rho_k_nonlin", [0, 1, 2, 3, 4])):
        b = np.linalg.lstsq(X[:, cols], d_full.omega, rcond=None)[0]
        e = d_full.omega - X[:, cols] @ b
        res.setdefault("regression", {})[nm] = {"coef": [round(float(v), 4) for v in b], "R2": round(float(1 - e.var() / d_full.omega.var()), 4)}
    dev = d_full.omega - d_full.omega_eq
    res["vs_equicorrelated"] = {"R2_omega_on_omega_eq": round(float(np.corrcoef(d_full.omega, d_full.omega_eq)[0, 1] ** 2), 4),
                                "mean_dev_by_k": dev.groupby(d_full.k).mean().round(4).to_dict(),
                                "mean_rel_dev_by_k": (dev / d_full.omega_eq).groupby(d_full.k).mean().round(4).to_dict()}
    # empirical (kNN) vs GC at k = 3, 4 (existing files)
    tri = pd.read_csv(os.path.join(ROOT, "processed", "triplet_oinfo_20markets_tiesafe.csv"), encoding="utf-8-sig")
    inc = pd.read_csv(os.path.join(ROOT, "processed", "incremental_oinfo_3to4_20markets_tiesafe.csv"), encoding="utf-8-sig")
    q = inc.drop_duplicates(subset=["omega_quadruplet_knn"])
    kcol = [c for c in tri.columns if "knn" in c.lower() and "k5" in c.lower()] or [c for c in tri.columns if "knn" in c.lower()]
    gcol = [c for c in tri.columns if "gc" in c.lower() or "gauss" in c.lower()]
    dec = {}
    for k, emp, gc in ((3, tri[kcol[0]], tri[gcol[0]]), (4, inc.omega_quadruplet_knn, inc.omega_quadruplet_gc)):
        dec[k] = {"emp_mean": round(float(emp.mean()), 4), "gauss_mean": round(float(gc.mean()), 4),
                  "BG_mean": round(float((emp - gc).mean()), 4),
                  "BG_over_emp_mean_ratio": round(float((emp - gc).mean() / emp.mean()), 4),
                  "columns_used": [kcol[0], gcol[0]] if k == 3 else ["omega_quadruplet_knn", "omega_quadruplet_gc"]}
    res["emp_vs_gauss_k34"] = dec
    # states
    G = xj.mean(1)
    qg = np.quantile(G, [0.2, 0.3, 0.7])
    vix = pd.read_csv(os.path.join(ROOT, "data_external", "VIX_History_CBOE.csv"), parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    vlag = vix.shift(1).reindex(df.Date).ffill().to_numpy()
    vq = np.nanquantile(vlag, [1 / 3, 2 / 3])
    states = {"G_normal": (G > qg[1]) & (G < qg[2]), "G_down": G <= qg[0],
              "VIX_low": vlag <= vq[0], "VIX_high": vlag >= vq[1]}
    st = {}
    for nm, m in states.items():
        _, s = curve(xj[m], SUB)
        s.round(5).to_csv(os.path.join(OUT, f"omega_system_size_{nm}.csv"))
        st[nm] = {"n_obs": int(m.sum()), "mean_by_k": s["mean"].round(4).to_dict(), "mean_rho_k3": round(float(s.loc[3, "mean_rho"]), 3),
                  "fits": fits(s)}
    res["states"] = st
    # stability
    h = len(xj) // 2
    _, a = curve(xj[:h], SUB); _, b = curve(xj[h:], SUB)
    wk = df.set_index("Date")[mk].groupby(pd.Grouper(freq="W-FRI")).sum()
    wk = wk[df.set_index("Date")[mk].groupby(pd.Grouper(freq="W-FRI")).size() > 0]
    _, w = curve(h1.apply_tie_safe_jitter(wk.to_numpy(float), seed=h1.TIE_JITTER_SEED), SUB)
    shape = lambda s: (s["mean"] / s.loc[20, "mean"]).to_numpy()
    res["stability"] = {
        "H1_vs_H2": {"corr_mean_curve": round(float(np.corrcoef(a["mean"], b["mean"])[0, 1]), 4),
                     "max_abs_diff_normalized_shape": round(float(np.abs(shape(a) - shape(b)).max()), 4),
                     "power_exp": [fits(a)["power"]["exponent"], fits(b)["power"]["exponent"]],
                     "mean_k3_k10_k20_H1": a["mean"].loc[[3, 10, 20]].round(3).tolist(), "mean_k3_k10_k20_H2": b["mean"].loc[[3, 10, 20]].round(3).tolist()},
        "daily_vs_weekly": {"corr_mean_curve": round(float(np.corrcoef(s_full["mean"], w["mean"])[0, 1]), 4),
                            "max_abs_diff_normalized_shape": round(float(np.abs(shape(s_full) - shape(w)).max()), 4),
                            "power_exp": [res["fits_full"]["power"]["exponent"], fits(w)["power"]["exponent"]],
                            "mean_k3_k10_k20_weekly": w["mean"].loc[[3, 10, 20]].round(3).tolist()}}
    json.dump(res, open(os.path.join(OUT, "omega_system_size_results.json"), "w"), indent=1, default=str)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].fill_between(s_full.index, s_full.p5, s_full.p95, alpha=0.2, label="full 5-95%")
        ax[0].plot(s_full.index, s_full["mean"], "k-", label="full mean")
        for nm, sty in (("G_normal", "g--"), ("G_down", "r--"), ("VIX_low", "g:"), ("VIX_high", "r:")):
            ax[0].plot(KS, list(st[nm]["mean_by_k"].values()), sty, label=nm)
        ax[0].set_xlabel("system size k"); ax[0].set_ylabel("mean Omega_GC (nats)"); ax[0].legend(fontsize=7)
        ax[1].plot(s_full.index, s_full["mean"], "k-", label="daily full")
        ax[1].plot(a.index, a["mean"], "b--", label="1st half"); ax[1].plot(b.index, b["mean"], "c--", label="2nd half")
        ax[1].plot(w.index, w["mean"], "m:", label="weekly"); ax[1].set_xlabel("k"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "omega_system_size.png"), dpi=130)
    except Exception as exc:  # plotting is optional
        res["plot_error"] = str(exc)
    print(s_full.round(4).to_string())
    print(json.dumps({k: v for k, v in res.items() if k not in ("universe",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
