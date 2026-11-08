"""Final inference for the O-information paper (C1-C6). Pre-specified 2026-11-05; no new estimators.

Fixed design
- Universe, tie-safe jitter, Gaussian-copula (GC) normal scores, O-information definition: as in
  quick_omega_system_size.py / routeA_composition.py / routeC_local_oinfo.py. Subset lists are imported from
  those scripts unchanged (system size: seed 20261105, <=5000 per k; Route A: build_subsets).
- O-information is evaluated in batch with the identity Omega = -logdet(S) - 0.5 * sum_i log (S^-1)_ii
  (equal to T - D for a Gaussian with unit variances); checked against entropy_estimators.o_information.
- Bootstrap: stationary block bootstrap (Politis-Romano), mean block length 20, B = 999, seed 20261106.
  Normal scores z stay fixed; rows are resampled, R is re-estimated in every replicate, state labels travel with
  their days. 95% percentile intervals.
- C1 independence surrogates: independent random circular shift of each market's normal-score column
  (keeps marginals and autocorrelation, destroys cross-dependence), 999 draws, seed 20261107.
  p = (1 + #{null >= obs}) / (1 + B).
- C5 Gaussian reference: T = 3792 iid draws from N(0, R_hat), passed through the same normal-score transform,
  R re-estimated, local omega computed, share(omega_t < 0) recorded; 999 draws, seed 20261108. Calibration only.
- C6 states: previous-day VIX terciles (as in Route C) and crisis windows GFC / COVID (as coded). Differences in
  share(omega_t<0): high - low VIX, GFC - non-crisis, COVID - non-crisis, with Newey-West (20 lags) SE from OLS of
  the indicator on a group dummy (descriptive difference), cross-checked with the bootstrap. Omega(S_k)
  high/low VIX ratio at k = 10 and 20, from the bootstrap.
"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from entropy_estimators import correlation_matrix, gaussian_copula_transform, o_information  # noqa: E402
from quick_omega_system_size import KS, fits, subsets  # noqa: E402
from routeA_composition import build_subsets  # noqa: E402
from routeC_local_oinfo import CRISES, local_omega  # noqa: E402

OUT = os.path.join(h1.ROOT, "raw_audit")
B, BLOCK = int(os.environ.get("FI_B", 999)), 20
KR = (3, 5, 10, 15, 20)


def omega_batch(R, S):
    """O-information for an (n, k) array of index subsets of correlation matrix R."""
    M = R[S[:, :, None], S[:, None, :]]
    ld = np.linalg.slogdet(M)[1]
    dg = np.diagonal(np.linalg.inv(M), axis1=1, axis2=2)
    return -ld - 0.5 * np.log(dg).sum(1)


def omega_eq_vec(k, r):
    ld = lambda m: (m - 1) * np.log(1 - r) + np.log(1 + (m - 1) * r)
    return -0.5 * ld(k) - 0.5 * (k * ld(k - 1) - (k - 1) * ld(k))


def stat_boot_idx(T, rng):
    idx = np.empty(T, dtype=int)
    i = 0
    while i < T:
        L = rng.geometric(1 / BLOCK)
        s = rng.integers(T)
        n = min(L, T - i)
        idx[i:i + n] = (s + np.arange(n)) % T
        i += n
    return idx


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [None, None]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def nw_diff(y, d, L=20):
    X = np.column_stack([np.ones(len(y)), d.astype(float)])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    u = X * (y - X @ b)[:, None]
    S = u.T @ u
    for lag in range(1, L + 1):
        G = u[lag:].T @ u[:-lag]
        S += (1 - lag / (L + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi
    se = float(np.sqrt(V[1, 1]))
    return {"diff": round(float(b[1]), 4), "nw_se": round(se, 4), "ci": [round(float(b[1] - 1.96 * se), 4), round(float(b[1] + 1.96 * se), 4)],
            "z": round(float(b[1] / se), 2)}


class Curve:
    """System-size subsets grouped by k as index arrays."""

    def __init__(self, sub):
        self.S = {k: np.array(v) for k, v in sub.items()}

    def means(self, R, ks=KS):
        return {k: float(omega_batch(R, self.S[k]).mean()) for k in ks}


class RouteA:
    def __init__(self, subs):
        self.groups = {}
        for k, cls, s in subs:
            self.groups.setdefault(k, {}).setdefault(cls, []).append(s)
        self.groups = {k: {c: np.array(v) for c, v in g.items()} for k, g in self.groups.items()}

    def stats(self, R):
        out = {}
        for k in KR:
            rel_all, sd_all = [], []
            for cls, S in self.groups[k].items():
                om = omega_batch(R, S)
                iu = np.triu_indices(k, 1)
                P = R[S[:, :, None], S[:, None, :]][:, iu[0], iu[1]]
                rel = (om - omega_eq_vec(k, P.mean(1))) / omega_eq_vec(k, P.mean(1))
                out[f"k{k}_{cls}"] = float(rel.mean())
                rel_all.append(rel); sd_all.append(P.std(1))
            rel, sd = np.concatenate(rel_all), np.concatenate(sd_all)
            out[f"k{k}_pooled"] = float(rel.mean())
            if 3 < k < 20:
                out[f"k{k}_R2_rhosd"] = float(np.corrcoef(rel, sd)[0, 1] ** 2)
        return out


def share_neg_by_state(w, st):
    return {nm: float((w[m] < 0).mean()) if m.any() else np.nan for nm, m in st.items()}


def main():
    t0 = time.time()
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    xj = h1.apply_tie_safe_jitter(df[mk].to_numpy(float), seed=h1.TIE_JITTER_SEED)
    z = gaussian_copula_transform(xj)
    R = correlation_matrix(z)
    T = len(z)
    cur = Curve(subsets())
    subsA, _ = build_subsets(mk)
    ra = RouteA(subsA)
    # check batched estimator
    chk = [(s, o_information(R, list(s))[0], omega_batch(R, np.array([s]))[0]) for s in cur.S[5][:3].tolist() + [tuple(range(20))]]
    assert all(abs(a - b) < 1e-8 for _, a, b in chk), chk
    vix = pd.read_csv(os.path.join(h1.ROOT, "data_external", "VIX_History_CBOE.csv"), parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    vl = vix.shift(1).reindex(df.Date).ffill().to_numpy()
    q = np.nanquantile(vl, [1 / 3, 2 / 3])
    st = {"VIX_low": vl <= q[0], "VIX_mid": (vl > q[0]) & (vl < q[1]), "VIX_high": vl >= q[1]}
    for c, (a, b) in CRISES.items():
        st[c] = df.Date.between(a, b).to_numpy()
    st["non_crisis"] = ~(st["GFC"] | st["COVID"])
    # point estimates
    w = local_omega(z, R)
    om20 = float(o_information(R, list(range(20)))[0])
    pt_curve = cur.means(R)
    pt_fit = fits(pd.DataFrame({"mean": pd.Series(pt_curve)}))
    pt_A = ra.stats(R)
    pt_share = share_neg_by_state(w, st)
    pt_share["all"] = float((w < 0).mean())
    Rl, Rh = correlation_matrix(z[st["VIX_low"]]), correlation_matrix(z[st["VIX_high"]])
    ml, mh = cur.means(Rl, (10, 20)), cur.means(Rh, (10, 20))
    pt_ratio = {k: mh[k] / ml[k] for k in (10, 20)}
    res = {"vix_terciles": [round(float(v), 2) for v in q], "omega20": round(om20, 4), "mean_local": round(float(w.mean()), 4)}
    # C1 + C2 null: circular-shift surrogates
    rng = np.random.default_rng(20261107)
    null20, nullc = [], []
    for _ in range(B):
        zs = np.column_stack([np.roll(z[:, j], rng.integers(T)) for j in range(20)])
        Rs = correlation_matrix(zs)
        null20.append(o_information(Rs, list(range(20)))[0])
        nullc.append(cur.means(Rs, (3, 10, 20)))
    null20 = np.array(null20)
    print(f"surrogates done {time.time() - t0:.0f}s", flush=True)
    # bootstrap
    rng = np.random.default_rng(20261106)
    bt = {"om20": [], "curve": [], "exp": [], "pairs": [], "A": [], "share": [], "ratio": [], "d_hl": [], "d_gfc": [], "d_cov": []}
    for b in range(B):
        idx = stat_boot_idx(T, rng)
        zb = z[idx]
        Rb = correlation_matrix(zb)
        bt["om20"].append(o_information(Rb, list(range(20)))[0])
        cm = cur.means(Rb)
        bt["curve"].append(cm)
        bt["exp"].append(fits(pd.DataFrame({"mean": pd.Series(cm)}))["power"]["exponent"])
        bt["pairs"].append({k: cm[k] / math.comb(k, 2) for k in KS})
        bt["A"].append(ra.stats(Rb))
        wb = local_omega(zb, Rb)
        stb = {nm: m[idx] for nm, m in st.items()}
        s = share_neg_by_state(wb, stb)
        s["all"] = float((wb < 0).mean())
        bt["share"].append(s)
        bt["d_hl"].append(s["VIX_high"] - s["VIX_low"])
        bt["d_gfc"].append(s["GFC"] - s["non_crisis"])
        bt["d_cov"].append(s["COVID"] - s["non_crisis"])
        a, h = cur.means(correlation_matrix(zb[stb["VIX_low"]]), (10, 20)), cur.means(correlation_matrix(zb[stb["VIX_high"]]), (10, 20))
        bt["ratio"].append({k: h[k] / a[k] for k in (10, 20)})
        if (b + 1) % 100 == 0:
            print(f"boot {b + 1} {time.time() - t0:.0f}s", flush=True)
    # C5 Gaussian reference
    rng = np.random.default_rng(20261108)
    Lc = np.linalg.cholesky(R)
    gref = []
    for _ in range(B):
        zg = gaussian_copula_transform(rng.standard_normal((T, 20)) @ Lc.T)
        wg = local_omega(zg, correlation_matrix(zg))
        gref.append(float((wg < 0).mean()))
    gref = np.array(gref)
    print(f"gauss ref done {time.time() - t0:.0f}s", flush=True)
    col = lambda key, sub: [d[sub] for d in bt[key]]
    res["C1"] = {"omega20": round(om20, 4), "boot_ci": ci(bt["om20"]), "null_mean": round(float(null20.mean()), 4),
                 "null_p99": round(float(np.percentile(null20, 99)), 4), "null_max": round(float(null20.max()), 4),
                 "p_value": round(float((1 + (null20 >= om20).sum()) / (1 + B)), 4)}
    res["C2"] = {"mean_by_k": {k: {"est": round(pt_curve[k], 4), "ci": ci(col("curve", k)),
                                   "per_pair": round(pt_curve[k] / math.comb(k, 2), 4), "per_pair_ci": ci(col("pairs", k))} for k in KS},
                 "power_exponent": {"est": pt_fit["power"]["exponent"], "ci": ci(bt["exp"])},
                 "null_band_mean": {k: [round(float(np.percentile([d[k] for d in nullc], p)), 4) for p in (2.5, 97.5)] for k in (3, 10, 20)}}
    res["C3"] = {key: {"est": round(v, 4), "ci": ci(col("A", key))} for key, v in pt_A.items()}
    res["C4"] = {"share_all": round(pt_share["all"], 4), "ci": ci(col("share", "all")),
                 "by_state": {nm: {"est": round(pt_share[nm], 4), "ci": ci(col("share", nm))} for nm in st}}
    res["C5"] = {"gauss_ref_mean": round(float(gref.mean()), 4), "gauss_ref_95": ci(gref),
                 "empirical": round(pt_share["all"], 4), "emp_over_ref": round(float(pt_share["all"] / gref.mean()), 3),
                 "share_ref_draws_ge_emp": round(float((gref >= pt_share["all"]).mean()), 4)}
    y = (w < 0).astype(float)
    sel = lambda a, b: st[a] | st[b]
    res["C6"] = {
        "high_minus_low": {**nw_diff(y[sel("VIX_high", "VIX_low")], st["VIX_high"][sel("VIX_high", "VIX_low")]), "boot_ci": ci(bt["d_hl"])},
        "GFC_minus_noncrisis": {**nw_diff(y[sel("GFC", "non_crisis")], st["GFC"][sel("GFC", "non_crisis")]), "boot_ci": ci(bt["d_gfc"])},
        "COVID_minus_noncrisis": {**nw_diff(y[sel("COVID", "non_crisis")], st["COVID"][sel("COVID", "non_crisis")]), "boot_ci": ci(bt["d_cov"])},
        "omega_curve_ratio_high_low": {k: {"est": round(pt_ratio[k], 3), "ci": ci(col("ratio", k)),
                                           "low": round(ml[k], 3), "high": round(mh[k], 3)} for k in (10, 20)}}
    res["design"] = {"B": B, "mean_block": BLOCK, "runtime_s": round(time.time() - t0)}
    json.dump(res, open(os.path.join(OUT, "final_inference_results.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "C2"}, indent=1, default=str))
    print(json.dumps({"C2_power": res["C2"]["power_exponent"], "C2_k": {k: res["C2"]["mean_by_k"][k] for k in (3, 5, 10, 15, 20)},
                      "C2_null": res["C2"]["null_band_mean"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
