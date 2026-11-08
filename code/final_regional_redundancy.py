"""Regional-redundancy check (pre-specified 2026-11-05). Same GC O-information, universe, regions, VIX states.

Test 1: per region (EU 9, AS 7, AM 3): k, Omega, mean rho, Omega/C(k,2), Omega_eq(k, mean rho), rel_dev.
Test 2: k = 3 and k = 5, exhaustive subsets; class = within-region (all markets in EU, AS or AM) vs mixed-region
        (>= 2 regions; SATop40 counts as its own region). Mean/median Omega, mean rho, Omega/C(k,2) per class (and
        per region for within). Bootstrap CI of mean(within) - mean(mixed). Descriptive OLS across subsets:
        Omega ~ 1 + rho + I(within)   (as specified)  and, as a nonlinearity control, + rho^2;
        plus the rel_dev (vs Omega_eq) difference within - mixed.
Test 3: low vs high VIX (previous-day terciles): Omega and Omega/C(k,2) for each region and the 20-market system,
        plus within vs mixed mean Omega at k = 3, 5; high/low ratios.
Inference: stationary block bootstrap, mean block 20, B = 999 (seed 20261109), normal scores fixed, R re-estimated
per replicate, state labels travel with days.
"""
import json
import math
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from entropy_estimators import correlation_matrix, gaussian_copula_transform  # noqa: E402
from final_inference import ci, omega_batch, omega_eq_vec, stat_boot_idx  # noqa: E402
from routeA_composition import REG  # noqa: E402

B = int(os.environ.get("FI_B", 999))


def mean_rho(R, S):
    k = S.shape[1]
    iu = np.triu_indices(k, 1)
    return R[S[:, :, None], S[:, None, :]][:, iu[0], iu[1]].mean(1)


def ols(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b


def test2_stats(R, groups):
    out = {}
    for k, g in groups.items():
        S, w, lab = g["S"], g["within"], g["lab"]
        om, rho = omega_batch(R, S), mean_rho(R, S)
        rel = (om - omega_eq_vec(k, rho)) / omega_eq_vec(k, rho)
        out[f"k{k}_diff_mean"] = om[w].mean() - om[~w].mean()
        X1 = np.column_stack([np.ones(len(om)), rho, w])
        X2 = np.column_stack([X1, rho ** 2])
        out[f"k{k}_beta_within_lin"] = ols(om, X1)[2]
        out[f"k{k}_beta_within_quad"] = ols(om, X2)[2]
        out[f"k{k}_reldev_diff"] = rel[w].mean() - rel[~w].mean()
        out[f"_k{k}"] = (om, rho, rel)
    return out


def main():
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    xj = h1.apply_tie_safe_jitter(df[mk].to_numpy(float), seed=h1.TIE_JITTER_SEED)
    z = gaussian_copula_transform(xj)
    R = correlation_matrix(z)
    T = len(z)
    reg = np.array([next(r for r, m in REG.items() if c in m) for c in mk])
    regions = {r: np.array([mk.index(m) for m in REG[r]]) for r in ("EU", "AS", "AM")}
    vix = pd.read_csv(os.path.join(h1.ROOT, "data_external", "VIX_History_CBOE.csv"), parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    vl = vix.shift(1).reindex(df.Date).ffill().to_numpy()
    q = np.nanquantile(vl, [1 / 3, 2 / 3])
    st = {"low": vl <= q[0], "high": vl >= q[1]}
    full = np.arange(20)[None, :]
    reg_S = {r: idx[None, :] for r, idx in regions.items()}
    groups = {}
    for k in (3, 5):
        S = np.array(list(combinations(range(20), k)))
        lab = np.array([reg[s[0]] if len(set(reg[s])) == 1 else "mixed" for s in S])
        groups[k] = {"S": S, "within": lab != "mixed", "lab": lab}

    def test1(Rm):
        out = {}
        for r, S in list(reg_S.items()) + [("ALL20", full)]:
            k = S.shape[1]
            om, rho = float(omega_batch(Rm, S)[0]), float(mean_rho(Rm, S)[0])
            oe = float(omega_eq_vec(k, rho))
            out[r] = {"k": k, "omega": om, "mean_rho": rho, "omega_per_pair": om / math.comb(k, 2), "omega_eq": oe, "rel_dev": (om - oe) / oe}
        return out

    def test3(zz, masks):
        out = {}
        for s, m in masks.items():
            Rs = correlation_matrix(zz[m])
            t1 = test1(Rs)
            for r, v in t1.items():
                out[f"{r}_{s}"] = v["omega"]
                out[f"{r}_{s}_pp"] = v["omega_per_pair"]
                out[f"{r}_{s}_rho"] = v["mean_rho"]
            for k, g in groups.items():
                om = omega_batch(Rs, g["S"])
                out[f"k{k}_within_{s}"] = om[g["within"]].mean()
                out[f"k{k}_mixed_{s}"] = om[~g["within"]].mean()
        for key in [k for k in out if k.endswith("_low")]:
            base = key[:-4]
            out[f"{base}_ratio"] = out[f"{base}_high"] / out[f"{base}_low"]
        for r in ("EU", "AS", "AM", "ALL20"):
            out[f"{r}_pp_ratio"] = out[f"{r}_high_pp"] / out[f"{r}_low_pp"]
        return out

    pt1, pt2, pt3 = test1(R), test2_stats(R, groups), test3(z, st)
    # descriptive tables (point)
    tab2 = {}
    for k, g in groups.items():
        om, rho, rel = pt2[f"_k{k}"]
        for lab in ["mixed", "within", "EU", "AS", "AM"]:
            m = g["within"] if lab == "within" else (g["lab"] == lab)
            if m.sum() == 0:
                continue
            tab2[f"k{k}_{lab}"] = {"n": int(m.sum()), "mean": round(float(om[m].mean()), 4), "median": round(float(np.median(om[m])), 4),
                                   "mean_rho": round(float(rho[m].mean()), 3), "per_pair": round(float(om[m].mean() / math.comb(k, 2)), 4),
                                   "rel_dev": round(float(rel[m].mean()), 3)}
    rng = np.random.default_rng(20261109)
    bt1, bt2, bt3 = [], [], []
    for b in range(B):
        idx = stat_boot_idx(T, rng)
        zb = z[idx]
        Rb = correlation_matrix(zb)
        bt1.append(test1(Rb))
        bt2.append({k: v for k, v in test2_stats(Rb, groups).items() if not k.startswith("_")})
        bt3.append(test3(zb, {s: m[idx] for s, m in st.items()}))
        if (b + 1) % 200 == 0:
            print("boot", b + 1, flush=True)
    res = {"vix_terciles": [round(float(v), 2) for v in q], "B": B,
           "test1": {r: {kk: round(vv, 4) for kk, vv in v.items()} | {"omega_ci": ci([d[r]["omega"] for d in bt1]),
                                                                       "per_pair_ci": ci([d[r]["omega_per_pair"] for d in bt1]),
                                                                       "rel_dev_ci": ci([d[r]["rel_dev"] for d in bt1])} for r, v in pt1.items()},
           "test2_table": tab2,
           "test2_inference": {k: {"est": round(float(v), 4), "ci": ci([d[k] for d in bt2])} for k, v in pt2.items() if not k.startswith("_")},
           "test3": {k: {"est": round(float(v), 4), "ci": ci([d[k] for d in bt3])} for k, v in pt3.items()}}
    json.dump(res, open(os.path.join(h1.ROOT, "raw_audit", "final_regional_redundancy.json"), "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
