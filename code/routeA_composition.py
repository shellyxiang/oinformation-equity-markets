"""Route A: how market composition shapes system-level redundancy across system size (descriptive).

Fixed before running (2026-11-05):
- Same universe, tie-safe jitter and Gaussian-copula O-information as quick_omega_system_size.py.
- Regions: EU (9), AS (7: HangSeng, LQ45, KOSPI200, Nifty50, ASX200, CSI300, TWSE), AM (3), AF (SATop40).
- Composition classes by number of distinct regions in S: R1 (single region), R2, R3+ (3 or 4 regions).
  R1 subsets enumerated exhaustively (EU, AS, AM); R2 and R3+ sampled at random, up to 1000 per k
  (seed 20261105), identical subset lists reused for every sample/state.
- Per subset: Omega_GC, mean pairwise GC corr rho, SD of pairwise corr (heterogeneity),
  Omega_eq(k, rho) (equicorrelated Gaussian), rel_dev = (Omega - Omega_eq)/Omega_eq, region-mixing entropy.
- Diversification (full sample, raw daily log returns): equal-weight subset portfolio
  VR = Var(EW)/mean Var_i ; ESR = ES95(EW loss)/mean ES95_i. Descriptive cross-subset associations at fixed k.
- States: previous-day VIX bottom vs top tercile. Stability: halves, weekly (W-FRI).
"""
import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from entropy_estimators import correlation_matrix, gaussian_copula_transform, o_information  # noqa: E402
from quick_omega_system_size import omega_eq  # noqa: E402

ROOT, OUT = h1.ROOT, os.path.join(h1.ROOT, "raw_audit")
REG = {"EU": ["AEX", "CAC40", "DAX", "FTSE100", "FTSEMIB", "IBEX35", "OMXS30", "OBX", "SMI"],
       "AS": ["HangSeng", "LQ45", "KOSPI200", "Nifty50", "ASX200", "CSI300", "TWSE"],
       "AM": ["SP500", "IPC", "TSX60"], "AF": ["SATop40"]}
KS, NS, SEED = list(range(3, 21)), 1000, 20261105


def build_subsets(mk):
    reg = np.array([next(r for r, m in REG.items() if c in m) for c in mk])
    rng = np.random.default_rng(SEED)
    out = []
    for k in KS:
        for r in ("EU", "AS", "AM"):
            idx = np.where(reg == r)[0]
            if len(idx) >= k:
                out += [(k, "R1", tuple(c)) for c in combinations(idx.tolist(), k)]
        for cls in ("R2", "R3+"):
            seen, tries = set(), 0
            while len(seen) < NS and tries < 200000:
                tries += 1
                s = tuple(sorted(rng.choice(20, k, replace=False).tolist()))
                nr = len(set(reg[list(s)]))
                if (cls == "R2" and nr == 2) or (cls == "R3+" and nr >= 3):
                    seen.add(s)
            out += [(k, cls, s) for s in sorted(seen)]
    return out, reg


def evaluate(x, subs, reg):
    R = correlation_matrix(gaussian_copula_transform(x))
    rows = []
    for k, cls, s in subs:
        sub = R[np.ix_(s, s)][np.triu_indices(k, 1)]
        rho = float(sub.mean())
        om, oe = o_information(R, s)[0], omega_eq(k, rho)
        p = pd.Series(reg[list(s)]).value_counts(normalize=True).to_numpy()
        rows.append((k, cls, om, rho, float(sub.std()), oe, (om - oe) / oe, float(-(p * np.log(p)).sum())))
    return pd.DataFrame(rows, columns=["k", "cls", "omega", "rho", "rho_sd", "omega_eq", "rel_dev", "mix_entropy"])


def table(d):
    return d.groupby(["k", "cls"]).agg(n=("omega", "size"), omega=("omega", "mean"), rho=("rho", "mean"),
                                       rho_sd=("rho_sd", "mean"), rel_dev=("rel_dev", "mean")).round(4)


def main():
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    X = df[mk].to_numpy(float)
    xj = h1.apply_tie_safe_jitter(X, seed=h1.TIE_JITTER_SEED)
    subs, reg = build_subsets(mk)
    full = evaluate(xj, subs, reg)
    tf = table(full)
    tf.to_csv(os.path.join(OUT, "routeA_composition_full.csv"))
    res = {"n_subsets_by_k_cls": tf["n"].unstack().fillna(0).astype(int).to_dict()}
    # composition vs heterogeneity: rel_dev ~ rho_sd + class dummies, per k
    reg_out = {}
    for k, g in full.groupby("k"):
        if g.cls.nunique() < 2:
            continue
        D = pd.get_dummies(g.cls, drop_first=True).astype(float)
        X1 = np.column_stack([np.ones(len(g)), g.rho_sd])
        X2 = np.column_stack([X1, D.to_numpy()])
        r2 = lambda Z: 1 - np.var(g.rel_dev - Z @ np.linalg.lstsq(Z, g.rel_dev, rcond=None)[0]) / np.var(g.rel_dev)
        reg_out[int(k)] = {"R2_rho_sd": round(float(r2(X1)), 3), "R2_rho_sd_plus_class": round(float(r2(X2)), 3),
                           "corr_reldev_mixentropy": round(float(np.corrcoef(g.rel_dev, g.mix_entropy)[0, 1]), 3)}
    res["reldev_explained_by_heterogeneity"] = reg_out
    # diversification link (full sample, raw returns)
    T = X - X.mean(0)
    var_i, es_i = X.var(0), np.array([-np.mean(np.sort(X[:, i])[: int(0.05 * len(X))]) for i in range(20)])
    div = []
    for (k, cls, s), om, rho in zip(subs, full.omega, full.rho):
        ew = X[:, list(s)].mean(1)
        div.append((k, cls, om, rho, ew.var() / var_i[list(s)].mean(),
                    -np.mean(np.sort(ew)[: int(0.05 * len(ew))]) / es_i[list(s)].mean()))
    dv = pd.DataFrame(div, columns=["k", "cls", "omega", "rho", "VR", "ESR"])
    link = {}
    for k in (3, 5, 8, 10, 15):
        g = dv[dv.k == k]
        out = {}
        for y in ("VR", "ESR"):
            r2r = np.corrcoef(g.rho, g[y])[0, 1] ** 2
            Z = np.column_stack([np.ones(len(g)), g.rho, g.omega])
            r2b = 1 - np.var(g[y] - Z @ np.linalg.lstsq(Z, g[y], rcond=None)[0]) / np.var(g[y])
            out[y] = {"corr_omega": round(float(np.corrcoef(g.omega, g[y])[0, 1]), 3), "R2_rho": round(float(r2r), 3),
                      "R2_rho_plus_omega": round(float(r2b), 3), "by_class_mean": g.groupby("cls")[y].mean().round(3).to_dict()}
        link[k] = out
    res["diversification_link"] = link
    # states
    vix = pd.read_csv(os.path.join(ROOT, "data_external", "VIX_History_CBOE.csv"), parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    vlag = vix.shift(1).reindex(df.Date).ffill().to_numpy()
    q = np.nanquantile(vlag, [1 / 3, 2 / 3])
    st = {nm: table(evaluate(xj[m], subs, reg)) for nm, m in (("VIX_low", vlag <= q[0]), ("VIX_high", vlag >= q[1]))}
    h = len(xj) // 2
    st["H1"], st["H2"] = table(evaluate(xj[:h], subs, reg)), table(evaluate(xj[h:], subs, reg))
    wk = df.set_index("Date")[mk].groupby(pd.Grouper(freq="W-FRI")).sum()
    wk = wk[df.set_index("Date")[mk].groupby(pd.Grouper(freq="W-FRI")).size() > 0]
    st["weekly"] = table(evaluate(h1.apply_tie_safe_jitter(wk.to_numpy(float), seed=h1.TIE_JITTER_SEED), subs, reg))
    for nm, t in st.items():
        t.to_csv(os.path.join(OUT, f"routeA_composition_{nm}.csv"))
    comp = lambda a, b, col: round(float(np.corrcoef(a[col], b.loc[a.index, col])[0, 1]), 4)
    res["stability"] = {"H1_vs_H2_reldev": comp(st["H1"], st["H2"], "rel_dev"), "daily_vs_weekly_reldev": comp(tf, st["weekly"], "rel_dev"),
                        "VIXlow_vs_high_reldev": comp(st["VIX_low"], st["VIX_high"], "rel_dev")}
    piv = lambda t: t["rel_dev"].unstack()
    res["rel_dev_table"] = {nm: piv(t).loc[[3, 4, 5, 6, 8, 10, 12, 15, 18, 20]].round(3).to_dict() for nm, t in
                            [("full", tf), ("VIX_low", st["VIX_low"]), ("VIX_high", st["VIX_high"]), ("weekly", st["weekly"]),
                             ("H1", st["H1"]), ("H2", st["H2"])]}
    res["omega_rho_table_full"] = {c: tf[c].unstack().loc[[3, 5, 8, 10, 15, 20]].round(3).to_dict() for c in ("omega", "rho", "rho_sd")}
    json.dump(res, open(os.path.join(OUT, "routeA_composition_results.json"), "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
