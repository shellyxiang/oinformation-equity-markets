"""Route C: Gaussian-copula local O-information (Scagliarini et al. 2022 formulation), quick diagnostic.

omega_t = sum_i h_i(t) - sum_i h_{-i}(t) + (n-2) h(t), with local entropies h_S(t) = -log p_S(z_S,t) of the
multivariate normal (correlation R of normal scores). Time-average of omega_t equals the global GC O-information
(checked). Normal scores from the tie-safe-jittered returns (same pipeline as Layer 1 / system-size runs).
States: previous-day VIX terciles (as before); crisis windows as already coded (GFC 2008-09-01..2009-06-30,
COVID 2020-02-15..2020-06-30). Diagnostics on raw returns: G = EW mean, |G|, G^2, cross-sectional SD; PC1 score.
Regions: EU, AS (Asia-Pacific incl. ASX200), AM. Stability: halves (R re-estimated per half), weekly (W-FRI).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from entropy_estimators import correlation_matrix, gaussian_copula_transform, o_information  # noqa: E402

OUT = os.path.join(h1.ROOT, "raw_audit")
REG = {"EU": ["AEX", "CAC40", "DAX", "FTSE100", "FTSEMIB", "IBEX35", "OMXS30", "OBX", "SMI"],
       "AS": ["HangSeng", "LQ45", "KOSPI200", "Nifty50", "ASX200", "CSI300", "TWSE"], "AM": ["SP500", "IPC", "TSX60"]}
CRISES = {"GFC": ("2008-09-01", "2009-06-30"), "COVID": ("2020-02-15", "2020-06-30")}


def local_omega(z, R):
    n = z.shape[1]
    q = lambda Ri, zi: np.einsum("ti,ij,tj->t", zi, np.linalg.inv(Ri), zi)
    ld = lambda Ri: np.linalg.slogdet(Ri)[1]
    h_full = 0.5 * (ld(R) + q(R, z))
    h_i = 0.5 * (z ** 2).sum(1)                       # sum of univariate local entropies (constants cancel)
    h_minus = np.zeros(len(z))
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        h_minus += 0.5 * (ld(R[np.ix_(idx, idx)]) + q(R[np.ix_(idx, idx)], z[:, idx]))
    return h_i - h_minus + (n - 2) * h_full


def gc(x):
    z = gaussian_copula_transform(x)
    return z, correlation_matrix(z)


def stats_(w):
    return {"mean": round(float(w.mean()), 3), "median": round(float(np.median(w)), 3), "sd": round(float(w.std()), 3),
            "p5": round(float(np.percentile(w, 5)), 3), "p95": round(float(np.percentile(w, 95)), 3),
            "share_pos": round(float((w > 0).mean()), 3), "share_neg": round(float((w < 0).mean()), 3), "n": int(len(w))}


def main():
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    X = df[mk].to_numpy(float)
    xj = h1.apply_tie_safe_jitter(X, seed=h1.TIE_JITTER_SEED)
    z, R = gc(xj)
    w = local_omega(z, R)
    res = {"global_check": {"mean_local": round(float(w.mean()), 4), "Omega_GC_full": round(float(o_information(R, list(range(20)))[0]), 4)},
           "full": stats_(w)}
    vix = pd.read_csv(os.path.join(h1.ROOT, "data_external", "VIX_History_CBOE.csv"), parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    vlag = vix.shift(1).reindex(df.Date).ffill().to_numpy()
    vnow = vix.reindex(df.Date).ffill().to_numpy()
    q = np.nanquantile(vlag, [1 / 3, 2 / 3])
    res["vix_terciles"] = [round(float(v), 2) for v in q]
    st = {"VIX_low": vlag <= q[0], "VIX_mid": (vlag > q[0]) & (vlag < q[1]), "VIX_high": vlag >= q[1]}
    for nm, (a, b) in CRISES.items():
        st[nm] = df.Date.between(a, b).to_numpy()
    st["non_crisis"] = ~(st["GFC"] | st["COVID"])
    res["states"] = {nm: stats_(w[m]) for nm, m in st.items()}
    G = X.mean(1)
    disp = X.std(1)
    pc1 = z @ np.linalg.eigh(R)[1][:, -1]
    diag = {"absG": np.abs(G), "G2": G ** 2, "dispersion": disp, "dispersion2": disp ** 2, "absPC1": np.abs(pc1), "PC1sq": pc1 ** 2,
            "sum_z2": (z ** 2).sum(1)}
    res["corr_with_omega"] = {k: round(float(np.corrcoef(w, v)[0, 1]), 3) for k, v in diag.items()}
    for nm, cols in (("PC1sq+sum_z2", ["PC1sq", "sum_z2"]), ("G2+dispersion2", ["G2", "dispersion2"])):
        Z = np.column_stack([np.ones(len(w))] + [diag[c] for c in cols])
        e = w - Z @ np.linalg.lstsq(Z, w, rcond=None)[0]
        res.setdefault("R2_omega_on", {})[nm] = round(float(1 - e.var() / w.var()), 3)
    # regions
    reg_w = {}
    for r, ms in REG.items():
        idx = [mk.index(m) for m in ms]
        zr, Rr = gc(xj[:, idx])
        reg_w[r] = local_omega(zr, Rr)
    res["regional"] = {r: {**stats_(v), "corr_with_global": round(float(np.corrcoef(v, w)[0, 1]), 3)} for r, v in reg_w.items()}
    neg = pd.DataFrame({r: v < 0 for r, v in reg_w.items()})
    res["regional_neg_cooccurrence"] = {"share_all3_neg": round(float(neg.all(axis=1).mean()), 3), "share_any_neg": round(float(neg.any(axis=1).mean()), 3),
                                        "corr_of_neg_indicators": neg.astype(float).corr().round(3).to_dict(),
                                        "share_global_neg_given_all3_neg": round(float((w[neg.all(axis=1).to_numpy()] < 0).mean()), 3) if neg.all(axis=1).any() else None}
    # top dates
    t = pd.DataFrame({"date": df.Date.dt.date.astype(str), "omega": w.round(3), "G_pct": (G * 100).round(3),
                      "VIX": np.round(vnow, 2), "disp_pct": (disp * 100).round(3)})
    res["top10_pos"] = t.nlargest(10, "omega").to_dict("records")
    res["top10_neg"] = t.nsmallest(10, "omega").to_dict("records")
    t.to_csv(os.path.join(OUT, "routeC_local_omega_daily.csv"), index=False)
    # stability
    h = len(xj) // 2
    res["halves"] = {}
    for nm, sl in (("H1", slice(0, h)), ("H2", slice(h, None))):
        zz, RR = gc(xj[sl])
        res["halves"][nm] = stats_(local_omega(zz, RR))
    wk = df.set_index("Date")[mk].groupby(pd.Grouper(freq="W-FRI")).sum()
    wk = wk[df.set_index("Date")[mk].groupby(pd.Grouper(freq="W-FRI")).size() > 0]
    zw, Rw = gc(h1.apply_tie_safe_jitter(wk.to_numpy(float), seed=h1.TIE_JITTER_SEED))
    res["weekly"] = stats_(local_omega(zw, Rw))
    json.dump(res, open(os.path.join(OUT, "routeC_local_oinfo_results.json"), "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
