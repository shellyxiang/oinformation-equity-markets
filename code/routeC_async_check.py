"""Asynchrony sanity check for Route C local O-information (same GC local estimator).

Variants (fixed before running, 2026-11-05):
  A baseline calendar-day returns;
  B Asia-Pacific return at t+1 (Europe, Americas at t), last row dropped;
  C Americas return at t-1 (Asia, Europe at t), first row dropped;
  D combined: non-overlapping 2-day returns (sum of t and t+1) for every market -> each market's window covers
    its own session plus the following one (standard simple de-synchronization remedy; chosen a priori).
States per row: previous-day VIX terciles (14.40/19.98) and crisis windows keyed on the row date (for D, the
block's first date). Regional omega is within-region, so it changes only under D; under B/C only the
cross-region synchronization of synergy days can change.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from routeC_local_oinfo import CRISES, REG, gc, local_omega  # noqa: E402

OUT = os.path.join(h1.ROOT, "raw_audit")


def variants(df, mk, xj):
    dates = df.Date.reset_index(drop=True)
    AS = [mk.index(m) for m in REG["AS"]]
    AMi = [mk.index(m) for m in REG["AM"]]
    B = xj[:-1].copy(); B[:, AS] = xj[1:, AS]
    C = xj[1:].copy(); C[:, AMi] = xj[:-1, AMi]
    n2 = len(xj) // 2 * 2
    D = xj[:n2].reshape(-1, 2, xj.shape[1]).sum(1)
    return {"A_baseline": (xj, dates), "B_Asia_t+1": (B, dates[:-1].reset_index(drop=True)),
            "C_Americas_t-1": (C, dates[1:].reset_index(drop=True)), "D_2day_nonoverlap": (D, dates[:n2:2].reset_index(drop=True))}


def main():
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    xj = h1.apply_tie_safe_jitter(df[mk].to_numpy(float), seed=h1.TIE_JITTER_SEED)
    vix = pd.read_csv(os.path.join(h1.ROOT, "data_external", "VIX_History_CBOE.csv"), parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    q = [14.40, 19.98]
    base = None
    res, tab, reg_tab = {}, [], []
    for nm, (X, d) in variants(df, mk, xj).items():
        z, R = gc(X)
        w = pd.Series(local_omega(z, R), index=d)
        vl = vix.shift(1).reindex(d).ffill().to_numpy()
        st = {"low_VIX": vl <= q[0], "high_VIX": vl >= q[1]}
        for c, (a, b) in CRISES.items():
            st[c] = d.between(a, b).to_numpy()
        st["non_crisis"] = ~(st["GFC"] | st["COVID"])
        row = {"variant": nm, "n": len(w), "mean": w.mean(), "median": w.median(), "share_neg": (w < 0).mean(),
               "p5": w.quantile(.05), "p95": w.quantile(.95)}
        row.update({f"neg_{k}": float((w.to_numpy()[m] < 0).mean()) for k, m in st.items()})
        if base is None:
            base = w
        common = base.index.intersection(w.index)
        row["corr_with_baseline"] = float(np.corrcoef(base.loc[common], w.loc[common])[0, 1]) if nm != "D_2day_nonoverlap" else np.nan
        tab.append(row)
        rw = {}
        for r, ms in REG.items():
            zr, Rr = gc(X[:, [mk.index(m) for m in ms]])
            rw[r] = local_omega(zr, Rr)
        rdf = pd.DataFrame(rw)
        neg = rdf < 0
        reg_tab.append({"variant": nm, **{f"share_neg_{r}": float(neg[r].mean()) for r in REG},
                        **{f"corr_omega_{a}_{b}": float(rdf[a].corr(rdf[b])) for a, b in (("EU", "AS"), ("EU", "AM"), ("AS", "AM"))},
                        **{f"corr_negind_{a}_{b}": float(neg[a].astype(float).corr(neg[b].astype(float))) for a, b in (("EU", "AS"), ("EU", "AM"), ("AS", "AM"))},
                        "share_all3_neg": float(neg.all(axis=1).mean())})
        res[nm] = w
    T = pd.DataFrame(tab).round(4)
    RG = pd.DataFrame(reg_tab).round(4)
    T.to_csv(os.path.join(OUT, "routeC_async_full.csv"), index=False)
    RG.to_csv(os.path.join(OUT, "routeC_async_regional.csv"), index=False)
    # extreme persistence
    bot = base.nsmallest(20)
    pers = []
    for nm, w in res.items():
        if nm == "A_baseline":
            continue
        if nm == "D_2day_nonoverlap":
            blk = pd.Series(w.values, index=w.index)
            bd = blk.index.to_numpy()
            pos = np.searchsorted(bd, bot.index.to_numpy(), side="right") - 1
            vals = blk.to_numpy()[pos]
            pct = [float((blk < v).mean()) for v in vals]
            pers.append({"variant": nm, "in_aligned_bottom20": int(np.isin(blk.index[pos], blk.nsmallest(20).index).sum()),
                         "median_pctile_of_containing_block": round(float(np.median(pct)), 3),
                         "share_containing_block_negative": round(float(np.mean(vals < 0)), 3)})
        else:
            ww = w.reindex(bot.index)
            pers.append({"variant": nm, "in_aligned_bottom20": int(bot.index.isin(w.nsmallest(20).index).sum()),
                         "spearman_on_bottom20_dates": round(float(spearmanr(bot.values, ww.values, nan_policy="omit")[0]), 3),
                         "share_still_negative": round(float((ww < 0).mean()), 3),
                         "median_pctile": round(float(np.median([(w < v).mean() for v in ww.dropna()])), 3)})
    crisis_dates = ["2008-10-28", "2008-10-14", "2009-01-07", "2020-03-20"]
    pers_dates = {nm: {cd: (round(float(w.get(pd.Timestamp(cd), np.nan)), 2)) for cd in crisis_dates} for nm, w in res.items() if nm != "D_2day_nonoverlap"}
    # lead-lag of regional EW returns (baseline calendar data)
    eq = {r: df[REG[r]].mean(1).to_numpy() for r in REG}
    ll = lambda a, b, lag: float(np.corrcoef(a[:len(a) - lag], b[lag:])[0, 1])
    lead = {"AS_t~EU_t": ll(eq["AS"], eq["EU"], 0), "AS_t~EU_t+1": ll(eq["AS"], eq["EU"], 1),
            "EU_t~AM_t": ll(eq["EU"], eq["AM"], 0), "EU_t~AM_t+1": ll(eq["EU"], eq["AM"], 1),
            "AM_t~AS_t+1": ll(eq["AM"], eq["AS"], 1), "AM_t~AS_t": ll(eq["AM"], eq["AS"], 0),
            "AM_t~EU_t+1": ll(eq["AM"], eq["EU"], 1)}
    out = {"full": T.to_dict("records"), "regional": RG.to_dict("records"), "extreme_persistence": pers,
           "crisis_synergy_dates_omega": pers_dates, "leadlag_regional_EW": {k: round(v, 3) for k, v in lead.items()}}
    json.dump(out, open(os.path.join(OUT, "routeC_async_results.json"), "w"), indent=1, default=str)
    print(T.to_string()); print(RG.to_string())
    print(json.dumps({k: out[k] for k in ("extreme_persistence", "crisis_synergy_dates_omega", "leadlag_regional_EW")}, indent=1))


if __name__ == "__main__":
    main()
