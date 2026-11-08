"""State-specific Gaussian calibration of the local-synergy share (pre-specified 2026-11-05).

States: previous-day VIX terciles (low / mid / high), as in final_inference.py C4/C6.
For each state s:
  R_s  = correlation of the full-sample normal scores restricted to the days in s (Methods 2.7 convention).
  Reference: 999 draws of (T_s x 20) from N(0, R_s) -> normal-score transform -> R re-estimated -> local omega -> share(<0)
             (same steps as C5). Seed 20261110 + state offset.
Observed shares reported against the same reference:
  A  existing C4 share: omega_t from full-sample z and full-sample R, averaged over state days (unchanged).
  B  state-R share: full-sample z restricted to s, omega_t evaluated under R_s.
  C  within-state pipeline: returns of s re-ranked to normal scores within s, R re-estimated -> identical to the
     reference pipeline (observed and simulated samples processed the same way).
Diagnostic: mean variance of the full-sample normal scores within s (scale differs from 1 in non-central states).

Reference matrix (env FSG_REF, revised 2026-11-06):
  Rc (default): reference drawn from N(0, R_c), R_c = correlation of the within-state re-ranked scores, i.e. the same
                matrix used for observed variant C. Output: raw_audit/final_state_gauss_ref_Rc.json.
  Rs          : original 2026-11-05 run (reference from R_s). Output: raw_audit/final_state_gauss_ref.json.
Seeds, B, states and all other steps are identical in both modes.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from entropy_estimators import correlation_matrix, gaussian_copula_transform  # noqa: E402
from routeC_local_oinfo import local_omega  # noqa: E402

B = int(os.environ.get("FI_B", 999))
REF = os.environ.get("FSG_REF", "Rc")
assert REF in ("Rc", "Rs"), REF
OUT_NAME = "final_state_gauss_ref_Rc.json" if REF == "Rc" else "final_state_gauss_ref.json"


def share(z, R):
    return float((local_omega(z, R) < 0).mean())


def main():
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    xj = h1.apply_tie_safe_jitter(df[mk].to_numpy(float), seed=h1.TIE_JITTER_SEED)
    z = gaussian_copula_transform(xj)
    R = correlation_matrix(z)
    w = local_omega(z, R)
    vix = pd.read_csv(os.path.join(h1.ROOT, "data_external", "VIX_History_CBOE.csv"), parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    vl = vix.shift(1).reindex(df.Date).ffill().to_numpy()
    q = np.nanquantile(vl, [1 / 3, 2 / 3])
    st = {"VIX_low": vl <= q[0], "VIX_mid": (vl > q[0]) & (vl < q[1]), "VIX_high": vl >= q[1]}
    iu = np.triu_indices(20, 1)
    res = {"vix_terciles": [round(float(v), 2) for v in q], "B": B, "reference_matrix": REF}
    for off, (nm, m) in enumerate(st.items()):
        zs = z[m]
        Rs = correlation_matrix(zs)
        zc = gaussian_copula_transform(xj[m])
        Rc = correlation_matrix(zc)
        obs = {"A_existing": float((w[m] < 0).mean()), "B_state_R": share(zs, Rs), "C_within_state": share(zc, Rc)}
        rng = np.random.default_rng(20261110 + off)
        Rref = Rc if REF == "Rc" else Rs
        L = np.linalg.cholesky(Rref)
        Ts = int(m.sum())
        ref = np.empty(B)
        for b in range(B):
            zg = gaussian_copula_transform(rng.standard_normal((Ts, 20)) @ L.T)
            ref[b] = share(zg, correlation_matrix(zg))
        rm = float(ref.mean())
        res[nm] = {"T": Ts, "mean_rho_Rs": round(float(Rs[iu].mean()), 3), "mean_var_fullsample_z": round(float(zs.var(0).mean()), 3),
                   "ref_mean": round(rm, 4), "ref_95": [round(float(v), 4) for v in np.percentile(ref, [2.5, 97.5])],
                   "observed": {k: {"share": round(v, 4), "ratio": round(v / rm, 3), "n_ref_ge_obs": int((ref >= v).sum())}
                                for k, v in obs.items()}}
        print(nm, json.dumps(res[nm]), flush=True)
    for k in ("A_existing", "B_state_R", "C_within_state"):
        h, lo = res["VIX_high"]["observed"][k], res["VIX_low"]["observed"][k]
        res[f"high_minus_low_{k}"] = {"observed_pp": round(100 * (h["share"] - lo["share"]), 2),
                                      "reference_pp": round(100 * (res["VIX_high"]["ref_mean"] - res["VIX_low"]["ref_mean"]), 2),
                                      "excess_over_ref_pp": round(100 * ((h["share"] - res["VIX_high"]["ref_mean"])
                                                                         - (lo["share"] - res["VIX_low"]["ref_mean"])), 2)}
    with open(os.path.join(h1.ROOT, "raw_audit", OUT_NAME), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
