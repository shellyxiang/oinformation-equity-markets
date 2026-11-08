"""Matched t-copula references for the regional systems and the VIX-state calibrations (robustness, 2026-11-07).

Each group uses exactly the sample and pipeline of its existing Gaussian reference:
  Regions (final_regional_gauss_ref.py): full-sample jittered returns of the region's markets -> normal scores ->
    R_r -> local omega -> share(<0). Gaussian ref: N(0, R_r), T = 3792, seed 20261108 + offset + 1.
  VIX states, variant C (final_state_gauss_ref.py, FSG_REF=Rc): jittered returns of the state's days re-ranked
    within the state -> R_c -> local omega -> share(<0). Gaussian ref: N(0, R_c), T_s, seed 20261110 + offset.
The Gaussian references are regenerated with their original seeds and must reproduce the saved values.

t copula per group (same procedure as tcopula_reference.py): P_tau = sin(pi tau / 2) from Kendall's tau of the
group's returns (main); nu by maximum pseudo-likelihood on u = rank/(n+1) within the group's sample, with a
profile-likelihood 95% interval. Sensitivity: P = the group's Gaussian-reference matrix (R_r or R_c), nu re-fitted.
Every simulated sample (length = group's sample size) passes through the identical pipeline. Mismatch diagnostic:
mean absolute off-diagonal difference between each replication's re-estimated R and the group's observed matrix.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from entropy_estimators import correlation_matrix, gaussian_copula_transform  # noqa: E402
from routeC_local_oinfo import REG  # noqa: E402
from tcopula_reference import fit_nu, kendall_P, share, simulate  # noqa: E402

OUT = os.path.join(h1.ROOT, "raw_audit")
B = int(os.environ.get("FI_B", 999))


def refs(rng, gen, target):
    iu = np.triu_indices(target.shape[0], 1)
    sh, mad = np.empty(B), np.empty(B)
    for b in range(B):
        sh[b], Rb = share(gen(rng))
        mad[b] = np.abs(Rb[iu] - target[iu]).mean()
    return sh, mad


def summ(sh, emp, mad=None):
    n_ge = int((sh >= emp).sum())
    out = {"mean": round(float(sh.mean()), 5), "band_95": [round(float(v), 5) for v in np.percentile(sh, [2.5, 97.5])],
           "ratio_emp_over_mean": round(emp / float(sh.mean()), 3), "n_ge_emp": n_ge, "p_mc": round((1 + n_ge) / (B + 1), 4)}
    if mad is not None:
        out["R_mismatch_mean_abs"] = round(float(mad.mean()), 4)
    return out


def run_group(name, x, gauss_seed, t_seed, saved):
    n, d = x.shape
    emp, Robs = share(x)
    assert abs(emp - saved["emp"]) < 1e-4, (name, emp, saved["emp"])
    U = stats.rankdata(x, axis=0) / (n + 1)
    P_tau, _ = kendall_P(x)
    assert np.linalg.eigvalsh(P_tau).min() > 0, name
    fit_tau, fit_obs = fit_nu(U, P_tau), fit_nu(U, Robs)
    t0 = time.time()
    Lg = np.linalg.cholesky(Robs)
    g, mad_g = refs(np.random.default_rng(gauss_seed), lambda r: r.standard_normal((n, d)) @ Lg.T, Robs)
    assert abs(g.mean() - saved["ref_mean"]) < 1e-4, (name, g.mean(), saved["ref_mean"])
    assert np.allclose(np.percentile(g, [2.5, 97.5]), saved["ref_95"], atol=1e-4), name
    tm, mad_t = refs(np.random.default_rng(t_seed), lambda r: simulate(r, P_tau, fit_tau["nu"], n, d), Robs)
    ts, mad_s = refs(np.random.default_rng(t_seed + 100), lambda r: simulate(r, Robs, fit_obs["nu"], n, d), Robs)
    iu = np.triu_indices(d, 1)
    res = {"n_obs": n, "dim": d, "empirical_share": round(emp, 5),
           "mean_offdiag_R_obs": round(float(Robs[iu].mean()), 4), "mean_offdiag_P_tau": round(float(P_tau[iu].mean()), 4),
           "nu_fit_P_tau": fit_tau, "nu_fit_R_obs": fit_obs,
           "gaussian_reference": summ(g, emp, mad_g), "t_reference_P_tau": summ(tm, emp, mad_t),
           "t_reference_R_obs_sensitivity": summ(ts, emp, mad_s)}
    print(name, f"{time.time() - t0:.0f}s", json.dumps({k: res[k] for k in ("empirical_share", "nu_fit_P_tau")}),
          res["t_reference_P_tau"], flush=True)
    rows = pd.DataFrame({"group": name, "rep": np.arange(1, B + 1), "share_gauss": g, "share_t_P_tau": tm,
                         "share_t_R_obs": ts, "mad_R_gauss": mad_g, "mad_R_t_P_tau": mad_t, "mad_R_t_R_obs": mad_s})
    return res, rows


def main():
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    xj = h1.apply_tie_safe_jitter(df[mk].to_numpy(float), seed=h1.TIE_JITTER_SEED)
    with open(os.path.join(OUT, "final_regional_gauss_ref.json")) as f:
        sreg = json.load(f)
    with open(os.path.join(OUT, "final_state_gauss_ref_Rc.json")) as f:
        sst = json.load(f)
    assert sst["reference_matrix"] == "Rc"
    vix = pd.read_csv(os.path.join(h1.ROOT, "data_external", "VIX_History_CBOE.csv"), parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    vl = vix.shift(1).reindex(df.Date).ffill().to_numpy()
    q = np.nanquantile(vl, [1 / 3, 2 / 3])
    states = {"VIX_low": vl <= q[0], "VIX_mid": (vl > q[0]) & (vl < q[1]), "VIX_high": vl >= q[1]}

    res, rows = {}, []
    for off, (r, ms) in enumerate(REG.items()):
        saved = {"emp": sreg[r]["empirical"], "ref_mean": sreg[r]["ref_mean"], "ref_95": sreg[r]["ref_95"]}
        res[r], rw = run_group(r, xj[:, [mk.index(m) for m in ms]], 20261108 + off + 1, 20261120 + off, saved)
        rows.append(rw)
    for off, (s, m) in enumerate(states.items()):
        d = sst[s]
        saved = {"emp": d["observed"]["C_within_state"]["share"], "ref_mean": d["ref_mean"], "ref_95": d["ref_95"]}
        res[s], rw = run_group(s, xj[m], 20261110 + off, 20261130 + off, saved)
        rows.append(rw)

    tab = []
    for k, v in res.items():
        g, t, s = v["gaussian_reference"], v["t_reference_P_tau"], v["t_reference_R_obs_sensitivity"]
        tab.append({"system": k, "n": v["n_obs"], "dim": v["dim"], "nu": v["nu_fit_P_tau"]["nu"],
                    "nu_lo": v["nu_fit_P_tau"]["profile_95"][0], "nu_hi": v["nu_fit_P_tau"]["profile_95"][1],
                    "emp_pct": 100 * v["empirical_share"], "gauss_mean_pct": 100 * g["mean"],
                    "gauss_lo_pct": 100 * g["band_95"][0], "gauss_hi_pct": 100 * g["band_95"][1],
                    "t_mean_pct": 100 * t["mean"], "t_lo_pct": 100 * t["band_95"][0], "t_hi_pct": 100 * t["band_95"][1],
                    "ratio_emp_gauss": g["ratio_emp_over_mean"], "ratio_emp_t": t["ratio_emp_over_mean"],
                    "n_t_ge_emp": t["n_ge_emp"], "p_t": t["p_mc"], "mismatch_t": t["R_mismatch_mean_abs"],
                    "mismatch_gauss": g["R_mismatch_mean_abs"], "nu_sens": v["nu_fit_R_obs"]["nu"],
                    "ratio_emp_t_sens": s["ratio_emp_over_mean"], "p_t_sens": s["p_mc"], "mismatch_t_sens": s["R_mismatch_mean_abs"]})
    pd.DataFrame(tab).round(4).to_csv(os.path.join(OUT, "tcopula_reference_groups_table.csv"), index=False)
    pd.concat(rows).to_csv(os.path.join(OUT, "tcopula_reference_groups_shares.csv"), index=False)
    with open(os.path.join(OUT, "tcopula_reference_groups_results.json"), "w") as f:
        json.dump({"design": {"B": B, "vix_terciles": [round(float(v), 2) for v in q]}, **res}, f, indent=1)
    print(pd.DataFrame(tab).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
