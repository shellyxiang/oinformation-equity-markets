"""t-copula reference for the full-sample local-synergy share (robustness diagnostic, 2026-11-07).

Question: is the empirical share of days with omega_t < 0 (9.86%) reproduced by a heavy-tailed elliptical
(Student-t copula) benchmark with the same correlation structure, i.e. by tail / scale-mixture dependence?

Data and pipeline: identical to final_inference C5 (same 20 markets, T = 3792, tie-safe jitter, rank -> normal
scores, Pearson R of the scores, local O-information under N(0, R), share(omega_t < 0)).

Estimation of the t copula (two-step, standard for elliptical copulas):
  P_tau : P_ij = sin(pi * tau_ij / 2), tau = Kendall's tau of the returns (Lindskog, McNeil & Schmock 2003).
          This is the consistent estimator of the t-copula's correlation (shape) matrix: for an elliptical copula
          tau = (2/pi) arcsin(P_ij) for every nu, while the normal-score Pearson R_hat is NOT the t-copula's P.
  nu    : maximum pseudo-likelihood with P fixed at P_tau; pseudo-observations u = rank/(T+1) of the jittered
          returns (Genest, Ghoudi & Rivest 1995). 95% interval from the profile log-likelihood (LR, chi2_1).
Sensitivity: P = R_hat (the baseline normal-score correlation), nu re-estimated by the same pseudo-likelihood.

Reference: 999 samples of T x 20 from t_nu(0, P) -> the SAME normal-score transform -> R re-estimated ->
local omega -> share(<0). Gaussian reference re-generated with the C5 seed (20261108) for replication-level
comparison; it must reproduce the saved C5 mean and band exactly.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from entropy_estimators import correlation_matrix, gaussian_copula_transform  # noqa: E402
from routeC_local_oinfo import local_omega  # noqa: E402

OUT = os.path.join(h1.ROOT, "raw_audit")
B = int(os.environ.get("FI_B", 999))
SEED_GAUSS, SEED_T, SEED_T_RHAT = 20261108, 20261112, 20261113


def share(x):
    z = gaussian_copula_transform(x)
    R = correlation_matrix(z)
    return float((local_omega(z, R) < 0).mean()), R


def kendall_P(x):
    n = x.shape[1]
    tau = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            tau[i, j] = tau[j, i] = stats.kendalltau(x[:, i], x[:, j]).statistic
    P = np.sin(np.pi * tau / 2)
    np.fill_diagonal(P, 1.0)
    return P, tau


def t_copula_loglik(nu, U, P):
    d = U.shape[1]
    x = stats.t.ppf(U, nu)
    Pinv = np.linalg.inv(P)
    q = np.einsum("ti,ij,tj->t", x, Pinv, x)
    logdet = np.linalg.slogdet(P)[1]
    lmv = (special.gammaln((nu + d) / 2) - special.gammaln(nu / 2) - 0.5 * d * np.log(nu * np.pi)
           - 0.5 * logdet - 0.5 * (nu + d) * np.log1p(q / nu))
    return float((lmv - stats.t.logpdf(x, nu).sum(1)).sum())


def fit_nu(U, P):
    f = lambda v: -t_copula_loglik(v, U, P)
    r = optimize.minimize_scalar(f, bounds=(2.05, 200.0), method="bounded", options={"xatol": 1e-4})
    nu, lmax = float(r.x), -float(r.fun)
    g = lambda v: 2 * (lmax - t_copula_loglik(v, U, P)) - stats.chi2.ppf(0.95, 1)
    lo = optimize.brentq(g, 2.05, nu) if g(2.05) > 0 else 2.05
    hi = optimize.brentq(g, nu, 200.0) if g(200.0) > 0 else 200.0
    lgauss = t_copula_loglik(1e6, U, P)  # nu -> infinity: Gaussian copula with the same P
    return {"nu": round(nu, 3), "profile_95": [round(lo, 3), round(hi, 3)], "loglik": round(lmax, 2),
            "loglik_gauss_same_P": round(lgauss, 2), "LR_vs_gauss": round(2 * (lmax - lgauss), 1)}


def simulate(rng, P, nu, T, d):
    L = np.linalg.cholesky(P)
    return (rng.standard_normal((T, d)) @ L.T) / np.sqrt(rng.chisquare(nu, size=(T, 1)) / nu)


def run_ref(rng, gen, Rhat, label):
    sh, mad = np.empty(B), np.empty(B)
    iu = np.triu_indices(Rhat.shape[0], 1)
    t0 = time.time()
    for b in range(B):
        sh[b], Rb = share(gen(rng))
        mad[b] = np.abs(Rb[iu] - Rhat[iu]).mean()
    print(f"{label}: {time.time() - t0:.0f}s", flush=True)
    return sh, mad


def summ(sh, emp):
    n_ge = int((sh >= emp).sum())
    return {"mean": round(float(sh.mean()), 5), "band_95": [round(float(v), 5) for v in np.percentile(sh, [2.5, 97.5])],
            "min": round(float(sh.min()), 5), "max": round(float(sh.max()), 5), "sd": round(float(sh.std(ddof=1)), 5),
            "ratio_emp_over_mean": round(emp / float(sh.mean()), 3), "n_ge_emp": n_ge, "p_mc": round((1 + n_ge) / (B + 1), 4)}


def main():
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    xj = h1.apply_tie_safe_jitter(df[mk].to_numpy(float), seed=h1.TIE_JITTER_SEED)
    T, d = xj.shape
    emp, Rhat = share(xj)
    with open(os.path.join(OUT, "final_inference_results.json")) as f:
        c5 = json.load(f)["C5"]
    assert abs(emp - c5["empirical"]) < 1e-4, (emp, c5["empirical"])
    U = (stats.rankdata(xj, axis=0)) / (T + 1)
    P_tau, tau = kendall_P(xj)
    assert np.linalg.eigvalsh(P_tau).min() > 0
    iu = np.triu_indices(d, 1)
    fit_tau, fit_R = fit_nu(U, P_tau), fit_nu(U, Rhat)
    print("fit P_tau", fit_tau, "\nfit R_hat", fit_R, flush=True)

    Lg = np.linalg.cholesky(Rhat)
    gauss, mad_g = run_ref(np.random.default_rng(SEED_GAUSS), lambda r: r.standard_normal((T, d)) @ Lg.T, Rhat, "gauss")
    assert abs(gauss.mean() - c5["gauss_ref_mean"]) < 1e-4, (gauss.mean(), c5["gauss_ref_mean"])
    assert np.allclose(np.percentile(gauss, [2.5, 97.5]), c5["gauss_ref_95"], atol=1e-4)
    t_main, mad_t = run_ref(np.random.default_rng(SEED_T), lambda r: simulate(r, P_tau, fit_tau["nu"], T, d), Rhat, "t P_tau")
    t_rhat, mad_tr = run_ref(np.random.default_rng(SEED_T_RHAT), lambda r: simulate(r, Rhat, fit_R["nu"], T, d), Rhat,
                             "t R_hat")

    ks = stats.ks_2samp(t_main, gauss)
    res = {
        "design": {"T": T, "markets": d, "B": B, "seeds": {"gauss": SEED_GAUSS, "t_P_tau": SEED_T, "t_R_hat": SEED_T_RHAT},
                   "pipeline": "rank -> normal scores -> Pearson R -> local omega under N(0,R) -> share(<0) (= C5)"},
        "empirical_share": round(emp, 5),
        "correlation_inputs": {"mean_offdiag_R_hat": round(float(Rhat[iu].mean()), 4),
                               "mean_offdiag_P_tau": round(float(P_tau[iu].mean()), 4),
                               "mean_abs_diff_P_tau_vs_R_hat": round(float(np.abs(P_tau[iu] - Rhat[iu]).mean()), 4),
                               "max_abs_diff_P_tau_vs_R_hat": round(float(np.abs(P_tau[iu] - Rhat[iu]).max()), 4)},
        "nu_fit_P_tau": fit_tau, "nu_fit_R_hat": fit_R,
        "gaussian_reference": summ(gauss, emp),
        "t_reference_P_tau": summ(t_main, emp),
        "t_reference_R_hat_sensitivity": summ(t_rhat, emp),
        "re_estimated_R_vs_R_hat_mean_abs_diff": {"gauss": round(float(mad_g.mean()), 4), "t_P_tau": round(float(mad_t.mean()), 4),
                                                  "t_R_hat": round(float(mad_tr.mean()), 4)},
        "t_vs_gauss": {"diff_means_pp": round(100 * float(t_main.mean() - gauss.mean()), 3),
                       "ks_stat": round(float(ks.statistic), 4), "ks_p": float(ks.pvalue),
                       "share_t_draws_above_gauss_p97.5": round(float((t_main > np.percentile(gauss, 97.5)).mean()), 4)},
    }
    pd.DataFrame({"rep": np.arange(1, B + 1), "share_gauss": gauss, "share_t_P_tau": t_main, "share_t_R_hat": t_rhat,
                  "mad_R_gauss": mad_g, "mad_R_t_P_tau": mad_t, "mad_R_t_R_hat": mad_tr}).to_csv(
        os.path.join(OUT, "tcopula_reference_shares.csv"), index=False)
    with open(os.path.join(OUT, "tcopula_reference_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
