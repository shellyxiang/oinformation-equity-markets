"""Figure 3 for Section 3.2 (deviation from the equicorrelated reference by size and composition).

(a) Mean relative deviation D_Omega = (Omega - Omega_eq) / Omega_eq (%) vs k.
    All subsets (black): system-size subset list (quick_omega_system_size.subsets, as in Section 3.1).
    By composition (single-region / two-region / three-or-more regions): Route A subsets (routeA_composition.build_subsets).
    95% stationary-bootstrap intervals at k = 3, 5, 10, 15, 20 only. Class intervals are read from final_inference C3;
    the all-subset intervals reproduce the final_inference bootstrap draws (seed 20261106; checked against C2 mean CIs).
(b) D_Omega vs SD(rho_ij) for the Route A subsets at k = 10, coloured by composition; R^2 = squared Pearson correlation
    (checked against final_inference C3 k10_R2_rhosd).
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from entropy_estimators import correlation_matrix, gaussian_copula_transform  # noqa: E402
from final_inference import Curve, omega_batch, omega_eq_vec, stat_boot_idx  # noqa: E402
from quick_omega_system_size import KS, subsets  # noqa: E402
from routeA_composition import build_subsets  # noqa: E402

B = 999
KB = (3, 5, 10, 15, 20)
OUT = os.path.join(h1.ROOT, "raw_audit")
FIG = os.path.join(h1.ROOT, "figures")
CLS = {"R1": "single-region", "R2": "two regions", "R3+": "three or more regions"}
COL = {"R1": "#2a78d6", "R2": "#eb6834", "R3+": "#1baf7a", "all": "#1f1f1f"}
MRK = {"R1": "o", "R2": "s", "R3+": "^", "all": "o"}
DX = {"all": 0.0, "R1": -0.25, "R2": 0.25, "R3+": -0.25}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e2"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 300, "legend.frameon": False})


def stats(R, S, k):
    om = omega_batch(R, S)
    iu = np.triu_indices(k, 1)
    P = R[S[:, :, None], S[:, None, :]][:, iu[0], iu[1]]
    oe = omega_eq_vec(k, P.mean(1))
    return om, (om - oe) / oe, P.std(1)


def all_subset_boot(z, cur):
    path = os.path.join(OUT, "fig3_allsubset_boot.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["rel"], d["om"]
    rng = np.random.default_rng(20261106)
    rel, om = [], []
    for b in range(B):
        Rb = correlation_matrix(z[stat_boot_idx(len(z), rng)])
        r = [stats(Rb, cur.S[k], k) for k in KB]
        rel.append([x[1].mean() for x in r])
        om.append([x[0].mean() for x in r])
        if (b + 1) % 200 == 0:
            print("boot", b + 1, flush=True)
    rel, om = np.array(rel), np.array(om)
    np.savez(path, rel=rel, om=om)
    return rel, om


def main():
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    z = gaussian_copula_transform(h1.apply_tie_safe_jitter(df[mk].to_numpy(float), seed=h1.TIE_JITTER_SEED))
    R = correlation_matrix(z)
    with open(os.path.join(OUT, "final_inference_results.json")) as f:
        fi = json.load(f)
    c2, c3 = fi["C2"]["mean_by_k"], fi["C3"]

    # all subsets (system-size list)
    cur = Curve(subsets())
    all_pt = np.array([stats(R, cur.S[k], k)[1].mean() for k in KS])
    rel_b, om_b = all_subset_boot(z, cur)
    ref = np.array([c2[str(k)]["ci"] for k in KB])
    assert np.allclose(ref, np.percentile(om_b, [2.5, 97.5], axis=0).T, atol=1e-4), "bootstrap does not reproduce C2"
    all_ci = np.percentile(rel_b, [2.5, 97.5], axis=0).T

    # composition classes (Route A list)
    subsA, _ = build_subsets(mk)
    grp = {}
    for k, cls, s in subsA:
        grp.setdefault((k, cls), []).append(s)
    cls_pt, k10 = {}, {}
    for (k, cls), ss in grp.items():
        om, rel, sd = stats(R, np.array(ss), k)
        cls_pt[(k, cls)] = (rel.mean(), len(ss))
        if k == 10:
            k10[cls] = (rel, sd)
    chk = pd.read_csv(os.path.join(OUT, "routeA_composition_full.csv"))
    for _, r in chk.iterrows():
        assert abs(cls_pt[(r.k, r.cls)][0] - r.rel_dev) < 1e-3, (r.k, r.cls)
    rel10 = np.concatenate([v[0] for v in k10.values()])
    sd10 = np.concatenate([v[1] for v in k10.values()])
    r2 = float(np.corrcoef(rel10, sd10)[0, 1] ** 2)
    assert abs(r2 - c3["k10_R2_rhosd"]["est"]) < 1e-3

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.30})
    a.axhline(0, color=MUTED, lw=1, ls="--", zorder=1)
    ka = np.array(KS)
    a.plot(ka, 100 * all_pt, color=COL["all"], lw=2, marker="o", ms=3.5, label="all subsets", zorder=4)
    a.errorbar(np.array(KB) + DX["all"], 100 * all_pt[[KS.index(k) for k in KB]],
               yerr=100 * np.abs(all_ci.T - all_pt[[KS.index(k) for k in KB]]), fmt="none", ecolor=COL["all"],
               elinewidth=1.2, capsize=2.5, zorder=4)
    for cls in ("R1", "R2", "R3+"):
        ks = sorted(k for (k, c) in cls_pt if c == cls)
        y = np.array([cls_pt[(k, cls)][0] for k in ks]) * 100
        a.plot(np.array(ks) + DX[cls], y, color=COL[cls], lw=1.5, marker=MRK[cls], ms=3.5, label=CLS[cls], zorder=3)
        kb = [k for k in KB if f"k{k}_{cls}" in c3]
        if kb:
            pt = np.array([c3[f"k{k}_{cls}"]["est"] for k in kb]) * 100
            ci = np.array([c3[f"k{k}_{cls}"]["ci"] for k in kb]) * 100
            a.errorbar(np.array(kb) + DX[cls], pt, yerr=np.abs(ci.T - pt), fmt="none", ecolor=COL[cls],
                       elinewidth=1.2, capsize=2.5, zorder=3)
    a.set_xlabel("system size $k$")
    a.set_ylabel(r"$D_\Omega$ (%)")
    a.set_xticks([3, 5, 7, 10, 15, 20])
    a.set_xlim(2.3, 20.7)
    a.legend(loc="lower right", fontsize=7.5)
    a.set_title("(a) Deviation by size and composition", loc="left", fontsize=9, color=INK)

    for cls in ("R3+", "R2"):
        rel, sd = k10[cls]
        b.scatter(sd, 100 * rel, s=9, marker=MRK[cls], color=COL[cls], alpha=0.35, lw=0,
                  label=f"{CLS[cls]} (n = {len(rel):,})", rasterized=True)
    bb = np.polyfit(sd10, 100 * rel10, 1)
    xx = np.linspace(sd10.min(), sd10.max(), 50)
    b.plot(xx, np.polyval(bb, xx), color=INK, lw=1.2)
    b.text(0.03, 0.97, rf"$k$ = 10,  $R^2$ = {r2:.2f}", transform=b.transAxes, va="top", fontsize=8, color=INK)
    b.set_xlabel(r"$SD(\rho_{ij})$ within subset")
    b.set_ylabel(r"$D_\Omega$ (%)")
    leg = b.legend(loc="lower right", fontsize=7.5, markerscale=2)
    for h in leg.legend_handles:
        h.set_alpha(1)
    b.set_title("(b) Deviation vs correlation dispersion", loc="left", fontsize=9, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig3_composition.{ext}"), bbox_inches="tight")
    plt.close(fig)

    res = {"all_subsets_point": {int(k): round(float(v), 4) for k, v in zip(KS, all_pt)},
           "all_subsets_ci": {int(k): [round(float(x), 4) for x in c] for k, c in zip(KB, all_ci)},
           "k10_n": {c: int(len(v[0])) for c, v in k10.items()}, "k10_R2": round(r2, 4),
           "k10_fit_slope_pct_per_sd": round(float(bb[0]), 2)}
    with open(os.path.join(OUT, "fig3_composition_summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
