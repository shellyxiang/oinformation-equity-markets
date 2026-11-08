"""Figures 1-2 for Section 3.1 (global redundancy, system-size scaling, first-order gradients).

Figure 1  left : mean Omega(S_k) vs k with 5-95% subset quantile band and bootstrap 95% CI of the mean;
                 circular-shift surrogate null (mean Omega(S_k), 999 surrogates) at k = 3, 10, 20 near zero (inset x1e4).
          right: local exponent d ln E[Omega] / d ln k between adjacent k, bootstrap 95% CI; horizontal line beta = 2.12.
Figure 2  left : horizontal bars of dOmega_j (full-system GC gradient), coloured by region.
          right: dOmega_j vs average pairwise correlation.

Bootstrap and surrogate draws reproduce final_inference.py exactly (same seeds, same rng call order):
bootstrap seed 20261106 (stationary, block 20), surrogates seed 20261107. Draws are cached in raw_audit/fig1_draws.npz.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402
from entropy_estimators import correlation_matrix, gaussian_copula_transform  # noqa: E402
from final_inference import Curve, omega_batch, stat_boot_idx  # noqa: E402
from quick_omega_system_size import KS, subsets  # noqa: E402

B = 999
OUT = os.path.join(h1.ROOT, "raw_audit")
FIG = os.path.join(h1.ROOT, "figures")
REGION = {"Europe": ["AEX", "CAC40", "DAX", "FTSE100", "FTSEMIB", "IBEX35", "OMXS30", "OBX", "SMI"],
          "Asia-Pacific": ["HangSeng", "LQ45", "KOSPI200", "Nifty50", "ASX200", "CSI300", "TWSE"],
          "Americas": ["SP500", "IPC", "TSX60"], "South Africa": ["SATop40"]}
COL = {"Europe": "#2a78d6", "Asia-Pacific": "#eb6834", "Americas": "#1baf7a", "South Africa": "#6b6b6b"}
MRK = {"Europe": "o", "Asia-Pacific": "s", "Americas": "^", "South Africa": "D"}
LABEL = {"CAC40": "CAC 40", "FTSE100": "FTSE 100", "FTSEMIB": "FTSE MIB", "IBEX35": "IBEX 35", "SP500": "S&P 500",
         "HangSeng": "Hang Seng", "KOSPI200": "KOSPI 200", "Nifty50": "Nifty 50", "ASX200": "ASX 200", "CSI300": "CSI 300",
         "TSX60": "TSX 60", "SATop40": "SA Top 40"}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e2"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 300, "legend.frameon": False})


def draws(z, cur):
    path = os.path.join(OUT, "fig1_draws.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["boot"], d["null"]
    T = len(z)
    rng = np.random.default_rng(20261107)
    null = []
    for _ in range(B):
        zs = np.column_stack([np.roll(z[:, j], rng.integers(T)) for j in range(20)])
        m = cur.means(correlation_matrix(zs), (3, 10, 20))
        null.append([m[3], m[10], m[20]])
    rng = np.random.default_rng(20261106)
    boot = []
    for b in range(B):
        m = cur.means(correlation_matrix(z[stat_boot_idx(T, rng)]))
        boot.append([m[k] for k in KS])
        if (b + 1) % 200 == 0:
            print("boot", b + 1, flush=True)
    boot, null = np.array(boot), np.array(null)
    np.savez(path, boot=boot, null=null)
    return boot, null


def local_exp(m):
    k = np.array(KS, float)
    return np.diff(np.log(m), axis=-1) / np.diff(np.log(k))


def figure1(R, cur, boot, null, beta, beta_ci):
    k = np.array(KS)
    allv = {kk: omega_batch(R, cur.S[kk]) for kk in KS}
    mean = np.array([allv[kk].mean() for kk in KS])
    q05, q95 = np.array([np.percentile(allv[kk], [5, 95]) for kk in KS]).T
    blo, bhi = np.percentile(boot, [2.5, 97.5], axis=0)
    le, le_b = local_exp(mean), local_exp(boot)
    llo, lhi = np.percentile(le_b, [2.5, 97.5], axis=0)
    kmid = np.sqrt(k[:-1] * k[1:])

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.32})
    a.fill_between(k, q05, q95, color=COL["Europe"], alpha=0.16, lw=0, label="5–95% of subsets")
    a.fill_between(k, blo, bhi, color=COL["Europe"], alpha=0.40, lw=0, label="95% bootstrap CI of mean")
    a.plot(k, mean, color=COL["Europe"], lw=2, marker="o", ms=3.5, label=r"mean $\Omega(S_k)$")
    a.axhline(0, color=MUTED, lw=0.8)
    a.plot([3, 10, 20], [0, 0, 0], ls="none", marker="x", ms=6, mew=1.4, color=INK, label="surrogate null (k = 3, 10, 20)")
    a.set_xlabel("system size $k$")
    a.set_ylabel(r"$\Omega(S_k)$ (nats)")
    a.set_ylim(-0.4, 7.6)
    a.set_xticks([3, 5, 10, 15, 20])
    a.set_xlim(2.5, 20.5)
    a.legend(loc="upper left", fontsize=7, handlelength=1.6)
    ins = a.inset_axes([0.62, 0.13, 0.35, 0.17])
    ins.boxplot(null * 1e4, positions=[0, 1, 2], widths=0.5, whis=(2.5, 97.5), showfliers=False,
                medianprops={"color": INK, "lw": 1}, boxprops={"color": MUTED}, whiskerprops={"color": MUTED},
                capprops={"color": MUTED})
    ins.set_xticks([0, 1, 2], ["3", "10", "20"], fontsize=6.5)
    ins.tick_params(labelsize=6, length=2, pad=1)
    ins.set_ylabel(r"null $\times10^{4}$", fontsize=6.5, color=MUTED, labelpad=1)
    ins.grid(False)
    ins.axhline(0, color=GRID, lw=0.6)
    a.set_title("(a) Redundancy vs system size", loc="left", fontsize=9, color=INK)

    b.axhspan(*beta_ci, color=MUTED, alpha=0.18, lw=0)
    b.axhline(beta, color=MUTED, lw=1.2, ls="--")
    b.text(19.8, beta + 0.06, rf"global fit $\beta$ = {beta:.2f}", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    b.fill_between(kmid, llo, lhi, color=COL["Asia-Pacific"], alpha=0.30, lw=0, label="95% bootstrap CI")
    b.plot(kmid, le, color=COL["Asia-Pacific"], lw=2, marker="o", ms=3.5, label="local exponent")
    b.set_xlabel("system size $k$ (midpoint of $k$, $k$+1)")
    b.set_ylabel(r"$\Delta \ln E[\Omega] \,/\, \Delta \ln k$")
    b.set_xticks([3, 5, 10, 15, 20])
    b.set_xlim(2.5, 20.5)
    b.legend(loc="upper right", fontsize=7.5)
    b.set_title("(b) Local scaling exponent", loc="left", fontsize=9, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig1_scaling.{ext}"), bbox_inches="tight")
    plt.close(fig)
    pair = lambda i, j: float(np.log(mean[j] / mean[i]) / np.log(k[j] / k[i]))
    return {"local_exponent": {f"{k[i]}-{k[i + 1]}": [round(float(le[i]), 3), round(float(llo[i]), 3), round(float(lhi[i]), 3)]
                               for i in range(len(le))},
            "exp_3_5": round(pair(0, 2), 3), "exp_5_10": round(pair(2, 7), 3), "exp_10_15": round(pair(7, 12), 3),
            "exp_15_20": round(pair(12, 17), 3),
            "null_95": {kk: [float(np.percentile(null[:, i], 2.5)), float(np.percentile(null[:, i], 97.5))]
                        for i, kk in enumerate((3, 10, 20))},
            "null_absmax": {kk: float(np.abs(null[:, i]).max()) for i, kk in enumerate((3, 10, 20))},
            "subset_q05_q95": {int(kk): [round(float(q05[i]), 4), round(float(q95[i]), 4)] for i, kk in enumerate(KS)},
            "min_subset_omega": float(min(v.min() for v in allv.values()))}


def figure2():
    g = pd.read_csv(os.path.join(OUT, "final_gc_gradient_markets.csv"))
    reg = {m: r for r, ms in REGION.items() for m in ms}
    g["region"] = g.market.map(reg)
    g["name"] = g.market.map(lambda m: LABEL.get(m, m))
    g = g.sort_values("dOmega")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.30})
    a.barh(g.name, g.dOmega, color=g.region.map(COL), height=0.72, edgecolor="white", linewidth=1)
    for y, v in enumerate(g.dOmega):
        a.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=6.5, color=MUTED)
    a.set_xlabel(r"$\Delta\Omega_j = \Omega(X) - \Omega(X_{-j})$ (nats)")
    a.grid(axis="y", visible=False)
    a.tick_params(axis="y", length=0, labelsize=7.5, labelcolor=INK)
    a.set_xlim(0, 1.18)
    a.legend(handles=[Patch(color=COL[r], label=r) for r in REGION], loc="lower right", fontsize=7.5)
    a.set_title("(a) First-order O-information gradient", loc="left", fontsize=9, color=INK)
    for r in REGION:
        s = g[g.region == r]
        b.scatter(s.avg_corr, s.dOmega, s=34, marker=MRK[r], color=COL[r], edgecolor="white", linewidth=1, label=r, zorder=3)
    for m in ("CAC40", "CSI300", "LQ45", "SATop40"):
        r = g[g.market == m].iloc[0]
        b.annotate(r["name"], (r.avg_corr, r.dOmega), xytext=(5, -3), textcoords="offset points", fontsize=6.5, color=MUTED)
    with open(os.path.join(OUT, "final_gc_gradient_summary.json")) as f:
        s = json.load(f)["vs_avg_corr"]
    b.text(0.03, 0.97, f"Spearman {s['spearman']:.2f}\nPearson {s['pearson']:.2f}", transform=b.transAxes, va="top", fontsize=7.5, color=INK)
    b.set_xlabel(r"average pairwise correlation of market $j$")
    b.set_ylabel(r"$\Delta\Omega_j$ (nats)")
    b.legend(handles=[Line2D([], [], ls="none", marker=MRK[r], color=COL[r], ms=6, label=r) for r in REGION],
             loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=7.5)
    b.set_title("(b) Gradient vs integration", loc="left", fontsize=9, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig2_gradient.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    z = gaussian_copula_transform(h1.apply_tie_safe_jitter(df[mk].to_numpy(float), seed=h1.TIE_JITTER_SEED))
    R = correlation_matrix(z)
    cur = Curve(subsets())
    boot, null = draws(z, cur)
    with open(os.path.join(OUT, "final_inference_results.json")) as f:
        c2 = json.load(f)["C2"]
    # consistency with final_inference.py
    ref = np.array([c2["mean_by_k"][str(kk)]["ci"] for kk in KS])
    got = np.round(np.percentile(boot, [2.5, 97.5], axis=0).T, 4)
    assert np.allclose(ref, got, atol=1e-4), "bootstrap draws do not reproduce final_inference"
    res = figure1(R, cur, boot, null, c2["power_exponent"]["est"], c2["power_exponent"]["ci"])
    figure2()
    with open(os.path.join(OUT, "fig1_scaling_summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
