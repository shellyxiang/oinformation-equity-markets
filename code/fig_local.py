"""Figure 4 for Section 3.4 (local O-information and the same-correlation Gaussian reference).

(a) Histogram of daily local O-information omega_t (20 markets, full-sample GC); zero, mean, median; synergy share shaded.
(b) Distribution of share(omega_t < 0) over the 999 same-correlation Gaussian reference draws (reproduces final_inference
    C5: seed 20261108); observed share with its 95% stationary-bootstrap interval (C4).
(c) Observed share with 95% bootstrap interval and Gaussian-reference 95% band for ALL20, EU, AS, AM.
    Regional bands from final_regional_gauss_ref.json. Regional bootstrap intervals reuse the final_inference bootstrap
    index draws (seed 20261106, mean block 20, B = 999; normal scores fixed, R re-estimated per replicate);
    the global interval is checked against C4.
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
from final_inference import stat_boot_idx  # noqa: E402
from routeC_local_oinfo import REG, local_omega  # noqa: E402

B = 999
OUT = os.path.join(h1.ROOT, "raw_audit")
FIG = os.path.join(h1.ROOT, "figures")
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NAMES = {"ALL20": "Global\n(20)", "EU": "Europe\n(9)", "AS": "Asia-Pacific\n(7)", "AM": "Americas\n(3)"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 300, "legend.frameon": False})


def draws(z, cols):
    path = os.path.join(OUT, "fig4_draws.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["gref"], d["boot"]
    T = len(z)
    rng = np.random.default_rng(20261108)
    L = np.linalg.cholesky(correlation_matrix(z))
    gref = []
    for _ in range(B):
        zg = gaussian_copula_transform(rng.standard_normal((T, 20)) @ L.T)
        gref.append(float((local_omega(zg, correlation_matrix(zg)) < 0).mean()))
    rng = np.random.default_rng(20261106)
    boot = []
    for b in range(B):
        zb = z[stat_boot_idx(T, rng)]
        boot.append([float((local_omega(zb[:, c], correlation_matrix(zb[:, c])) < 0).mean()) for c in cols.values()])
        if (b + 1) % 200 == 0:
            print("boot", b + 1, flush=True)
    gref, boot = np.array(gref), np.array(boot)
    np.savez(path, gref=gref, boot=boot)
    return gref, boot


def main():
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"])
    mk = [c for c in df.columns if c != "Date"]
    z = gaussian_copula_transform(h1.apply_tie_safe_jitter(df[mk].to_numpy(float), seed=h1.TIE_JITTER_SEED))
    cols = {"ALL20": np.arange(20)} | {r: np.array([mk.index(m) for m in ms]) for r, ms in REG.items()}
    w = local_omega(z, correlation_matrix(z))
    obs = {r: float((local_omega(z[:, c], correlation_matrix(z[:, c])) < 0).mean()) for r, c in cols.items()}
    gref, boot = draws(z, cols)
    with open(os.path.join(OUT, "final_inference_results.json")) as f:
        fi = json.load(f)
    with open(os.path.join(OUT, "final_regional_gauss_ref.json")) as f:
        rg = json.load(f)
    assert abs(gref.mean() - fi["C5"]["gauss_ref_mean"]) < 1e-4, "Gaussian reference does not reproduce C5"
    bci = np.percentile(boot, [2.5, 97.5], axis=0).T
    assert np.allclose(np.round(bci[0], 4), fi["C4"]["ci"], atol=1e-4), "bootstrap does not reproduce C4"
    for r in REG:
        assert abs(obs[r] - rg[r]["empirical"]) < 1e-4
    band = {"ALL20": [float(v) for v in np.percentile(gref, [2.5, 97.5])], **{r: rg[r]["ref_95"] for r in REG}}
    refm = {"ALL20": float(gref.mean()), **{r: rg[r]["ref_mean"] for r in REG}}

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.45, wspace=0.28)
    a, b, c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    mean, med, share = float(w.mean()), float(np.median(w)), obs["ALL20"]
    lo, hi = np.percentile(w, [0.1, 99.5])
    bins = np.linspace(min(lo, -5), hi, 90)
    n, e, patches = a.hist(w, bins=bins, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.3)
    for p, left in zip(patches, e[:-1]):
        if left < 0:
            p.set_facecolor(ORANGE)
            p.set_alpha(0.9)
    a.axvline(0, color=INK, lw=1.2)
    a.axvline(mean, color=INK, lw=1.1, ls="--")
    a.axvline(med, color=INK, lw=1.1, ls=":")
    ytop = n.max()
    a.set_ylim(0, ytop * 1.22)
    a.text(mean + 0.4, ytop * 1.10, f"mean {mean:.2f}", fontsize=7.5, color=INK, va="center")
    a.text(med - 0.4, ytop * 1.10, f"median {med:.2f}", fontsize=7.5, color=INK, ha="right", va="center",
           bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    a.text(-0.4, ytop * 0.45, rf"$\omega_t<0$: {100 * share:.1f}% of days", fontsize=7.5, color=ORANGE, ha="right")
    a.set_xlabel(r"local O-information $\omega_t$ (nats)")
    a.set_ylabel("days")
    a.set_title("(a) Daily local O-information, 20 markets", loc="left", fontsize=9, color=INK)
    n_clip = int((w > hi).sum() + (w < bins[0]).sum())

    b.hist(100 * gref, bins=30, color=MUTED, alpha=0.6, edgecolor="white", linewidth=0.3)
    yb = b.get_ylim()[1]
    b.axvline(100 * share, color=ORANGE, lw=1.6)
    b.plot(100 * bci[0], [yb * 0.85] * 2, color=ORANGE, lw=2.2, solid_capstyle="butt")
    b.text(100 * share, yb * 0.90, f"observed {100 * share:.1f}%", fontsize=7.5, color=INK, ha="center", va="bottom")
    b.text(100 * gref.mean(), yb * 1.02, f"reference\nmean {100 * gref.mean():.1f}%", fontsize=7, color=MUTED, ha="center", va="bottom")
    b.set_ylim(0, yb * 1.22)
    b.set_xlabel(r"share of days with $\omega_t<0$ (%)")
    b.set_ylabel("replications")
    b.set_title("(b) Gaussian reference, 20 markets", loc="left", fontsize=9, color=INK)

    order = ["ALL20", "EU", "AS", "AM"]
    for i, r in enumerate(order):
        c.fill_between([i - 0.3, i + 0.3], 100 * band[r][0], 100 * band[r][1], color=MUTED, alpha=0.25, lw=0)
        c.plot([i - 0.3, i + 0.3], [100 * refm[r]] * 2, color=MUTED, lw=1)
        c.errorbar(i, 100 * obs[r], yerr=[[100 * (obs[r] - bci[i, 0])], [100 * (bci[i, 1] - obs[r])]], fmt="o",
                   color=ORANGE, ms=5, elinewidth=1.4, capsize=3, zorder=3)
        c.text(i, 100 * max(bci[i, 1], band[r][1]) + 0.8, f"{obs[r] / refm[r]:.2f}×", fontsize=7.5, color=INK, ha="center")
    c.set_xticks(range(4), [NAMES[r] for r in order], fontsize=7.5)
    c.tick_params(axis="x", length=0)
    c.grid(axis="x", visible=False)
    c.set_xlim(-0.5, 3.5)
    c.set_ylim(0, 36)
    c.set_ylabel(r"share of days with $\omega_t<0$ (%)")
    c.set_title("(c) Observed vs Gaussian reference", loc="left", fontsize=9, color=INK)
    c.legend(handles=[plt.Line2D([], [], marker="o", color=ORANGE, ls="none", label="observed (95% CI)"),
                      plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.25, label="reference 95% band")],
             loc="center left", fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig4_local.{ext}"), bbox_inches="tight")
    plt.close(fig)

    res = {"mean": round(mean, 4), "median": round(med, 4), "share": round(share, 4), "hist_range": [round(float(bins[0]), 2), round(float(hi), 2)],
           "n_outside_hist": n_clip, "min": round(float(w.min()), 2), "max": round(float(w.max()), 2),
           "gref_mean": round(float(gref.mean()), 4), "gref_95": [round(v, 4) for v in band["ALL20"]],
           "observed": {r: round(v, 4) for r, v in obs.items()},
           "boot_ci": {r: [round(float(x), 4) for x in bci[i]] for i, r in enumerate(cols)},
           "ratio": {r: round(obs[r] / refm[r], 3) for r in order}}
    with open(os.path.join(OUT, "fig4_local_summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
