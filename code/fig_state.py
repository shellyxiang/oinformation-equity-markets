"""Figure 5 for Section 3.4 (local-synergy share by VIX state and crisis window).

(a) VIX states: observed share under full-sample normalisation (open, PRIMARY series since 2026-11-08; = final_inference C4)
    and, as a secondary sensitivity comparison, the within-state (state-calibrated, variant C) share (filled), with the state-specific Gaussian reference mean and 2.5-97.5% range
    (final_state_gauss_ref_Rc.json: reference drawn from the within-state R_c, matched to variant C; 2026-11-06).
    Open markers are taken from final_inference C4 (identical to A_existing).
(b) GFC / COVID / non-crisis shares (full-sample normalisation) with the saved 95% stationary-bootstrap intervals
    (final_inference C4; mean block 20, B = 999). Not recalibrated within windows.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_triplet_matched_null_H1_rankreordered as h1  # noqa: E402

OUT = os.path.join(h1.ROOT, "raw_audit")
FIG = os.path.join(h1.ROOT, "figures")
INK, MUTED, GRID, ORANGE = "#1f1f1f", "#6b6b6b", "#e4e4e2", "#eb6834"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 300, "legend.frameon": False})


def main():
    with open(os.path.join(OUT, "final_state_gauss_ref_Rc.json")) as f:
        sg = json.load(f)
    assert sg["reference_matrix"] == "Rc" and sg["B"] == 999
    with open(os.path.join(OUT, "final_inference_results.json")) as f:
        c4 = json.load(f)["C4"]
    # final values (Table 6) -- the figure must match the saved outputs
    exp = {"VIX_low": (1264, 0.0601, 0.1392, 0.1030, [0.0854, 0.1218], 1.35),
           "VIX_mid": (1263, 0.0879, 0.1029, 0.0640, [0.0507, 0.0792], 1.61),
           "VIX_high": (1265, 0.1478, 0.0719, 0.0229, [0.0142, 0.0324], 3.14)}
    for s, (n, op, ob, rm, band, rt) in exp.items():
        d, c = sg[s], sg[s]["observed"]["C_within_state"]
        assert d["T"] == n and abs(c4["by_state"][s]["est"] - op) < 1e-4 and abs(d["observed"]["A_existing"]["share"] - op) < 1e-4
        assert abs(c["share"] - ob) < 1e-4 and abs(d["ref_mean"] - rm) < 1e-4 and round(c["ratio"], 2) == rt
        assert all(abs(x - y) < 1e-4 for x, y in zip(d["ref_95"], band))
    for s, v in {"GFC": 0.2973, "COVID": 0.1667, "non_crisis": 0.0892}.items():
        assert abs(c4["by_state"][s]["est"] - v) < 1e-4
    assert abs(c4["share_all"] - 0.0986) < 1e-4
    df = pd.read_csv(h1.RETURNS_PATH, encoding="utf-8-sig", parse_dates=["Date"], usecols=["Date"])
    n_win = {"GFC": int(df.Date.between("2008-09-01", "2009-06-30").sum()),
             "COVID": int(df.Date.between("2020-02-15", "2020-06-30").sum())}
    n_win["non_crisis"] = len(df) - n_win["GFC"] - n_win["COVID"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.30})
    states = [("VIX_low", "Low VIX"), ("VIX_mid", "Intermediate VIX"), ("VIX_high", "High VIX")]
    # 2026-11-08 revision: full-sample normalisation is the primary series (main comparison); the state-specific
    # recalibration (observed + Gaussian reference) is a secondary sensitivity comparison. Ratio labels removed.
    LIGHT_ORANGE, LIGHT_GREY = "#f2b08f", "#b3b3b1"
    x_main, x_sens = [i - 0.10 for i in range(3)], [i + 0.22 for i in range(3)]
    main_y = [100 * c4["by_state"][s]["est"] for s, _ in states]
    sens_y = [100 * sg[s]["observed"]["C_within_state"]["share"] for s, _ in states]
    a.plot(x_sens, sens_y, color=LIGHT_ORANGE, lw=0.9, ls="--", zorder=2)
    for i, (s, _) in enumerate(states):
        d = sg[s]
        lo, hi = d["ref_95"]
        a.plot([x_sens[i]] * 2, [100 * lo, 100 * hi], color=LIGHT_GREY, lw=2.0, solid_capstyle="butt", zorder=2)
        a.plot(x_sens[i], 100 * d["ref_mean"], marker="D", color=LIGHT_GREY, ms=4, ls="none", zorder=3)
        a.plot(x_sens[i], sens_y[i], marker="o", color=LIGHT_ORANGE, ms=5, ls="none", zorder=3)
    a.plot(x_main, main_y, color=ORANGE, lw=2.0, zorder=4)
    a.plot(x_main, main_y, marker="o", mfc="white", mec=ORANGE, mew=2.0, ms=8.5, ls="none", zorder=5)
    for xm, ym in zip(x_main, main_y):
        a.text(xm - 0.13, ym, f"{ym:.1f}%", fontsize=7.5, color=INK, va="center", ha="right", fontweight="bold")
    a.set_xticks(range(3), [f"{lab}\n(n = {sg[s]['T']:,})" for s, lab in states], fontsize=7.5)
    a.tick_params(axis="x", length=0)
    a.grid(axis="x", visible=False)
    a.set_xlim(-0.6, 2.5)
    a.set_ylim(0, 21)
    a.set_ylabel(r"share of days with $\omega_t<0$ (%)")
    a.legend(handles=[plt.Line2D([], [], marker="o", mfc="white", mec=ORANGE, mew=2.0, ms=8, color=ORANGE, lw=2.0,
                                 label="full-sample normalisation (main)"),
                      plt.Line2D([], [], marker="o", color=LIGHT_ORANGE, ms=5, lw=0.9, ls="--",
                                 label="state-recalibrated observed (sensitivity)"),
                      plt.Line2D([], [], marker="D", color=LIGHT_GREY, ms=4, lw=2.0,
                                 label="state Gaussian reference (sensitivity)")],
             loc="upper left", fontsize=6.8)
    a.set_title("(a) VIX states", loc="left", fontsize=9, color=INK)

    wins = [("GFC", "GFC"), ("COVID", "COVID-19"), ("non_crisis", "Non-crisis")]
    for i, (s, _) in enumerate(wins):
        v, (lo, hi) = c4["by_state"][s]["est"], c4["by_state"][s]["ci"]
        b.errorbar(i, 100 * v, yerr=[[100 * (v - lo)], [100 * (hi - v)]], fmt="o", mfc="white", mec=ORANGE, mew=1.6,
                   ms=7, ecolor=ORANGE, elinewidth=1.4, capsize=3)
        b.text(i + 0.12, 100 * v, f"{100 * v:.1f}%", fontsize=7, color=INK, va="center")
    b.axhline(100 * c4["share_all"], color=MUTED, lw=1, ls="--")
    b.text(-0.45, 100 * c4["share_all"] + 0.6, f"full sample {100 * c4['share_all']:.1f}%", fontsize=7, color=MUTED, ha="left")
    b.set_xticks(range(3), [f"{lab}\n(n = {n_win[s]:,})" for s, lab in wins], fontsize=7.5)
    b.tick_params(axis="x", length=0)
    b.grid(axis="x", visible=False)
    b.set_xlim(-0.5, 2.5)
    b.set_ylim(0, 40)
    b.set_ylabel(r"share of days with $\omega_t<0$ (%)")
    b.set_title("(b) Crisis windows (full-sample normalisation)", loc="left", fontsize=9, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig5_state.{ext}"), bbox_inches="tight")
    plt.close(fig)
    assert n_win == {"GFC": 148, "COVID": 66, "non_crisis": 3578}, n_win
    print(n_win)


if __name__ == "__main__":
    main()
