# -*- coding: utf-8 -*-
"""论文主图 (matplotlib → paper/figures/*.pdf)
fig_scatter: 精度-参数量散点 (PQ 与 mPQ+ 双面板)
fig_ablation: 消融柱状图 (PQ, mean±std)
fig_tissue: 逐组织 PQ 热图 (补充材料)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent
TAB = Path(r"c:\Users\HP\OneDrive\Desktop\Paper\paper\tables")
FIG = Path(r"c:\Users\HP\OneDrive\Desktop\Paper\paper\figures")
FIG.mkdir(parents=True, exist_ok=True)

facts = json.load(open(PKG.parent.parent / "paper" / "paper_facts.json", encoding="utf-8"))

CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}

METHODS = {  # run → (label, color, marker)
    "full":         ("Ours (full)", CB["red"], "*"),
    "full_nokd":    ("Ours (w/o KD)", CB["orange"], "D"),
    "own_baseline": ("Ours (base)", CB["orange"], "o"),
    "hovernet":     ("HoVer-Net", CB["blue"], "s"),
    "cellvit_sam256": ("CellViT-256-SAM", CB["green"], "^"),
    "cellvit_t256": ("CellViT-256", CB["green"], "v"),
    "nulite_s":     ("NuLite-S", CB["purple"], "P"),
    "nulite_t":     ("NuLite-T", CB["purple"], "X"),
    "ddbunet":      ("DDBU-Net", CB["grey"], "p"),
}
PARAMS = facts["params"]
P2P = {"full": "own_full", "full_nokd": "own_full", "own_baseline": "own_base",
       "hovernet": "hovernet", "cellvit_sam256": "cellvit_sam256",
       "cellvit_t256": "cellvit_t256", "nulite_s": "nulite_s",
       "nulite_t": "nulite_t", "ddbunet": "ddbunet"}


def fig_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, key, ylab in ((axes[0], "pq", "PQ (%)"),
                          (axes[1], "mpq_plus", "mPQ+ (%)")):
        for run, (lab, c, mk) in METHODS.items():
            x = PARAMS[P2P[run]] / 1e6
            y = facts["main"][run][key]
            std = facts["main_std"][run][key]
            size = 190 if run == "full" else 60
            ax.errorbar(x, y, yerr=std, fmt=mk, color=c, markersize=9 if run == "full" else 6,
                        capsize=2, lw=1, markeredgecolor="white", markeredgewidth=0.6,
                        label=lab, zorder=5 if run == "full" else 3)
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (M, log scale)")
        ax.set_ylabel(ylab)
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
        ax.set_xticks([5, 10, 30, 50, 150])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0].annotate("ours", xy=(PARAMS["own_full"]/1e6, facts["main"]["full"]["pq"]),
                     xytext=(8, -14), textcoords="offset points", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7.6, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(FIG / "fig_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_scatter.pdf")


def fig_ablation():
    rows = [("own_baseline", "Base"), ("abl_sg", "+SG"), ("abl_rab", "+RAB"),
            ("abl_bnd", "+Bnd"), ("abl_sgrab", "+SG+RAB"),
            ("abl_sgbnd", "+SG+Bnd"), ("abl_rabbnd", "+RAB+Bnd"),
            ("full_nokd", "Full\n(w/o KD)"), ("full", "Full\n(+BGKD)")]
    xs = np.arange(len(rows))
    means = [facts["main"][r]["pq"] for r, _ in rows]
    stds = [facts["main_std"][r]["pq"] for r, _ in rows]
    colors = [CB["orange"]] * 7 + [CB["blue"], CB["red"]]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    bars = ax.bar(xs, means, yerr=stds, capsize=3, color=colors, width=0.62,
                  error_kw=dict(lw=0.9))
    ax.axhline(means[0], color=CB["orange"], ls="--", lw=0.9, alpha=0.7)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.09, f"{m:.2f}", ha="center", fontsize=7.4)
    ax.set_xticks(xs, [l for _, l in rows], fontsize=8)
    ax.set_ylabel("PQ (%)")
    ax.set_ylim(21.2, 24.9)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_ablation.pdf")


def fig_tissue():
    methods = ["full", "own_baseline", "hovernet", "cellvit_sam256", "nulite_s"]
    labels = ["Ours\n(full)", "Ours\n(base)", "HoVer-\nNet", "CellViT\n-SAM", "NuLite-S"]
    df = pd.read_csv(TAB / "per_tissue_pq.csv", index_col=0)
    M = df[methods].to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 7.0))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(methods)), labels, fontsize=8)
    ax.set_yticks(range(len(df.index)), df.index, fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                    fontsize=6.4, color="white" if M[i, j] < M.mean() else "black")
    fig.colorbar(im, ax=ax, shrink=0.75, label="PQ (%)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_tissue.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_tissue.pdf")


if __name__ == "__main__":
    fig_scatter()
    fig_ablation()
    fig_tissue()
