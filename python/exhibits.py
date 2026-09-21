"""
Figures for the manuscript, from the analysis CSVs. Writes output/figures/*.png and .pdf.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
import config  # noqa: E402

P = common.p6config.PALETTE
T = config.TABLES
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": P["muted"], "axes.labelcolor": P["ink"],
                     "xtick.color": P["muted"], "ytick.color": P["muted"], "axes.grid": True,
                     "grid.color": P["rule"], "grid.linewidth": 0.5, "figure.dpi": 120})


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(config.FIGURES / f"{name}.{ext}", dpi=common.p6config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def figure_frontier():
    t = pd.read_csv(T / "table3_extraction.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fams = {"penalized": ("Coding penalty", P["linear"]), "dro": ("Worst-subgroup penalty", P["glm"]),
            "combined": ("Cap + coding penalty + worst-subgroup penalty", P["fair"])}
    for fam, (lab, col) in fams.items():
        d = t[t["family"] == fam].sort_values("param")
        base = t[t["key"] == "cms"]
        x = np.concatenate([base["extraction"].to_numpy(), d["extraction"].to_numpy()]) / 1000
        y = np.concatenate([base["r2_ungamed"].to_numpy(), d["r2_ungamed"].to_numpy()])
        ax.plot(x, y, "-o", color=col, ms=3.5, lw=1.4, label=lab)
    for key, lab, col, mk in (("cms", "CMS form", P["ink"], "s"), ("gbm", "Tweedie boosting", P["gbm"], "D"),
                              ("fair", "Fair stacked (paper 6)", P["nn"], "^"),
                              ("cms_robust_adv", "Robust CMS form, trained against the plan", P["fair"], "*"),
                              ("gbm_robust_adv", "Robust boosting, trained against the plan", P["gbm"], "*")):
        d = t[t["key"] == key]
        if len(d):
            ax.plot(d["extraction"] / 1000, d["r2_ungamed"], mk, color=col, ms=8 if mk == "*" else 6, label=lab, zorder=5)
    ax.set_xlabel("Extractable payment, $ thousand per 1,000 enrollees")
    ax.set_ylabel("R² on ungamed data")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    _save(fig, "fig1_frontier")


def figure_group_gaps():
    g = pd.read_csv(T / "table6_group_gaps.csv")
    keys = [k for k in ("cms", "cms_pen_1", "cms_robust_adv", "gbm", "gbm_robust_adv") if k in set(g["key"])]
    groups = ["Needs ADL or IADL help", "Age 65 and over", "Uninsured all year", "Income below 200% FPL",
              "Mental health condition", "Non-Hispanic Black", "Hispanic"]
    labels = dict(zip(g["key"], g["label"]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    cols = [P["ink"], P["linear"], P["fair"], P["gbm"], P["nn"]]
    for ax, col, title in zip(axes, ("nc_ungamed", "nc_post"), ("Before the plan responds", "After")):
        y = np.arange(len(groups))
        h = 0.8 / len(keys)
        for i, k in enumerate(keys):
            d = g[(g["key"] == k)].set_index("group").reindex(groups)
            ax.barh(y + i * h - 0.4 + h / 2, d[col] / 1000, height=h, color=cols[i], label=labels[k])
        ax.axvline(0, color=P["ink"], lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(groups)
        ax.set_xlabel("Net compensation, $ thousand per person-year")
        ax.set_title(title, fontsize=9)
        ax.invert_yaxis()
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    _save(fig, "fig2_group_gaps")


def figure_calibration():
    c = pd.read_csv(T / "table4_calibration.csv")
    cal = json.loads((config.DERIVED / "calibration.json").read_text())
    d = c[(c["channel"] == "coding") & (c["cost_per_code"] == cal["cost_per_code"])]
    w = d.pivot_table(index="max_codes", columns="reach", values="payment_rise_pct")
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    im = ax.imshow(w.to_numpy(), cmap="Reds", aspect="auto", vmin=0, vmax=min(120, np.nanmax(w.to_numpy())))
    ax.set_xticks(range(len(w.columns)))
    ax.set_xticklabels([f"{100 * r:g}%" for r in w.columns])
    ax.set_yticks(range(len(w.index)))
    ax.set_yticklabels([int(i) for i in w.index])
    ax.set_xlabel("Chart-review reach, share of enrollees")
    ax.set_ylabel("Codes per person")
    ax.grid(False)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            v = w.to_numpy()[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if v > 60 else P["ink"])
    fig.colorbar(im, ax=ax, label="Payment rise, CMS form")
    _save(fig, "fig3_calibration")


def figure_path():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cols = {"cms_adv": P["ink"], "cms_pen_adv": P["linear"], "cms_robust_adv": P["fair"], "gbm_robust_adv": P["gbm"]}
    t3 = pd.read_csv(T / "table3_extraction.csv")
    labels = dict(zip(t3["key"], t3["label"]))
    for k, col in cols.items():
        f = T / f"path_{k}.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f).groupby("iter")["extraction"].mean()
        ax.plot(p.index, p.to_numpy() / 1000, "-o", ms=3, lw=1.3, color=col, label=labels.get(k, k))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Extraction on training rows, $ thousand per 1,000")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7.5)
    _save(fig, "fig4_path")


def figure_codes():
    t = pd.read_csv(T / "table8_code_increments.csv").head(20)
    fig, ax = plt.subplots(figsize=(6.6, 5))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t["increment_cms"] / 1000, height=0.4, color=P["ink"], label="CMS form")
    ax.barh(y + 0.2, t["increment_cms_robust_adv"] / 1000, height=0.4, color=P["fair"],
            label="Robust CMS form, trained against the plan")
    ax.plot(t["incremental_cost"] / 1000, y, "|", color=P["gbm"], ms=10, mew=1.5, label="Incremental cost")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{f}: {l[:38]}" for f, l in zip(t["flag"], t["label"])], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Payment rise per added code, $ thousand")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    _save(fig, "fig5_codes")


def main():
    for f in (figure_frontier, figure_group_gaps, figure_calibration, figure_path, figure_codes):
        f()
        print(f"  {f.__name__}")


if __name__ == "__main__":
    main()
