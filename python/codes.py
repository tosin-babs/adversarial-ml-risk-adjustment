"""
What each formula pays per code, before and after robustness: Table 8.

Formulas are fitted on all rows (an exhibit, not an evaluation), with the
plan's cost model cross-fitted over all rows for the adversarial fit. For each
CCSR flag: prevalence, incremental cost, and the payment increment under the
CMS form, the coding-penalized form, the combined robust form and the
adversarially trained robust form, with the rank under each.

    ../.venv/bin/python python/codes.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import adversarial  # noqa: F401  (registers the adversarial keys)
import common
import config
import formulas as F


class _AllRows:
    """A Context over all rows, for full-sample exhibit fits."""

    def __init__(self, fs):
        X, y, w, cl, st, attrs = common.build(fs)
        self.cols = list(X.columns)
        self.pool = common.ccsr_columns(self.cols)
        self.Xtr, self.ytr, self.wtr, self.cltr = X.to_numpy(np.float32), y, w, cl
        self.attrs_tr = attrs
        self.masks_tr = common.masks(attrs)
        self.Ptr = common.plausibility(self.Xtr, self.cols, self.pool, attrs)
        f = config.DERIVED / "plan_all_rows.npy"
        if not f.exists():
            oof = common.fairness._cross_fit_scores(lambda: common.make(config.PLAN_MODEL),
                                                    common.build(config.PLAN_MODEL_FEATURES)[0].to_numpy(np.float32),
                                                    y, w, cl, k=config.PLAN_CROSS_FIT_K, seed=config.SEED)
            np.save(f, oof)
        self.cost_tr = np.load(f)
        self.cal = F.calibration()
        self.plan_tr = F.plan_for(self.cost_tr, self.Ptr, self.pool, self.cal)
        j = {c: self.cols.index(c) for c in self.pool}
        self.pool_weights = {c: float(np.average(self.Ptr[:, q] & (self.Xtr[:, j[c]] == 0), weights=self.wtr))
                             for q, c in enumerate(self.pool)}
        self.pool_weights.update({config.COUNT_COL: 1.0, config.SYSTEM_COL: 1.0})
        self.caps = common.robust.incremental_costs(self.Xtr, self.ytr, self.wtr, self.cols, self.pool,
                                                    n_conditions_col=config.COUNT_COL)


def main():
    ctx = _AllRows(config.PRIMARY_FEATURE_SET)
    fits = {}
    for key in ("cms", "cms_pen_1", "cms_cap_pen_dro_1"):
        fits[key] = F.FORMULAS[key][2](ctx).fit(ctx.Xtr, ctx.ytr, ctx.wtr, clusters=ctx.cltr)
    make = F.FORMULAS["cms_robust_adv"][2]
    fits["cms_robust_adv"], path = common.adversary.train(lambda: make(ctx), ctx.plan_tr, ctx.Xtr, ctx.ytr,
                                                          ctx.wtr, ctx.cols, iters=config.ADV_ITERS,
                                                          tol=config.ADV_TOL)
    pd.DataFrame(path).to_csv(config.TABLES / "path_all_rows_cms_robust_adv.csv", index=False)
    labels = pd.read_csv(config.PAPER6 / "data" / "derived" / "ccsr_labels.csv")
    lab = dict(zip(labels.iloc[:, 0], labels.iloc[:, 1]))
    j = {c: ctx.cols.index(c) for c in ctx.pool}
    tab = pd.DataFrame({
        "flag": [c[5:] for c in ctx.pool],
        "label": [lab.get(c[5:].split("+")[0], "") for c in ctx.pool],
        "prevalence_pct": [100 * float(np.average(ctx.Xtr[:, j[c]] > 0, weights=ctx.wtr)) for c in ctx.pool],
        "plausible_share_pct": [100 * ctx.pool_weights[c] for c in ctx.pool],
        "incremental_cost": [ctx.caps[c] for c in ctx.pool],
    })
    for key, f in fits.items():
        # the payment rise per person when the flag is switched on for those
        # who could plausibly receive it, counts updated: what the plan sees
        G = common.adversary.gain_matrix(f.predict, ctx.Xtr, ctx.cols, ctx.pool, ctx.Ptr,
                                         config.COUNT_COL, config.SYSTEM_COL)
        inc = [float(np.nanmean(G[:, q])) if np.isfinite(G[:, q]).any() else np.nan for q in range(len(ctx.pool))]
        tab[f"increment_{key}"] = inc
        tab[f"rank_{key}"] = tab[f"increment_{key}"].rank(ascending=False).astype(int)
    tab["increment_over_cost_cms"] = tab["increment_cms"] / tab["incremental_cost"].replace(0, np.nan)
    tab = tab.sort_values("increment_cms", ascending=False)
    tab.to_csv(config.TABLES / "table8_code_increments.csv", index=False)
    # coefficients of the count columns under each formula
    counts = pd.DataFrame([{"formula": k, **{c: f.coefficients().get(c, np.nan) for c in (config.COUNT_COL, config.SYSTEM_COL)}}
                           for k, f in fits.items()])
    counts.to_csv(config.TABLES / "table8b_count_coefficients.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:,.0f}".format):
        print(tab.head(20)[["flag", "prevalence_pct", "incremental_cost", "increment_cms", "increment_cms_pen_1",
                            "increment_cms_cap_pen_dro_1", "increment_cms_robust_adv"]].to_string(index=False))
        print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
