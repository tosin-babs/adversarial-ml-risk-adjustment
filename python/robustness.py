"""
Robustness rows: Table 10.

  65 and over          the stored out-of-fold results restricted to persons
                       65 and over (no refit)
  F3 for the CMS form  the coding-penalized and robust forms with prior use
  squared-error boosting
  DRO alpha 0.05 / 0.20
  leave-one-panel-out  the CMS form, the penalized form and boosting refitted
                       with one MEPS panel held out in turn

    ../.venv/bin/python python/robustness.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import benchmark
import common
import config
import formulas as F
import summarise

KEYS_65 = ("cms", "cms_pen_1", "cms_cap_pen_dro_1", "gbm", "gbm_cap_dro", "cms_pen_adv", "cms_robust_adv")


def over_65():
    X, y, w, cl, st, attrs = common.build(config.PRIMARY_FEATURE_SET)
    m = attrs["age"].to_numpy(float) >= 65
    rows = []
    for k in KEYS_65:
        f = summarise.OOF / f"{k}.npz"
        if not f.exists():
            continue
        z, _ = summarise.load(k)
        idx, mult = np.flatnonzero(m), np.ones(int(m.sum()))
        rows.append({"variant": "persons 65 and over", "key": k, "label": F.FORMULAS[k][3],
                     "r2_ungamed": summarise.r2_full(z, idx, mult, "p0"),
                     "coding": summarise.per_1000(z, idx, mult, "coding"),
                     "selection": summarise.per_1000(z, idx, mult, "selection")})
    return rows


def refits():
    rows = []
    X, y, w, cl, st, attrs = common.build(config.PRIMARY_FEATURE_SET)
    splits = common.folds(cl, st, repeats=1)
    extra = {
        "cms_pen_1_F3": ("F3 (prior use) for the penalized form", "F3", F.FORMULAS["cms_pen_1"][2]),
        "cms_cap_pen_dro_1_F3": ("F3 (prior use) for the robust form", "F3", F.FORMULAS["cms_cap_pen_dro_1"][2]),
        "cms_cap_pen_dro_1_a05": ("DRO alpha 0.05", "F2", F._combo(1.0, 1.0, capped=True, alpha=0.05)),
        "cms_cap_pen_dro_1_a20": ("DRO alpha 0.20", "F2", F._combo(1.0, 1.0, capped=True, alpha=0.20)),
        "gbm_mse": ("squared-error boosting", "F3", F._plain("gbm_mse")),
    }
    for key, (variant, fs, make) in extra.items():
        F.FORMULAS[key] = ("robustness", fs, make, variant, None)
        df = benchmark.run(key, splits)
        rows.append({"variant": variant, "key": key, "label": variant, "r2_ungamed": df["r2_ungamed"].mean(),
                     "coding": df["coding"].mean(), "selection": df["selection"].mean()})
    # leave one panel out
    panel = attrs["panel"].to_numpy()
    for key in ("cms", "cms_pen_1", "gbm"):
        fam, fs, make, label, param = F.FORMULAS[key]
        Xf, yf, wf, clf, stf, af = common.build(fs)
        Xn = Xf.to_numpy(np.float32)
        parts = []
        for p in sorted(np.unique(panel)):
            tr, te = np.flatnonzero(panel != p), np.flatnonzero(panel == p)
            # the plan's cost model for this split, cross-fitted on the training panels
            oof = common.fairness._cross_fit_scores(lambda: common.make(config.PLAN_MODEL),
                                                    common.build("F3")[0].to_numpy(np.float32)[tr], yf[tr], wf[tr],
                                                    clf[tr], k=config.PLAN_CROSS_FIT_K, seed=config.SEED)
            mplan = common.make(config.PLAN_MODEL).fit(common.build("F3")[0].to_numpy(np.float32)[tr], yf[tr], wf[tr], clusters=clf[tr])
            cost_te = mplan.predict(common.build("F3")[0].to_numpy(np.float32)[te])
            np.savez(config.DERIVED / "plan" / f"rep9_fold{p}.npz", train_idx=tr, test_idx=te,
                     cost_train_oof=oof, cost_test=cost_te)
            ctx = F.Context(fs, 9, int(p), tr, te)
            f = make(ctx).fit(ctx.Xtr, ctx.ytr, ctx.wtr, clusters=ctx.cltr)
            row, _ = benchmark.evaluate(f, ctx)
            parts.append(row)
        d = pd.DataFrame(parts)
        rows.append({"variant": "leave one panel out", "key": key, "label": label,
                     "r2_ungamed": d["r2_ungamed"].mean(), "coding": d["coding"].mean(),
                     "selection": d["selection"].mean()})
        print(f"  panel-out {label}: R2 {d['r2_ungamed'].mean():.4f}, coding ${d['coding'].mean():,.0f}, "
              f"selection ${d['selection'].mean():,.0f}", flush=True)
    return rows


def main():
    rows = over_65() + refits()
    df = pd.DataFrame(rows)
    df["extraction"] = df["coding"] + df["selection"]
    df.to_csv(config.TABLES / "table10_robustness.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:,.3f}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
