"""
Sensitivity of extraction to the plan: Table 9.

For the CMS form, the coding-penalized form and (when it exists) the
adversarially trained robust form, the plan's parameters are swept on the
test folds, one at a time from the calibrated point:

  cost per code, codes per person, chart-review reach, selection tilt,
  selection rule (threshold or smooth), plausibility rule, the plan's own
  cost model (WLS in place of boosting: a less able plan), and the share q of
  added codes that are real (raising the person's cost by the code's
  incremental cost), which is coding as discovery rather than as gaming.

Formulas are refitted per fold on ungamed data (or adversarially, for the
adversarial key) and the plan varied at evaluation.

    ../.venv/bin/python python/sweeps.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import adversarial
import benchmark
import common
import config
import formulas as F

KEYS = ("cms", "cms_pen_1", "cms_robust_adv", "gbm")


def plan_set_cost(name, ctx):
    return F.load_set(name, ctx.rep, ctx.fold, ctx.tr, ctx.te)[1]


def real_coding(ctx, f, q, rng, G=None):
    """Evaluate with a share q of added codes real: the person's cost rises by
    the code's incremental cost."""
    plan = ctx.plan_te
    added, s, Xc = plan.respond(f.predict, ctx.Xte, ctx.wte, ctx.cols, G=G)
    real = added & (rng.random(added.shape) < q)
    extra = np.zeros(len(ctx.yte))
    for qq, c in enumerate(ctx.pool):
        extra += real[:, qq] * ctx.caps.get(c, 0.0)
    y = ctx.yte + extra
    p0, p1 = f.predict(ctx.Xte), f.predict(Xc)
    w = ctx.wte
    codes = float(np.sum(w * added.sum(axis=1)))
    k = 1000.0 / w.sum()
    coding = (float(np.sum(w * (p1 - p0 - extra))) - float(np.sum(w * plan.row_cost_))) * k
    selection = float(np.sum(w * (s - 1.0) * (p1 - y))) * k
    return {"coding": coding, "selection": selection, "extraction": coding + selection,
            "r2_ungamed": common.metrics.r2(ctx.yte, p0, ctx.wte),
            "distinct_codes": int((added.sum(axis=0) > 0).sum())}


def main():
    X, y, w, cl, st, attrs = common.build(config.PRIMARY_FEATURE_SET)
    splits = common.folds(cl, st)
    cal = F.calibration()
    rows = []
    for rep, fold, tr, te in splits:
        ctx2 = {rule: F.Context("F2", rep, fold, tr, te, plausibility=rule) for rule in config.PLAUSIBILITY_RULES}
        ctx3 = F.Context("F3", rep, fold, tr, te)
        for key in KEYS:
            fs = F.FORMULAS[key][1]
            ctx = ctx2[config.PLAUSIBILITY] if fs == "F2" else ctx3
            make = F.FORMULAS[key][2]
            if key in adversarial.ADVERSARIAL:
                f, _ = common.adversary.train(lambda: make(ctx), ctx.plan_tr, ctx.Xtr, ctx.ytr, ctx.wtr,
                                              ctx.cols, iters=config.ADV_ITERS, tol=config.ADV_TOL, groups=ctx.cltr, seed=config.SEED)
                adversarial.renormalize(f, ctx)
            else:
                f = make(ctx).fit(ctx.Xtr, ctx.ytr, ctx.wtr, clusters=ctx.cltr)
            G = common.adversary.gain_matrix(f.predict, ctx.Xte, ctx.cols, ctx.pool, ctx.Pte,
                                             config.COUNT_COL, config.SYSTEM_COL)

            def ev(name, value, plan, c=ctx, g=G):
                row, _ = benchmark.evaluate(f, c, plan, G=g)
                rows.append({"key": key, "sweep": name, "value": value, "rep": rep, "fold": fold, **row})
            for v in config.COST_PER_CODE_GRID:
                ev("cost per code", v, F.plan_for(ctx.cost_te, ctx.Pte, ctx.pool, cal, cost_per_code=v))
            for v in config.MAX_CODES_GRID:
                ev("codes per person", v, F.plan_for(ctx.cost_te, ctx.Pte, ctx.pool, cal, max_codes=v))
            for v in (0.02, 0.06, 0.10, 0.25, 0.50, 1.0):
                ev("reach", v, F.plan_for(ctx.cost_te, ctx.Pte, ctx.pool, cal, reach=v))
            for v in config.AUDIT_GRID:
                ev("audit exposure", v, F.plan_for(ctx.cost_te, ctx.Pte, ctx.pool, cal, audit=v))
            for v in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
                ev("tilt", v, F.plan_for(ctx.cost_te, ctx.Pte, ctx.pool, cal, tilt=v))
            for name, label in (("plan_F3", "boosting on F3 only"),
                                ("plan_wls_F3", "WLS on F3")):
                ev("plan cost model", label, F.plan_for(plan_set_cost(name, ctx), ctx.Pte, ctx.pool, cal))
            if fs == "F2":
                for rule, c2 in ctx2.items():
                    if rule == config.PLAUSIBILITY:
                        continue
                    g2 = common.adversary.gain_matrix(f.predict, c2.Xte, c2.cols, c2.pool, c2.Pte,
                                                      config.COUNT_COL, config.SYSTEM_COL)
                    ev("plausibility", rule, F.plan_for(c2.cost_te, c2.Pte, c2.pool, cal), c=c2, g=g2)
            for q in (0.0, 0.25, 0.5, 1.0):
                r = real_coding(ctx, f, q, np.random.default_rng(config.SEED + fold), G)
                rows.append({"key": key, "sweep": "share of added codes real", "value": q, "rep": rep, "fold": fold, **r})
        print(f"  rep {rep} fold {fold} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES / "sweeps_by_fold.csv", index=False)
    g = df.groupby(["key", "sweep", "value"], sort=False).agg(
        r2_ungamed=("r2_ungamed", "mean"), coding=("coding", "mean"), selection=("selection", "mean"),
        extraction=("extraction", "mean"), distinct_codes=("distinct_codes", "mean")).reset_index()
    g["label"] = g["key"].map(lambda k: F.FORMULAS[k][3])
    g.to_csv(config.TABLES / "table9_sweeps.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:,.0f}".format):
        print(g.pivot_table(index=["sweep", "value"], columns="key", values="extraction", sort=False).to_string())


if __name__ == "__main__":
    main()
