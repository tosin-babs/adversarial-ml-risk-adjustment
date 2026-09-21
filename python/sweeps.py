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

KEYS = ("cms", "cms_pen_1", "cms_robust_adv")


def less_able_cost(ctx):
    """A plan whose cost model is unconstrained WLS on F3."""
    X3 = common.build("F3")[0].to_numpy(np.float32)
    m = common.make("wls").fit(X3[ctx.tr], ctx.ytr, ctx.wtr)
    return m.predict(X3[ctx.te])


def real_coding(ctx, f, q, rng):
    """Evaluate with a share q of added codes real: the person's cost rises by
    the code's incremental cost."""
    plan = ctx.plan_te
    added, s, Xc = plan.respond(f.predict, ctx.Xte, ctx.wte, ctx.cols)
    real = added & (rng.random(added.shape) < q)
    extra = np.zeros(len(ctx.yte))
    for qq, c in enumerate(ctx.pool):
        extra += real[:, qq] * ctx.caps.get(c, 0.0)
    y = ctx.yte + extra
    p0, p1 = f.predict(ctx.Xte), f.predict(Xc)
    w = ctx.wte
    codes = float(np.sum(w * added.sum(axis=1)))
    k = 1000.0 / w.sum()
    coding = (float(np.sum(w * (p1 - p0 - extra))) - plan.cost_per_code * codes) * k
    selection = float(np.sum(w * (s - 1.0) * (p1 - y))) * k
    return {"coding": coding, "selection": selection, "extraction": coding + selection,
            "r2_ungamed": common.metrics.r2(ctx.yte, p0, ctx.wte)}


def main():
    X, y, w, cl, st, attrs = common.build(config.PRIMARY_FEATURE_SET)
    splits = common.folds(cl, st)
    cal = F.calibration()
    rows = []
    for rep, fold, tr, te in splits:
        ctxs = {rule: F.Context(config.PRIMARY_FEATURE_SET, rep, fold, tr, te, plausibility=rule)
                for rule in config.PLAUSIBILITY_RULES}
        ctx = ctxs[config.PLAUSIBILITY]
        fits = {}
        for key in KEYS:
            make = F.FORMULAS[key][2]
            if key in adversarial.ADVERSARIAL:
                fits[key], _ = common.adversary.train(lambda: make(ctx), ctx.plan_tr, ctx.Xtr, ctx.ytr, ctx.wtr,
                                                      ctx.cols, iters=config.ADV_ITERS, tol=config.ADV_TOL)
            else:
                fits[key] = make(ctx).fit(ctx.Xtr, ctx.ytr, ctx.wtr, clusters=ctx.cltr)
        variants = []
        for c in config.COST_PER_CODE_GRID:
            variants.append(("cost per code", c, dict(cost_per_code=c)))
        for k in config.MAX_CODES_GRID:
            variants.append(("codes per person", k, dict(max_codes=k)))
        for r in config.REACH_GRID:
            variants.append(("reach", r, dict(reach=r)))
        for t in config.TILT_GRID:
            variants.append(("tilt", t, dict(tilt=t)))
        variants.append(("selection rule", "smooth", dict(rule="smooth", tilt=cal["tilt"] * 10)))
        for key, f in fits.items():
            for name, value, kw in variants:
                plan = F.plan_for(ctx.cost_te, ctx.Pte, ctx.pool, cal, **kw)
                row, _ = benchmark.evaluate(f, ctx, plan)
                rows.append({"key": key, "sweep": name, "value": value, "rep": rep, "fold": fold, **row})
            for rule, c2 in ctxs.items():
                plan = F.plan_for(c2.cost_te, c2.Pte, c2.pool, cal)
                row, _ = benchmark.evaluate(f, c2, plan)
                rows.append({"key": key, "sweep": "plausibility", "value": rule, "rep": rep, "fold": fold, **row})
            plan = F.plan_for(less_able_cost(ctx), ctx.Pte, ctx.pool, cal)
            row, _ = benchmark.evaluate(f, ctx, plan)
            rows.append({"key": key, "sweep": "plan cost model", "value": "WLS", "rep": rep, "fold": fold, **row})
            for q in (0.0, 0.25, 0.5, 1.0):
                r = real_coding(ctx, f, q, np.random.default_rng(config.SEED + fold))
                rows.append({"key": key, "sweep": "share of added codes real", "value": q, "rep": rep, "fold": fold, **r})
        print(f"  rep {rep} fold {fold} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES / "sweeps_by_fold.csv", index=False)
    g = df.groupby(["key", "sweep", "value"], sort=False).agg(
        r2_ungamed=("r2_ungamed", "mean"), coding=("coding", "mean"), selection=("selection", "mean"),
        extraction=("extraction", "mean")).reset_index()
    g["label"] = g["key"].map(lambda k: F.FORMULAS[k][3])
    g.to_csv(config.TABLES / "table9_sweeps.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:,.0f}".format):
        print(g.pivot_table(index=["sweep", "value"], columns="key", values="extraction", sort=False).to_string())


if __name__ == "__main__":
    main()
