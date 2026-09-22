"""
Payload for the "let the plan game it" panel of the public explorer
(Paper 6's page, https://fair-risk-adjustment.vercel.app).

For each formula and each of the five test folds of repeat 0, the formula is
fitted (adversarially where its key says so), the plan's gain matrix is
computed once, and the plan's response is evaluated over a grid of
chart-review reach, cost per code and selection tilt, one code per reviewed
person. The page reads extraction by channel and the net compensation and
enrollment share of four groups, averaged over the folds. No person-level
data leave this script.

Writes ../paper6/tool/plan_game.json.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd

import adversarial
import common
import config
import formulas as F

KEYS = ["cms", "cms_cap", "cms_pen_1", "cms_dro_5", "cms_cap_pen_dro_1", "gbm", "fair",
        "cms_pen_adv", "cms_robust_adv"]
AUDIT = None  # the calibrated audit exposure
REACH = [0.0, 0.02, 0.04, 0.06, 0.10, 0.25, 0.50, 1.0]
COST = [0.0, 500.0, 1000.0, 2000.0, 3000.0]
TILT = [0.0, 0.05, 0.10, 0.20, 0.25, 0.30, 0.50]
GROUPS = ["Needs ADL or IADL help", "Age 65 and over", "Uninsured all year", "Income below 200% FPL"]
OUT = config.PAPER6 / "tool" / "plan_game.json"


def fit(key, ctx):
    fam, fs, make, label, param = F.FORMULAS[key]
    if key in adversarial.ADVERSARIAL:
        f, _ = common.adversary.train(lambda: make(ctx), ctx.plan_tr, ctx.Xtr, ctx.ytr, ctx.wtr, ctx.cols,
                                      iters=config.ADV_ITERS, tol=config.ADV_TOL, groups=ctx.cltr, seed=config.SEED)
        adversarial.renormalize(f, ctx)
        return f
    return make(ctx).fit(ctx.Xtr, ctx.ytr, ctx.wtr, clusters=ctx.cltr)


def main():
    cal = F.calibration()
    X, y, w, cl, st, attrs = common.build(config.PRIMARY_FEATURE_SET)
    splits = [s for s in common.folds(cl, st) if s[0] == 0]
    rows, r2 = [], {}
    for key in KEYS:
        fs = F.FORMULAS[key][1]
        t0 = time.time()
        acc = []
        for rep, fold, tr, te in splits:
            ctx = F.Context(fs, rep, fold, tr, te)
            f = fit(key, ctx)
            Xte, yte, wte = ctx.Xte, ctx.yte, ctx.wte
            p0 = f.predict(Xte)
            acc.append(common.metrics.r2(yte, p0, wte))
            G = common.adversary.gain_matrix(f.predict, Xte, ctx.cols, ctx.pool, ctx.Pte,
                                             config.COUNT_COL, config.SYSTEM_COL)
            base = float(np.sum(wte * p0))
            k = 1000.0 / wte.sum()
            masks = {g: ctx.masks_te[g] for g in GROUPS if g in ctx.masks_te}
            coded = {}
            for reach, cost in itertools.product(REACH, COST):
                plan = F.plan_for(ctx.cost_te, ctx.Pte, ctx.pool, cal, reach=reach, cost_per_code=cost,
                                  max_codes=1, tilt=0.0)
                added, _ = plan.code(f.predict, Xte, wte, ctx.cols, G=G)
                Xc = common.adversary.apply_codes(Xte, ctx.cols, ctx.pool, added, config.COUNT_COL, config.SYSTEM_COL)
                coded[(reach, cost)] = (added, f.predict(Xc), plan.row_cost_.copy())
            for (reach, cost), (added, p1, row_cost) in coded.items():
                coding = (float(np.sum(wte * (p1 - p0))) - float(np.sum(wte * row_cost))) * k
                for tilt in TILT:
                    plan = F.plan_for(ctx.cost_te, ctx.Pte, ctx.pool, cal, tilt=tilt)
                    s = plan.select(p1, wte, ctx.cost_te)
                    sel = float(np.sum(wte * (s - 1.0) * (p1 - yte))) * k
                    rec = {"key": key, "fold": fold, "reach": reach, "cost": cost, "tilt": tilt,
                           "coding": coding, "selection": sel, "base": base * k}
                    for g, m in masks.items():
                        ww = wte[m] * s[m]
                        rec[f"nc|{g}"] = float(np.sum(ww * (p1[m] - yte[m])) / ww.sum())
                        rec[f"share|{g}"] = float(np.sum(wte[m] * s[m]) / np.sum(wte[m]))
                    rows.append(rec)
        r2[key] = float(np.mean(acc))
        print(f"  {F.FORMULAS[key][3]:<55} {time.time() - t0:5.0f}s", flush=True)
    df = pd.DataFrame(rows)
    g = df.drop(columns="fold").groupby(["key", "reach", "cost", "tilt"], sort=False).mean().reset_index()
    g["extraction"] = g["coding"] + g["selection"]
    g["pct"] = 100 * g["extraction"] / g["base"]
    out = {
        "formulas": [{"key": k, "label": F.FORMULAS[k][3], "r2": round(r2[k], 4),
                      "trained": k in adversarial.ADVERSARIAL} for k in KEYS],
        "grid": {"reach": REACH, "cost": COST, "tilt": TILT},
        "groups": GROUPS,
        "calibrated": {"reach": cal["reach"], "cost": cal["cost_per_code"], "tilt": cal["tilt"]},
        "cells": [[k, r, c, t, round(e), round(cd), round(sl), round(p, 2)]
                  + [round(v) for v in row[[f"nc|{x}" for x in GROUPS]]]
                  + [round(v, 3) for v in row[[f"share|{x}" for x in GROUPS]]]
                  for (_, row), k, r, c, t, e, cd, sl, p in zip(
                      g.iterrows(), g["key"], g["reach"], g["cost"], g["tilt"], g["extraction"],
                      g["coding"], g["selection"], g["pct"])],
        "note": ("Repeat 0 of the five survey folds; one code per reviewed person, with the calibrated audit "
                 "exposure spreading the plan's codes; formulas normalized to total cost; the plan selects on "
                 "its own cross-fitted cost model, which also uses health-risk-assessment information."),
        "paper": "Adversarial Machine Learning for Medicare Risk Adjustment: Training Payment Formulas Against a Strategic Health Plan",
        "repository": "https://github.com/tosin-babs/adversarial-ml-risk-adjustment",
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(out['cells'])} cells)")


if __name__ == "__main__":
    main()
