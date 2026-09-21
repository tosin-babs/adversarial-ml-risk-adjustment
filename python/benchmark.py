"""
Every formula against the calibrated plan, out of sample.

For each formula and each of Paper 6's 15 splits: fit on the training rows
(ungamed), then on the test rows record accuracy on ungamed data, the plan's
response (codes added, selection weights), accuracy on the data the response
produces, what the plan extracts by channel, and the net compensation of
every evaluated group before and after the response.

Per-row contributions are saved to data/derived/oof/<key>.npz so that
`summarise` can bootstrap sums and paired differences over PSUs.

    ../.venv/bin/python python/benchmark.py            # all formulas
    P7_ONLY=cms,cms_cap ../.venv/bin/python python/benchmark.py
    P7_FAMILY=linear ../.venv/bin/python python/benchmark.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import common
import config
import formulas as F

OOF = config.DERIVED / "oof"
OOF.mkdir(exist_ok=True)


def evaluate(f, ctx, plan=None, tag=""):
    """One fitted formula on one test fold: metrics and per-row contributions."""
    plan = plan or ctx.plan_te
    A = common.adversary
    X, y, w = ctx.Xte, ctx.yte, ctx.wte
    p0 = f.predict(X)
    added, s, Xc = plan.respond(f.predict, X, w, ctx.cols)
    p1 = f.predict(Xc)
    codes = added.sum(axis=1).astype(float)
    row = {"rep": ctx.rep, "fold": ctx.fold,
           "r2_ungamed": common.metrics.r2(y, p0, w),
           "r2_post": common.metrics.r2(y, p1, w * s),
           **A.extraction(f.predict, X, Xc, w, s, y, added, plan.cost_per_code)}
    g0 = common.metrics.group_fairness(y, p0, w, ctx.masks_te)
    g1 = common.metrics.group_fairness(y, p1, w * s, ctx.masks_te)
    for name in g0:
        row[f"nc_ungamed|{name}"] = g0[name]["nc"]
        row[f"nc_post|{name}"] = g1[name]["nc"]
        row[f"sel_share|{name}"] = float(np.sum(w[ctx.masks_te[name]] * s[ctx.masks_te[name]])
                                         / np.sum(w[ctx.masks_te[name]]))
    per_row = {"p0": p0, "p1": p1, "s": s, "codes": codes,
               "coding": w * (p1 - p0) - plan.cost_per_code * w * codes,
               "selection": w * (s - 1.0) * (p1 - y)}
    return row, per_row


def run(key, splits, plausibility=None, cal=None, tag=None):
    family, fs, make, label, param = F.FORMULAS[key]
    name = tag or key
    out = OOF / f"{name}.npz"
    if out.exists() and os.environ.get("P7_REFIT") != "1":
        return pd.read_pickle(OOF / f"{name}.rows.pkl")
    X, y, w, cl, st, attrs = common.build(fs)
    n = len(y)
    reps = sorted({r for r, *_ in splits})
    store = {k: np.full((len(reps), n), np.nan) for k in ("p0", "p1", "s", "codes", "coding", "selection")}
    rows, t0 = [], time.time()
    for rep, fold, tr, te in splits:
        ctx = F.Context(fs, rep, fold, tr, te, plausibility, cal)
        f = make(ctx).fit(ctx.Xtr, ctx.ytr, ctx.wtr, clusters=ctx.cltr)
        row, per = evaluate(f, ctx)
        rows.append({"key": key, "label": label, "family": family, "feature_set": fs, "param": param, **row})
        for k in store:
            store[k][rep, te] = per[k]
    np.savez(out, y=y, w=w, clusters=cl, strata=st, **store)
    df = pd.DataFrame(rows)
    df.to_pickle(OOF / f"{name}.rows.pkl")
    print(f"  {label:<52} R2 {df['r2_ungamed'].mean():.4f} -> {df['r2_post'].mean():.4f}  "
          f"extraction ${df['extraction'].mean():,.0f}/1000 (coding ${df['coding'].mean():,.0f}, "
          f"selection ${df['selection'].mean():,.0f})  {time.time() - t0:.0f}s", flush=True)
    return df


def main():
    X, y, w, cl, st, attrs = common.build(config.PRIMARY_FEATURE_SET)
    splits = common.folds(cl, st)
    only = os.environ.get("P7_ONLY")
    fam = os.environ.get("P7_FAMILY")
    keys = only.split(",") if only else (F.LINEAR if fam == "linear" else F.BOOSTED if fam == "boosted"
                                          else list(F.FORMULAS))
    parts = [run(k, splits) for k in keys]
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(config.TABLES / "benchmark_by_fold.csv", index=False)


if __name__ == "__main__":
    main()
