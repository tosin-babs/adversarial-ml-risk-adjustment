"""
Formulas trained against the plan.

For each split the formula is fitted on the training rows, the plan responds
(with its out-of-fold cost predictions for those rows), the formula is refitted
on the coded features and tilted weights with the true cost, and so on up to
ADV_ITERS times or until payments move by less than ADV_TOL. The final formula
is then evaluated on the test rows exactly as in benchmark.py, so its row in
Table 3 is comparable. The path of extraction and accuracy on the training
rows is written for Table 7.

    ../.venv/bin/python python/adversarial.py                 # linear formulas, 15 splits
    P7_ADV=gbm ../.venv/bin/python python/adversarial.py      # boosted, one repeat
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import benchmark
import common
import config
import formulas as F

# key -> (base formula key, label)
ADVERSARIAL = {
    "cms_adv": ("cms", "CMS form, trained against the plan"),
    "cms_pen_adv": ("cms_pen_1", "CMS form, coding-penalized, trained against the plan"),
    "cms_robust_adv": ("cms_cap_pen_dro_1", "CMS form, capped + penalized + DRO, trained against the plan"),
    "gbm_robust_adv": ("gbm_cap_dro", "Boosting, capped + DRO, trained against the plan"),
}
for k, (base, label) in ADVERSARIAL.items():
    fam, fs, make, _, param = F.FORMULAS[base]
    F.FORMULAS[k] = ("adversarial", fs, make, label, param)


def renormalize(f, ctx):
    """Each refit is normalized to cost on the coded, tilted training rows,
    which leaves the final formula paying about 10% below cost on ungamed
    records: a built-in coding adjustment. The regulator's budget rule is
    applied to the final formula instead, on the ungamed training rows, so
    every formula in the comparison pays total cost before the plan responds."""
    k = float(np.sum(ctx.wtr * ctx.ytr) / np.sum(ctx.wtr * f.predict(ctx.Xtr)))
    f.factor_ *= k
    return k


def run(key, splits, iters=None, tag=None):
    base, label = ADVERSARIAL[key]
    fam, fs, make, _, param = F.FORMULAS[base]
    name = tag or key
    out = benchmark.OOF / f"{name}.npz"
    if out.exists() and os.environ.get("P7_REFIT") != "1":
        return pd.read_pickle(benchmark.OOF / f"{name}.rows.pkl")
    X, y, w, cl, st, attrs = common.build(fs)
    n = len(y)
    reps = sorted({r for r, *_ in splits})
    store = {k: np.full((len(reps), n), np.nan) for k in ("p0", "p1", "s", "codes", "row_cost", "coding", "selection")}
    rows, paths, t0 = [], [], time.time()
    for rep, fold, tr, te in splits:
        ctx = F.Context(fs, rep, fold, tr, te)
        f, path = common.adversary.train(lambda: make(ctx), ctx.plan_tr, ctx.Xtr, ctx.ytr, ctx.wtr, ctx.cols,
                                         iters=iters or config.ADV_ITERS, tol=config.ADV_TOL,
                                         groups=ctx.cltr, seed=config.SEED + 17 * rep + fold)
        renormalize(f, ctx)
        # accuracy on ungamed training data along the path is what the
        # regulator gives up; evaluate the final formula out of sample
        for p in path:
            paths.append({"key": key, "rep": rep, "fold": fold, **p})
        row, per = benchmark.evaluate(f, ctx)
        rows.append({"key": key, "label": label, "family": "adversarial", "feature_set": fs, "param": param,
                     "iterations": len(path), "converged": path[-1]["change"] < config.ADV_TOL, **row})
        for k in store:
            store[k][reps.index(rep), te] = per[k]
        print(f"    rep {rep} fold {fold}: {len(path)} iterations, extraction on training rows "
              f"${path[0]['extraction']:,.0f} -> ${path[-1]['extraction']:,.0f}", flush=True)
    np.savez(out, y=y, w=w, clusters=cl, strata=st, **store)
    df = pd.DataFrame(rows)
    df.to_pickle(benchmark.OOF / f"{name}.rows.pkl")
    pd.DataFrame(paths).to_csv(config.TABLES / f"path_{name}.csv", index=False)
    print(f"  {label:<52} R2 {df['r2_ungamed'].mean():.4f} -> {df['r2_post'].mean():.4f}  "
          f"extraction ${df['extraction'].mean():,.0f}/1000 (coding ${df['coding'].mean():,.0f}, "
          f"selection ${df['selection'].mean():,.0f})  {time.time() - t0:.0f}s", flush=True)
    return df


def merge_reps(key, reps):
    """Combine per-repeat runs of one adversarial formula into its main files."""
    parts = [np.load(benchmark.OOF / f"{key}_rep{r}.npz", allow_pickle=True) for r in reps]
    base = parts[0]
    out = {k: base[k] for k in ("y", "w", "clusters", "strata")}
    for f in ("p0", "p1", "s", "codes", "row_cost", "coding", "selection"):
        out[f] = np.vstack([p[f] for p in parts])
    np.savez(benchmark.OOF / f"{key}.npz", **out)
    rows = pd.concat([pd.read_pickle(benchmark.OOF / f"{key}_rep{r}.rows.pkl") for r in reps], ignore_index=True)
    rows.to_pickle(benchmark.OOF / f"{key}.rows.pkl")
    paths = pd.concat([pd.read_csv(config.TABLES / f"path_{key}_rep{r}.csv") for r in reps], ignore_index=True)
    paths["key"] = key
    paths.to_csv(config.TABLES / f"path_{key}.csv", index=False)
    print(f"merged {key}: {len(rows)} folds")


def main():
    X, y, w, cl, st, attrs = common.build(config.PRIMARY_FEATURE_SET)
    which = os.environ.get("P7_ADV", "linear")
    if which == "gbm":
        # one process per repeat (P7_ADV_REPS), merged by merge_reps()
        for r in [int(x) for x in os.environ.get("P7_ADV_REPS", "0").split(",")]:
            splits = [s for s in common.folds(cl, st) if s[0] == r]
            run("gbm_robust_adv", splits, iters=10, tag=f"gbm_robust_adv_rep{r}")
    elif which == "merge":
        merge_reps("gbm_robust_adv", [0, 1, 2])
    else:
        splits = common.folds(cl, st)
        for k in os.environ.get("P7_ADV_KEYS", "cms_adv,cms_pen_adv,cms_robust_adv").split(","):
            run(k, splits)


if __name__ == "__main__":
    main()
