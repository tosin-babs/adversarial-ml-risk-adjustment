"""
Cross-fitted cost models per fold, for the plan and for the regulator.

For each of the 15 (repeat, fold) splits and each model set:

  plan_F4      the plan's primary cost model: Tweedie boosting on F4
  ref_F3       the regulator's reference: Tweedie boosting on F3 (used by the
               worst-subgroup penalty, and as the "plan sees only F3" row)
  plan_wls_F3  a less able plan: unconstrained WLS on F3

each fitted out of fold inside the training rows (k = 3, PSU-disjoint) for
use in training, and on all training rows for the test rows. The test fold
is never seen. Writes data/derived/plan/<set>/rep{r}_fold{f}.npz with
train_idx, test_idx, cost_train_oof, cost_test.

    ../.venv/bin/python python/build_plan.py
"""

from __future__ import annotations

import time

import numpy as np

import common
import config

SPECS = {
    "plan_F4": (config.PLAN_MODEL_FEATURES, "gbm", config.SEED + config.PLAN_SEED_OFFSET),
    "ref_F3": (config.REF_MODEL_FEATURES, "gbm", config.SEED),
    "plan_wls_F3": ("F3", "wls", config.SEED),
    "plan_F3": ("F3", "gbm", config.SEED + 2),
}


def make(key, seed):
    if key == "gbm":
        return common.models.GBM("tweedie", seed=seed)
    return common.make(key)


def main():
    for name, (fs, key, seed) in SPECS.items():
        out = config.DERIVED / "plan" / name
        out.mkdir(parents=True, exist_ok=True)
        X, y, w, cl, st, attrs = common.build(fs)
        Xn = X.to_numpy(np.float32)
        for rep, fold, tr, te in common.folds(cl, st):
            f = out / f"rep{rep}_fold{fold}.npz"
            if f.exists():
                continue
            t0 = time.time()
            oof = common.fairness._cross_fit_scores(lambda: make(key, seed), Xn[tr], y[tr], w[tr], cl[tr],
                                                    k=config.PLAN_CROSS_FIT_K, seed=seed + rep)
            m = make(key, seed).fit(Xn[tr], y[tr], w[tr], clusters=cl[tr])
            np.savez(f, train_idx=tr, test_idx=te, cost_train_oof=oof, cost_test=m.predict(Xn[te]))
            print(f"  {name} rep {rep} fold {fold}: R2 oof {common.metrics.r2(y[tr], oof, w[tr]):.3f}, "
                  f"test {common.metrics.r2(y[te], m.predict(Xn[te]), w[te]):.3f}, {time.time() - t0:.0f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
