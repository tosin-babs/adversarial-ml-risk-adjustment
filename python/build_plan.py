"""
The plan's beliefs, per fold: its own cost model, cross-fitted.

For each of Paper 6's 15 (repeat, fold) splits the plan's cost model (Tweedie
LightGBM on F3, the best predictor Paper 6 found) is fitted

  * out of fold inside the training rows (k = 3, PSU-disjoint), giving the
    plan's expected cost for every training person without that person's
    own outcome, for use inside adversarial training; and
  * on all training rows, applied to the test rows, for evaluation.

The regulator's test fold is never seen by either. Writes
data/derived/plan/rep{r}_fold{f}.npz with `train_idx`, `test_idx`,
`cost_train_oof`, `cost_test`.

    ../.venv/bin/python python/build_plan.py
"""

from __future__ import annotations

import time

import numpy as np

import common
import config

OUT = config.DERIVED / "plan"
OUT.mkdir(exist_ok=True)


def main():
    X, y, w, cl, st, attrs = common.build(config.PLAN_MODEL_FEATURES)
    Xn = X.to_numpy(np.float32)
    for rep, fold, tr, te in common.folds(cl, st):
        f = OUT / f"rep{rep}_fold{fold}.npz"
        if f.exists():
            continue
        t0 = time.time()
        make = lambda: common.make(config.PLAN_MODEL)  # noqa: E731
        oof = common.fairness._cross_fit_scores(make, Xn[tr], y[tr], w[tr], cl[tr],
                                                k=config.PLAN_CROSS_FIT_K, seed=config.SEED + rep)
        m = make().fit(Xn[tr], y[tr], w[tr], clusters=cl[tr])
        np.savez(f, train_idx=tr, test_idx=te, cost_train_oof=oof, cost_test=m.predict(Xn[te]))
        r2_oof = common.metrics.r2(y[tr], oof, w[tr])
        r2_te = common.metrics.r2(y[te], m.predict(Xn[te]), w[te])
        print(f"  rep {rep} fold {fold}: plan model R2 oof-train {r2_oof:.3f}, test {r2_te:.3f}, "
              f"{time.time() - t0:.0f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
