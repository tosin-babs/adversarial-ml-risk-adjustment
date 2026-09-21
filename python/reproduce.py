"""
Week-1 gate: Paper 6's payment-form and boosting results reproduce from the
sibling directory before anything is built on them.

    ../.venv/bin/python python/reproduce.py

Refits the CMS-form formula (PaymentWLS, F2 and F3) on Paper 6's folds and
compares out-of-fold R2 with Paper 6's Table 2b, and reproduces the
own-targeted flag gains of Table 7c for the payment form on one fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common
import config


def main():
    ref = pd.read_csv(config.PAPER6 / "output" / "tables" / "table2b_feature_sets.csv")
    ref = ref[ref["model_key"] == "pwls"].set_index("feature_set")["r2"]
    ok = True
    for fs in config.FEATURE_SETS:
        X, y, w, cl, st, attrs = common.build(fs)
        Xn = X.to_numpy(np.float32)
        fl = common.folds(cl, st)
        reps = sorted({r for r, *_ in fl})
        preds = np.full((len(reps), len(y)), np.nan)
        for rep, fold, tr, te in fl:
            m = common.make("pwls", X.columns).fit(Xn[tr], y[tr], w[tr], clusters=cl[tr])
            preds[rep, te] = m.predict(Xn[te])
        r2 = float(np.mean([common.metrics.r2(y, preds[r], w) for r in reps]))
        print(f"  PaymentWLS {fs}: R2 {r2:.4f}  paper 6 {ref[fs]:.4f}  diff {r2 - ref[fs]:+.5f}")
        ok &= abs(r2 - ref[fs]) < 1e-6

    # own-targeted flags on fold 0, payment form F3, as coding.py found them
    own = pd.read_csv(config.PAPER6 / "output" / "tables" / "table5c_own_targeted_flags.csv")
    own = own[(own["model"].str.startswith("Payment")) & (own["feature_set"] == "F3") & (own["fold"] == 0)]
    X, y, w, cl, st, attrs = common.build("F3")
    Xn = X.to_numpy(np.float32)
    cols = list(X.columns)
    rep, fold, tr, te = [f for f in common.folds(cl, st) if f[0] == 0 and f[1] == 0][0]
    m = common.make("pwls", cols).fit(Xn[tr], y[tr], w[tr], clusters=cl[tr])
    probe = np.random.default_rng(config.SEED + 100 * 0 + 3).choice(tr, size=min(3000, len(tr)), replace=False)
    gains = common.coding.flag_gains(m.predict, Xn[probe], w[probe], cols, common.ccsr_columns(cols),
                                     count_col=config.COUNT_COL, system_col=config.SYSTEM_COL)
    top = sorted(gains, key=gains.get, reverse=True)[:5]
    print("  own-targeted flags, payment form F3, fold 0:")
    for (c, g), (_, r) in zip([(c, gains[c]) for c in top], own.sort_values("rank").iterrows()):
        print(f"    {c[5:]:<10} {g:>10,.0f}   paper 6 {r['flag']:<10} {r['training_gain']:>10,.0f}")
        ok &= c[5:] == r["flag"] and abs(g - r["training_gain"]) < 1.0
    print("\nREPRODUCED" if ok else "\nMISMATCH: do not build on this")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
