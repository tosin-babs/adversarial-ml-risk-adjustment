"""
Calibrate the plan against the CMS-form formula, jointly, on training rows.

For each of the 15 splits the training rows are split into two
cluster-disjoint halves; the CMS-form formula (payment-form WLS on F2) is
fitted on one half and the plan responds on the other, using its
out-of-fold cost predictions, over a grid of chart-review
reach and selection tilt, with one code per reviewed person at
PRIMARY_COST_PER_CODE and audit exposure AUDIT_PER_PCT. The chosen (reach,
tilt) minimizes the squared distance of (coding gain, selection gain), both
as shares of base payment and averaged over splits, to MedPAC's
(CODING_TARGET, SELECTION_TARGET). Coding is on when selection is measured.
The test folds are not used.

Writes table4_calibration.csv, calibration.json and table_codable_pool.csv.

    ../.venv/bin/python python/calibrate.py
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

import common
import config
import formulas as F


def main():
    X, y, w, cl, st, attrs = common.build(config.PRIMARY_FEATURE_SET)
    rows = []
    for rep, fold, tr, te in common.folds(cl, st):
        ctx = F.Context(config.PRIMARY_FEATURE_SET, rep, fold, tr, te,
                        cal={"cost_per_code": config.PRIMARY_COST_PER_CODE, "max_codes": 1, "reach": 0.1,
                             "tilt": 0.2, "rule": config.SELECTION_RULE, "audit": config.AUDIT_PER_PCT})
        # The plan responds on each half of the training rows to the CMS form
        # fitted on the other half, so it games out-of-sample payments, as it
        # does on the test folds.
        halves = common.adversary._halves(ctx.cltr, config.SEED + 17 * rep + fold)
        parts = []
        for h, other in ((halves[0], halves[1]), (halves[1], halves[0])):
            f = F.Normalized(common.make("pwls", ctx.cols)).fit(ctx.Xtr[other], ctx.ytr[other], ctx.wtr[other])
            Xs, ys, ws, P, cost = ctx.Xtr[h], ctx.ytr[h], ctx.wtr[h], ctx.Ptr[h], ctx.cost_tr[h]
            p0 = f.predict(Xs)
            G = common.adversary.gain_matrix(f.predict, Xs, ctx.cols, ctx.pool, P, config.COUNT_COL, config.SYSTEM_COL)
            parts.append((f, Xs, ys, ws, P, cost, p0, G))
        for reach in config.REACH_GRID:
            coded = []
            for f, Xs, ys, ws, P, cost, p0, G in parts:
                plan = F.plan_for(cost, P, ctx.pool, ctx.cal, reach=reach, tilt=0.0)
                added, _ = plan.code(f.predict, Xs, ws, ctx.cols, G=G)
                Xc = common.adversary.apply_codes(Xs, ctx.cols, ctx.pool, added, config.COUNT_COL, config.SYSTEM_COL)
                coded.append((added, f.predict(Xc), plan.row_cost_.copy()))
            for tilt in config.TILT_GRID:
                base = coding = sel = rise = codes = 0.0
                counts = np.zeros(len(ctx.pool))
                for (f, Xs, ys, ws, P, cost, p0, G), (added, p1, row_cost) in zip(parts, coded):
                    s_ = F.plan_for(cost, P, ctx.pool, ctx.cal, reach=reach, tilt=tilt).select(p1, ws, cost)
                    base += float(np.sum(ws * p0))
                    rise += float(np.sum(ws * (p1 - p0)))
                    coding += float(np.sum(ws * (p1 - p0))) - float(np.sum(ws * row_cost))
                    sel += float(np.sum(ws * (s_ - 1.0) * (p1 - ys)))
                    codes += float(np.sum(ws * added.sum(axis=1)))
                    counts += added.sum(axis=0)
                rows.append({"rep": rep, "fold": fold, "reach": reach, "tilt": tilt,
                             "coding_pct": 100 * coding / base, "selection_pct": 100 * sel / base,
                             "payment_rise_pct": 100 * rise / base,
                             "codes_per_1000": 1000 * codes / ctx.wtr.sum(),
                             "distinct_codes": int((counts > 0).sum()),
                             "top_code_share": float(counts.max() / max(counts.sum(), 1))})
        print(f"  rep {rep} fold {fold}", flush=True)
    df = pd.DataFrame(rows)
    g = df.groupby(["reach", "tilt"]).mean(numeric_only=True).drop(columns=["rep", "fold"]).reset_index()
    g["dist"] = ((g["coding_pct"] / 100 - config.CODING_TARGET) ** 2
                 + (g["selection_pct"] / 100 - config.SELECTION_TARGET) ** 2) ** 0.5
    g.to_csv(config.TABLES / "table4_calibration.csv", index=False)
    best = g.sort_values("dist").iloc[0]
    chosen = {"cost_per_code": config.PRIMARY_COST_PER_CODE, "max_codes": config.PRIMARY_MAX_CODES,
              "audit": config.AUDIT_PER_PCT, "reach": float(best["reach"]), "tilt": float(best["tilt"]),
              "rule": config.SELECTION_RULE,
              "coding_pct": float(best["coding_pct"]), "selection_pct": float(best["selection_pct"]),
              "coding_target_pct": 100 * config.CODING_TARGET, "selection_target_pct": 100 * config.SELECTION_TARGET,
              "distinct_codes": float(best["distinct_codes"]), "top_code_share": float(best["top_code_share"]),
              "calibrated_on": "training rows, plan responding out of sample on each half"}
    (config.DERIVED / "calibration.json").write_text(json.dumps(chosen, indent=2))
    print(g.pivot_table(index="reach", columns="tilt", values="dist").round(3).to_string())
    print(f"\nchosen: {chosen}")

    # codable pool: prevalence, plausible share, coefficient, incremental cost
    Xn = X.to_numpy(np.float32)
    cols = list(X.columns)
    pool = common.ccsr_columns(cols)
    cp = common.cell_prevalence(Xn, cols, pool, w)
    P = common.plausibility(Xn, cols, pool, attrs, cell_prev=cp)
    f = common.make("pwls", cols).fit(Xn, y, w, clusters=cl)
    coef = f.coefficients()
    ic = common.robust.incremental_costs(Xn, y, w, cols, pool, n_conditions_col=config.COUNT_COL)
    labels = pd.read_csv(config.PAPER6 / "data" / "derived" / "ccsr_labels.csv")
    lab = dict(zip(labels.iloc[:, 0], labels.iloc[:, 1]))
    j = {c: cols.index(c) for c in pool}
    tab = pd.DataFrame({
        "flag": [c[5:] for c in pool],
        "label": [lab.get(c[5:].split("+")[0], "") for c in pool],
        "prevalence_pct": [100 * float(np.average(Xn[:, j[c]] > 0, weights=w)) for c in pool],
        "plausible_share_pct": [100 * float(np.average(P[:, q] & (Xn[:, j[c]] == 0), weights=w)) for q, c in enumerate(pool)],
        "cms_form_increment": [coef[c] for c in pool],
        "incremental_cost": [ic[c] for c in pool],
    })
    tab["increment_over_cost"] = tab["cms_form_increment"] / tab["incremental_cost"].replace(0, np.nan)
    tab.sort_values("cms_form_increment", ascending=False).to_csv(config.TABLES / "table_codable_pool.csv", index=False)


if __name__ == "__main__":
    main()
