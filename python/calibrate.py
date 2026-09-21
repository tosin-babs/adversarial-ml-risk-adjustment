"""
Calibrate the plan against the CMS-form formula and write the codable pool.

For the payment-form WLS on F2 (the closest analogue to CMS-HCC), on each of
the 15 test folds, the plan's coding response is run over the grid of cost
per code, codes per person and chart-review reach with selection off, and
its selection response over the tilt grid with coding off. The primary
triple is the grid point whose gross payment rise is closest to
CODING_TARGET, and the primary tilt the one whose selection profit share is
closest to SELECTION_TARGET.

Writes table4_calibration.csv (the full grid), calibration.json (the chosen
parameters) and table_codable_pool.csv.

    ../.venv/bin/python python/calibrate.py
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

import common
import config


class _Fixed:
    """A cost model that returns stored predictions for the rows it is given."""

    def __init__(self, values):
        self.values = values

    def predict(self, X):
        return self.values


def fold_formulas():
    """The CMS-form formula fitted on each training fold, with the plan's
    test-fold cost predictions."""
    X, y, w, cl, st, attrs = common.build(config.PRIMARY_FEATURE_SET)
    Xn = X.to_numpy(np.float32)
    cols = list(X.columns)
    pool = common.ccsr_columns(cols)
    out = []
    for rep, fold, tr, te in common.folds(cl, st):
        f = common.make("pwls", cols).fit(Xn[tr], y[tr], w[tr], clusters=cl[tr])
        plan = np.load(config.DERIVED / "plan" / f"rep{rep}_fold{fold}.npz")
        assert np.array_equal(plan["test_idx"], te)
        P = common.plausibility(Xn[te], cols, pool, attrs.iloc[te])
        out.append((rep, fold, f, Xn[te], y[te], w[te], P, plan["cost_test"]))
    return cols, pool, out


def main():
    cols, pool, folds = fold_formulas()
    rows = []
    for rep, fold, f, Xte, yte, wte, P, cost in folds:
        base = float(np.sum(wte * f.predict(Xte)))
        G = common.adversary.gain_matrix(f.predict, Xte, cols, pool, P, config.COUNT_COL, config.SYSTEM_COL)
        for c, k, r in itertools.product(config.COST_PER_CODE_GRID, config.MAX_CODES_GRID, config.REACH_GRID):
            plan = common.adversary.Plan(cost_per_code=c, max_codes=k, reach=r, tilt=0.0, pool=pool,
                                         plausible=P, count_col=config.COUNT_COL, system_col=config.SYSTEM_COL)
            added, _ = plan.code(f.predict, Xte, wte, cols)
            Xc = common.adversary.apply_codes(Xte, cols, pool, added, config.COUNT_COL, config.SYSTEM_COL)
            gross = float(np.sum(wte * (f.predict(Xc) - f.predict(Xte))))
            codes = float(np.sum(wte * added.sum(axis=1)))
            rows.append({"rep": rep, "fold": fold, "channel": "coding", "cost_per_code": c, "max_codes": k,
                         "reach": r, "tilt": np.nan, "payment_rise_pct": 100 * gross / base,
                         "net_rise_pct": 100 * (gross - c * codes) / base,
                         "codes_per_1000": 1000 * codes / wte.sum()})
        pay = f.predict(Xte)
        for tilt in config.TILT_GRID:
            plan = common.adversary.Plan(cost_model=_Fixed(cost), tilt=tilt, rule=config.SELECTION_RULE)
            s = plan.select(pay, wte, cost)
            profit = float(np.sum(wte * (s - 1.0) * (pay - yte)))
            rows.append({"rep": rep, "fold": fold, "channel": "selection", "cost_per_code": np.nan,
                         "max_codes": np.nan, "reach": np.nan, "tilt": tilt,
                         "payment_rise_pct": 100 * float(np.sum(wte * (s - 1.0) * pay)) / base,
                         "profit_share_pct": 100 * profit / base})
    df = pd.DataFrame(rows)
    keys = ["channel", "cost_per_code", "max_codes", "reach", "tilt"]
    g = df.groupby(keys, dropna=False).agg(payment_rise_pct=("payment_rise_pct", "mean"),
                                           net_rise_pct=("net_rise_pct", "mean"),
                                           profit_share_pct=("profit_share_pct", "mean"),
                                           codes_per_1000=("codes_per_1000", "mean")).reset_index()
    g.to_csv(config.TABLES / "table4_calibration.csv", index=False)

    cod = g[g["channel"] == "coding"].copy()
    cod["dist"] = (cod["payment_rise_pct"] / 100 - config.CODING_TARGET).abs()
    # one code per reviewed person at the stated cost; reach is what calibrates
    prim = cod[(cod["max_codes"] == config.PRIMARY_MAX_CODES) & (cod["cost_per_code"] == config.PRIMARY_COST_PER_CODE)]
    best = prim.sort_values("dist").iloc[0]
    sel = g[g["channel"] == "selection"].copy()
    sel["dist"] = (sel["profit_share_pct"] / 100 - config.SELECTION_TARGET).abs()
    best_s = sel.sort_values("dist").iloc[0]
    chosen = {"cost_per_code": float(best["cost_per_code"]), "max_codes": int(best["max_codes"]),
              "reach": float(best["reach"]), "coding_payment_rise_pct": float(best["payment_rise_pct"]),
              "coding_target_pct": 100 * config.CODING_TARGET,
              "tilt": float(best_s["tilt"]), "selection_profit_share_pct": float(best_s["profit_share_pct"]),
              "selection_target_pct": 100 * config.SELECTION_TARGET, "rule": config.SELECTION_RULE}
    (config.DERIVED / "calibration.json").write_text(json.dumps(chosen, indent=2))
    print("coding grid (mean over folds), payment rise %:")
    print(cod.pivot_table(index=["cost_per_code", "max_codes"], columns="reach", values="payment_rise_pct").round(1).to_string())
    print("\nselection, profit share %:")
    print(sel[["tilt", "payment_rise_pct", "profit_share_pct"]].round(2).to_string(index=False))
    print(f"\nchosen: {chosen}")

    # codable pool: prevalence, plausible share, CMS-form increment, incremental cost
    X, y, w, cl, st, attrs = common.build(config.PRIMARY_FEATURE_SET)
    Xn = X.to_numpy(np.float32)
    P = common.plausibility(Xn, cols, pool, attrs)
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
    print(f"\ncodable pool: {len(tab)} flags; "
          f"{(tab['cms_form_increment'] > tab['incremental_cost']).sum()} pay more than their incremental cost")


if __name__ == "__main__":
    main()
