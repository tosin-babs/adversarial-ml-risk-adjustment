"""
Render the manuscript's tables from the analysis CSVs. Writes manuscript/tables.md.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config

OUT = config.ROOT / "manuscript" / "tables.md"
T = config.TABLES

MAIN_KEYS = ["cms", "cms_F3", "wls", "gbm", "fair", "cms_cap", "cms_pen_1", "cms_dro_5", "cms_cap_pen_dro_1",
             "gbm_cap_dro", "cms_adv", "cms_pen_adv", "cms_robust_adv", "gbm_robust_adv"]
GROUPS = ["Needs ADL or IADL help", "Age 65 and over", "Uninsured all year", "Income below 200% FPL",
          "Mental health condition", "Non-Hispanic Black", "Hispanic", "Non-Hispanic Asian"]


def num(d=2):
    return lambda x: "" if pd.isna(x) else f"{x:,.{d}f}"


def dollars(d=0):
    return lambda x: "" if pd.isna(x) else (f"-${abs(x):,.{d}f}" if x < 0 else f"${x:,.{d}f}")


def pct(d=1):
    return lambda x: "" if pd.isna(x) else f"{x:,.{d}f}%"


def interval(lo, hi, f=dollars(0)):
    return lambda r: f"({f(r[lo])} to {f(r[hi])})"


def render(df, cols, fmts, headers=None):
    out = pd.DataFrame({c: df[c].map(f) if f else df[c].astype(str) for c, f in zip(cols, fmts)})
    out.columns = headers or cols
    align = ["---" if i == 0 else "---:" for i in range(len(out.columns))]
    lines = ["| " + " | ".join(out.columns) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for _, r in out.iterrows()]
    return "\n".join(lines)


def caption(n, title, note=None):
    s = f"\n**Table {n}.** {title}\n"
    return s + (f"\n*{note}*\n" if note else "")


def main():
    cal = json.loads((config.DERIVED / "calibration.json").read_text())
    parts = ["# Tables\n",
             "*Generated from `output/tables/*.csv` by `python/make_tables.py`. Spending is total "
             "expenditure from all payers in 2024 dollars; extraction is per 1,000 enrollees of survey "
             "weight. Every model quantity is out of fold from the five survey folds and three repeats of "
             "the Paper 6 benchmark, with primary sampling units kept whole. Intervals are from a Rao-Wu "
             "rescaled bootstrap over PSUs within strata (200 resamples), paired across formulas.*\n"]

    # 1 formulas
    t3 = pd.read_csv(T / "table3_extraction.csv")
    rob = {"reference": "none", "capped": "cap at incremental cost", "penalized": "coding penalty",
           "dro": "worst-subgroup penalty", "combined": "cap, coding penalty, worst-subgroup penalty",
           "boosted": "cap, worst-subgroup penalty on the linear layer",
           "adversarial": "as the base formula"}
    f1 = t3[t3["key"].isin(MAIN_KEYS)].copy()
    f1["robust"] = f1["family"].map(rob)
    f1["trained"] = np.where(f1["family"] == "adversarial", "yes", "no")
    f1["order"] = f1["key"].map({k: i for i, k in enumerate(MAIN_KEYS)})
    f1 = f1.sort_values("order")
    parts += [caption(1, "The formulas compared.",
                      "F2 is demographics and CCSR condition flags; F3 adds year-1 use and spending. The CMS "
                      "form is weighted least squares with a base rate for each age band and sex, no intercept, "
                      "and non-negative increments for everything else. Penalties are at lambda = 1 unless the "
                      "label says otherwise; the DRO alpha is 0.10."),
              render(f1, ["label", "feature_set", "robust", "trained"], [None, None, None, None],
                     ["Formula", "Features", "Robust term", "Trained against the plan"])]

    # 2 calibration
    c = pd.read_csv(T / "table4_calibration.csv")
    cod = c[(c["channel"] == "coding") & (c["max_codes"] == cal["max_codes"]) & (c["cost_per_code"] == cal["cost_per_code"])]
    sel = c[c["channel"] == "selection"]
    parts += [caption(2, "Calibrating the plan against the CMS form.",
                      f"Coding: one code per reviewed person at ${cal['cost_per_code']:,.0f} per code, by chart-review "
                      f"reach; the calibrated reach is {100 * cal['reach']:.0f}%, where the gross payment rise is "
                      f"closest to the target of {cal['coding_target_pct']:.1f}% (MedPAC's 16% coding intensity less "
                      f"the 5.9% statutory adjustment). Selection: threshold rule by tilt; the calibrated tilt is "
                      f"{cal['tilt']:.2f}, where the plan's profit is closest to {cal['selection_target_pct']:.1f}% of "
                      f"payment (MedPAC's $44 billion of favorable selection on about $500 billion of payment). "
                      "Means over the 15 test folds."),
              render(cod, ["reach", "payment_rise_pct", "net_rise_pct", "codes_per_1000"],
                     [pct(0) if False else (lambda x: f"{100 * x:.0f}%"), pct(1), pct(1), num(0)],
                     ["Reach", "Payment rise", "Net of code cost", "Codes per 1,000"]),
              "",
              render(sel, ["tilt", "payment_rise_pct", "profit_share_pct"], [num(2), pct(1), pct(1)],
                     ["Tilt", "Payment shift", "Profit, share of payment"])]

    # 3 extraction
    m = t3[t3["key"].isin(MAIN_KEYS)].copy()
    m["order"] = m["key"].map({k: i for i, k in enumerate(MAIN_KEYS)})
    m = m.sort_values("order")
    m["ci_c"] = m.apply(interval("coding_lo", "coding_hi"), axis=1)
    m["ci_s"] = m.apply(interval("selection_lo", "selection_hi"), axis=1)
    m["ci_e"] = m.apply(interval("extraction_lo", "extraction_hi"), axis=1)
    parts += [caption(3, "What the calibrated plan extracts from each formula, per 1,000 enrollees.",
                      "Coding: payment for added codes, net of the cost of adding them, for codes that change "
                      "nothing about cost. Selection: payment above cost from tilting enrollment toward the "
                      "half of enrollees the plan expects to be overpaid. Share is of base payment. 95% "
                      "bootstrap intervals in parentheses."),
              render(m, ["label", "coding", "ci_c", "selection", "ci_s", "extraction", "ci_e", "extraction_pct"],
                     [None, dollars(), None, dollars(), None, dollars(), None, pct(1)],
                     ["Formula", "Coding", "", "Selection", "", "Total", "", "Share"])]

    # 4 accuracy
    m["ci_r"] = m.apply(interval("r2_ungamed_lo", "r2_ungamed_hi", num(3)), axis=1)
    m["ci_p"] = m.apply(interval("r2_post_lo", "r2_post_hi", num(3)), axis=1)
    parts += [caption(4, "Accuracy before and after the plan responds.",
                      "R² on ungamed data is what the regulator gives up; R² on post-response data, with the "
                      "plan's coded features and tilted weights, is what it experiences."),
              render(m, ["label", "r2_ungamed", "ci_r", "r2_post", "ci_p", "r2_ungamed_vs_cms"],
                     [None, num(3), None, num(3), None, num(3)],
                     ["Formula", "R², ungamed", "", "R², post-response", "", "Difference from CMS form"])]

    # 5 frontier
    fr = t3[t3["family"].isin(["penalized", "dro", "combined", "reference", "capped"]) &
            t3["key"].isin(["cms", "cms_cap"] + [k for k in t3["key"] if k.startswith(("cms_pen_", "cms_dro_", "cms_cap_pen_dro_"))])].copy()
    order = {"reference": 0, "capped": 1, "penalized": 2, "dro": 3, "combined": 4}
    fr["o"] = fr["family"].map(order)
    fr = fr.sort_values(["o", "param"])
    parts += [caption(5, "The price of robustness: accuracy against extraction along each penalty grid.",
                      "Linear payment-form formulas on F2. Lambda is the penalty weight; the DRO alpha is 0.10."),
              render(fr, ["label", "r2_ungamed", "coding", "selection", "extraction", "extraction_pct"],
                     [None, num(3), dollars(), dollars(), dollars(), pct(1)],
                     ["Formula", "R², ungamed", "Coding", "Selection", "Total", "Share"])]

    # 6 group gaps
    g = pd.read_csv(T / "table6_group_gaps.csv")
    keys6 = [k for k in ("cms", "cms_pen_1", "cms_robust_adv", "gbm", "gbm_robust_adv", "fair") if k in set(g["key"])]
    g6 = g[g["key"].isin(keys6) & g["group"].isin(GROUPS)].copy()
    wide = g6.pivot_table(index="group", columns="key", values=["nc_ungamed", "nc_post"]).reindex(GROUPS)
    rows = []
    for grp in GROUPS:
        r = {"group": grp}
        for k in keys6:
            r[f"u_{k}"] = wide.loc[grp, ("nc_ungamed", k)]
            r[f"p_{k}"] = wide.loc[grp, ("nc_post", k)]
        rows.append(r)
    g6w = pd.DataFrame(rows)
    labels = dict(zip(t3["key"], t3["label"]))
    cols = ["group"] + [c for k in keys6 for c in (f"u_{k}", f"p_{k}")]
    heads = ["Group"] + [h for k in keys6 for h in (f"{labels[k]}, ungamed", "post-response")]
    parts += [caption(6, "Net compensation by group before and after the plan responds: predicted minus "
                         "observed spending per person-year.",
                      "Negative is underpayment. Post-response values use the plan's coded features and tilted "
                      "weights, so they include the coding rise. Race and ethnicity are evaluated only."),
              render(g6w, cols, [None] + [dollars()] * (len(cols) - 1), heads)]

    # 7 training path
    paths = []
    for k in ("cms_adv", "cms_pen_adv", "cms_robust_adv", "gbm_robust_adv"):
        f = T / f"path_{k}.csv"
        if f.exists():
            p = pd.read_csv(f)
            a = p.groupby("iter").agg(extraction=("extraction", "mean"), coding=("coding", "mean"),
                                      selection=("selection", "mean"), change=("change", "mean"),
                                      folds=("rep", "size")).reset_index()
            a["formula"] = labels.get(k, k)
            paths.append(a[a["iter"].isin([1, 2, 3, 4, 5, 10, 15, 20])])
    if paths:
        parts += [caption(7, "Training against the plan: extraction on the training rows by iteration.",
                          "Iteration 1 is the plan's response to the formula fitted on ungamed data. Change is "
                          "the mean absolute movement of payments as a share of mean payment; the loop stops "
                          "below 1%. Folds is the number of splits still iterating."),
                  render(pd.concat(paths), ["formula", "iter", "extraction", "coding", "selection", "change", "folds"],
                         [None, num(0), dollars(), dollars(), dollars(), pct(2) if False else (lambda x: f"{100 * x:.2f}%"), num(0)],
                         ["Formula", "Iteration", "Extraction", "Coding", "Selection", "Change", "Folds"])]

    # 8 codes
    t8 = pd.read_csv(T / "table8_code_increments.csv").head(20)
    parts += [caption(8, "The 20 codes the CMS form pays most for, and what the robust formulas pay for them.",
                      "Full-sample fits. Increment is the payment rise when the code is added to a person who "
                      "could plausibly receive it. Incremental cost is the weighted difference in year-2 "
                      "spending between people with and without the code, within age, sex and number of other "
                      "conditions."),
              render(t8, ["flag", "label", "prevalence_pct", "incremental_cost", "increment_cms", "increment_cms_pen_1",
                          "increment_cms_cap_pen_dro_1", "increment_cms_robust_adv"],
                     [None, None, pct(1), dollars(), dollars(), dollars(), dollars(), dollars()],
                     ["CCSR", "Condition", "Prevalence", "Incremental cost", "CMS form", "Penalized", "Robust",
                      "Robust, trained"])]
    t8b = pd.read_csv(T / "table8b_count_coefficients.csv")
    parts += [caption("8b", "Payment per condition and per body system under each formula (full-sample fits)."),
              render(t8b, ["formula", "n_conditions", "n_body_systems"], [None, dollars(), dollars()],
                     ["Formula", "Per condition", "Per body system"])]

    # 9 sweeps
    if (T / "table9_sweeps.csv").exists():
        s = pd.read_csv(T / "table9_sweeps.csv")
        # the smooth rule has no bound on the tilt and is not comparable
        s = s[s["sweep"] != "selection rule"]
        w = s.pivot_table(index=["sweep", "value"], columns="key", values="extraction", sort=False).reset_index()
        ks = [k for k in ("cms", "cms_pen_1", "cms_robust_adv") if k in w.columns]
        parts += [caption(9, "Sensitivity of extraction to the plan, one parameter at a time from the calibrated point.",
                          "Per 1,000 enrollees; mean over the 15 test folds. The last block makes a share of the "
                          "added codes real, raising the person's cost by the code's incremental cost."),
                  render(w, ["sweep", "value"] + ks, [None, None] + [dollars()] * len(ks),
                         ["Parameter", "Value"] + [labels[k] for k in ks])]

    # 10 robustness
    if (T / "table10_robustness.csv").exists():
        r = pd.read_csv(T / "table10_robustness.csv")
        parts += [caption(10, "Robustness rows.",
                          "The 65-and-over rows restrict the stored out-of-fold results to that age group. "
                          "Other rows are refits with one repeat of the five folds; leave-one-panel-out holds each "
                          "MEPS panel out in turn."),
                  render(r, ["variant", "label", "r2_ungamed", "coding", "selection", "extraction"],
                         [None, None, num(3), dollars(), dollars(), dollars()],
                         ["Variant", "Formula", "R², ungamed", "Coding", "Selection", "Total"])]

    # A1 codable pool
    parts += ["\n\n# Appendix tables\n"]
    a = pd.read_csv(T / "table_codable_pool.csv").head(25)
    parts += [caption("A1", "The codable pool: the 25 codes with the largest CMS-form increments.",
                      "Plausible share is the weighted share of people who lack the code and could receive it "
                      "under the primary rule (a code in the same body system already present, or year-1 "
                      "spending in the top half)."),
              render(a, ["flag", "label", "prevalence_pct", "plausible_share_pct", "cms_form_increment", "incremental_cost"],
                     [None, None, pct(1), pct(1), dollars(), dollars()],
                     ["CCSR", "Condition", "Prevalence", "Plausible share", "CMS-form coefficient", "Incremental cost"])]

    OUT.write_text("\n".join(parts) + "\n")
    n = sum(1 for p in parts if p.startswith("\n**Table"))
    print(f"wrote {OUT.relative_to(config.ROOT)} with {n} tables")


if __name__ == "__main__":
    main()
