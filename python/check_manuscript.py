"""
Check that the manuscript's numbers still match the analysis output, that every
table cited is rendered and vice versa, that every figure exists, and that the
prose has no em dashes or placeholders. Writes the word count.

    ../.venv/bin/python python/check_manuscript.py
"""

from __future__ import annotations

import json
import re
import sys

import pandas as pd

import config

MS = config.ROOT / "manuscript" / "Paper7_manuscript.md"
T = config.TABLES


def d(x):
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def checks():
    t = pd.read_csv(T / "table3_extraction.csv").set_index("key")
    cal = json.loads((config.DERIVED / "calibration.json").read_text())
    g = pd.read_csv(T / "table6_group_gaps.csv").set_index(["key", "group"])
    c8 = pd.read_csv(T / "table8_code_increments.csv").set_index("flag")
    c8b = pd.read_csv(T / "table8b_count_coefficients.csv").set_index("formula")
    pool = pd.read_csv(T / "table_codable_pool.csv")
    sw = pd.read_csv(T / "table9_sweeps.csv")

    def _norm(v):
        try:
            return str(float(v))
        except ValueError:
            return str(v)
    sw["value"] = sw["value"].map(_norm)
    stk = pd.read_pickle(config.DERIVED / "oof" / "cms_pen_stack.rows.pkl")

    def fd(k):
        rr = pd.read_pickle(config.DERIVED / "oof" / f"{k}.rows.pkl").set_index(["rep", "fold"])
        return rr["coding"] + rr["selection"]
    swd = sw.set_index(["key", "sweep", "value"])["distinct_codes"]
    sw = sw.set_index(["key", "sweep", "value"])["extraction"]
    rb = pd.read_csv(T / "table10_robustness.csv")
    rb["extraction"] = rb["coding"] + rb["selection"]
    rb = rb.set_index(["variant", "key"])["extraction"]
    r = lambda k, col: t.loc[k, col]  # noqa: E731
    adl = "Needs ADL or IADL help"
    lin = [k for k in t.index if t.loc[k, "family"] in ("reference", "capped", "penalized", "dro", "combined", "adversarial", "stackelberg")
           and t.loc[k, "feature_set"] == "F2"]
    ns = lambda k: 100 * (1 - g.loc[(k, adl), "enrollment_share_post"])  # noqa: E731
    paths = {k: pd.read_csv(T / f"path_{k}.csv") for k in ("cms_adv", "cms_pen_adv", "cms_robust_adv", "gbm_robust_adv")}
    rows = {k: pd.read_pickle(config.DERIVED / "oof" / f"{k}.rows.pkl") for k in paths}
    gp = paths["gbm_robust_adv"]
    first = gp[gp["iter"] == 1].set_index(["rep", "fold"])["extraction"]
    last = gp.sort_values("iter").groupby(["rep", "fold"])["extraction"].last()
    c = {
        "cal": f"reviews {100 * cal['reach']:.0f}% of enrollees with a tilt of {cal['tilt']:.2f}, giving coding of {cal['coding_pct']:.1f}% and selection of {cal['selection_pct']:.1f}% of base payment",
        "cal codes": f"a mean of {cal['distinct_codes']:.0f} distinct codes per fold on the training halves, and the most used code accounts for {100 * cal['top_code_share']:.0f}%",
        "abs cms": f"extracts {d(r('cms', 'extraction'))} per 1,000 enrollees a year from the CMS-form formula, {r('cms', 'extraction_pct'):.1f}% of base payment: {d(r('cms', 'coding'))} from coding and {d(r('cms', 'selection'))} from selection",
        "abs pen": f"cutting extraction to {d(r('cms_pen_1', 'extraction'))}",
        "cms total": f"extracts {d(r('cms', 'extraction'))} per 1,000 enrollees a year, {r('cms', 'extraction_pct'):.1f}% of base payment ({d(r('cms', 'extraction_lo'))} to {d(r('cms', 'extraction_hi'))}), or {d(r('cms', 'extraction') / 1000)} per enrollee",
        "cms folds": f"ranges from {d(r('cms', 'extraction_fold_min'))} to {d(r('cms', 'extraction_fold_max'))}",
        "cms split": f"of which {d(r('cms', 'selection_ungamed'))} comes from tilting enrollment on the formula's ungamed payment and {d(r('cms', 'selection_interaction'))} from tilting",
        "cms codes": f"uses a mean of {r('cms', 'distinct_codes'):.0f} distinct codes per fold, the most used code takes {100 * r('cms', 'top_code_share'):.0f}% of additions",
        "cms F3": f"extracts {d(r('cms_F3', 'extraction'))}",
        "fair": f"{d(r('fair', 'extraction'))} per 1,000, {r('fair', 'extraction_pct'):.1f}% of payment, with {d(r('fair', 'coding'))} from coding",
        "gbm": f"extracts {d(r('gbm', 'extraction'))} in total, {r('gbm', 'extraction_pct'):.1f}% of payment, with selection of {d(r('gbm', 'selection'))}, of which only {d(r('gbm', 'selection_ungamed'))}",
        "gbm cod": f"Its coding gain, {d(r('gbm', 'coding'))}",
        "cap": f"extracts {d(r('cms_cap', 'extraction'))}, a difference from the CMS form of {d(r('cms_cap', 'extraction_vs_cms'))} ({d(r('cms_cap', 'extraction_vs_cms_lo'))} to {d(r('cms_cap', 'extraction_vs_cms_hi'))})",
        "cap ratio": f"a median {pool['increment_over_cost'].median():.2f} of it",
        "codes above cost": f"on that basis {int((c8['increment_cms'] > c8['incremental_cost']).sum())} codes pay more than their incremental cost, {int(((c8['increment_cms'] > c8['incremental_cost']) & (c8['incremental_cost'] == 0)).sum())} of them",
        "pen": f"cuts the coding gain from {d(r('cms', 'coding'))} to {d(r('cms_pen_1', 'coding'))}, {100 * (1 - r('cms_pen_1', 'coding') / r('cms', 'coding')):.0f}%, while R² on ungamed data falls from {r('cms', 'r2_ungamed'):.3f} to {r('cms_pen_1', 'r2_ungamed'):.3f}",
        "pen total": f"Total extraction falls to {d(r('cms_pen_1', 'extraction'))}, {r('cms_pen_1', 'extraction_pct'):.1f}% of payment, {d(-r('cms_pen_1', 'extraction_vs_cms'))} less than the CMS form ({d(-r('cms_pen_1', 'extraction_vs_cms_hi'))} to {d(-r('cms_pen_1', 'extraction_vs_cms_lo'))})",
        "pen spread": f"the most used code takes {100 * r('cms_pen_1', 'top_code_share'):.0f}% of additions",
        "pen inter": f"falls from {d(r('cms', 'selection_interaction'))} to {d(r('cms_pen_1', 'selection_interaction'))}",
        "pen5": f"λ = 5 gives {d(r('cms_pen_5', 'extraction'))} at R² {r('cms_pen_5', 'r2_ungamed'):.3f} and λ = 10 gives {d(r('cms_pen_10', 'extraction'))} at {r('cms_pen_10', 'r2_ungamed'):.3f}",
        "pen50": f"back up to {d(r('cms_pen_50', 'extraction'))} at R² {r('cms_pen_50', 'r2_ungamed'):.3f}",
        "dro": f"to R² {r('cms_dro_5', 'r2_ungamed'):.3f}",
        "dro cod": f"cuts coding to {d(r('cms_dro_5', 'coding'))}",
        "dro range": f"between {d(min(r(k, 'extraction') for k in t.index if k.startswith('cms_dro_')))} and {d(max(r(k, 'extraction') for k in t.index if k.startswith('cms_dro_')))}",
        "combo": f"rises from {d(r('cms_cap_pen_dro_0', 'extraction'))} with no worst-subgroup penalty to {d(r('cms_cap_pen_dro_1', 'extraction'))} at a weight of 1 and {d(r('cms_cap_pen_dro_50', 'extraction'))} at 50",
        "sel range": f"stays between {d(t.loc[lin, 'selection_ungamed'].min())} and {d(t.loc[lin, 'selection_ungamed'].max())} per 1,000",
        "pen adv": f"in a mean of {rows['cms_pen_adv']['iterations'].mean():.1f} rounds, and extracts {d(r('cms_pen_adv', 'extraction'))} per 1,000, against {d(r('cms_pen_1', 'extraction'))}",
        "robust adv": f"in a mean of {rows['cms_robust_adv']['iterations'].mean():.1f} rounds and extracts {d(r('cms_robust_adv', 'extraction'))}, against {d(r('cms_cap_pen_dro_1', 'extraction'))}",
        "cms adv": f"extracts {d(r('cms_adv', 'extraction'))} per 1,000, {r('cms_adv', 'extraction_pct'):.1f}% of payment, {d(-r('cms_adv', 'extraction_vs_cms'))} less than the CMS form ({d(-r('cms_adv', 'extraction_vs_cms_hi'))} to {d(-r('cms_adv', 'extraction_vs_cms_lo'))})",
        "gbm adv": f"extracts {d(r('gbm_robust_adv', 'extraction'))} per 1,000, {r('gbm_robust_adv', 'extraction_pct'):.1f}% of payment, the least of any formula, against {d(r('gbm_cap_dro', 'extraction'))} for the same formula fitted once, with coding of {d(r('gbm_robust_adv', 'coding'))} and R² of {r('gbm_robust_adv', 'r2_ungamed'):.3f}",
        "gbm adv folds": f"ranges from {d(r('gbm_robust_adv', 'extraction_fold_min'))} to {d(r('gbm_robust_adv', 'extraction_fold_max'))}",
        "adl": f"The shortfall is {d(-g.loc[('cms', adl), 'nc_ungamed'])} per person-year under the CMS form, {d(-g.loc[('cms_pen_1', adl), 'nc_ungamed'])} under the coding penalty and {d(-g.loc[('cms_pen_stack', adl), 'nc_ungamed'])} under the Stackelberg-tuned penalty, against {d(-g.loc[('gbm', adl), 'nc_ungamed'])} under boosting",
        "adl fair": f"overpays it, by {d(g.loc[('fair', adl), 'nc_ungamed'])}",
        "adl share": f"It enrolls {ns('cms'):.0f}% fewer people needing help under the CMS form, {ns('cms_pen_1'):.0f}% fewer under the coding penalty and {ns('cms_pen_stack'):.0f}% fewer under the Stackelberg-tuned penalty",
        "adl share fair": f"it enrolls {-ns('fair'):.0f}% more",
        "adl post": f"the shortfall is {d(-g.loc[('cms', adl), 'nc_post'])} per person-year under the CMS form and {d(-g.loc[('cms_pen_1', adl), 'nc_post'])} under the coding penalty",
        "unins": f"enrolls {100 * (g.loc[('gbm', 'Uninsured all year'), 'enrollment_share_post'] - 1):.0f}% more of the uninsured",
        "65 fair": f"overpays them by {d(g.loc[('fair', 'Age 65 and over'), 'nc_ungamed'])}",
        "code top": f"pays {d(c8.loc['NVS015', 'increment_cms'])} when polyneuropathy is added",
        "code top pen": f"the coding penalty pays {d(c8.loc['NVS015', 'increment_cms_pen_1'])}",
        "code range": f"pays between {d(c8.head(20)['increment_cms_robust_adv'].min())} and {d(c8.head(20)['increment_cms_robust_adv'].max())}",
        "code mean": f"falls from {d(c8['increment_cms'].mean())} to {d(c8['increment_cms_pen_1'].mean())}",
        "counts": f"rises from {d(c8b.loc['cms', 'n_conditions'])} per condition under the CMS form to {d(c8b.loc['cms_pen_1', 'n_conditions'])} under the penalty and {d(c8b.loc['cms_robust_adv', 'n_conditions'])}",
        "sw cost": f"moves between {d(sw[('cms', 'cost per code', '0.0')])} and {d(sw[('cms', 'cost per code', '3000.0')])} per 1,000",
        "sw reach": f"the CMS form loses {d(sw[('cms', 'reach', '0.25')])}, the penalized formula {d(sw[('cms_pen_1', 'reach', '0.25')])} and boosting {d(sw[('gbm', 'reach', '0.25')])}",
        "sw codes": f"the CMS form loses {d(sw[('cms', 'codes per person', '3.0')])}, the penalized formula {d(sw[('cms_pen_1', 'codes per person', '3.0')])} and boosting {d(sw[('gbm', 'codes per person', '3.0')])}",
        "sw audit": f"takes {d(sw[('cms', 'audit exposure', '0.0')])}, and at $5,000 per 1% it uses {swd[('cms', 'audit exposure', '5000.0')]:.0f} codes and takes {d(sw[('cms', 'audit exposure', '5000.0')])}",
        "sw tilt": f"the CMS form loses {d(sw[('cms', 'tilt', '0.0')])}, the penalized formula {d(sw[('cms_pen_1', 'tilt', '0.0')])} and boosting {d(sw[('gbm', 'tilt', '0.0')])}",
        "sw any": f"the CMS form loses {d(sw[('cms', 'plausibility', 'any')])}",
        "sw ref": f"takes {d(sw[('cms', 'plan cost model', 'boosting on F3 only')])} from the CMS form, {d(sw[('cms_pen_1', 'plan cost model', 'boosting on F3 only')])} from the penalized formula and {d(sw[('gbm', 'plan cost model', 'boosting on F3 only')])} from boosting",
        "sw wls": f"takes {d(sw[('cms', 'plan cost model', 'WLS on F3')])} from the CMS form, {d(sw[('cms_pen_1', 'plan cost model', 'WLS on F3')])} from the penalized formula and {d(sw[('gbm', 'plan cost model', 'WLS on F3')])} from boosting",
        "sw real": f"the CMS form loses {d(sw[('cms', 'share of added codes real', '1.0')])} per 1,000, the penalized formula {d(sw[('cms_pen_1', 'share of added codes real', '1.0')])}, the robust formula trained against the plan {d(sw[('cms_robust_adv', 'share of added codes real', '1.0')])} and boosting {d(sw[('gbm', 'share of added codes real', '1.0')])}",
        "rob panel": f"gives {d(rb[('leave one panel out', 'cms')])} from the CMS form and {d(rb[('leave one panel out', 'cms_pen_1')])} from the penalized formula",
        "rob F3": f"the penalized formula extracts {d(rb[('F3 (prior use) for the penalized form', 'cms_pen_1_F3')])}, and squared-error boosting extracts {d(rb[('squared-error boosting', 'gbm_mse')])}",
        "rob alpha": f"between {d(rb[('DRO alpha 0.20', 'cms_cap_pen_dro_1_a20')])} at α = 0.20 and {d(rb[('DRO alpha 0.05', 'cms_cap_pen_dro_1_a05')])} at α = 0.05",
        "rob 65": f"gives {d(rb[('persons 65 and over (subgroup of all-age results)', 'cms')])} per 1,000 from the CMS form and {d(rb[('persons 65 and over (subgroup of all-age results)', 'cms_pen_1')])} from the penalized formula",
        "abs stack": f"cuts it slightly further, to {d(r('cms_pen_stack', 'extraction'))}",
        "stack lam": f"chooses λ = {stk['lambda_chosen'].mode().iloc[0]:g} in {int((stk['lambda_chosen'] == stk['lambda_chosen'].mode().iloc[0]).sum())} of the 15 folds",
        "stack": f"extracts {d(r('cms_pen_stack', 'extraction'))}, {r('cms_pen_stack', 'extraction_pct'):.1f}% of payment, at R² {r('cms_pen_stack', 'r2_ungamed'):.3f}",
        "stack vs pen": f"in {int((fd('cms_pen_stack') - fd('cms_pen_1') < 0).sum())} of 15 folds, by {d(-(fd('cms_pen_stack') - fd('cms_pen_1')).mean())} per 1,000 on average",
        "cms adv coding": f"Its coding gain falls to {d(r('cms_adv', 'coding'))}",
        "gbm adv diff": f"lower in {int((fd('gbm_robust_adv') - fd('gbm_cap_dro') < 0).sum())} of the 15 folds, by {d(-(fd('gbm_robust_adv') - fd('gbm_cap_dro')).mean())} on average with a standard deviation across folds of {d((fd('gbm_robust_adv') - fd('gbm_cap_dro')).std())}",
        "mental": f"underpays people with a mental health condition by {d(-g.loc[('cms_pen_stack', 'Mental health condition'), 'nc_ungamed'])}",
        "robust systems": f"falls to {d(c8b.loc['cms_robust_adv', 'n_body_systems'])} under the robust formula trained against the plan",
    }
    return c


def cross_reference(raw):
    tb = config.ROOT / "manuscript" / "tables.md"
    rendered = set(re.findall(r"\*\*Table ([0-9A-Za-z]+)\.\*\*", tb.read_text()))
    tok = r"([0-9]+[a-f]?|A[0-9]+[a-f]?)"
    cited = set()
    for m in re.finditer(rf"Tables? {tok}(?:(?:,\s*|\s+and\s+){tok})*", raw):
        cited.update(re.findall(tok, m.group(0)))
    problems = [f"Table {t} is cited but not rendered" for t in sorted(cited - rendered)]
    problems += [f"Table {t} is rendered but never cited" for t in sorted(rendered - cited)]
    print(f"  {len(rendered)} tables rendered, {len(cited)} cited")
    for p in problems:
        print(f"      {p}")
    return problems


def figures(raw):
    missing = [f for f in re.findall(r"\]\(\.\./(output/figures/[^)]+)\)", raw) if not (config.ROOT / f).exists()]
    for f in missing:
        print(f"      figure missing: {f}")
    return missing


def body_words(raw):
    body = raw.split("## References")[0].split("## 1. Introduction", 1)[-1].split("## Declarations")[0]
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", body)
    body = re.sub(r"^\*\*Figure.*$", "", body, flags=re.M)
    return len(re.findall(r"\b[\w'’-]+\b", body))


def main():
    raw = MS.read_text()
    n = body_words(raw)
    new = re.sub(r"\*\*Word count\.\*\* [0-9,]+", f"**Word count.** {n:,}", raw)
    if new != raw:
        MS.write_text(new)
        raw = new
    text = raw.replace("−", "-")
    c = checks()
    bad = [(a, b) for a, b in c.items() if b.replace("−", "-") not in text]
    width = max(len(a) for a in c)
    for a, b in c.items():
        print(f"  {'ok ' if (a, b) not in bad else 'MISSING'}  {a:<{width}}  {b}")
    xref, figs = cross_reference(raw), figures(raw)
    dashes = raw.count("—")
    left = re.findall(r"\b(?:TODO|XXX|VERIFY|ROBUSTNESS_PARAGRAPH)\b", raw)
    print(f"  body word count {n:,}")
    if dashes:
        print(f"  {dashes} em dash(es)")
    if left:
        print(f"  placeholders left: {sorted(set(left))}")
    if bad or xref or figs or dashes or left:
        print(f"\n{len(bad)} figure(s) do not appear in {MS.name}.")
        sys.exit(1)
    print(f"\nAll {len(c)} figures match the current tables.")


if __name__ == "__main__":
    main()
