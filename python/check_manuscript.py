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
    sw = pd.read_csv(T / "table9_sweeps.csv")
    def _norm(v):
        try:
            return str(float(v))
        except ValueError:
            return str(v)
    sw["value"] = sw["value"].map(_norm)
    sw = sw.set_index(["key", "sweep", "value"])["extraction"]
    r = lambda k, col: t.loc[k, col]  # noqa: E731
    adl = "Needs ADL or IADL help"
    lin = [k for k in t.index if t.loc[k, "family"] in ("reference", "capped", "penalized", "dro", "combined", "adversarial")
           and t.loc[k, "feature_set"] == "F2"]
    c = {
        "cal reach": f"reviewing {100 * cal['reach']:.0f}% of enrollees raises payment by {cal['coding_payment_rise_pct']:.1f}%",
        "cal tilt": f"A tilt of {cal['tilt']:.2f} gives the plan a profit of {cal['selection_profit_share_pct']:.1f}% of payment",
        "cms total": f"extracts {d(r('cms', 'extraction'))} per 1,000 enrollees, {r('cms', 'extraction_pct'):.1f}% of base payment ({d(r('cms', 'extraction_lo'))} to {d(r('cms', 'extraction_hi'))})",
        "cms channels": f"{d(r('cms', 'coding'))} from coding and {d(r('cms', 'selection'))} from selection",
        "cms F3": f"extracts less, {d(r('cms_F3', 'extraction'))}",
        "fair": f"{d(r('fair', 'extraction'))} per 1,000, {r('fair', 'extraction_pct'):.1f}% of payment, with {d(r('fair', 'coding'))} from coding",
        "gbm sel": f"Its selection gain is {d(r('gbm', 'selection'))} per 1,000",
        "gbm cod": f"Its coding gain, {d(r('gbm', 'coding'))}",
        "cap": f"extracts {d(r('cms_cap', 'extraction'))}, a difference from the CMS form of {d(r('cms_cap', 'extraction_vs_cms'))} ({d(r('cms_cap', 'extraction_vs_cms_lo'))} to {d(r('cms_cap', 'extraction_vs_cms_hi'))})",
        "codes above cost": f"Of the 124 codes, {int((c8['increment_cms'] > c8['incremental_cost']).sum())} pay more than their incremental cost",
        "pen coding": f"cuts the coding gain from {d(r('cms', 'coding'))} to {d(r('cms_pen_1', 'coding'))}, {100 * (1 - r('cms_pen_1', 'coding') / r('cms', 'coding')):.0f}%",
        "pen r2": f"falls from {r('cms', 'r2_ungamed'):.3f} to {r('cms_pen_1', 'r2_ungamed'):.3f}",
        "pen total": f"Total extraction falls to {d(r('cms_pen_1', 'extraction'))}, {r('cms_pen_1', 'extraction_pct'):.1f}% of payment",
        "pen50": f"R² {r('cms_pen_50', 'r2_ungamed'):.3f} at λ = 50",
        "dro r2": f"to R² {r('cms_dro_5', 'r2_ungamed'):.3f}",
        "dro coding": f"cuts coding by {100 * (1 - r('cms_dro_5', 'coding') / r('cms', 'coding')):.0f}% to {d(r('cms_dro_5', 'coding'))}, but selection is untouched at {d(r('cms_dro_5', 'selection'))}",
        "combo": f"gives R² {r('cms_cap_pen_dro_1', 'r2_ungamed'):.3f} with {d(r('cms_cap_pen_dro_1', 'coding'))} of coding and {d(r('cms_cap_pen_dro_1', 'extraction'))} in total",
        "sel range": f"selection stays between {d(t.loc[lin, 'selection'].min())} and {d(t.loc[lin, 'selection'].max())} per 1,000",
        "pen adv": f"extracts {d(r('cms_pen_adv', 'extraction'))} per 1,000 ({r('cms_pen_adv', 'extraction_pct'):.1f}% of payment), {d(-r('cms_pen_adv', 'extraction_vs_cms'))} less than the CMS form ({d(-r('cms_pen_adv', 'extraction_vs_cms_hi'))} to {d(-r('cms_pen_adv', 'extraction_vs_cms_lo'))}), at an R² cost of {-r('cms_pen_adv', 'r2_ungamed_vs_cms'):.3f}",
        "robust adv": f"extracts {d(r('cms_robust_adv', 'extraction'))}, with {d(r('cms_robust_adv', 'coding'))} of coding and {d(r('cms_robust_adv', 'selection'))} of selection",
        "cms adv": f"the final formula extracts {d(r('cms_adv', 'extraction'))}, only {100 * (1 - r('cms_adv', 'extraction') / r('cms', 'extraction')):.0f}% less",
        "gbm adv": f"it extracts {d(r('gbm_robust_adv', 'extraction'))}, against {d(r('gbm', 'extraction'))} before training, with coding rising to {d(r('gbm_robust_adv', 'coding'))} and R² falling from {r('gbm', 'r2_ungamed'):.3f} to {r('gbm_robust_adv', 'r2_ungamed'):.3f}",
        "adl": f"The shortfall is {d(-g.loc[('cms', adl), 'nc_ungamed'])} per person-year under the CMS form, {d(-g.loc[('cms_pen_1', adl), 'nc_ungamed'])} under the coding penalty and {d(-g.loc[('cms_robust_adv', adl), 'nc_ungamed'])} under the robust formula trained against the plan, against {d(-g.loc[('gbm', adl), 'nc_ungamed'])} under boosting",
        "adl fair": f"overpays it, by {d(g.loc[('fair', adl), 'nc_ungamed'])}",
        "adl share": f"enrolls {100 * (1 - g.loc[('cms_robust_adv', adl), 'enrollment_share_post']):.0f}% fewer people needing help",
        "adl share gbm": f"it enrolls {100 * (g.loc[('gbm', adl), 'enrollment_share_post'] - 1):.0f}% more",
        "65": f"from an overpayment of {d(g.loc[('cms', 'Age 65 and over'), 'nc_ungamed'])} under the CMS form to an underpayment of {d(-g.loc[('cms_robust_adv', 'Age 65 and over'), 'nc_ungamed'])}",
        "code top": f"pays {d(c8.loc['NVS015', 'increment_cms'])} when polyneuropathy is added",
        "code top adv": f"the robust formula trained against the plan pays {d(c8.loc['NVS015', 'increment_cms_robust_adv'])}",
        "code mean": f"falls from {d(c8['increment_cms'].mean())} to {d(c8['increment_cms_robust_adv'].mean())}",
        "counts": f"rises from {d(c8b.loc['cms', 'n_conditions'])} per condition under the CMS form to {d(c8b.loc['cms_robust_adv', 'n_conditions'])}",
        "systems": f"falls from {d(c8b.loc['cms', 'n_body_systems'])} to {d(c8b.loc['cms_robust_adv', 'n_body_systems'])}",
        "sweep cost": f"moves between {d(sw[('cms', 'cost per code', '0.0')])} and {d(sw[('cms', 'cost per code', '3000.0')])} per 1,000",
        "sweep reach": f"loses {d(sw[('cms', 'reach', '0.25')])} per 1,000 and the penalized formula {d(sw[('cms_pen_1', 'reach', '0.25')])}",
        "sweep tilt0": f"the CMS form loses {d(sw[('cms', 'tilt', '0.0')])} and the penalized formula {d(sw[('cms_pen_1', 'tilt', '0.0')])}",
        "sweep any": f"the CMS form loses {d(sw[('cms', 'plausibility', 'any')])}",
        "sweep wls": f"takes {d(sw[('cms', 'plan cost model', 'WLS')])}",
        "sweep real": f"the CMS form loses {d(sw[('cms', 'share of added codes real', '1.0')])} per 1,000 and the robust formula trained against the plan {d(sw[('cms_robust_adv', 'share of added codes real', '1.0')])}",
        "sweep real pen": f"({d(sw[('cms_pen_1', 'share of added codes real', '1.0')])})",
    }
    for k, (key, col) in {"path gbm": ("gbm_robust_adv", None)}.items():
        p = pd.read_csv(T / f"path_{key}.csv").groupby("iter")["extraction"].mean()
        c[k] = f"rises from {d(p.iloc[0])} to {d(p.iloc[-1])} per 1,000"
    ra = pd.read_pickle(config.DERIVED / "oof" / "cms_robust_adv.rows.pkl")
    c["robust adv conv"] = f"in {int(ra['converged'].sum())} of {len(ra)} folds"
    rb = T / "table10_robustness.csv"
    if rb.exists() and "ROBUSTNESS_PARAGRAPH" not in MS.read_text():
        rr = pd.read_csv(rb).set_index(["variant", "key"])
        rr["extraction"] = rr["coding"] + rr["selection"]
        c["rob 65"] = (f"{d(rr.loc[('persons 65 and over', 'cms'), 'extraction'])} per 1,000 from the CMS form and "
                       f"{d(rr.loc[('persons 65 and over', 'cms_pen_adv'), 'extraction'])} from the penalized formula trained against the plan")
        c["rob panel"] = (f"{d(rr.loc[('leave one panel out', 'cms'), 'extraction'])} from the CMS form and "
                          f"{d(rr.loc[('leave one panel out', 'cms_pen_1'), 'extraction'])} from the penalized formula")
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
