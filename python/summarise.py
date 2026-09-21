"""
Tables from the benchmark: accuracy, extraction by channel with Rao-Wu PSU
bootstrap intervals paired across formulas, and group compensation before
and after the plan responds.

Headline accuracy follows Paper 6: the mean over repeats of the R2 of each
repeat's full out-of-fold prediction vector. Extraction is the sum of the
per-row contributions per 1,000 enrollees, bootstrapped over PSUs with the
same resamples for every formula so that differences are paired.

    ../.venv/bin/python python/summarise.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common
import config
import adversarial  # noqa: F401  (registers the adversarial keys)
import formulas as F

OOF = config.DERIVED / "oof"


def load(key):
    z = dict(np.load(OOF / f"{key}.npz", allow_pickle=True))
    z["_c"] = contributions(z)
    rows = pd.read_pickle(OOF / f"{key}.rows.pkl")
    return z, rows


def resamples(cl, st, reps=None):
    rng = np.random.default_rng(config.SEED)
    return [common.metrics.bootstrap_index(cl, st, rng) for _ in range(reps or config.BOOT_REPS)]


def contributions(z):
    """Per-row contributions (carrying w) from the stored arrays, so the
    definitions live here and older runs need no refit."""
    w, y = z["w"], z["y"]
    cost = z["cost_per_code"] if "cost_per_code" in z else float(F.calibration()["cost_per_code"])
    return {"coding": w * (z["p1"] - z["p0"]) - cost * w * z["codes"],
            "selection": w * (z["s"] - 1.0) * (z["p1"] - y),
            "selection_shift": w * (z["s"] - 1.0) * z["p1"]}


def per_1000(z, idx, mult, field):
    """Sum of a per-row field over a resample, per 1,000 of weight, mean over repeats."""
    w = z["w"][idx] * mult
    vals = z["_c"][field][:, idx] * mult
    return float(np.mean(vals.sum(axis=1)) * 1000.0 / w.sum())


def r2_full(z, idx, mult, field):
    y, w = z["y"][idx], z["w"][idx] * mult
    if field == "p1":
        return float(np.mean([common.metrics.r2(y, z["p1"][r, idx], w * z["s"][r, idx]) for r in range(z["p1"].shape[0])]))
    return float(np.mean([common.metrics.r2(y, z["p0"][r, idx], w) for r in range(z["p0"].shape[0])]))


def main():
    keys = [k for k in F.FORMULAS if (OOF / f"{k}.npz").exists()]
    zs = {k: load(k) for k in keys}
    z0 = zs[keys[0]][0]
    cl, st = z0["clusters"], z0["strata"]
    n = len(z0["y"])
    full = (np.arange(n), np.ones(n))
    boots = resamples(cl, st)

    def stat(k, fn, field):
        z = zs[k][0]
        point = fn(z, *full, field)
        b = np.array([fn(z, idx, mult, field) for idx, mult in boots])
        return point, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), b

    rows, draws = [], {}
    for k in keys:
        family, fs, _, label, param = F.FORMULAS[k]
        rec = {"key": k, "label": label, "family": family, "feature_set": fs, "param": param}
        for name, fn, field in (("r2_ungamed", r2_full, "p0"), ("r2_post", r2_full, "p1"),
                                ("coding", per_1000, "coding"), ("selection", per_1000, "selection"),
                                ("selection_shift", per_1000, "selection_shift")):
            p, lo, hi, b = stat(k, fn, field)
            rec[name], rec[f"{name}_lo"], rec[f"{name}_hi"] = p, lo, hi
            draws[(k, name)] = b
        z = zs[k][0]
        rec["extraction"] = rec["coding"] + rec["selection"]
        b = draws[(k, "coding")] + draws[(k, "selection")]
        rec["extraction_lo"], rec["extraction_hi"] = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
        draws[(k, "extraction")] = b
        rec["payment_base"] = per_1000(z, *full, "p0") if False else float(np.mean((z["p0"] * z["w"]).sum(axis=1)) * 1000 / z["w"].sum())
        rec["extraction_pct"] = 100 * rec["extraction"] / rec["payment_base"]
        rec["codes_per_1000"] = float(np.mean((z["codes"] * z["w"]).sum(axis=1)) * 1000 / z["w"].sum())
        rows.append(rec)
    df = pd.DataFrame(rows)
    # paired differences against the CMS form
    if "cms" in keys:
        for name in ("r2_ungamed", "extraction", "coding", "selection"):
            d = {k: draws[(k, name)] - draws[("cms", name)] for k in keys}
            df[f"{name}_vs_cms"] = df["key"].map(lambda k: float(df.set_index("key").loc[k, name] - df.set_index("key").loc["cms", name]))
            df[f"{name}_vs_cms_lo"] = df["key"].map(lambda k: float(np.percentile(d[k], 2.5)))
            df[f"{name}_vs_cms_hi"] = df["key"].map(lambda k: float(np.percentile(d[k], 97.5)))
    df.to_csv(config.TABLES / "table3_extraction.csv", index=False)

    # group compensation before and after, mean over folds with fold spread
    grp = []
    for k in keys:
        rows_k = zs[k][1]
        for col in [c for c in rows_k.columns if c.startswith("nc_ungamed|")]:
            g = col.split("|", 1)[1]
            grp.append({"key": k, "label": F.FORMULAS[k][3], "family": F.FORMULAS[k][0], "group": g,
                        "nc_ungamed": rows_k[col].mean(), "nc_post": rows_k[f"nc_post|{g}"].mean(),
                        "nc_ungamed_fold_min": rows_k[col].min(), "nc_ungamed_fold_max": rows_k[col].max(),
                        "nc_post_fold_min": rows_k[f"nc_post|{g}"].min(), "nc_post_fold_max": rows_k[f"nc_post|{g}"].max(),
                        "enrollment_share_post": rows_k[f"sel_share|{g}"].mean()})
    pd.DataFrame(grp).to_csv(config.TABLES / "table6_group_gaps.csv", index=False)

    show = df[["label", "r2_ungamed", "r2_post", "coding", "selection", "extraction", "extraction_pct", "codes_per_1000"]].copy()
    with pd.option_context("display.width", 220, "display.float_format", "{:,.3f}".format):
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
