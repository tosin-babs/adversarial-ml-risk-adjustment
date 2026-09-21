"""
Bridge to Paper 6: its configuration, data, feature sets, folds, models and
the riskfair package (with the adversary and robust modules this paper adds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402



def _load_paper6():
    """Import Paper 6's config, features and common under their own names.
    They import each other as `config`, `features` and `common`, the same
    names this paper uses, so they are loaded with those names bound to
    Paper 6's modules and then restored."""
    import importlib
    saved = {k: sys.modules.get(k) for k in ("config", "features", "common")}
    for k in saved:
        sys.modules.pop(k, None)
    path = list(sys.path)
    sys.path[:0] = [str(config.PAPER6 / "python"), str(config.PAPER6)]
    try:
        p6 = {k: importlib.import_module(k) for k in ("config", "features", "common")}
    finally:
        sys.path[:] = path
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)
    return p6["config"], p6["features"], p6["common"]


p6config, p6features, p6common = _load_paper6()
sys.path.insert(0, str(config.PAPER6))
from riskfair import adversary, coding, fairness, metrics, models, robust, surveycv  # noqa: E402,F401

CACHE = {}


def data():
    if "data" not in CACHE:
        CACHE["data"] = p6features.load()
    return CACHE["data"]


def build(feature_set=None):
    """X (DataFrame), y, w, clusters, strata, attrs, plus year-1 spending for
    the plausibility rule."""
    fs = feature_set or config.PRIMARY_FEATURE_SET
    key = f"build_{fs}"
    if key not in CACHE:
        d, c = data()
        X, y, w, cl, st, attrs = p6features.build(fs, data=(d, c))
        attrs = attrs.assign(spend_y1=d["spend_y1"].fillna(0).to_numpy(float))
        CACHE[key] = (X, y, w, cl, st, attrs)
    return CACHE[key]


def folds(cl, st, repeats=None):
    return p6common.folds(cl, st, repeats)


def make(key, columns=None):
    return p6common.make(key, columns)


def masks(attrs):
    return p6features.group_masks(attrs)


def ccsr_columns(columns):
    return [c for c in columns if c.startswith("ccsr_")]


def plausibility(X, columns, pool, attrs, rule=None):
    """(n, len(pool)) bool matrix under the named rule."""
    rule = rule or config.PLAUSIBILITY
    Xn = np.asarray(X, float)
    by_system = adversary.plausible_by_system(Xn, columns, pool)
    spend = attrs["spend_y1"].to_numpy(float)
    top_half = spend >= np.median(spend[spend > 0]) if (spend > 0).any() else np.zeros(len(spend), bool)
    by_use = np.repeat(top_half[:, None], len(pool), axis=1)
    if rule == "system":
        return by_system
    if rule == "use":
        return by_use
    if rule == "any":
        return np.ones_like(by_system)
    return by_system | by_use
