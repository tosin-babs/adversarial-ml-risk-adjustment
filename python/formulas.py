"""
The formulas compared, and the per-fold context they need.

`context(feature_set, rep, fold, tr, te)` returns everything a formula or a
plan needs on one split: the plausibility matrices, the codable-pool weights
(share of people each flag could be added to, with the two count columns at
one because every added code raises them), the caps (incremental cost of each
flag on the training rows), and a Plan for the training rows (with the plan's
out-of-fold cost predictions) and for the test rows.

`FORMULAS` maps a key to how to build the formula from a context. Robust
penalties are scaled so lam is comparable across the linear and boosted
classes: the objective is the weighted mean square plus lam times the penalty.
"""

from __future__ import annotations

import json

import numpy as np

import common
import config


class Fixed:
    """A cost model that returns stored predictions for the rows it is given."""

    def __init__(self, values):
        self.values = np.asarray(values, float)

    def predict(self, X):
        assert len(X) == len(self.values), "Fixed cost model applied to the wrong rows"
        return self.values


def calibration():
    return json.loads((config.DERIVED / "calibration.json").read_text())


def plan_for(cost, plausible, pool, cal=None, tilt=None, cost_per_code=None, max_codes=None,
             reach=None, rule=None):
    cal = cal or calibration()
    return common.adversary.Plan(
        cost_model=Fixed(cost),
        cost_per_code=cal["cost_per_code"] if cost_per_code is None else cost_per_code,
        max_codes=cal["max_codes"] if max_codes is None else max_codes,
        reach=cal["reach"] if reach is None else reach,
        tilt=cal["tilt"] if tilt is None else tilt,
        rule=cal["rule"] if rule is None else rule,
        pool=pool, plausible=plausible, count_col=config.COUNT_COL, system_col=config.SYSTEM_COL)


class Context:
    def __init__(self, feature_set, rep, fold, tr, te, plausibility=None, cal=None):
        X, y, w, cl, st, attrs = common.build(feature_set)
        self.fs, self.rep, self.fold, self.tr, self.te = feature_set, rep, fold, tr, te
        self.cols = list(X.columns)
        self.pool = common.ccsr_columns(self.cols)
        Xn = X.to_numpy(np.float32)
        self.Xtr, self.ytr, self.wtr, self.cltr = Xn[tr], y[tr], w[tr], cl[tr]
        self.Xte, self.yte, self.wte = Xn[te], y[te], w[te]
        self.attrs_tr, self.attrs_te = attrs.iloc[tr], attrs.iloc[te]
        self.masks_tr = common.masks(self.attrs_tr)
        self.masks_te = common.masks(self.attrs_te)
        rule = plausibility or config.PLAUSIBILITY
        self.Ptr = common.plausibility(self.Xtr, self.cols, self.pool, self.attrs_tr, rule)
        self.Pte = common.plausibility(self.Xte, self.cols, self.pool, self.attrs_te, rule)
        plan = np.load(config.DERIVED / "plan" / f"rep{rep}_fold{fold}.npz")
        assert np.array_equal(plan["train_idx"], tr) and np.array_equal(plan["test_idx"], te)
        self.cost_tr, self.cost_te = plan["cost_train_oof"], plan["cost_test"]
        self.cal = cal or calibration()
        self.plan_tr = plan_for(self.cost_tr, self.Ptr, self.pool, self.cal)
        self.plan_te = plan_for(self.cost_te, self.Pte, self.pool, self.cal)
        # codable-pool weights: share of training people each flag could be
        # added to; the count columns at one
        j = {c: self.cols.index(c) for c in self.pool}
        self.pool_weights = {c: float(np.average(self.Ptr[:, q] & (self.Xtr[:, j[c]] == 0), weights=self.wtr))
                             for q, c in enumerate(self.pool)}
        self.pool_weights.update({config.COUNT_COL: 1.0, config.SYSTEM_COL: 1.0})
        self.caps = common.robust.incremental_costs(self.Xtr, self.ytr, self.wtr, self.cols, self.pool,
                                                    n_conditions_col=config.COUNT_COL)


class Normalized:
    """Any formula, normalized so that total payment equals total cost on the
    rows it was fitted to, as CMS applies a normalization factor. Formulas
    that already normalize (the DRO fits) are unchanged by it."""

    def __init__(self, model):
        self.model = model
        self.name = getattr(model, "name", type(model).__name__)

    def fit(self, X, y, w, clusters=None):
        # A column that is zero on every training row has an arbitrary
        # coefficient; it is zeroed at prediction too. This happens only when
        # a whole MEPS panel is held out: the two dental CCSR categories first
        # appear in panel 27.
        self.dead_ = np.all(np.asarray(X) == 0, axis=0)
        self.model.fit(X, y, w, clusters=clusters)
        p = self.model.predict(X)
        self.factor_ = float(np.sum(w * y) / np.sum(w * p))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        if self.dead_.any():
            X = X.copy()
            X[:, self.dead_] = 0.0
        return self.factor_ * self.model.predict(X)

    def coefficients(self):
        return {k: self.factor_ * v for k, v in self.model.coefficients().items()}

    def __getattr__(self, name):
        return getattr(self.model, name)


def _cms(ctx):
    return common.make("pwls", ctx.cols)


def _pen(lam):
    return lambda ctx: common.robust.PenalizedPaymentWLS(lam, ctx.pool_weights).set_columns(ctx.cols)


def _cap(ctx):
    return common.robust.CappedPaymentWLS(ctx.caps).set_columns(ctx.cols)


def _dro(lam, alpha=None):
    # the reference expected cost is the regulator's cross-fitted flexible
    # model on the training rows (the same model the plan uses, in the
    # primary specification)
    return lambda ctx: common.robust.DROPaymentWLS(lam, alpha or config.DRO_ALPHA,
                                                   reference=ctx.cost_tr).set_columns(ctx.cols)


def _combo(lam_code, lam_dro, capped=False, alpha=None):
    return lambda ctx: common.robust.DROPaymentWLS(
        lam_dro, alpha or config.DRO_ALPHA, ctx.caps if capped else None,
        lam_code, ctx.pool_weights, reference=ctx.cost_tr).set_columns(ctx.cols)


def _stacked_robust(lam_dro, capped=True, lam_code=0.0):
    return lambda ctx: common.robust.StackedRobustGBM(
        ctx.cols, lam_dro, config.DRO_ALPHA, ctx.caps if capped else None, lam_code, ctx.pool_weights,
        seed=config.SEED)


class _Fair:
    """Paper 6's constrained stacked estimator, given the target masks."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.m = common.fairness.StackedFairGBM(constrained=True, seed=config.SEED)

    def fit(self, X, y, w, clusters=None):
        masks = [self.ctx.masks_tr[g] for g in common.p6config.FAIR_TARGET_GROUPS if g in self.ctx.masks_tr]
        self.m.fit(X, y, w, masks, clusters=clusters)
        return self

    def predict(self, X):
        return self.m.predict(X)


def _plain(key):
    return lambda ctx: common.make(key, ctx.cols)


# key -> (family, feature set, constructor, label, parameter)
FORMULAS = {
    "cms": ("reference", "F2", _cms, "CMS form", None),
    "cms_F3": ("reference", "F3", _cms, "CMS form, with prior use", None),
    "wls": ("reference", "F3", _plain("wls"), "Unconstrained WLS", None),
    "gbm": ("reference", "F3", _plain("gbm"), "Tweedie boosting", None),
    "fair": ("reference", "F3", _Fair, "Fair stacked (paper 6)", None),
    "cms_cap": ("capped", "F2", _cap, "CMS form, cost-capped", None),
}
for lam in config.PENALTY_GRID[1:]:
    FORMULAS[f"cms_pen_{lam:g}"] = ("penalized", "F2", _pen(lam), f"CMS form, coding-penalized (lambda={lam:g})", lam)
for lam in config.DRO_LAMBDA_GRID[1:]:
    FORMULAS[f"cms_dro_{lam:g}"] = ("dro", "F2", _dro(lam), f"CMS form, DRO (lambda={lam:g})", lam)
for lam in config.PENALTY_GRID[1:]:
    FORMULAS[f"cms_cap_pen_dro_{lam:g}"] = ("combined", "F2", _combo(lam, lam, capped=True),
                                            f"CMS form, capped + penalized + DRO (lambda={lam:g})", lam)
FORMULAS["gbm_cap_dro"] = ("boosted", "F3", _stacked_robust(config.DRO_LAMBDA_GRID[3]),
                           "Boosting, capped + DRO", config.DRO_LAMBDA_GRID[3])

for _k, (_fam, _fs, _make, _label, _param) in list(FORMULAS.items()):
    FORMULAS[_k] = (_fam, _fs, (lambda m: (lambda ctx: Normalized(m(ctx))))(_make), _label, _param)

LINEAR = [k for k, v in FORMULAS.items() if v[0] not in ("boosted",) and k not in ("gbm", "fair")]
BOOSTED = ["gbm", "fair", "gbm_cap_dro"]
