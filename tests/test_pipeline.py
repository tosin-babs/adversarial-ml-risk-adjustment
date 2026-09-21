"""
Tests for the Paper 7 pipeline that need no microdata.

    ../.venv/bin/python -m pytest -q tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import config  # noqa: E402

DATA = (config.PAPER6 / "data" / "derived" / "prospective.pkl").exists()


def test_primary_feature_set_has_no_prior_use():
    assert config.PRIMARY_FEATURE_SET == "F2"


def test_calibration_targets_are_medpac_magnitudes():
    assert abs(config.CODING_TARGET - (0.16 - 0.059)) < 1e-12
    assert abs(config.SELECTION_TARGET - 44 / 500) < 1e-12


@pytest.mark.skipif(not DATA, reason="Paper 6 data absent")
def test_plausibility_rules_nest():
    import common
    X, y, w, cl, st, attrs = common.build("F2")
    cols = list(X.columns)
    pool = common.ccsr_columns(cols)
    Xn = X.to_numpy(np.float32)[:2000]
    a = attrs.iloc[:2000]
    P_sys = common.plausibility(Xn, cols, pool, a, "system")
    P_use = common.plausibility(Xn, cols, pool, a, "use")
    P_either = common.plausibility(Xn, cols, pool, a, "either")
    P_any = common.plausibility(Xn, cols, pool, a, "any")
    assert np.array_equal(P_either, P_sys | P_use)
    assert P_any.all()
    assert P_sys.mean() < P_either.mean() <= 1.0


@pytest.mark.skipif(not DATA, reason="Paper 6 data absent")
def test_plan_cost_model_never_sees_test_rows():
    import common
    X, y, w, cl, st, attrs = common.build("F2")
    for rep, fold, tr, te in common.folds(cl, st)[:3]:
        f = config.DERIVED / "plan" / f"rep{rep}_fold{fold}.npz"
        if not f.exists():
            pytest.skip("plan models not built")
        z = np.load(f)
        assert np.array_equal(z["train_idx"], tr) and np.array_equal(z["test_idx"], te)
        assert not set(cl[tr]) & set(cl[te]), "a PSU is in both training and test rows"
        assert len(z["cost_train_oof"]) == len(tr) and len(z["cost_test"]) == len(te)


@pytest.mark.skipif(not DATA, reason="Paper 6 data absent")
def test_race_never_enters_features():
    import common
    X = common.build("F2")[0]
    assert not any("race" in c.lower() or "hisp" in c.lower() for c in X.columns)
