"""
Every analytic choice in Paper 7, in one place.

The data, folds, feature sets and models come from Paper 6 (the sibling
directory `../paper6`, whose `riskfair` package this paper extends). Nothing
about the benchmark is rebuilt here.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER6 = ROOT.parent / "paper6"
DERIVED = ROOT / "data" / "derived"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"
for _p in (DERIVED, TABLES, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 2026

# ----------------------------------------------------------- formulas ----
# F2 (demographics and conditions, no prior use) is the primary feature set
# for every payment-form formula, as a payment formula does not see prior
# spending; F3 is reported alongside where the conclusion could depend on it.
PRIMARY_FEATURE_SET = "F2"
FEATURE_SETS = ("F2", "F3")
COUNT_COL, SYSTEM_COL = "n_conditions", "n_body_systems"

# ---------------------------------------------------------------- plan ----
# The plan's own cost model: the best predictor Paper 6 found, cross-fitted
# inside each training fold so it never sees the regulator's test fold.
PLAN_MODEL = "gbm"
PLAN_MODEL_FEATURES = "F3"
PLAN_CROSS_FIT_K = 3

# Plausibility: which codes a plan could add to which person.
#   "system"  a flag in the same body system is already present
#   "use"     the person is in the top half of year-1 spending
#   "either"  system or use (primary)
#   "any"     no restriction
PLAUSIBILITY = "either"
PLAUSIBILITY_RULES = ("either", "system", "use", "any")

# Coding response: cost per added code, codes per person, chart-review reach.
# The primary triple is calibrated in calibrate.py so that the plan's response
# to the CMS-form formula raises payment by CODING_TARGET; the grid is swept.
COST_PER_CODE_GRID = (0.0, 250.0, 500.0, 1000.0, 2000.0, 3000.0)
MAX_CODES_GRID = (1, 2, 3)
REACH_GRID = (0.02, 0.04, 0.05, 0.06, 0.075, 0.10, 0.25, 0.50, 1.00)
# Primary coding plan: one code per reviewed person at a stated cost, with
# reach calibrated; the cost does not bind because the formula pays far more
# than any code costs to add, which the paper reports.
PRIMARY_MAX_CODES = 1
PRIMARY_COST_PER_CODE = 1000.0
# MedPAC (March 2025, ch. 11): MA risk scores about 16% above FFS before the 5.9%
# statutory coding adjustment; the net is the target.
CODING_TARGET = 0.16 - 0.059

# Selection response: tilt in the threshold rule, and the smooth alternative.
TILT_GRID = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
SELECTION_RULE = "threshold"
# MedPAC (March 2025): favorable selection about $44 billion of the $84 billion;
# MA payments of roughly $500 billion; selection profit as a share of payment.
SELECTION_TARGET = 44.0 / 500.0

# ------------------------------------------------------------ robust ----
PENALTY_GRID = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
DRO_ALPHA = 0.10
DRO_ALPHAS = (0.05, 0.10, 0.20)
DRO_LAMBDA_GRID = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)

# -------------------------------------------------------- adversarial ----
ADV_ITERS = 20
ADV_TOL = 0.01
ADV_DAMPING = 0.0

# ------------------------------------------------------------ bootstrap ----
BOOT_REPS = 200
