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
# The plan's own cost model: Tweedie boosting on F4, which adds to the
# regulator's F3 what a plan learns from its own health risk assessments
# (self-rated physical and mental health, help with ADLs and IADLs, income
# and insurance history; never race). Cross-fitted inside each training fold
# so it never sees the regulator's test fold. The regulator's reference model
# for the worst-subgroup penalty is boosting on F3, a different model with a
# different seed. A plan that sees only F3, and a WLS plan, are sensitivity rows.
PLAN_MODEL = "gbm"
PLAN_MODEL_FEATURES = "F4"
PLAN_SEED_OFFSET = 1
REF_MODEL_FEATURES = "F3"
PLAN_CROSS_FIT_K = 3
PLAN_SET = "plan_F4"
PLAN_SETS = ("plan_F4", "ref_F3", "plan_wls_F3", "plan_F3")

# Plausibility: which codes a plan could add to which person.
#   "system"  a flag in the same body system is already present
#   "use"     the person is in the top half of year-1 spending
#   "either"  system or use (primary)
#   "any"     no restriction
# Every rule except "any" also requires the code to occur in at least
# CELL_MIN_PREVALENCE of training people of the same age band and sex.
PLAUSIBILITY = "either"
PLAUSIBILITY_RULES = ("either", "system", "use", "any")
CELL_MIN_PREVALENCE = 0.002

# Coding response. One code per reviewed person at a stated cost, with audit
# exposure: each 1% of enrollees already given a code raises the cost of
# giving it again by AUDIT_PER_PCT dollars, so the plan spreads its codes.
# Reach and tilt are calibrated jointly in calibrate.py.
COST_PER_CODE_GRID = (0.0, 250.0, 500.0, 1000.0, 2000.0, 3000.0)
MAX_CODES_GRID = (1, 2, 3)
REACH_GRID = (0.02, 0.04, 0.05, 0.06, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.50, 1.00)
PRIMARY_MAX_CODES = 1
PRIMARY_COST_PER_CODE = 1000.0
AUDIT_PER_PCT = 2000.0
AUDIT_GRID = (0.0, 500.0, 2000.0, 5000.0)

# Selection response: tilt in the threshold rule.
TILT_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50)
SELECTION_RULE = "threshold"

# MedPAC (March 2025, ch. 11): $84 billion of excess payment, 20% of
# fee-for-service-equivalent spending (about $420 billion), of which about
# $40 billion is coding intensity and $44 billion favorable selection. Both
# targets are shares of the same fee-for-service-equivalent base, which is
# what a formula normalized to cost pays before the plan responds.
CODING_TARGET = 40.0 / 420.0
SELECTION_TARGET = 44.0 / 420.0

# ------------------------------------------------------------ robust ----
PENALTY_GRID = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
DRO_ALPHA = 0.10
DRO_ALPHAS = (0.05, 0.10, 0.20)
DRO_LAMBDA_GRID = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
# The combined formula holds the coding penalty at COMBO_LAMBDA_CODE and
# varies the worst-subgroup weight over DRO_LAMBDA_GRID.
COMBO_LAMBDA_CODE = 1.0

# -------------------------------------------------------- adversarial ----
ADV_ITERS = 20
ADV_TOL = 0.01
ADV_DAMPING = 0.0

# ------------------------------------------------------------ bootstrap ----
BOOT_REPS = 200
