"""
Run the whole analysis, in order.

    ../.venv/bin/python python/run_all.py

Needs Paper 6 built in the sibling directory ../paper6 (its prospective
dataset and derived files). Each step runs in its own process, as in Paper 6,
because LightGBM and scikit-learn each ship an OpenMP runtime.

Roughly: the plan models 5 minutes, the linear benchmark 15 minutes, the
boosted formulas 15 minutes, the adversarial linear formulas 20 minutes, the
adversarial boosted formula 3 to 4 hours (one repeat, five folds), the
sweeps 2 hours, robustness 1 hour.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("Reproduce Paper 6's payment-form results", "reproduce", {}),
    ("The plan's cost model, cross-fitted per fold", "build_plan", {}),
    ("Calibrate the plan and write the codable pool", "calibrate", {}),
    ("Linear formulas against the plan", "benchmark", {"P7_FAMILY": "linear"}),
    ("Boosted formulas against the plan", "benchmark", {"P7_FAMILY": "boosted"}),
    ("Linear formulas trained against the plan", "adversarial", {}),
    ("Boosted formula trained against the plan", "adversarial", {"P7_ADV": "gbm"}),
    ("Tables from the benchmark", "summarise", {}),
    ("Payment per code before and after", "codes", {}),
    ("Sweeps of the plan", "sweeps", {}),
    ("Robustness rows", "robustness", {}),
    ("Tables from the benchmark (again, with every row)", "summarise", {}),
    ("Figures", "exhibits", {}),
    ("Manuscript tables", "make_tables", {}),
]


def main():
    import os
    t0 = time.time()
    for title, mod, env in STEPS:
        print(f"\n== {title} ({mod}.py)", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, str(HERE / f"{mod}.py")], cwd=HERE.parent,
                           env={**os.environ, **env})
        if r.returncode:
            raise SystemExit(f"{mod}.py failed with code {r.returncode}")
        print(f"   {time.time() - t:,.0f}s", flush=True)
    print(f"\nall steps done in {(time.time() - t0) / 60:,.0f} min")


if __name__ == "__main__":
    main()
