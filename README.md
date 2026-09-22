# Adversarial Machine Learning for Medicare Risk Adjustment

Reproduction code for *Adversarial Machine Learning for Medicare Risk
Adjustment: Training Payment Formulas Against a Strategic Health Plan*.

A simulated health plan best-responds to any payment formula by adding
diagnosis codes and by tilting enrollment toward people it expects to be
overpaid, calibrated so that its response to a formula in the CMS form matches
MedPAC's published coding and selection magnitudes. Payment formulas are then
made robust to it (a coding penalty, caps at incremental cost, a worst-subgroup
penalty) and trained against it by repeated risk minimization. Everything runs
on the public MEPS benchmark of the companion paper and is out of sample.

It is a simulation of incentives on survey data, not evidence about any plan.

## Headline results

Per 1,000 enrollees of survey weight a year, out of fold, 95% Rao-Wu PSU
bootstrap intervals paired across formulas. The plan sees health-risk-assessment
information the formulas do not, pays for every code it adds, responds out of
sample, and is calibrated on training data to MedPAC's $40 billion of coding and
$44 billion of selection. Every formula pays the same budget.

| Formula | R², ungamed | Extracted by the plan | Share of payment |
|---|---:|---:|---:|
| CMS form (payment-form WLS, no prior use) | 0.098 | $1,337,599 | 17.1% |
| CMS form, cost-capped | 0.098 | $1,328,701 | 17.0% |
| CMS form, coding penalty (λ = 1) | 0.096 | $743,433 | 9.5% |
| CMS form, coding penalty tuned against the plan (Stackelberg) | 0.091 | $731,169 | 9.4% |
| CMS form, worst-subgroup penalty | 0.102 | $953,648 | 12.2% |
| CMS form, retrained against the plan (never converges) | 0.099 | $1,122,389 | 14.4% |
| Coding penalty, retrained against the plan | 0.097 | $748,865 | 9.6% |
| Tweedie boosting (with prior use) | 0.187 | $755,299 | 9.7% |
| Boosting, retrained against the plan (never converges) | 0.186 | $675,774 | 8.7% |
| Fair stacked estimator (companion paper) | 0.169 | $1,557,041 | 20.1% |

- A coding penalty removes 88% of the coding channel; caps at incremental cost
  remove none of it, because the formula already pays codes less than those caps.
- No formula built on diagnoses closes the selection channel.
- Retraining against the plan helps unprotected formulas but never settles, and
  adds nothing once the penalty is in place.
- Every formula that resists coding underpays people needing help with daily
  activities by more than the CMS form ($9,915 per person-year), up to $13,760,
  and the plan enrolls fewer of them.

## Reproducing

Needs the companion repository built alongside, as `../paper6`
([fair-ml-health-risk-adjustment](https://github.com/tosin-babs/fair-ml-health-risk-adjustment)),
whose `riskfair` package (version 0.3.0) carries the `adversary` and `robust`
modules this paper adds, and whose pipeline builds the MEPS analysis file.

```bash
../.venv/bin/python python/run_all.py
```

| Step | Script | Output |
|---|---|---|
| Reproduce the companion paper's payment-form results | `reproduce.py` | pass/fail gate |
| The plan's cross-fitted cost model per fold | `build_plan.py` | `data/derived/plan/` |
| Calibrate the plan, write the codable pool | `calibrate.py` | Table 2, Table A1 |
| Every formula against the plan | `benchmark.py` | per-fold rows |
| Formulas trained against the plan | `adversarial.py` | training paths |
| Tables with paired bootstrap | `summarise.py` | Tables 3 to 6 |
| Payment per code | `codes.py` | Table 8 |
| Sweeps of the plan | `sweeps.py` | Table 9 |
| Robustness rows | `robustness.py` | Table 10 |
| Figures, manuscript tables, prose check | `exhibits.py`, `make_tables.py`, `check_manuscript.py` | |
| Submission documents | `make_manuscript.py` | DOCX and PDF |

Seeded throughout. No MEPS microdata are in this repository.

## Citation

Ologunbaba, T., Adiegwu, C. G., & Temitope, A. (2026). *Adversarial Machine
Learning for Medicare Risk Adjustment: Training Payment Formulas Against a
Strategic Health Plan*. Working paper.

## Authors

- Temitope Ologunbaba, Faculty of Engineering, Federal University of Technology, Akure, Nigeria, ologunbabatope@gmail.com (corresponding)
- Chisom G. Adiegwu, Department of Actuarial Science and Quantitative Risk Analysis and Management, Georgia State University, Atlanta, GA, USA, Cadiegwu1@student.gsu.edu
- Adebolu Temitope, Department of Epidemiology and Medical Statistics, University of Ibadan, Ibadan, Nigeria, temitopeadebolu5@gmail.com

## License

MIT for the code. MEPS data are governed by AHRQ's terms of use.
