# Tables

*Generated from `output/tables/*.csv` by `python/make_tables.py`. Spending is total expenditure from all payers in 2024 dollars; extraction is per 1,000 enrollees of survey weight. Every model quantity is out of fold from the five survey folds and three repeats of the Paper 6 benchmark, with primary sampling units kept whole. Intervals are from a Rao-Wu rescaled bootstrap over PSUs within strata (200 resamples), paired across formulas.*


**Table 1.** The formulas compared.

*F2 is demographics and CCSR condition flags; F3 adds year-1 use and spending. The CMS form is weighted least squares with a base rate for each age band and sex, no intercept, and non-negative increments for everything else. Penalties are at lambda = 1 unless the label says otherwise; the DRO alpha is 0.10.*

| Formula | Features | Robust term | Trained against the plan |
|---|---:|---:|---:|
| CMS form | F2 | none | no |
| CMS form, with prior use | F3 | none | no |
| Unconstrained WLS | F3 | none | no |
| Tweedie boosting | F3 | none | no |
| Fair stacked (paper 6) | F3 | none | no |
| CMS form, cost-capped | F2 | cap at incremental cost | no |
| CMS form, coding-penalized (lambda=1) | F2 | coding penalty | no |
| CMS form, DRO (lambda=5) | F2 | worst-subgroup penalty | no |
| CMS form, capped + penalized + DRO (lambda=1) | F2 | cap, coding penalty, worst-subgroup penalty | no |
| Boosting, capped + DRO | F3 | cap, worst-subgroup penalty on the linear layer | no |
| CMS form, trained against the plan | F2 | as the base formula | yes |
| CMS form, coding-penalized, trained against the plan | F2 | as the base formula | yes |
| CMS form, capped + penalized + DRO, trained against the plan | F2 | as the base formula | yes |
| Boosting, capped + DRO, trained against the plan | F3 | as the base formula | yes |

**Table 2.** Calibrating the plan against the CMS form.

*Coding: one code per reviewed person at $1,000 per code, by chart-review reach; the calibrated reach is 6%, where the gross payment rise is closest to the target of 10.1% (MedPAC's 16% coding intensity less the 5.9% statutory adjustment). Selection: threshold rule by tilt; the calibrated tilt is 0.20, where the plan's profit is closest to 8.8% of payment (MedPAC's $44 billion of favorable selection on about $500 billion of payment). Means over the 15 test folds.*

| Reach | Payment rise | Net of code cost | Codes per 1,000 |
|---|---:|---:|---:|
| 2% | 3.5% | 3.2% | 20 |
| 4% | 6.9% | 6.4% | 40 |
| 5% | 8.6% | 8.0% | 50 |
| 6% | 10.4% | 9.6% | 60 |
| 8% | 13.0% | 12.0% | 75 |
| 10% | 17.3% | 16.0% | 100 |
| 25% | 43.1% | 39.9% | 250 |
| 50% | 82.7% | 76.3% | 500 |
| 100% | 97.7% | 88.7% | 707 |

| Tilt | Payment shift | Profit, share of payment |
|---|---:|---:|
| 0.00 | 0.0% | 0.0% |
| 0.05 | 1.4% | 2.1% |
| 0.10 | 2.8% | 4.2% |
| 0.20 | 5.6% | 8.5% |
| 0.30 | 8.4% | 12.7% |
| 0.50 | 13.9% | 21.2% |

**Table 3.** What the calibrated plan extracts from each formula, per 1,000 enrollees.

*Coding: payment for added codes, net of the cost of adding them, for codes that change nothing about cost. Selection: payment above cost from tilting enrollment toward the half of enrollees the plan expects to be overpaid. Share is of base payment. 95% bootstrap intervals in parentheses.*

| Formula | Coding |  | Selection |  | Total |  | Share |
|---|---:|---:|---:|---:|---:|---:|---:|
| CMS form | $748,542 | ($723,936 to $771,061) | $769,238 | ($716,652 to $810,016) | $1,517,780 | ($1,460,172 to $1,568,914) | 19.5% |
| CMS form, with prior use | $653,070 | ($631,884 to $673,387) | $703,912 | ($658,812 to $754,219) | $1,356,983 | ($1,296,406 to $1,405,978) | 17.4% |
| Unconstrained WLS | $677,427 | ($652,390 to $708,232) | $832,084 | ($784,154 to $881,841) | $1,509,511 | ($1,457,682 to $1,576,770) | 19.3% |
| Tweedie boosting | $646,352 | ($620,422 to $674,848) | $120,911 | ($75,171 to $164,084) | $767,263 | ($710,335 to $824,487) | 9.9% |
| Fair stacked (paper 6) | $1,037,444 | ($998,124 to $1,073,769) | $649,716 | ($594,976 to $701,265) | $1,687,160 | ($1,618,672 to $1,763,305) | 21.7% |
| CMS form, cost-capped | $748,628 | ($728,018 to $773,942) | $765,216 | ($712,357 to $811,450) | $1,513,844 | ($1,456,859 to $1,567,525) | 19.4% |
| CMS form, coding-penalized (lambda=1) | $92,274 | ($89,176 to $95,668) | $683,864 | ($629,016 to $736,720) | $776,138 | ($720,404 to $829,849) | 10.0% |
| CMS form, DRO (lambda=5) | $404,936 | ($391,269 to $418,282) | $669,386 | ($621,229 to $721,810) | $1,074,322 | ($1,027,400 to $1,129,768) | 13.8% |
| CMS form, capped + penalized + DRO (lambda=1) | $174,519 | ($168,995 to $181,340) | $668,490 | ($621,787 to $712,724) | $843,010 | ($796,975 to $888,107) | 10.8% |
| Boosting, capped + DRO | $645,067 | ($619,426 to $673,805) | $121,814 | ($82,028 to $171,847) | $766,880 | ($714,405 to $824,581) | 9.9% |
| CMS form, trained against the plan | $675,101 | ($655,033 to $697,493) | $711,243 | ($656,587 to $763,313) | $1,386,344 | ($1,326,577 to $1,440,619) | 19.7% |
| CMS form, coding-penalized, trained against the plan | $77,193 | ($74,844 to $80,062) | $646,544 | ($597,475 to $695,659) | $723,737 | ($675,732 to $772,690) | 10.3% |
| CMS form, capped + penalized + DRO, trained against the plan | $119,568 | ($115,284 to $123,703) | $628,614 | ($578,513 to $677,850) | $748,183 | ($700,106 to $797,069) | 10.7% |
| Boosting, capped + DRO, trained against the plan | $1,060,061 | ($1,008,162 to $1,126,272) | $254,853 | ($201,956 to $306,191) | $1,314,914 | ($1,229,106 to $1,409,117) | 17.1% |

**Table 4.** Accuracy before and after the plan responds.

*R² on ungamed data is what the regulator gives up; R² on post-response data, with the plan's coded features and tilted weights, is what it experiences.*

| Formula | R², ungamed |  | R², post-response |  | Difference from CMS form |
|---|---:|---:|---:|---:|---:|
| CMS form | 0.098 | (0.076 to 0.130) | 0.075 | (0.060 to 0.098) | 0.000 |
| CMS form, with prior use | 0.108 | (0.085 to 0.141) | 0.088 | (0.071 to 0.115) | 0.010 |
| Unconstrained WLS | 0.123 | (0.102 to 0.155) | 0.094 | (0.079 to 0.118) | 0.025 |
| Tweedie boosting | 0.187 | (0.156 to 0.242) | 0.176 | (0.148 to 0.225) | 0.089 |
| Fair stacked (paper 6) | 0.169 | (0.142 to 0.215) | 0.124 | (0.098 to 0.160) | 0.072 |
| CMS form, cost-capped | 0.098 | (0.076 to 0.130) | 0.074 | (0.057 to 0.097) | 0.000 |
| CMS form, coding-penalized (lambda=1) | 0.096 | (0.074 to 0.127) | 0.089 | (0.071 to 0.115) | -0.002 |
| CMS form, DRO (lambda=5) | 0.102 | (0.078 to 0.135) | 0.091 | (0.072 to 0.117) | 0.004 |
| CMS form, capped + penalized + DRO (lambda=1) | 0.098 | (0.075 to 0.130) | 0.090 | (0.071 to 0.117) | 0.000 |
| Boosting, capped + DRO | 0.187 | (0.156 to 0.242) | 0.176 | (0.148 to 0.227) | 0.089 |
| CMS form, trained against the plan | 0.098 | (0.075 to 0.129) | 0.083 | (0.065 to 0.109) | -0.000 |
| CMS form, coding-penalized, trained against the plan | 0.094 | (0.072 to 0.124) | 0.091 | (0.071 to 0.117) | -0.003 |
| CMS form, capped + penalized + DRO, trained against the plan | 0.096 | (0.073 to 0.127) | 0.092 | (0.072 to 0.119) | -0.002 |
| Boosting, capped + DRO, trained against the plan | 0.175 | (0.141 to 0.227) | 0.155 | (0.127 to 0.201) | 0.077 |

**Table 5.** The price of robustness: accuracy against extraction along each penalty grid.

*Linear payment-form formulas on F2. Lambda is the penalty weight; the DRO alpha is 0.10.*

| Formula | R², ungamed | Coding | Selection | Total | Share |
|---|---:|---:|---:|---:|---:|
| CMS form | 0.098 | $748,542 | $769,238 | $1,517,780 | 19.5% |
| CMS form, cost-capped | 0.098 | $748,628 | $765,216 | $1,513,844 | 19.4% |
| CMS form, coding-penalized (lambda=0.5) | 0.097 | $109,009 | $677,627 | $786,636 | 10.1% |
| CMS form, coding-penalized (lambda=1) | 0.096 | $92,274 | $683,864 | $776,138 | 10.0% |
| CMS form, coding-penalized (lambda=2) | 0.095 | $75,625 | $678,335 | $753,961 | 9.7% |
| CMS form, coding-penalized (lambda=5) | 0.091 | $48,464 | $673,976 | $722,441 | 9.3% |
| CMS form, coding-penalized (lambda=10) | 0.085 | $23,077 | $690,668 | $713,744 | 9.2% |
| CMS form, coding-penalized (lambda=20) | 0.074 | $0 | $747,044 | $747,044 | 9.6% |
| CMS form, coding-penalized (lambda=50) | 0.060 | $0 | $836,286 | $836,286 | 10.7% |
| CMS form, DRO (lambda=1) | 0.102 | $428,440 | $669,483 | $1,097,923 | 14.1% |
| CMS form, DRO (lambda=2) | 0.102 | $413,126 | $683,405 | $1,096,531 | 14.1% |
| CMS form, DRO (lambda=5) | 0.102 | $404,936 | $669,386 | $1,074,322 | 13.8% |
| CMS form, DRO (lambda=10) | 0.102 | $402,085 | $679,813 | $1,081,898 | 13.9% |
| CMS form, DRO (lambda=20) | 0.101 | $400,597 | $678,900 | $1,079,496 | 13.9% |
| CMS form, DRO (lambda=50) | 0.101 | $399,511 | $678,964 | $1,078,476 | 13.8% |
| CMS form, capped + penalized + DRO (lambda=0.5) | 0.099 | $177,473 | $670,134 | $847,607 | 10.9% |
| CMS form, capped + penalized + DRO (lambda=1) | 0.098 | $174,519 | $668,490 | $843,010 | 10.8% |
| CMS form, capped + penalized + DRO (lambda=2) | 0.098 | $173,019 | $669,959 | $842,978 | 10.8% |
| CMS form, capped + penalized + DRO (lambda=5) | 0.098 | $172,127 | $669,480 | $841,607 | 10.8% |
| CMS form, capped + penalized + DRO (lambda=10) | 0.098 | $171,687 | $668,972 | $840,660 | 10.8% |
| CMS form, capped + penalized + DRO (lambda=20) | 0.098 | $171,450 | $674,861 | $846,311 | 10.9% |
| CMS form, capped + penalized + DRO (lambda=50) | 0.098 | $171,484 | $667,726 | $839,209 | 10.8% |

**Table 6.** Net compensation by group before and after the plan responds: predicted minus observed spending per person-year.

*Negative is underpayment. Post-response values use the plan's coded features and tilted weights, so they include the coding rise. Race and ethnicity are evaluated only.*

| Group | CMS form, ungamed | post-response | CMS form, coding-penalized (lambda=1), ungamed | post-response | CMS form, capped + penalized + DRO, trained against the plan, ungamed | post-response | Tweedie boosting, ungamed | post-response | Boosting, capped + DRO, trained against the plan, ungamed | post-response | Fair stacked (paper 6), ungamed | post-response |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Needs ADL or IADL help | -$9,915 | -$5,764 | -$12,462 | -$9,661 | -$14,002 | -$11,348 | -$5,677 | -$2,856 | -$7,668 | -$1,975 | $847 | $6,270 |
| Age 65 and over | $22 | $2,952 | -$14 | $1,322 | -$1,761 | -$437 | -$455 | $1,455 | -$993 | $2,260 | $2,629 | $6,279 |
| Uninsured all year | $1,212 | $1,645 | $1,233 | $1,590 | $830 | $1,120 | $567 | $778 | $681 | $1,004 | -$532 | -$42 |
| Income below 200% FPL | $331 | $1,888 | -$32 | $888 | -$777 | $101 | $6 | $789 | -$120 | $1,194 | $209 | $2,100 |
| Mental health condition | $4 | $2,735 | -$381 | $1,065 | -$1,251 | $222 | $550 | $2,069 | $96 | $2,692 | $382 | $3,922 |
| Non-Hispanic Black | $86 | $1,379 | -$123 | $623 | -$759 | -$23 | -$5 | $740 | -$24 | $1,146 | -$138 | $1,325 |
| Hispanic | $294 | $1,470 | $255 | $962 | -$303 | $350 | $92 | $517 | $137 | $759 | -$576 | $331 |
| Non-Hispanic Asian | $591 | $1,657 | $832 | $1,357 | $191 | $736 | $301 | $753 | $409 | $1,161 | -$47 | $947 |

**Table 7.** Training against the plan: extraction on the training rows by iteration.

*Iteration 1 is the plan's response to the formula fitted on ungamed data. Change is the mean absolute movement of payments as a share of mean payment; the loop stops below 1%. Folds is the number of splits still iterating.*

| Formula | Iteration | Extraction | Coding | Selection | Change | Folds |
|---|---:|---:|---:|---:|---:|---:|
| CMS form, trained against the plan | 1 | $1,442,450 | $747,474 | $694,976 | 9.60% | 15 |
| CMS form, trained against the plan | 2 | $1,238,117 | $606,915 | $631,203 | 5.36% | 15 |
| CMS form, trained against the plan | 3 | $1,317,293 | $674,253 | $643,040 | 5.22% | 15 |
| CMS form, trained against the plan | 4 | $1,219,582 | $595,613 | $623,968 | 5.33% | 15 |
| CMS form, trained against the plan | 5 | $1,331,504 | $686,250 | $645,254 | 5.18% | 15 |
| CMS form, trained against the plan | 10 | $1,237,005 | $608,746 | $628,259 | 5.33% | 15 |
| CMS form, trained against the plan | 15 | $1,319,431 | $677,506 | $641,925 | 5.25% | 15 |
| CMS form, trained against the plan | 20 | $1,229,594 | $605,590 | $624,004 | 5.20% | 15 |
| CMS form, coding-penalized, trained against the plan | 1 | $748,994 | $92,121 | $656,873 | 9.74% | 15 |
| CMS form, coding-penalized, trained against the plan | 2 | $697,944 | $76,414 | $621,530 | 0.75% | 15 |
| CMS form, capped + penalized + DRO, trained against the plan | 1 | $827,468 | $174,218 | $653,250 | 10.16% | 15 |
| CMS form, capped + penalized + DRO, trained against the plan | 2 | $724,489 | $118,630 | $605,859 | 1.48% | 15 |
| CMS form, capped + penalized + DRO, trained against the plan | 3 | $728,848 | $122,975 | $605,873 | 1.37% | 14 |
| CMS form, capped + penalized + DRO, trained against the plan | 4 | $719,005 | $116,053 | $602,952 | 1.47% | 12 |
| CMS form, capped + penalized + DRO, trained against the plan | 5 | $728,842 | $123,224 | $605,617 | 1.57% | 11 |
| CMS form, capped + penalized + DRO, trained against the plan | 10 | $726,482 | $119,482 | $607,000 | 1.81% | 6 |
| CMS form, capped + penalized + DRO, trained against the plan | 15 | $739,999 | $126,796 | $613,203 | 2.04% | 6 |
| CMS form, capped + penalized + DRO, trained against the plan | 20 | $729,940 | $124,977 | $604,963 | 1.83% | 4 |
| Boosting, capped + DRO, trained against the plan | 1 | $484,724 | $694,796 | -$210,072 | 8.36% | 5 |
| Boosting, capped + DRO, trained against the plan | 2 | $794,135 | $941,850 | -$147,715 | 8.28% | 5 |
| Boosting, capped + DRO, trained against the plan | 3 | $1,157,610 | $1,205,242 | -$47,632 | 8.59% | 5 |
| Boosting, capped + DRO, trained against the plan | 4 | $858,967 | $1,004,497 | -$145,530 | 8.95% | 5 |
| Boosting, capped + DRO, trained against the plan | 5 | $1,152,394 | $1,221,096 | -$68,702 | 11.56% | 5 |
| Boosting, capped + DRO, trained against the plan | 10 | $1,176,476 | $1,269,434 | -$92,958 | 9.57% | 5 |

**Table 8.** The 20 codes the CMS form pays most for, and what the robust formulas pay for them.

*Full-sample fits. Increment is the payment rise when the code is added to a person who could plausibly receive it. Incremental cost is the weighted difference in year-2 spending between people with and without the code, within age, sex and number of other conditions.*

| CCSR | Condition | Prevalence | Incremental cost | CMS form | Penalized | Robust | Robust, trained |
|---|---:|---:|---:|---:|---:|---:|---:|
| NVS015 | Polyneuropathies | 0.8% | $16,955 | $12,514 | $2,157 | $2,646 | $2,292 |
| MUS003 | Rheumatoid arthritis and related disease | 1.3% | $12,555 | $11,071 | $2,074 | $3,006 | $2,668 |
| NVS009 | Epilepsy; convulsions | 0.9% | $12,003 | $10,419 | $2,103 | $2,806 | $2,526 |
| GEN006 | Other specified and unspecified diseases of kidney and ureters | 0.8% | $14,081 | $9,896 | $2,139 | $2,748 | $2,494 |
| CIR019 | Heart failure | 0.4% | $15,142 | $9,619 | $1,873 | $2,233 | $2,021 |
| MUS026 | Muscle disorders | 0.7% | $10,878 | $8,417 | $1,903 | $2,442 | $2,145 |
| NEO030 | Breast cancer - all other types | 0.7% | $7,707 | $7,382 | $2,080 | $2,396 | $2,091 |
| EYE005 | Retinal and vitreous conditions | 1.1% | $8,523 | $7,311 | $2,089 | $2,353 | $2,131 |
| RSP008 | Chronic obstructive pulmonary disease and bronchiectasis | 1.3% | $10,448 | $6,472 | $2,026 | $2,503 | $2,321 |
| MBD003 | Bipolar and related disorders | 0.8% | $8,095 | $6,326 | $1,965 | $2,415 | $2,153 |
| INF009 | Parasitic, other specified and unspecified infections | 0.5% | $9,288 | $6,104 | $2,015 | $2,186 | $1,932 |
| GEN008 | Urinary incontinence | 0.5% | $10,028 | $5,977 | $2,013 | $2,115 | $1,890 |
| END002+END005 | Diabetes mellitus without complication | 7.8% | $5,399 | $5,608 | $2,308 | $3,797 | $2,892 |
| SKN001 | Skin and subcutaneous tissue infections | 1.1% | $7,835 | $5,576 | $2,077 | $2,509 | $2,098 |
| NEO039 | Male reproductive system cancers - prostate | 0.5% | $5,066 | $5,543 | $2,033 | $2,220 | $1,941 |
| DIG008 | Other specified and unspecified disorders of stomach and duodenum | 1.1% | $7,304 | $5,415 | $2,027 | $2,265 | $2,014 |
| EYE001 | Cornea and external disease | 0.8% | $5,811 | $5,101 | $2,011 | $1,985 | $1,717 |
| MUS013 | Osteoporosis | 0.5% | $5,842 | $4,899 | $1,833 | $1,883 | $1,712 |
| BLD003 | Aplastic anemia | 0.5% | $8,140 | $4,742 | $2,105 | $2,157 | $1,890 |
| SYM016 | Other general signs and symptoms | 3.0% | $7,707 | $4,495 | $2,150 | $2,738 | $2,428 |

**Table 8b.** Payment per condition and per body system under each formula (full-sample fits).

| Formula | Per condition | Per body system |
|---|---:|---:|
| cms | $575 | $536 |
| cms_pen_1 | $1,505 | $544 |
| cms_cap_pen_dro_1 | $1,619 | $401 |
| cms_robust_adv | $1,560 | $194 |

**Table 9.** Sensitivity of extraction to the plan, one parameter at a time from the calibrated point.

*Per 1,000 enrollees; mean over the 15 test folds. The last block makes a share of the added codes real, raising the person's cost by the code's incremental cost.*

| Parameter | Value | CMS form | CMS form, coding-penalized (lambda=1) | CMS form, capped + penalized + DRO, trained against the plan |
|---|---:|---:|---:|---:|
| cost per code | 0.0 | $1,575,687 | $836,511 | $808,452 |
| cost per code | 250.0 | $1,561,434 | $821,478 | $793,429 |
| cost per code | 500.0 | $1,547,230 | $806,446 | $778,407 |
| cost per code | 1000.0 | $1,517,990 | $776,380 | $748,363 |
| cost per code | 2000.0 | $1,456,158 | $715,024 | $687,570 |
| cost per code | 3000.0 | $1,395,834 | $675,959 | $626,072 |
| codes per person | 1 | $1,517,990 | $776,380 | $748,363 |
| codes per person | 2 | $2,294,024 | $867,272 | $882,289 |
| codes per person | 3 | $2,998,860 | $964,563 | $1,002,833 |
| reach | 1 | $8,264,403 | $1,559,494 | $1,860,493 |
| reach | 0.02 | $944,580 | $712,029 | $660,096 |
| reach | 0.04 | $1,229,359 | $745,379 | $703,036 |
| reach | 0.05 | $1,372,112 | $761,550 | $726,194 |
| reach | 0.06 | $1,517,990 | $776,380 | $748,363 |
| reach | 0.075 | $1,732,554 | $798,234 | $783,506 |
| reach | 0.1 | $2,086,192 | $836,722 | $837,268 |
| reach | 0.25 | $4,197,896 | $1,074,023 | $1,134,928 |
| reach | 0.5 | $7,413,432 | $1,392,125 | $1,608,201 |
| tilt | 0.0 | $748,204 | $92,275 | $119,713 |
| tilt | 0.05 | $940,646 | $263,297 | $276,872 |
| tilt | 0.1 | $1,133,091 | $434,322 | $434,033 |
| tilt | 0.5 | $2,672,753 | $1,802,630 | $1,691,414 |
| tilt | 0.2 | $1,517,990 | $776,380 | $748,363 |
| tilt | 0.3 | $1,902,899 | $1,118,451 | $1,062,703 |
| plausibility | either | $1,517,990 | $776,380 | $748,363 |
| plausibility | system | $1,465,552 | $744,533 | $739,308 |
| plausibility | use | $1,514,200 | $774,861 | $745,868 |
| plausibility | any | $1,530,294 | $787,621 | $755,384 |
| plan cost model | WLS | $1,269,086 | $560,526 | $560,051 |
| share of added codes real | 0.0 | $1,517,990 | $776,380 | $748,363 |
| share of added codes real | 1 | $305,254 | $424,523 | $245,223 |
| share of added codes real | 0.25 | $1,221,646 | $688,929 | $618,754 |
| share of added codes real | 0.5 | $919,341 | $602,447 | $489,975 |

**Table 10.** Robustness rows.

*The 65-and-over rows restrict the stored out-of-fold results to that age group. Other rows are refits with one repeat of the five folds; leave-one-panel-out holds each MEPS panel out in turn.*

| Variant | Formula | R², ungamed | Coding | Selection | Total |
|---|---:|---:|---:|---:|---:|
| persons 65 and over | CMS form | 0.066 | $1,498,673 | $1,454,734 | $2,953,407 |
| persons 65 and over | CMS form, coding-penalized (lambda=1) | 0.068 | $103,279 | $1,243,391 | $1,346,671 |
| persons 65 and over | CMS form, capped + penalized + DRO (lambda=1) | 0.070 | $202,353 | $1,137,991 | $1,340,344 |
| persons 65 and over | Tweedie boosting | 0.167 | $1,591,765 | $281,886 | $1,873,650 |
| persons 65 and over | Boosting, capped + DRO | 0.167 | $1,591,398 | $355,983 | $1,947,381 |
| persons 65 and over | CMS form, coding-penalized, trained against the plan | 0.065 | $88,450 | $1,163,155 | $1,251,606 |
| persons 65 and over | CMS form, capped + penalized + DRO, trained against the plan | 0.066 | $146,424 | $1,088,392 | $1,234,816 |
| F3 (prior use) for the penalized form | F3 (prior use) for the penalized form | 0.112 | $11,501 | $650,823 | $662,324 |
| F3 (prior use) for the robust form | F3 (prior use) for the robust form | 0.115 | $91,877 | $654,880 | $746,757 |
| DRO alpha 0.05 | DRO alpha 0.05 | 0.105 | $225,862 | $711,355 | $937,216 |
| DRO alpha 0.20 | DRO alpha 0.20 | 0.104 | $121,656 | $614,743 | $736,399 |
| squared-error boosting | squared-error boosting | 0.201 | $826,294 | $319,149 | $1,145,443 |
| leave one panel out | CMS form | 0.105 | $735,783 | $768,858 | $1,504,641 |
| leave one panel out | CMS form, coding-penalized (lambda=1) | 0.103 | $92,222 | $657,519 | $749,741 |
| leave one panel out | Tweedie boosting | 0.190 | $793,246 | $184,161 | $977,407 |


# Appendix tables


**Table A1.** The codable pool: the 25 codes with the largest CMS-form increments.

*Plausible share is the weighted share of people who lack the code and could receive it under the primary rule (a code in the same body system already present, or year-1 spending in the top half).*

| CCSR | Condition | Prevalence | Plausible share | CMS-form coefficient | Incremental cost |
|---|---:|---:|---:|---:|---:|
| NVS015 | Polyneuropathies | 0.8% | 42.1% | $11,532 | $16,955 |
| MUS003 | Rheumatoid arthritis and related disease | 1.3% | 44.2% | $10,203 | $12,555 |
| NVS009 | Epilepsy; convulsions | 0.9% | 42.1% | $9,438 | $12,003 |
| GEN006 | Other specified and unspecified diseases of kidney and ureters | 0.8% | 41.6% | $8,879 | $14,081 |
| CIR019 | Heart failure | 0.4% | 46.3% | $8,772 | $15,142 |
| MUS026 | Muscle disorders | 0.7% | 44.9% | $7,555 | $10,878 |
| NEO030 | Breast cancer - all other types | 0.7% | 41.0% | $6,327 | $7,707 |
| EYE005 | Retinal and vitreous conditions | 1.1% | 42.0% | $6,300 | $8,523 |
| RSP008 | Chronic obstructive pulmonary disease and bronchiectasis | 1.3% | 46.4% | $5,550 | $10,448 |
| MBD003 | Bipolar and related disorders | 0.8% | 44.0% | $5,386 | $8,095 |
| INF009 | Parasitic, other specified and unspecified infections | 0.5% | 44.3% | $5,088 | $9,288 |
| GEN008 | Urinary incontinence | 0.5% | 41.9% | $4,964 | $10,028 |
| END002+END005 | Diabetes mellitus without complication | 7.8% | 39.2% | $4,721 | $5,399 |
| SKN001 | Skin and subcutaneous tissue infections | 1.1% | 42.4% | $4,552 | $7,835 |
| NEO039 | Male reproductive system cancers - prostate | 0.5% | 41.1% | $4,490 | $5,066 |
| DIG008 | Other specified and unspecified disorders of stomach and duodenum | 1.1% | 42.3% | $4,428 | $7,304 |
| EYE001 | Cornea and external disease | 0.8% | 42.2% | $4,095 | $5,811 |
| MUS013 | Osteoporosis | 0.5% | 45.0% | $4,037 | $5,842 |
| BLD003 | Aplastic anemia | 0.5% | 40.4% | $3,632 | $8,140 |
| SYM016 | Other general signs and symptoms | 3.0% | 41.6% | $3,531 | $7,707 |
| RSP016 | Other specified and unspecified lower respiratory disease | 0.7% | 46.9% | $3,435 | $6,600 |
| FAC010 | Other aftercare encounter | 0.6% | 45.0% | $3,223 | $7,177 |
| DIG001 | Intestinal infection | 0.6% | 42.8% | $3,217 | $5,371 |
| MBD014 | Neurodevelopmental disorders | 3.3% | 41.5% | $2,990 | $4,693 |
| GEN004 | Urinary tract infections | 2.4% | 40.0% | $2,643 | $4,729 |
