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
| CMS form, coding penalty tuned against the plan (Stackelberg) | F2 | coding penalty, weight chosen against the plan's response | tuned |
| CMS form, DRO (lambda=5) | F2 | worst-subgroup penalty | no |
| CMS form, capped + coding penalty + DRO (DRO lambda=1) | F2 | cap, coding penalty, worst-subgroup penalty | no |
| Boosting, capped + DRO | F3 | cap, worst-subgroup penalty on the linear layer | no |
| CMS form, trained against the plan | F2 | as the base formula | retrained |
| CMS form, coding-penalized, trained against the plan | F2 | as the base formula | retrained |
| CMS form, capped + penalized + DRO, trained against the plan | F2 | as the base formula | retrained |
| Boosting, capped + DRO, trained against the plan | F3 | as the base formula | retrained |

**Table 2.** Calibrating the plan against the CMS form, on training rows.

*One code per reviewed person at $1,000 a code, with audit exposure of $2,000 per 1% of enrollees already given the same code. Targets: coding 9.5% and selection 10.5% of base payment (MedPAC's $40 billion and $44 billion over a fee-for-service-equivalent base of about $420 billion). The calibrated point is 6% reach and tilt 0.20. Upper panel: by reach at the calibrated tilt. Lower panel: by tilt at the calibrated reach. Means over the training rows of the 15 splits.*

| Reach | Coding | Selection | Codes per 1,000 | Distinct codes | Top code share | Distance (points) |
|---|---:|---:|---:|---:|---:|---:|
| 2% | 3.4% | 9.4% | 20 | 4.7 | 47% | 6.2 |
| 4% | 6.3% | 9.8% | 40 | 7.3 | 39% | 3.2 |
| 5% | 7.7% | 10.1% | 50 | 8.3 | 35% | 1.9 |
| 6% | 8.9% | 10.3% | 60 | 8.9 | 32% | 0.6 |
| 7.5% | 10.7% | 10.5% | 75 | 10.3 | 29% | 1.2 |
| 10% | 13.4% | 11.0% | 100 | 12.3 | 25% | 3.9 |
| 12.5% | 15.9% | 11.4% | 125 | 14.7 | 22% | 6.4 |
| 15% | 18.2% | 11.8% | 150 | 16.3 | 20% | 8.7 |
| 20% | 22.2% | 12.5% | 200 | 21.3 | 17% | 12.8 |
| 25% | 25.7% | 13.1% | 250 | 26.1 | 15% | 16.3 |
| 50% | 37.2% | 15.6% | 500 | 56.7 | 9% | 28.2 |
| 100% | 40.6% | 15.5% | 686 | 82.7 | 7% | 31.5 |

| Tilt | Coding | Selection | Distance (points) |
|---|---:|---:|---:|
| 0.00 | 8.9% | 0.0% | 10.5 |
| 0.05 | 8.9% | 2.6% | 7.9 |
| 0.10 | 8.9% | 5.1% | 5.4 |
| 0.15 | 8.9% | 7.7% | 2.9 |
| 0.20 | 8.9% | 10.3% | 0.6 |
| 0.25 | 8.9% | 12.8% | 2.4 |
| 0.30 | 8.9% | 15.4% | 4.9 |
| 0.50 | 8.9% | 25.6% | 15.2 |

**Table 3.** What the calibrated plan extracts from each formula, per 1,000 enrollees.

*Coding: payment for added codes, net of the cost of adding them, for codes that change nothing about cost. Selection: payment above cost from tilting enrollment toward the half of enrollees the plan expects to be overpaid, split in the lower panel into the part on the formula's ungamed payment and the part from tilting toward people the plan coded. Share is of base payment. 95% bootstrap intervals in parentheses; fold range over the 15 test folds. Distinct codes and top code share describe the plan's coding per fold.*

| Formula | Coding |  | Selection |  | Total |  | Share |
|---|---:|---:|---:|---:|---:|---:|---:|
| CMS form | $566,291 | ($548,206 to $584,798) | $771,309 | ($721,612 to $813,610) | $1,337,599 | ($1,282,598 to $1,385,782) | 17.1% |
| CMS form, with prior use | $463,722 | ($446,878 to $477,951) | $696,466 | ($647,977 to $736,800) | $1,160,188 | ($1,105,496 to $1,203,527) | 14.9% |
| Unconstrained WLS | $482,100 | ($464,715 to $502,726) | $829,713 | ($781,623 to $871,768) | $1,311,813 | ($1,260,470 to $1,365,028) | 16.8% |
| Tweedie boosting | $520,346 | ($497,570 to $543,906) | $234,953 | ($199,704 to $273,988) | $755,299 | ($711,739 to $803,995) | 9.7% |
| Fair stacked (paper 6) | $902,143 | ($868,774 to $935,203) | $654,898 | ($604,981 to $710,089) | $1,557,041 | ($1,487,068 to $1,634,806) | 20.1% |
| CMS form, cost-capped | $566,257 | ($549,586 to $582,161) | $762,444 | ($712,941 to $804,833) | $1,328,701 | ($1,275,496 to $1,375,657) | 17.0% |
| CMS form, coding-penalized (lambda=1) | $66,357 | ($64,343 to $68,767) | $677,077 | ($625,573 to $726,535) | $743,433 | ($692,036 to $792,697) | 9.5% |
| CMS form, coding penalty tuned against the plan (Stackelberg) | $39,308 | ($37,688 to $40,978) | $691,861 | ($639,506 to $742,985) | $731,169 | ($678,528 to $783,013) | 9.4% |
| CMS form, DRO (lambda=5) | $286,729 | ($277,475 to $295,253) | $666,919 | ($618,912 to $725,718) | $953,648 | ($902,913 to $1,010,814) | 12.2% |
| CMS form, capped + coding penalty + DRO (DRO lambda=1) | $106,259 | ($102,168 to $111,387) | $667,791 | ($617,238 to $713,874) | $774,050 | ($725,096 to $819,090) | 9.9% |
| Boosting, capped + DRO | $519,037 | ($496,548 to $542,129) | $241,331 | ($205,183 to $281,063) | $760,367 | ($717,896 to $809,795) | 9.8% |
| CMS form, trained against the plan | $391,941 | ($379,132 to $405,833) | $730,448 | ($684,348 to $775,746) | $1,122,389 | ($1,075,731 to $1,175,787) | 14.4% |
| CMS form, coding-penalized, trained against the plan | $67,540 | ($65,098 to $69,963) | $681,326 | ($631,950 to $728,484) | $748,865 | ($700,016 to $795,941) | 9.6% |
| CMS form, capped + penalized + DRO, trained against the plan | $104,752 | ($100,637 to $109,665) | $672,676 | ($623,149 to $719,772) | $777,428 | ($727,893 to $824,058) | 10.0% |
| Boosting, capped + DRO, trained against the plan | $479,388 | ($457,384 to $504,496) | $196,385 | ($162,548 to $236,885) | $675,774 | ($631,538 to $726,524) | 8.7% |

| Formula | Selection on ungamed payment | Selection on coded people | Lowest fold | Highest fold | Distinct codes used | Top code share |
|---|---:|---:|---:|---:|---:|---:|
| CMS form | $651,605 | $119,704 | $1,251,632 | $1,407,072 | 6.0 | 41% |
| CMS form, with prior use | $599,792 | $96,674 | $1,078,185 | $1,233,882 | 5.8 | 42% |
| Unconstrained WLS | $721,039 | $108,675 | $1,150,251 | $1,468,181 | 6.9 | 43% |
| Tweedie boosting | $105,659 | $129,294 | $448,410 | $977,015 | 13.5 | 43% |
| Fair stacked (paper 6) | $441,823 | $213,075 | $1,090,419 | $1,931,086 | 7.5 | 43% |
| CMS form, cost-capped | $643,341 | $119,102 | $1,238,148 | $1,423,687 | 6.0 | 41% |
| CMS form, coding-penalized (lambda=1) | $667,466 | $9,611 | $657,834 | $853,432 | 124.0 | 5% |
| CMS form, coding penalty tuned against the plan (Stackelberg) | $687,387 | $4,474 | $630,942 | $866,181 | 107.0 | 3% |
| CMS form, DRO (lambda=5) | $610,309 | $56,610 | $854,928 | $1,095,242 | 9.9 | 26% |
| CMS form, capped + coding penalty + DRO (DRO lambda=1) | $649,004 | $18,787 | $684,372 | $901,776 | 28.2 | 16% |
| Boosting, capped + DRO | $112,596 | $128,735 | $431,773 | $972,914 | 13.4 | 43% |
| CMS form, trained against the plan | $651,317 | $79,131 | $1,016,261 | $1,327,339 | 9.2 | 30% |
| CMS form, coding-penalized, trained against the plan | $671,672 | $9,654 | $674,514 | $863,037 | 124.0 | 6% |
| CMS form, capped + penalized + DRO, trained against the plan | $657,428 | $15,247 | $675,619 | $895,435 | 27.3 | 16% |
| Boosting, capped + DRO, trained against the plan | $88,310 | $108,075 | $392,387 | $1,134,344 | 18.1 | 32% |

**Table 4.** Accuracy before and after the plan responds.

*R² on ungamed data is what the regulator gives up; R² on post-response data, with the plan's coded features and tilted weights, is what it experiences.*

| Formula | R², ungamed |  | R², post-response |  | Difference from CMS form |
|---|---:|---:|---:|---:|---:|
| CMS form | 0.098 | (0.076 to 0.130) | 0.078 | (0.061 to 0.103) | 0.000 |
| CMS form, with prior use | 0.108 | (0.085 to 0.141) | 0.091 | (0.074 to 0.119) | 0.010 |
| Unconstrained WLS | 0.123 | (0.102 to 0.155) | 0.099 | (0.084 to 0.122) | 0.025 |
| Tweedie boosting | 0.187 | (0.156 to 0.242) | 0.175 | (0.148 to 0.225) | 0.089 |
| Fair stacked (paper 6) | 0.169 | (0.142 to 0.215) | 0.123 | (0.095 to 0.161) | 0.072 |
| CMS form, cost-capped | 0.098 | (0.076 to 0.130) | 0.079 | (0.062 to 0.105) | 0.000 |
| CMS form, coding-penalized (lambda=1) | 0.096 | (0.074 to 0.127) | 0.089 | (0.070 to 0.116) | -0.002 |
| CMS form, coding penalty tuned against the plan (Stackelberg) | 0.091 | (0.069 to 0.118) | 0.085 | (0.066 to 0.109) | -0.007 |
| CMS form, DRO (lambda=5) | 0.102 | (0.078 to 0.135) | 0.092 | (0.072 to 0.121) | 0.004 |
| CMS form, capped + coding penalty + DRO (DRO lambda=1) | 0.099 | (0.076 to 0.130) | 0.091 | (0.072 to 0.118) | 0.001 |
| Boosting, capped + DRO | 0.187 | (0.156 to 0.242) | 0.176 | (0.149 to 0.225) | 0.089 |
| CMS form, trained against the plan | 0.099 | (0.076 to 0.130) | 0.087 | (0.069 to 0.115) | 0.001 |
| CMS form, coding-penalized, trained against the plan | 0.097 | (0.074 to 0.127) | 0.089 | (0.070 to 0.116) | -0.001 |
| CMS form, capped + penalized + DRO, trained against the plan | 0.099 | (0.075 to 0.130) | 0.091 | (0.072 to 0.118) | 0.001 |
| Boosting, capped + DRO, trained against the plan | 0.186 | (0.154 to 0.239) | 0.182 | (0.152 to 0.237) | 0.088 |

**Table 5.** The price of robustness: accuracy against extraction along each penalty grid.

*Linear payment-form formulas on F2. Lambda is the penalty weight; the DRO alpha is 0.10.*

| Formula | R², ungamed | Coding | Selection | Total | Share |
|---|---:|---:|---:|---:|---:|
| CMS form | 0.098 | $566,291 | $771,309 | $1,337,599 | 17.1% |
| CMS form, cost-capped | 0.098 | $566,257 | $762,444 | $1,328,701 | 17.0% |
| CMS form, coding-penalized (lambda=0.5) | 0.097 | $73,566 | $682,237 | $755,802 | 9.7% |
| CMS form, coding-penalized (lambda=1) | 0.096 | $66,357 | $677,077 | $743,433 | 9.5% |
| CMS form, coding-penalized (lambda=2) | 0.095 | $58,186 | $680,053 | $738,239 | 9.5% |
| CMS form, coding-penalized (lambda=5) | 0.091 | $38,242 | $686,361 | $724,602 | 9.3% |
| CMS form, coding-penalized (lambda=10) | 0.085 | $16,084 | $706,459 | $722,543 | 9.3% |
| CMS form, coding-penalized (lambda=20) | 0.075 | $0 | $758,024 | $758,024 | 9.7% |
| CMS form, coding-penalized (lambda=50) | 0.060 | $0 | $845,534 | $845,534 | 10.8% |
| CMS form, DRO (lambda=1) | 0.102 | $306,274 | $658,789 | $965,064 | 12.4% |
| CMS form, DRO (lambda=2) | 0.102 | $294,272 | $675,197 | $969,469 | 12.4% |
| CMS form, DRO (lambda=5) | 0.102 | $286,729 | $666,919 | $953,648 | 12.2% |
| CMS form, DRO (lambda=10) | 0.102 | $284,341 | $678,875 | $963,216 | 12.4% |
| CMS form, DRO (lambda=20) | 0.101 | $283,146 | $673,380 | $956,526 | 12.3% |
| CMS form, DRO (lambda=50) | 0.101 | $282,189 | $678,777 | $960,967 | 12.3% |
| CMS form, capped + coding penalty + DRO (DRO lambda=0) | 0.096 | $66,539 | $681,910 | $748,448 | 9.6% |
| CMS form, capped + coding penalty + DRO (DRO lambda=1) | 0.099 | $106,259 | $667,791 | $774,050 | 9.9% |
| CMS form, capped + coding penalty + DRO (DRO lambda=2) | 0.099 | $129,402 | $665,366 | $794,768 | 10.2% |
| CMS form, capped + coding penalty + DRO (DRO lambda=5) | 0.100 | $167,971 | $660,533 | $828,504 | 10.6% |
| CMS form, capped + coding penalty + DRO (DRO lambda=10) | 0.101 | $199,405 | $662,486 | $861,892 | 11.1% |
| CMS form, capped + coding penalty + DRO (DRO lambda=20) | 0.101 | $227,654 | $670,684 | $898,338 | 11.5% |
| CMS form, capped + coding penalty + DRO (DRO lambda=50) | 0.101 | $254,619 | $662,489 | $917,107 | 11.8% |

**Table 6.** Net compensation by group before and after the plan responds: predicted minus observed spending per person-year.

*Negative is underpayment. Post-response values use the plan's coded features and tilted weights, so they include the coding rise. Race and ethnicity are evaluated only.*

| Group | CMS form, ungamed | post-response | CMS form, coding-penalized (lambda=1), ungamed | post-response | CMS form, capped + penalized + DRO, trained against the plan, ungamed | post-response | Tweedie boosting, ungamed | post-response | Boosting, capped + DRO, trained against the plan, ungamed | post-response | Fair stacked (paper 6), ungamed | post-response |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Needs ADL or IADL help | -$9,915 | -$5,921 | -$12,452 | -$10,109 | -$11,885 | -$9,518 | -$5,677 | -$2,284 | -$5,795 | -$1,790 | $847 | $6,639 |
| Age 65 and over | $22 | $3,118 | -$14 | $1,346 | $106 | $1,317 | -$455 | $1,678 | -$437 | $1,440 | $2,629 | $6,299 |
| Uninsured all year | $1,212 | $1,652 | $1,233 | $1,561 | $1,110 | $1,459 | $567 | $675 | $569 | $657 | -$532 | -$50 |
| Income below 200% FPL | $331 | $1,757 | -$30 | $854 | $9 | $936 | $6 | $851 | -$7 | $799 | $209 | $2,035 |
| Mental health condition | $4 | $2,437 | -$384 | $1,011 | $162 | $1,720 | $550 | $2,204 | $491 | $2,193 | $382 | $3,820 |
| Non-Hispanic Black | $86 | $1,352 | -$122 | $600 | -$109 | $714 | -$5 | $858 | -$31 | $729 | -$138 | $1,333 |
| Hispanic | $294 | $1,354 | $255 | $939 | $164 | $900 | $92 | $542 | $85 | $459 | -$576 | $306 |
| Non-Hispanic Asian | $591 | $1,614 | $830 | $1,273 | $710 | $1,287 | $301 | $731 | $338 | $730 | -$47 | $913 |

**Table 7.** Training against the plan: extraction on the training rows by iteration.

*Iteration 1 is the plan's response to the formula fitted on ungamed data. Change is the mean absolute movement of payments as a share of mean payment; the loop stops below 1%. Folds is the number of splits still iterating.*

| Formula | Iteration | Extraction | Coding | Selection | Change | Folds |
|---|---:|---:|---:|---:|---:|---:|
| CMS form, trained against the plan | 1 | $1,495,273 | $695,315 | $799,958 | 11.96% | 15 |
| CMS form, trained against the plan | 2 | $1,331,604 | $592,681 | $738,922 | 4.97% | 15 |
| CMS form, trained against the plan | 3 | $1,372,302 | $625,556 | $746,747 | 4.72% | 15 |
| CMS form, trained against the plan | 4 | $1,357,626 | $605,158 | $752,468 | 4.06% | 15 |
| CMS form, trained against the plan | 5 | $1,358,551 | $617,584 | $740,967 | 4.14% | 15 |
| CMS form, trained against the plan | 10 | $1,364,698 | $617,223 | $747,475 | 3.86% | 15 |
| CMS form, trained against the plan | 15 | $1,361,876 | $610,885 | $750,991 | 3.96% | 15 |
| CMS form, trained against the plan | 20 | $1,367,274 | $613,974 | $753,300 | 3.74% | 15 |
| CMS form, coding-penalized, trained against the plan | 1 | $746,962 | $66,541 | $680,421 | 9.57% | 15 |
| CMS form, coding-penalized, trained against the plan | 2 | $700,984 | $54,498 | $646,486 | 0.91% | 15 |
| CMS form, coding-penalized, trained against the plan | 3 | $691,430 | $53,439 | $637,991 | 1.13% | 5 |
| CMS form, coding-penalized, trained against the plan | 4 | $680,884 | $52,896 | $627,988 | 0.58% | 3 |
| CMS form, capped + penalized + DRO, trained against the plan | 1 | $763,381 | $106,889 | $656,492 | 10.18% | 15 |
| CMS form, capped + penalized + DRO, trained against the plan | 2 | $700,892 | $87,755 | $613,138 | 0.99% | 15 |
| CMS form, capped + penalized + DRO, trained against the plan | 3 | $688,777 | $83,253 | $605,524 | 1.16% | 7 |
| CMS form, capped + penalized + DRO, trained against the plan | 4 | $685,098 | $84,469 | $600,629 | 0.96% | 5 |
| CMS form, capped + penalized + DRO, trained against the plan | 5 | $654,229 | $78,223 | $576,005 | 1.10% | 2 |
| Boosting, capped + DRO, trained against the plan | 1 | $856,461 | $559,878 | $296,583 | 7.59% | 15 |
| Boosting, capped + DRO, trained against the plan | 2 | $875,078 | $579,997 | $295,081 | 7.57% | 15 |
| Boosting, capped + DRO, trained against the plan | 3 | $921,997 | $610,273 | $311,725 | 7.03% | 15 |
| Boosting, capped + DRO, trained against the plan | 4 | $915,471 | $601,386 | $314,086 | 6.03% | 15 |
| Boosting, capped + DRO, trained against the plan | 5 | $854,809 | $564,087 | $290,722 | 6.85% | 15 |
| Boosting, capped + DRO, trained against the plan | 10 | $886,192 | $583,949 | $302,243 | 6.71% | 15 |

**Table 8.** The 20 codes the CMS form pays most for, and what the robust formulas pay for them.

*Full-sample fits. Increment is the payment rise when the code is added to a person who could plausibly receive it. Incremental cost is the weighted difference in year-2 spending between people with and without the code, within age, sex and number of other conditions.*

| CCSR | Condition | Prevalence | Incremental cost | CMS form | Penalized | Robust | Robust, trained |
|---|---:|---:|---:|---:|---:|---:|---:|
| NVS015 | Polyneuropathies | 0.8% | $16,955 | $12,504 | $2,196 | $2,754 | $2,640 |
| MUS003 | Rheumatoid arthritis and related disease | 1.3% | $12,555 | $11,038 | $2,110 | $3,208 | $3,101 |
| NVS009 | Epilepsy; convulsions | 0.9% | $12,003 | $10,419 | $2,097 | $2,789 | $2,718 |
| GEN006 | Other specified and unspecified diseases of kidney and ureters | 0.8% | $14,081 | $9,883 | $2,159 | $2,854 | $2,920 |
| CIR019 | Heart failure | 0.4% | $15,142 | $9,534 | $1,835 | $2,368 | $2,446 |
| MUS026 | Muscle disorders | 0.7% | $10,878 | $8,394 | $1,888 | $2,483 | $2,486 |
| NEO030 | Breast cancer - all other types | 0.7% | $7,707 | $7,382 | $2,197 | $2,782 | $2,997 |
| EYE005 | Retinal and vitreous conditions | 1.1% | $8,523 | $7,301 | $2,144 | $2,488 | $2,581 |
| RSP008 | Chronic obstructive pulmonary disease and bronchiectasis | 1.3% | $10,448 | $6,500 | $2,127 | $2,732 | $2,817 |
| MBD003 | Bipolar and related disorders | 0.8% | $8,095 | $6,326 | $1,971 | $2,445 | $2,376 |
| INF009 | Parasitic, other specified and unspecified infections | 0.5% | $9,288 | $6,108 | $2,019 | $2,185 | $2,183 |
| GEN008 | Urinary incontinence | 0.5% | $10,028 | $5,955 | $2,045 | $2,189 | $2,238 |
| END002+END005 | Diabetes mellitus without complication | 7.8% | $5,399 | $5,593 | $2,326 | $3,854 | $3,799 |
| SKN001 | Skin and subcutaneous tissue infections | 1.1% | $7,835 | $5,576 | $2,071 | $2,493 | $2,424 |
| NEO039 | Male reproductive system cancers - prostate | 0.5% | $5,066 | $5,528 | $2,135 | $2,737 | $2,438 |
| DIG008 | Other specified and unspecified disorders of stomach and duodenum | 1.1% | $7,304 | $5,412 | $2,024 | $2,266 | $2,259 |
| EYE001 | Cornea and external disease | 0.8% | $5,811 | $5,104 | $2,009 | $1,977 | $1,941 |
| MUS013 | Osteoporosis | 0.5% | $5,842 | $4,838 | $1,810 | $1,870 | $1,923 |
| BLD003 | Aplastic anemia | 0.5% | $8,140 | $4,742 | $2,116 | $2,173 | $2,169 |
| SYM016 | Other general signs and symptoms | 3.0% | $7,707 | $4,495 | $2,143 | $2,707 | $2,740 |

**Table 8b.** Payment per condition and per body system under each formula (full-sample fits).

| Formula | Per condition | Per body system |
|---|---:|---:|
| cms | $575 | $536 |
| cms_pen_1 | $1,499 | $544 |
| cms_cap_pen_dro_1 | $1,595 | $403 |
| cms_robust_adv | $1,636 | $337 |

**Table 9.** Sensitivity of extraction to the plan, one parameter at a time from the calibrated point.

*Per 1,000 enrollees; mean over the 15 test folds. The last block makes a share of the added codes real, raising the person's cost by the code's incremental cost.*

| Parameter | Value | CMS form | CMS form, coding-penalized (lambda=1) | CMS form, capped + penalized + DRO, trained against the plan |
|---|---:|---:|---:|---:|
| cost per code | 0.0 | $1,398,127 | $803,732 | $839,793 |
| cost per code | 250.0 | $1,383,112 | $788,719 | $824,779 |
| cost per code | 500.0 | $1,368,096 | $773,705 | $809,765 |
| cost per code | 1000.0 | $1,338,065 | $743,678 | $779,736 |
| cost per code | 2000.0 | $1,278,003 | $684,728 | $719,680 |
| cost per code | 3000.0 | $1,217,941 | $675,384 | $676,172 |
| codes per person | 1 | $1,338,065 | $743,678 | $779,736 |
| codes per person | 2 | $1,843,407 | $811,706 | $867,188 |
| codes per person | 3 | $2,261,311 | $876,479 | $940,814 |
| reach | 1 | $3,524,712 | $1,017,130 | $1,161,849 |
| reach | 0.02 | $929,098 | $702,991 | $716,878 |
| reach | 0.06 | $1,338,065 | $743,678 | $779,736 |
| reach | 0.1 | $1,660,530 | $788,323 | $833,189 |
| reach | 0.25 | $2,552,429 | $919,956 | $982,015 |
| reach | 0.5 | $3,372,877 | $1,017,130 | $1,147,951 |
| audit exposure | 0.0 | $1,534,681 | $773,097 | $875,244 |
| audit exposure | 500.0 | $1,460,675 | $751,739 | $819,433 |
| audit exposure | 2000.0 | $1,338,065 | $743,678 | $779,736 |
| audit exposure | 5000.0 | $1,219,642 | $739,504 | $754,451 |
| tilt | 0.0 | $566,196 | $66,357 | $105,244 |
| tilt | 0.1 | $952,123 | $405,009 | $442,485 |
| tilt | 0.5 | $2,495,985 | $1,759,786 | $1,791,554 |
| tilt | 0.05 | $759,157 | $235,681 | $273,863 |
| tilt | 0.2 | $1,338,065 | $743,678 | $779,736 |
| tilt | 0.3 | $1,724,023 | $1,082,364 | $1,116,998 |
| plan cost model | boosting on F3 only | $1,317,983 | $745,281 | $771,508 |
| plan cost model | WLS on F3 | $1,076,106 | $531,229 | $572,959 |
| plausibility | system | $1,276,705 | $720,251 | $761,136 |
| plausibility | use | $1,340,802 | $744,701 | $781,099 |
| plausibility | any | $1,351,512 | $754,713 | $792,207 |
| share of added codes real | 0.0 | $1,338,065 | $743,678 | $779,736 |
| share of added codes real | 1 | $273,908 | $356,605 | $274,838 |
| share of added codes real | 0.25 | $1,084,901 | $647,558 | $655,368 |
| share of added codes real | 0.5 | $819,672 | $553,968 | $520,915 |

**Table 10.** Robustness rows.

*The 65-and-over rows restrict the stored out-of-fold results to that age group. Other rows are refits with one repeat of the five folds; leave-one-panel-out holds each MEPS panel out in turn.*

| Variant | Formula | R², ungamed | Coding | Selection | Total |
|---|---:|---:|---:|---:|---:|
| persons 65 and over (subgroup of all-age results) | CMS form | 0.066 | $1,379,219 | $1,513,814 | $2,893,033 |
| persons 65 and over (subgroup of all-age results) | CMS form, coding-penalized (lambda=1) | 0.068 | $88,120 | $1,270,904 | $1,359,024 |
| persons 65 and over (subgroup of all-age results) | CMS form, capped + coding penalty + DRO (DRO lambda=1) | 0.070 | $58,081 | $1,174,195 | $1,232,276 |
| persons 65 and over (subgroup of all-age results) | Tweedie boosting | 0.167 | $1,314,639 | $459,713 | $1,774,352 |
| persons 65 and over (subgroup of all-age results) | Boosting, capped + DRO | 0.167 | $1,315,851 | $491,998 | $1,807,850 |
| persons 65 and over (subgroup of all-age results) | CMS form, coding-penalized, trained against the plan | 0.068 | $86,911 | $1,277,228 | $1,364,139 |
| persons 65 and over (subgroup of all-age results) | CMS form, capped + penalized + DRO, trained against the plan | 0.070 | $48,978 | $1,220,698 | $1,269,676 |
| F3 (prior use) for the penalized form | F3 (prior use) for the penalized form | 0.112 | $554 | $635,168 | $635,722 |
| F3 (prior use) for the robust form | F3 (prior use) for the robust form | 0.115 | $29,817 | $646,680 | $676,497 |
| DRO alpha 0.05 | DRO alpha 0.05 | 0.105 | $148,342 | $684,459 | $832,801 |
| DRO alpha 0.20 | DRO alpha 0.20 | 0.104 | $67,978 | $613,115 | $681,093 |
| squared-error boosting | squared-error boosting | 0.201 | $642,547 | $309,196 | $951,744 |
| leave one panel out | CMS form | 0.105 | $558,635 | $750,517 | $1,309,152 |
| leave one panel out | CMS form, coding-penalized (lambda=1) | 0.103 | $66,399 | $685,679 | $752,078 |
| leave one panel out | Tweedie boosting | 0.190 | $662,542 | $256,702 | $919,244 |


# Appendix tables


**Table A1.** The codable pool: the 25 codes with the largest CMS-form increments.

*Plausible share is the weighted share of people who lack the code and could receive it under the primary rule (a code in the same body system already present, or year-1 spending in the top half).*

| CCSR | Condition | Prevalence | Plausible share | CMS-form coefficient | Incremental cost |
|---|---:|---:|---:|---:|---:|
| NVS015 | Polyneuropathies | 0.8% | 34.0% | $11,532 | $16,955 |
| MUS003 | Rheumatoid arthritis and related disease | 1.3% | 34.4% | $10,203 | $12,555 |
| NVS009 | Epilepsy; convulsions | 0.9% | 42.1% | $9,438 | $12,003 |
| GEN006 | Other specified and unspecified diseases of kidney and ureters | 0.8% | 34.4% | $8,879 | $14,081 |
| CIR019 | Heart failure | 0.4% | 28.7% | $8,772 | $15,142 |
| MUS026 | Muscle disorders | 0.7% | 39.3% | $7,555 | $10,878 |
| NEO030 | Breast cancer - all other types | 0.7% | 16.5% | $6,327 | $7,707 |
| EYE005 | Retinal and vitreous conditions | 1.1% | 27.3% | $6,300 | $8,523 |
| RSP008 | Chronic obstructive pulmonary disease and bronchiectasis | 1.3% | 31.2% | $5,550 | $10,448 |
| MBD003 | Bipolar and related disorders | 0.8% | 39.0% | $5,386 | $8,095 |
| INF009 | Parasitic, other specified and unspecified infections | 0.5% | 40.5% | $5,088 | $9,288 |
| GEN008 | Urinary incontinence | 0.5% | 21.3% | $4,964 | $10,028 |
| END002+END005 | Diabetes mellitus without complication | 7.8% | 35.9% | $4,721 | $5,399 |
| SKN001 | Skin and subcutaneous tissue infections | 1.1% | 42.4% | $4,552 | $7,835 |
| NEO039 | Male reproductive system cancers - prostate | 0.5% | 10.3% | $4,490 | $5,066 |
| DIG008 | Other specified and unspecified disorders of stomach and duodenum | 1.1% | 40.2% | $4,428 | $7,304 |
| EYE001 | Cornea and external disease | 0.8% | 41.4% | $4,095 | $5,811 |
| MUS013 | Osteoporosis | 0.5% | 20.3% | $4,037 | $5,842 |
| BLD003 | Aplastic anemia | 0.5% | 30.5% | $3,632 | $8,140 |
| SYM016 | Other general signs and symptoms | 3.0% | 41.6% | $3,531 | $7,707 |
| RSP016 | Other specified and unspecified lower respiratory disease | 0.7% | 43.2% | $3,435 | $6,600 |
| FAC010 | Other aftercare encounter | 0.6% | 39.8% | $3,223 | $7,177 |
| DIG001 | Intestinal infection | 0.6% | 42.3% | $3,217 | $5,371 |
| MBD014 | Neurodevelopmental disorders | 3.3% | 37.2% | $2,990 | $4,693 |
| GEN004 | Urinary tract infections | 2.4% | 37.1% | $2,643 | $4,729 |
