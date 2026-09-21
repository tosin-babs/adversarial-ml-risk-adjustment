# Adversarial Machine Learning for Medicare Risk Adjustment: Training Payment Formulas Against a Strategic Health Plan

**Temitope Ologunbaba**¹ *(corresponding author)*, **Chisom Adiegwu**², **Adebolu Temitope**³

¹ Faculty of Engineering, Federal University of Technology, Akure, Nigeria. ologubabatopeeee2351@futa.edu.ng

² Department of Actuarial Science and Quantitative Risk Analysis and Management, Georgia State University, Atlanta, GA, USA.

³ Department of Epidemiology and Medical Statistics, University of Ibadan, Ibadan, Nigeria.

**Word count.** 4,599 excluding abstract, tables, figure captions and references.

---

## Abstract

**Background.** Risk-adjustment formulas are fitted to data that health plans have not yet responded to, and plans respond. The Medicare Payment Advisory Commission estimates that Medicare Advantage is paid about 20% more than fee-for-service Medicare would spend on the same people, through two channels: coding intensity and favorable selection. We ask whether machine-learning methods built for adversarial settings can produce a payment formula that holds up once a plan has done what the formula rewards.

**Methods.** On the public Medical Expenditure Panel Survey benchmark of Babalola et al. (2026), 43,963 persons in panels 23 to 27 with year-1 predictors and year-2 spending, we simulate a strategic plan. It adds diagnosis codes where the payment rise exceeds the cost of adding them, among codes plausible for the person and within a limited chart-review reach, and it tilts enrollment toward people its own cross-fitted cost model expects to be overpaid. Both channels are calibrated to MedPAC's published magnitudes. We then compare a payment formula in the form used by CMS with three robust versions (a coding penalty, caps at each code's incremental cost, and a distributionally robust penalty on the worst-priced subgroup), with gradient boosting, and with versions of each trained against the plan by repeated risk minimization. Every quantity is out of sample, with a PSU bootstrap paired across formulas.

**Results.** The calibrated plan extracts $1,517,780 per 1,000 enrollees from the CMS-form formula, 19.5% of base payment, in roughly equal parts from coding and selection. A coding penalty removes 88% of the coding channel at an R² cost of 0.002. Caps at incremental cost remove nothing, because an added code changes nothing about cost. No linear formula closes the selection channel; boosting on prior use does, but stays open to coding. The penalized formula trained against the plan converges in two rounds and halves extraction, to $723,737; the unpenalized formula trained the same way never converges, and boosting trained against the plan becomes more gameable, not less. Every formula that resists coding underpays people who need help with daily activities by more than the CMS form does.

**Conclusions.** Coding is cheap to defend against; selection is an information problem that a diagnosis-based formula cannot solve; and adversarial retraining helps only when the formula is constrained. Robustness to gaming and fair payment for the frail pull in opposite directions and have to be set together.

**Keywords.** risk adjustment; adversarial machine learning; strategic classification; performative prediction; distributionally robust optimization; upcoding; risk selection; Medicare Advantage; MEPS

---

## 1. Introduction

Medicare pays private Medicare Advantage plans a monthly amount for each enrollee, scaled by a risk score built from age, sex and recorded diagnoses (Pope et al., 2004). The formula behind the score is a regression of annual spending on demographic cells and condition categories, fitted to fee-for-service claims, where no plan has an interest in what is recorded. It is then applied to plans that do. The Medicare Payment Advisory Commission projected that Medicare would spend 20% more on Medicare Advantage enrollees in 2025 than it would have spent on the same people in fee-for-service Medicare, about $84 billion. It attributed $40 billion to coding intensity, with risk scores about 16% above those of comparable fee-for-service beneficiaries before a 5.9% statutory adjustment, and $44 billion to favorable selection (MedPAC, 2025).

The two channels are well documented. Risk scores rise when the same people move into Medicare Advantage (Geruso and Layton, 2020), and coding intensity has been measured for more than a decade (Kronick and Welch, 2014). When CMS refined its formula, plans selected on the risk that remained unpriced (Brown et al., 2014). Plans can also shape who enrolls through networks and benefit design (Shepard, 2022). In response, CMS rebuilt its model for 2024 to 2026, cutting the diagnosis codes that map to payment and dropping conditions whose coding it judged discretionary (CMS, 2023).

What is less developed is a way to build a formula that anticipates the response. The economics of risk adjustment has long recognized that the best predictor of cost is not the best payment formula when plans can select (Glazer and McGuire, 2000), and it measures a formula by the profits it leaves available to a plan (Layton et al., 2017). Constrained and penalized regressions can remove chosen underpayments (Zink and Rose, 2020; McGuire, Zink and Rose, 2021), and data transformations can reduce selection incentives (Bergquist et al., 2019). But these methods fit the formula to data a plan has not yet gamed. Machine learning has a literature built for exactly this problem. Strategic classification trains a model knowing that its inputs will be manipulated at a cost (Hardt et al., 2016). Performative prediction treats a deployed model as something that changes the data it will later be evaluated on, and repeated retraining as the way to find a stable point (Perdomo et al., 2020). Distributionally robust optimization bounds a model's worst performance over any subgroup of a given size (Hashimoto et al., 2018; Duchi and Namkoong, 2021). These tools have not been applied to health plan payment, where manipulation is costly, partly legitimate, and has published magnitudes to calibrate against.

This paper applies them. A companion study benchmarked ten risk-adjustment models on the Medical Expenditure Panel Survey (MEPS) and measured how much each pays for an added diagnosis code and which groups each underpays (Babalola et al., 2026). It treated the plan as passive. Here the plan is strategic: it best-responds to any formula by coding and by selection, calibrated so that its response to a CMS-form formula matches MedPAC's figures. We ask four questions.

- RQ1. How much can a strategic plan extract from each kind of formula, and through which channel?
- RQ2. Can a formula be trained against the plan's response, and what does robustness cost in accuracy?
- RQ3. What does robustness to gaming do to what each group is paid relative to what it costs?
- RQ4. Which codes lose payment under a robust formula?

The main findings follow. The calibrated plan takes 19.5% of payment from the CMS-form formula, half by coding and half by selection. A penalty on the codes a plan can add removes almost all of the coding channel at almost no loss of accuracy; capping each code at its incremental cost removes none of it. No formula built on diagnoses closes the selection channel, because the plan selects on information the formula does not use; boosting on prior use nearly closes it, and pays the price in coding. Training against the plan works for the penalized formula, which settles in two rounds at half the extraction of the CMS form, and fails for the unpenalized formula, which cycles, and for boosting, which gets worse. Every formula that resists coding pays less for people who need help with daily activities, a group every formula already underpays.

## 2. Background

### 2.1 Risk adjustment when plans respond

A risk-adjustment formula f maps an enrollee's recorded characteristics x to a payment. Fitted by least squares to cost y on data where x is recorded without regard to payment, f is the best linear predictor of cost. Glazer and McGuire (2000) show that when a plan can select on characteristics that f does not pay for, the efficient formula deliberately departs from the best predictor, overpaying observable signals of the unpriced risk. Layton et al. (2017) formalize the evaluation: a formula is judged by the predictable profit it leaves at the level of groups a plan can target. Payment formulas in practice constrain the regression: CMS-HCC gives every condition a non-negative increment, uses separate base rates for each age and sex cell, and has no intercept (Pope et al., 2004).

Coding enters because x is not fixed. A diagnosis recorded for payment purposes changes the formula's input without changing the person's cost. Geruso and Layton (2020) estimate the effect from people switching between fee-for-service and Medicare Advantage, and MedPAC (2025) puts the net effect at about 10% of payment after the statutory adjustment. CMS's own responses are blunt: an across-the-board adjustment, and, in the 2024 model, removal or constraint of codes it judged discretionary (CMS, 2023).

### 2.2 Machine learning against a strategic adversary

Three strands of machine learning are relevant. Strategic classification (Hardt et al., 2016) models an agent who changes its features at a cost to obtain a better outcome from a classifier and trains the classifier knowing this; Miller, Milli and Hardt (2020) show that the distinction between gaming and genuine improvement is causal, which matters here because some added codes are real diagnoses. Hu, Immorlica and Vaughan (2019) show that the costs of strategic manipulation fall unevenly across groups. Performative prediction (Perdomo et al., 2020) generalizes the idea: when a deployed model changes the distribution of its inputs, repeatedly refitting on the data it induces converges, under conditions, to a performatively stable model. Distributionally robust optimization bounds the loss over every subpopulation of at least a given weight share (Hashimoto et al., 2018; Duchi and Namkoong, 2021); for a mean-type loss the worst subgroup of share α is the upper α tail, so the bound is the conditional value at risk of the residual (Rockafellar and Uryasev, 2000).

### 2.3 The gap

The health-economics literature has the right objective, profit left to a plan, but fits formulas to ungamed data and evaluates them on groups chosen in advance. The machine-learning literature has the right machinery, training against a response and bounding the worst subgroup, but has not been applied to plan payment. No study trains a payment formula against a calibrated strategic plan on public data, reports what robustness costs, and asks what it does to the groups that are already underpaid.

## 3. Data

Everything in this section comes from the companion benchmark (Babalola et al., 2026) and is reused without change, so every result here is comparable with it. The data are five two-year MEPS panels (23 to 27, 2018 to 2023), 43,963 persons, with year-1 characteristics predicting year-2 total spending in 2024 dollars. People who died in year 2 are kept, with spending annualized by months in scope and weights multiplied by the same fraction. Conditions are the 124 three-digit Clinical Classifications Software Refined (CCSR) categories held by at least 0.5% of the sample, with counts of conditions and of body systems. Two feature sets are used: F2 (age band by sex, region, condition flags and counts) and F3 (F2 plus year-1 use and spending). F2 is primary for every payment-form formula, because payment formulas exclude prior spending; F3 versions are reported alongside.

Cross-validation uses five folds formed from whole primary sampling units within strata, repeated three times, giving 15 test folds; the reproduction of the companion benchmark's payment-form accuracy is exact (R² 0.0980 on F2 and 0.1073 on F3). Intervals are from a Rao-Wu rescaled bootstrap over PSUs within strata (Rao and Wu, 1988), 200 resamples, with the same resamples for every formula so that differences are paired. Group membership (people needing help with activities of daily living, the uninsured, people aged 65 and over, income below 200% of poverty, a mental health condition, and race and ethnicity) is used only to evaluate payment, never as a feature and never as something the plan can see.

## 4. The strategic plan

### 4.1 Coding

For a formula f and a person i without code j, the payment rise from adding j is Δᵢⱼ = f(xᵢ with code j) − f(xᵢ), with the condition and body-system counts updated. The plan adds code j when Δᵢⱼ exceeds the cost c of adding it, only for codes plausible for the person, at most k codes per person in order of gain, and only for the share r of enrollees with most to gain. A code is plausible for a person who already carries a code in the same body system or whose year-1 spending is in the top half; the robustness section uses each rule alone and no restriction. The plausible pool is large: across the 124 codes a median of about 42% of people lacking a code could receive it (Table A1).

### 4.2 Selection

The plan cannot see cost. It sees its own prediction, from a Tweedie gradient-boosting model on F3 cross-fitted within each training fold, so that it never sees the rows it is applied to. Its expected margin on person i is the payment f(xᵢ) less its predicted cost. It recruits the half of enrollees with the highest expected margin at weight 1 + τ and the other half at 1 − τ, holding its size fixed.

### 4.3 What the plan extracts

On a test fold, the plan's coding gain is the payment rise from its added codes less their cost, and its selection gain is the payment above true cost that the tilt brings in, Σ wᵢ(sᵢ − 1)(f(x̃ᵢ) − yᵢ), where sᵢ is the tilt and x̃ᵢ the coded record. Both are reported per 1,000 enrollees of survey weight and as a share of base payment. The added codes change nothing about cost in the primary specification; the robustness section lets a share of them be real.

### 4.4 Calibration

The plan is calibrated to the CMS-form formula, the payment-form weighted least squares of the companion study on F2. With one code per reviewed person at $1,000 a code, reviewing 6% of enrollees raises payment by 10.4%, against MedPAC's net coding effect of 10.1% (16% less the 5.9% adjustment). A tilt of 0.20 gives the plan a profit of 8.5% of payment from selection, against MedPAC's $44 billion of favorable selection on roughly $500 billion of Medicare Advantage payment (Table 2, Figure 3). The grid shows why reach is the lever: the cost per code barely matters, because the formula pays more than $1,000 for most plausible codes, while payment rises almost in proportion to the share of charts reviewed.

![](../output/figures/fig3_calibration.png)

**Figure 3.** Payment rise under the CMS-form formula by chart-review reach and codes added per person, at $1,000 per code. Means over 15 test folds.

## 5. Robust and adversarially trained formulas

All formulas keep the payment form: a base rate for each age band and sex, non-negative increments for everything else, and no intercept. Every formula is normalized so that total payment equals total cost on its training data, as CMS normalizes its scores, so no formula gains or loses by changing the size of the budget.

**Coding penalty.** A ridge penalty λ Σⱼ pⱼ βⱼ² on the increments of codable flags, where pⱼ is the share of people for whom code j is plausible, so the easiest codes are shrunk most. The two count variables are penalized with weight one, because every added code raises them; a penalty on the flags alone moves their payment into the counts, which are exactly as gameable.

**Cost caps.** Each code's increment is bounded above by its incremental cost, the weighted difference in year-2 spending between people with and without the code within cells of age, sex and number of other conditions. This is the logic of CMS's 2024 constraints applied to every code.

**Worst-subgroup penalty.** A penalty λ(CVaR_α(m − f)² + CVaR_α(f − m)²), where m is a flexible model's cross-fitted expected cost and CVaR_α the mean over the largest α = 0.10 share. The two terms are the largest expected underpayment and overpayment of any subgroup of a tenth of enrollees that can be assembled from what the flexible model sees, which is what a plan selecting on observables can reach. The penalty is on expected rather than realized residuals: the largest realized residuals belong to people who had a catastrophic year, whom no formula can predict and no plan can pick out. The conditional value at risk is written in the Rockafellar-Uryasev form with a smoothed hinge, so the objective is smooth and the solver does not stall at the kinks of the hard top-share set.

**Boosting.** Tweedie gradient boosting on F3, and a robust version that feeds its cross-fitted score into the capped and penalized linear layer.

**Training against the plan.** Starting from the formula fitted on ungamed data, the plan responds on the training rows, the formula is refitted to the coded records and tilted weights with the true cost, and the loop repeats up to 20 times or until payments move by less than 1% of mean payment. This is repeated risk minimization (Perdomo et al., 2020). The refit uses true cost on manipulated records, which is the regulator's information set: it sees what plans report and what care costs. A code added to many cheap people then predicts lower cost, its increment falls, and the plan moves to other codes; the loop finds where that settles, if it does. The final formula is evaluated on the test fold against a fresh response by the plan.

Table 1 lists the formulas compared.

## 6. Results

### 6.1 What the plan extracts (RQ1)

Against the CMS-form formula the calibrated plan extracts $1,517,780 per 1,000 enrollees, 19.5% of base payment ($1,460,172 to $1,568,914), made up of $748,542 from coding and $769,238 from selection (Table 3). The formula with prior use extracts less, $1,356,983, and unconstrained least squares about the same as the CMS form. The paper 6 fair stacked estimator is the most gameable formula in the benchmark, at $1,687,160 per 1,000, 21.7% of payment, with $1,037,444 from coding alone: the constraints that close its target groups' gaps load payment onto condition flags.

Boosting behaves differently on the two channels. Its selection gain is $120,911 per 1,000, a sixth of the CMS form's, because it prices the prior use that the plan's cost model sees; the plan has little left to select on. Its coding gain, $646,352, is close to the CMS form's.

Caps at incremental cost remove nothing: the cost-capped formula extracts $1,513,844, a difference from the CMS form of −$3,935 (−$31,563 to $30,978). This is not a failure of estimation but of logic. A code added for payment does not change the person's cost, so any positive increment is extractable, whether or not it exceeds what the code is worth among people who genuinely have it. Of the 124 codes, 30 pay more than their incremental cost under the CMS form; the plan does not need those.

### 6.2 The price of robustness (RQ2)

A coding penalty at λ = 1 cuts the coding gain from $748,542 to $92,274, 88%, while R² on ungamed data falls from 0.098 to 0.096 (Table 4, Table 5, Figure 1). Total extraction falls to $776,138, 10.0% of payment. Larger penalties push coding to zero at λ = 20 but cost accuracy steadily, to R² 0.060 at λ = 50, while selection rises: the penalized formula's flatter payments leave more room between payment and cost.

The worst-subgroup penalty behaves differently. It raises accuracy slightly, to R² 0.102, because it pulls payment toward a better predictor of cost, and it cuts coding by 46% to $404,936, but selection is untouched at $669,386. The effect saturates at small λ: every penalty from 1 to 50 gives the same extraction within about $25,000 per 1,000. Combining the cap, the coding penalty and the worst-subgroup penalty at λ = 1 gives R² 0.098 with $174,519 of coding and $843,010 in total.

No linear formula closes the selection channel. Across every penalty, combination and trained version, selection stays between $628,614 and $836,286 per 1,000. The plan selects on its own model's prediction of cost, which uses prior use and nonlinearities that a diagnosis-based payment form cannot represent, and no penalty on the formula's coefficients gives the formula that information.

![](../output/figures/fig1_frontier.png)

**Figure 1.** R² on ungamed data against extractable payment per 1,000 enrollees, along each penalty grid, with the reference formulas and the formulas trained against the plan.

### 6.3 Training against the plan

Trained against the plan with the coding penalty, the CMS form converges in two rounds in every fold and extracts $723,737 per 1,000 (10.3% of payment), $794,043 less than the CMS form ($750,404 to $835,152), at an R² cost of 0.003 (Table 3, Table 7, Figure 4). It is the least gameable formula in the study. The combined robust formula trained the same way extracts $748,183, with $119,568 of coding and $628,614 of selection, converging within 20 rounds in 11 of 15 folds.

Without a penalty, training against the plan does not converge. In no fold does the CMS form settle: on the training rows extraction alternates between about $1,230,000 and $1,320,000 per 1,000 for all 20 rounds, and out of sample the final formula extracts $1,386,344, only 9% less than the formula it started from. The mechanism is visible in the coefficients. After the plan adds a code to many people, the refit pays less for it; the plan then prefers other codes, the first code's payment recovers, and the cycle repeats. The penalty damps the cycle by keeping every codable increment small.

Boosting trained against the plan gets worse. Over ten rounds its extraction on the training rows rises from $484,724 to $1,176,476 per 1,000, and out of sample it extracts $1,314,914, against $767,263 before training, with coding rising to $1,060,061 and R² falling from 0.187 to 0.175. A flexible model refitted to coded records learns the coded pattern as a signal of cost, which rewards the next round of coding. Repeated retraining is stable here only for a formula whose response to a code is constrained.

![](../output/figures/fig4_path.png)

**Figure 4.** Extraction on the training rows by round of training against the plan. Round 1 is the plan's response to the formula fitted on ungamed data.

### 6.4 Who pays for robustness (RQ3)

Every formula underpays people who need help with activities of daily living, and every formula that resists coding underpays them more (Table 6, Figure 2). The shortfall is $9,915 per person-year under the CMS form, $12,462 under the coding penalty and $14,002 under the robust formula trained against the plan, against $5,677 under boosting; only the fair stacked estimator, which targets this group directly, overpays it, by $847. The codes the penalty flattens are the codes that mark expensive people, and functional limitation is expensive whether or not it is coded.

The plan's response moves the gaps further. Although it cannot see functional status, the plan enrolls 5% fewer people needing help under the robust formula trained against it, because its cost model predicts that they cost more than the formula pays; under boosting, which pays them more, it enrolls 8% more. People aged 65 and over go from an overpayment of $22 under the CMS form to an underpayment of $1,761 under the robust formula trained against the plan. Before the plan responds, race and ethnicity gaps stay within about $850 in either direction under every formula. Selection that no one designed lands on the groups the formula underpays.

![](../output/figures/fig2_group_gaps.png)

**Figure 2.** Net compensation (payment minus cost) per person-year by group, before and after the plan responds.

### 6.5 What the robust formula stops paying for (RQ4)

The CMS form pays $12,514 when polyneuropathy is added to a person who could plausibly receive it; the robust formula trained against the plan pays $2,292 (Table 8, Figure 5). The same compression holds across the codes the CMS form pays most for, rheumatoid arthritis, epilepsy, kidney disease and heart failure among them: under the robust formulas every one pays between about $1,800 and $3,000. The average increment across the 124 codes falls from $2,612 to $1,861. Payment moves from individual codes to the count of conditions, whose coefficient rises from $575 per condition under the CMS form to $1,560 under the robust formula trained against the plan, while payment per body system falls from $536 to $194 (Table 8b). A count pays the same for any added code, so it rewards coding less selectively than a schedule of code-specific increments.

![](../output/figures/fig5_codes.png)

**Figure 5.** Payment rise per added code for the 20 codes the CMS form pays most for, under the CMS form and the robust formula trained against the plan, with each code's incremental cost.

### 6.6 Sensitivity and robustness

The ranking of formulas holds across every setting of the plan (Table 9). Cost per code matters little: from $0 to $3,000 a code, extraction from the CMS form moves between $1,575,687 and $1,395,834 per 1,000, and the penalized formula stays at about half. Reach matters most: at 25% of charts reviewed the CMS form loses $4,197,896 per 1,000 and the penalized formula $1,074,023. With no selection at all the CMS form loses $748,204 and the penalized formula $92,275, which isolates the coding defense. The plausibility rule barely matters: with no restriction on which codes can be added the CMS form loses $1,530,294. A less able plan whose cost model is unconstrained least squares takes $1,269,086. The smooth selection rule places no bound on the tilt and is not comparable with the threshold rule, so it is not reported.

If some added codes are real diagnoses, raising the person's cost by the code's incremental cost, extraction falls: with every added code real the CMS form loses $305,254 per 1,000 and the robust formula trained against the plan $245,223. The penalized formula then loses more than the CMS form ($424,523), because it pays less than a real diagnosis costs. How much of coding intensity is discovery rather than gaming decides how hard a formula should resist it, which is the causal question Miller, Milli and Hardt (2020) pose.

The ranking of formulas also survives changes to the benchmark itself (Table 10). Among persons aged 65 and over, the population Medicare Advantage covers, the plan extracts more from every formula because costs and payments are higher: $2,953,407 per 1,000 from the CMS form and $1,251,606 from the penalized formula trained against the plan. Boosting loses most of its advantage in this group, at $1,873,650, because its coding gain there is the largest of any formula. With prior use in the payment form the penalized formula extracts $662,324, and squared-error boosting extracts $1,145,443. The worst-subgroup share moves the combined formula between $736,399 at α = 0.20 and $937,216 at α = 0.05. Holding out each MEPS panel in turn gives $1,504,641 from the CMS form and $749,741 from the penalized formula, close to the cross-validated figures.

## 7. Discussion

**For CMS.** The 2024 model constrained some condition categories and removed discretionary codes. The results here say which of those levers work. Capping a code at what it is worth does not deter coding, because a code added for payment changes nothing about cost; any positive payment is extractable. Shrinking the payment for codes in proportion to how easily they can be added does deter it, at almost no loss of accuracy, provided the condition counts are shrunk with them. Refitting against the coding the formula induces helps when the formula is constrained, and fails, by cycling or by amplifying the gaming, when it is not. None of this touches favorable selection, which is half the problem. Selection is driven by what the plan knows and the formula does not, and the remedies lie outside the formula: risk corridors, reinsurance, or payment on a broader information set, with the incentive problems prior use brings (Babalola et al., 2026).

**For machine learning in payment.** Flexible models are not a defense. Boosting closes the selection channel by pricing prior use, keeps the coding channel open, and, retrained on gamed data, becomes more gameable. The machinery of performative prediction is useful here as a diagnostic: whether repeated retraining converges tells a regulator whether its formula has a stable point under the incentives it creates.

**For fairness.** Robustness to gaming and fair payment for the frail pull in opposite directions. The formulas that resist coding pay less for people needing help with daily activities, and the plan's selection, which never sees functional status, then enrolls fewer of them. A regulator choosing a robust formula is also choosing whom it underpays, and the two decisions have to be made together.

**Questions a reader will ask.** *Is the plan realistic?* It is a model, calibrated to MedPAC's two aggregate magnitudes and swept widely; the claims are about the ranking of formulas, which holds at every setting, rather than about the level of extraction. *Why not HCCs and claims?* The benchmark is public and reproducible, which claims data are not; MEPS conditions are self-reported CCSR categories, not diagnosis codes in claims, and the population is all ages. *Why true cost in the refit?* The regulator observes what care costs; refitting on ungamed fee-for-service data only, as CMS does, is the formula at round 1. *Why does the worst-subgroup penalty improve accuracy?* Because it pulls payment toward a better predictor of cost, which is what the reference model is.

**Limitations.** The plan is a stylized model with one coding cost, one reach and a threshold tilt; real plans differ in all three, and their coding and selection interact in ways the model treats as separable. The added codes are CCSR categories in a survey, not diagnosis codes in claims, and the population is not the Medicare Advantage population. Spending is total spending from all payers. The selection model tilts weights on an existing population rather than modeling enrollment decisions, networks or benefit design. Boosting trained against the plan was run on one repeat of five folds because of its cost. The analysis is a simulation of incentives on survey data and is not evidence about the behavior of any plan.

## 8. Conclusion

A strategic plan calibrated to MedPAC's magnitudes takes about a fifth of payment from a formula in the form CMS uses, half by coding and half by selection. A penalty on the codes a plan can add removes nearly all of the coding at almost no cost in accuracy, and training that penalized formula against the plan settles quickly at half the CMS form's extraction. Caps at incremental cost do not help, flexible models do not help, and retraining without a constraint cycles or diverges. Selection survives every formula built on diagnoses. The formulas that resist gaming underpay the frail, and the plan's selection compounds it, so robustness and fairness have to be chosen together.

---

## Declarations

**Data availability.** The Medical Expenditure Panel Survey public-use files are available from the Agency for Healthcare Research and Quality. No microdata are redistributed. The derived analysis file is built by the code of Babalola et al. (2026).

**Code availability.** Complete, seeded code at https://github.com/tosin-babs/adversarial-ml-risk-adjustment. The strategic plan and the robust formulas are released in version 0.3.0 of the `riskfair` Python package, at https://github.com/tosin-babs/fair-ml-health-risk-adjustment.

**Competing interests.** None declared.

**Ethics.** The analysis uses de-identified public survey data and did not require ethical approval.

---

## References

1. Babalola, O. D., Adiegwu, C., & Olamilekan, E. Z. (2026). Interpretable and fair machine learning for health-cost prediction and risk adjustment: a reproducible benchmark on public data and an open-source toolkit. Working paper, Georgia State University.
2. Bergquist, S. L., Layton, T. J., McGuire, T. G., & Rose, S. (2019). Data transformations to improve the performance of health plan payment methods. *Journal of Health Economics*, 66, 195–207. doi:10.1016/j.jhealeco.2019.05.005
3. Brown, J., Duggan, M., Kuziemko, I., & Woolston, W. (2014). How does risk selection respond to risk adjustment? New evidence from the Medicare Advantage program. *American Economic Review*, 104(10), 3335–3364. doi:10.1257/aer.104.10.3335
4. Centers for Medicare & Medicaid Services (2023). *Announcement of Calendar Year (CY) 2024 Medicare Advantage (MA) Capitation Rates and Part C and Part D Payment Policies*. 31 March 2023. Baltimore, MD: CMS.
5. Duchi, J. C., & Namkoong, H. (2021). Learning models with uniform performance via distributionally robust optimization. *Annals of Statistics*, 49(3), 1378–1406. doi:10.1214/20-AOS2004
6. Geruso, M., & Layton, T. (2020). Upcoding: evidence from Medicare on squishy risk adjustment. *Journal of Political Economy*, 128(3), 984–1026. doi:10.1086/704756
7. Glazer, J., & McGuire, T. G. (2000). Optimal risk adjustment in markets with adverse selection: an application to managed care. *American Economic Review*, 90(4), 1055–1071. doi:10.1257/aer.90.4.1055
8. Hardt, M., Megiddo, N., Papadimitriou, C., & Wootters, M. (2016). Strategic classification. In *Proceedings of the 2016 ACM Conference on Innovations in Theoretical Computer Science*, 111–122. doi:10.1145/2840728.2840730
9. Hashimoto, T., Srivastava, M., Namkoong, H., & Liang, P. (2018). Fairness without demographics in repeated loss minimization. In *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 1929–1938. https://proceedings.mlr.press/v80/hashimoto18a.html
10. Hu, L., Immorlica, N., & Vaughan, J. W. (2019). The disparate effects of strategic manipulation. In *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 259–268. doi:10.1145/3287560.3287597
11. Kronick, R., & Welch, W. P. (2014). Measuring coding intensity in the Medicare Advantage program. *Medicare & Medicaid Research Review*, 4(2), E1–E19. doi:10.5600/mmrr.004.02.sa06
12. Layton, T. J., Ellis, R. P., McGuire, T. G., & van Kleef, R. (2017). Measuring efficiency of health plan payment systems in managed competition health insurance markets. *Journal of Health Economics*, 56, 237–255. doi:10.1016/j.jhealeco.2017.05.004
13. McGuire, T. G., Zink, A. L., & Rose, S. (2021). Improving the performance of risk adjustment systems: constrained regressions, reinsurance, and variable selection. *American Journal of Health Economics*, 7(4), 497–521. doi:10.1086/716199
14. Medicare Payment Advisory Commission (2025). The Medicare Advantage program: status report. In *Report to the Congress: Medicare Payment Policy*, March 2025, chapter 11, pp. 317–406. Washington, DC: MedPAC.
15. Miller, J., Milli, S., & Hardt, M. (2020). Strategic classification is causal modeling in disguise. In *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 6917–6926. https://proceedings.mlr.press/v119/miller20b.html
16. Perdomo, J. C., Zrnic, T., Mendler-Dünner, C., & Hardt, M. (2020). Performative prediction. In *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 7599–7609. https://proceedings.mlr.press/v119/perdomo20a.html
17. Pope, G. C., Kautter, J., Ellis, R. P., et al. (2004). Risk adjustment of Medicare capitation payments using the CMS-HCC model. *Health Care Financing Review*, 25(4), 119–141.
18. Rao, J. N. K., & Wu, C. F. J. (1988). Resampling inference with complex survey data. *Journal of the American Statistical Association*, 83(401), 231–241. doi:10.1080/01621459.1988.10478591
19. Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–41. doi:10.21314/JOR.2000.038
20. Shepard, M. (2022). Hospital network competition and adverse selection: evidence from the Massachusetts health insurance exchange. *American Economic Review*, 112(2), 578–615. doi:10.1257/aer.20201453
21. Zink, A., & Rose, S. (2020). Fair regression for health care spending. *Biometrics*, 76(3), 973–982. doi:10.1111/biom.13206
22. Agency for Healthcare Research and Quality. *Medical Expenditure Panel Survey, Panel 23–27 Longitudinal Data Files and Medical Conditions Files*.

*DOIs were verified against the Crossref REST API. PMLR proceedings have no DOI and are cited by their proceedings pages. MedPAC figures are those checked against the chapter text in Babalola et al. (2026).*
