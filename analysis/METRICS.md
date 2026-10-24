# Metric definitions

Every metric reported in the thesis is computed with `analysis/metrics.py`; the tests are in `analysis/tests/`
(`python -m unittest discover -s analysis/tests -t .`). This document defines, for every evaluated task, the unit of
decision, the positive and negative classes, and how decisions are aggregated.

## 1. Confusion-matrix metrics

For a set of binary decisions with TP true positives, FP false positives, FN false negatives and TN true negatives:

| Metric | Definition |
|---|---|
| Precision | TP / (TP + FP) |
| Recall (sensitivity) | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| F1 | 2 TP / (2 TP + FP + FN) |
| MCC | (TP·TN − FP·FN) / √((TP + FP)(TP + FN)(TN + FP)(TN + FN)) |

MCC (Matthews, 1975) is the correlation between the true and predicted labels. It ranges from −1 to 1, is 0 for a
classifier with no predictive value and uses all four cells of the confusion matrix (Chicco and Jurman, 2020). F1 does
not use TN, so it cannot register errors on the negative class. With 27 positives and 3 negatives, a classifier that
always predicts positive has precision 0.9, recall 1, F1 0.947 and MCC 0 (`AlwaysPositiveExample` in the tests).

For more than two classes, MCC is the R_K statistic of Gorodkin (2004), computed from the K × K confusion matrix. With two
classes it equals the binary MCC.

**Undefined values.** Precision is undefined when nothing is predicted positive, recall when there are no positives,
and specificity when there are no negatives. MCC is undefined when the true labels contain only one class, because
there is then nothing to discriminate; such a set is described by its recall or its specificity alone. Undefined values
are reported as undefined, never as 0 or 1. When the true labels contain both classes and the prediction is the same for
every decision, MCC is 0.

**Two properties used in interpreting results.**

1. For fixed TP, FP and FN, MCC increases strictly with TN (`test_mcc_increases_with_true_negatives`).
2. When TN is much larger than TP, FP and FN, MCC approaches √(precision × recall)
   (`test_large_negative_universe_approaches_fowlkes_mallows`). For extraction tasks with a large vocabulary of possible
   answers, MCC therefore adds little to precision and recall. MCC is most informative for decisions whose negatives
   are not overwhelmingly numerous, in particular the scenario-level detection decision (Section 2.1).

## 2. Units of decision: Chapters 4 and 5

A scenario is one news article evaluated for one monitored firm. The same article can be positive for one firm and
negative for another. All string comparisons use `normalize` (lower case, `&` and `and` removed, `labour` read as
`labor`, white space collapsed).

A path runs from the monitored firm through its suppliers to a supplier at Tier 1 to Tier 4 and contains no company
twice. A path is disrupted when its last company is a named disrupted company, or is located in a disrupted country
and, when the disruption is confined to named industries, operates in one of them (`offline.kg.is_disrupted`).

### 2.1 Scenario-level detection

| | |
|---|---|
| Unit | Scenario |
| Positive | The reference contains at least one disrupted path from the monitored firm, Tier 1 to Tier 4 |
| Predicted positive | The knowledge-graph result of the pipeline contains at least one disrupted path (`alert`) |
| Metrics | 2 × 2 confusion matrix: precision, recall, specificity, F1, MCC |
| Baseline | Always alert |

The 30 scenarios of Chapter 5 comprise 21 positive and 9 negative scenarios: the SC004, SC008 and SC010 templates for
each firm (`test_ground_truth_v1_2_has_21_alerts_and_9_non_alerts`).

### 2.2 Disruption Monitoring Agent

| Output | Unit | Classes or universe |
|---|---|---|
| Ongoing disruption | Article | Binary: the article reports a disruption under way (`ongoing_disruption`) |
| Disruption type | Article with an ongoing disruption | The disruption types of the ground truth; multiclass MCC |
| Scope | Article with an ongoing disruption | Binary: confined to the named industries, or the countries generally |
| Countries | (scenario, country) | All countries in the knowledge graph |
| Industries | (scenario, industry) | All industries in the knowledge graph |
| Companies | (scenario, company) | All companies in the knowledge graph |

For the entity fields each element of the universe is one decision: the element is predicted present or absent and is
present or absent in the ground truth. Predicted and ground-truth names are compared after the resolution the pipeline
applies (`entity_keys`, `offline/names.py`): 'United States' and 'USA' are one country, 'Telstra' and 'Telstra Group Ltd'
one company. A name that does not resolve is compared as its normalised string and added to the universe, so a
predicted entity that does not exist in the graph is a false positive (`Confusion.from_sets`). Baseline: the majority
class. In the 30 scenarios every article reports an ongoing disruption, so the ongoing decision has one class
there and is reported as accuracy only.

### 2.3 Knowledge Graph Query Agent

| | |
|---|---|
| Unit | (scenario, path), over every path of the monitored firm's network from Tier 1 to Tier 4 |
| Positive | The path is disrupted in the ground truth |
| Matching | A path is the order-insensitive set of its (company, country, industry) nodes. A predicted path matches a reference path when the two node sets are equal (`match_sets`) |
| TN | Number of paths in the firm's network minus TP, FP and FN (`Confusion.from_matches`); the network path counts come from `offline/` |

### 2.4 Risk Manager Agent

| | |
|---|---|
| Unit | (scenario, Tier-1 supplier of the monitored firm), over all Tier-1 suppliers |
| Classes | NONE (no score: not exposed), LOW (< 0.45), MEDIUM (0.45 to < 0.60), HIGH (≥ 0.60), from the raw score (`risk_level`) |
| Metrics | Multiclass MCC and accuracy over the four levels |
| Baseline | Always NONE |

### 2.5 CSCO Agent

| | |
|---|---|
| Unit | (scenario, Tier-1 supplier of the monitored firm), over all Tier-1 suppliers |
| Classes | MAINTAIN_STANDARD, INCREASE_MONITORING, REPLACE_SUPPLIER |
| Ground truth | `expected_action` of the ground-truth risk score; a supplier without a score is MAINTAIN_STANDARD |
| Prediction | The action the agent gives; a supplier the agent does not address is MAINTAIN_STANDARD |
| Metrics | Multiclass MCC and accuracy |
| Baseline | Always MAINTAIN_STANDARD (accuracy equal to the MAINTAIN_STANDARD share, MCC 0) |

## 3. Units of decision: Chapter 3

| Task | Unit | Positive |
|---|---|---|
| Entity recognition | (document, candidate entity, entity type) | The candidate has that type in the ground truth |
| Relation extraction | (document, ordered pair of candidate nodes, relation type) | The ground truth contains that relation between the pair |

Candidates are the distinct normalised entity strings in the ground truth or in the output for the document. Strings
named by neither are not counted, so TN is a lower bound and, by property 1 of Section 1, the reported MCC is a lower
bound on the MCC over any larger universe. A chance baseline for each entity type assigns that type at random to
candidates at the rate the system assigns it (expected MCC 0).

## 4. Aggregation

**Pooling.** Precision, recall, F1 and MCC are computed on the confusion matrix pooled over all decisions of the
evaluation set, because a per-scenario MCC is undefined for every scenario whose reference contains no positive decision
(Section 1).

**Repeated runs.** For evaluations with several runs per scenario, each metric is computed per run over all scenarios
and reported as the mean and standard deviation over runs.

## 5. Intervals

- **Proportions** (sensitivity, specificity of each negative type, audit error rates): exact two-sided 95% interval
  (Clopper and Pearson, 1934; `clopper_pearson`). With no errors in 150 items the upper limit is 2.4%.
- **MCC and F1:** 95% percentile bootstrap, 10,000 resamples (`bootstrap`). The resampled unit is the document, with
  all its decisions and runs: a Wikipedia document in Chapter 3, a benchmark item in Chapter 4, and in Chapter 5 an
  article with its three firm scenarios. On the
  benchmark the resampling is stratified by item type (P, N1 to N4), so that every resample contains both classes.
  Resamples on which a metric is undefined are counted and reported.

## 6. Performance at lower prevalence

Real monitoring streams contain far fewer relevant articles than any benchmark. Two analyses describe scenario-level
detection at lower prevalence.

1. **Observed** (`subsample_at_prevalence`): all negatives are kept and positives are drawn at random without
   replacement to reach the target prevalence (down to 1% with 997 negatives, i.e. 10 positives), repeated 1,000
   times. The reported range reflects which positives are drawn. The benchmark scorer also reports a stratified
   bootstrap interval at the target prevalence (positives drawn with replacement, negatives resampled within type),
   which adds sampling uncertainty.
2. **Projected** (`project`): below 1%, precision, F1 and MCC follow from sensitivity s, specificity c and prevalence π:
   with q = πs + (1 − π)(1 − c), MCC = (s + c − 1)·√(π(1 − π)) / √(q(1 − q)). Sensitivity and the specificity of each
   negative type are measured on the benchmark and assumed to hold at the lower prevalence; the negative types are
   mixed with stated weights (`mixed_specificity`). Uncertainty is propagated by drawing each proportion from its
   Jeffreys posterior, Beta(k + ½, n − k + ½) (Brown, Cai and DasGupta, 2001). The result is a projection under these
   assumptions, not a measurement.

For fixed sensitivity and specificity, MCC and precision fall as prevalence falls
(`test_mcc_falls_with_prevalence_for_fixed_sensitivity_and_specificity`).

## References

- Brown, L. D., Cai, T. T. and DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101–133.
- Chicco, D. and Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21, 6.
- Clopper, C. J. and Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika*, 26(4), 404–413.
- Gorodkin, J. (2004). Comparing two K-category assignments by a K-category correlation coefficient. *Computational Biology and Chemistry*, 28(5–6), 367–374.
- Matthews, B. W. (1975). Comparison of the predicted and observed secondary structure of T4 phage lysozyme. *Biochimica et Biophysica Acta (BBA) – Protein Structure*, 405(2), 442–451.
