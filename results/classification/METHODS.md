# Binary nested-CV methods and reporting scope

## Current two-candidate evaluation

[`evaluate_binary.py`](../../code/classification/evaluate_binary.py) runs the
three within-participant binary comparisons. Its default pool contains
`fb_ts_lr` (filter-bank tangent-space features and logistic regression)
and `fbcsp_mibif_lda` (filter-bank CSP, mutual-information feature selection,
and LDA). This was the two-candidate shortlist used in the preceding comparison
on these recordings. The reduced pool and the previous six-candidate pool
were compared on the same outer folds.

The input profile and split rules below also apply to the current evaluation.
For each outer fold, mean accuracy over the four inner validation folds ranks
the candidates. Fixed pool order breaks exact ties. All learned transforms and
the classifier are fitted on the outer training partition before prediction.
The feature selector's random state is controlled separately for inner CV and
the outer fit. Test labels are used for stratification and metric calculation.
Within-recording, label-free epoch curation still precedes the split.

With `--full-comparison`, every candidate is fitted once on each outer training
partition and scored on the same outer test trials. For each candidate removed
from the six-candidate pool, the highest-ranked remaining candidate by inner
accuracy supplies the held-out results. The same test predictions support the
two-candidate, six-candidate, fixed-model, and leave-one-candidate-out comparisons.
These are exploratory comparisons on the ten original recordings.

### Metrics and tables

- **Accuracy, balanced accuracy, per-class recall and precision, macro-F1,
  and MCC** use the model's default class decisions. Balanced accuracy equally
  weights the two recalls. A class precision is undefined when that class
  receives no predictions. An empty cell represents undefined precision,
  and precision averages use folds with defined values.
- **ROC-AUC, PR-AUC, and average precision (AP)** use continuous decision
  scores within each fold. For FgMDM the score is the second-class value from
  the model's distance-based probability output. The sign is reversed to
  compute first-class PR-AUC and AP. PR-AUC uses trapezoidal integration,
  while AP weights precision by each increase in recall. The macro versions
  give the two classes equal weight.
- Fold metrics are averaged first within each participant and then across
  the ten participants. Confusion counts (`tn`, `fp`, `fn`, `tp`) are summed.
  The second condition in each contrast is the positive class for these
  counts. Continuous scores are ranked within each fold before averaging
  the resulting metrics.

| File | Rows |
|---|---|
| [Fold metrics](binary_fold_metrics.csv) | 900 rows for 10 participants × 3 contrasts × 5 folds × 6 candidates. Includes inner accuracy, compact selection, test confusion counts, and test metrics. |
| [Participant metrics](binary_participant_metrics.csv) | Averages for each of 14 strategies across five outer folds, by participant and contrast. |
| [Group metrics](binary_group_metrics.csv) | Equal-participant averages for the same 14 strategies and three contrasts. |

`compact` and `full` use the inner-CV winner from two or six candidates.
`fixed:<name>` uses the named pipeline on every fold. `drop:<name>` removes
that candidate and chooses among the remaining five by inner accuracy.

### Matched candidate removal

Each cell shows the change from the full six-candidate pool in **balanced
accuracy (percentage points) / macro AP (0-1 scale)**. Models for all of
these strategies are chosen by inner validation accuracy.

| Candidate pool | Rest-Stand | Rest-Walk | Stand-Walk |
|---|---:|---:|---:|
| Two-candidate default | +1.41 / +0.007 | +0.82 / +0.004 | +1.72 / +0.017 |
| Drop `csp_lda` | +0.40 / +0.002 | −0.40 / +0.003 | +0.69 / +0.004 |
| Drop `ts_lr` | +1.00 / +0.006 | +0.30 / +0.003 | +0.60 / +0.006 |
| Drop `acm_d5_10_ts_lr` | +0.39 / +0.000 | −0.10 / −0.005 | −0.09 / +0.000 |
| Drop `fgmdm` | −1.49 / −0.003 | −0.11 / −0.007 | +0.23 / +0.002 |
| Drop `fb_ts_lr` | −0.01 / −0.002 | −0.42 / −0.001 | −0.57 / +0.003 |
| Drop `fbcsp_mibif_lda` | −0.12 / +0.001 | −0.32 / +0.000 | −1.69 / −0.009 |

Removing `ts_lr` improved both metrics across the three contrasts. Removing
`fbcsp_mibif_lda` reduced Stand-Walk performance on both metrics. Other
changes varied with the contrast. The fixed time-delay model performed
well on Rest-Walk (ROC-AUC 0.725, macro AP 0.753), and removing FgMDM
lowered Rest-Stand balanced accuracy in the larger pool. For Stand-Walk, the
two-candidate selector exceeded the fixed `fb_ts_lr` on balanced accuracy
(54.8% versus 53.0%) and ROC-AUC (0.563 versus 0.551). These results
support using a smaller default while retaining the six-pipeline benchmark.
They offer no independent estimate of the effect of pruning candidates.

Use authorized recordings and a compatible Python environment. This run
used NumPy 2.1.3, SciPy 1.15.1, scikit-learn 1.8.0, pyriemann 0.9, and MNE 1.10.2.

```bash
python code/classification/evaluate_binary.py \
  --data /path/to/authorized-recordings --out cache/binary_evaluation
# Add --full-comparison to generate the published six-candidate diagnostics.
```

## Earlier six-candidate results and source

The earlier results are the saved `kind=nested` records for RvS, RvW,
and SvW in `results/final_scores.csv`. They are the same thirty participant
estimates summarized in the September 29 source snapshot. The tables below
document that saved six-candidate evaluation.

| Participant | Rest-Stand | Rest-Walk | Stand-Walk |
|---|---:|---:|---:|
| S01 | 54.7 | 41.9 | 44.4 |
| S02 | 48.0 | 62.0 | 49.0 |
| S03 | 72.7 | 76.4 | 52.6 |
| S04 | 64.7 | 56.4 | 47.0 |
| S05 | 63.6 | 65.8 | 56.0 |
| S06 | 88.0 | 76.0 | 70.0 |
| S07 | 58.0 | 74.0 | 43.0 |
| S08 | 82.0 | 62.0 | 55.0 |
| S09 | 64.0 | 63.0 | 57.0 |
| S10 | 62.0 | 67.0 | 57.0 |

| Contrast | Mean accuracy | Participant-level 95% t interval |
|---|---:|---:|
| Rest-Stand | 65.8% | 57.1-74.5% |
| Rest-Walk | 64.5% | 57.1-71.9% |
| Stand-Walk | 53.1% | 47.5-58.7% |

Each participant score is the arithmetic mean of five outer-fold accuracies.
The group mean and 95% t interval use the ten participant scores, not pooled
trial counts. Unequal outer-fold sizes make this score potentially different
from one accuracy computed over all out-of-fold predictions. No binomial
correct-count inference is made from these averaged scores.

## Recorded input profile

The saved records specify eight channels, plain-band 8-30 Hz input,
task 0.5-4.5 s, and `ref=none` (the recorded reference, with no CAR).
The alternative filterbank view uses 4-8, 8-12, 12-16, 16-20, 20-24,
24-28, 28-32, and 32-40 Hz subbands. `steam_io.get_task_arrays` aligns
the plain and bank arrays to the same retained trial mask.

After recording-level keep rules, the original nested preparation also
rejects clipped epochs and participant-level log-variance outliers.
The saved binary sample sizes are 94-100, not a universal 100. The dB and
linear-probe analyses elsewhere use their separately documented retention
rules, windows, and features, and do not replay these nested decisions.

The artifact median/MAD statistics are estimated across the participant's
epoch set before CV, without using labels. This is label-free dataset
curation, but uses held-out signal distribution and is not a prospective
train-only rejection procedure. Nested classifier selection does not remove
this limitation or validate the broader choices of preprocessing and model
pool made while inspecting the same records.

## Earlier six-candidate pool

`code/02_final_eval.py:nested_score` defines five stratified outer folds
with shuffle=True and seed=0. Within each outer training set, four stratified
inner folds with seed=1 compare six candidates:

- `csp_lda`: shrinkage covariance, CSP, shrinkage LDA.
- `ts_lr`: covariance, Riemannian tangent-space features, scaler, logistic regression.
- `acm_d5_10_ts_lr`: time-delay covariance, tangent features, scaler, logistic regression.
- `fgmdm`: covariance and FgMDM.
- `fb_ts_lr`: filterbank covariance/tangent features, scaler, logistic regression.
- `fbcsp_mibif_lda`: per-band CSP, mutual-information feature selection, LDA.

Inner mean accuracy selects the pipeline for that outer fold. The selected
pipeline is freshly fitted on the outer training set and predicts the outer
test set. The candidate definitions are in `code/decoders.py`; learned
transforms are inside scikit-learn pipelines.

Test labels appear in stratified assignment and final scoring, not in
inner candidate selection or fitting on that held-out fold. This checks the
public code's structure; saved aggregate files do not independently prove
the complete past execution history or reproduce package internals.

## Reporting limits

This report's evaluation profile was chosen after comparing existing results.
The estimates remain exploratory and conditional on the same data and
analysis choices. Participant t intervals do not account for that selection.
The reported means are not a matched benchmark against a fixed CNN split:
models, folds, epoch rejection, windows, and retained sample sizes differ.

Nested pipeline selection is not new-person, new-day, or online BCI validation.
It does not identify a motor-imagery-specific cortical source. The separate
power/probe results characterize the recordings, not the causal features
used by each selected primary pipeline.

## Inspect the stored calculation

Use an isolated Python environment with NumPy and SciPy:

```bash
python -m unittest discover -s code/classification -p 'test_nested_summary.py'
python code/classification/summarize_nested.py \
  --scores results/final_scores.csv --out cache/nested_score_audit
```

The summary validates thirty binary participant/task keys, stored profile,
sample sizes, and the five saved candidate choices per participant. It exports
only the three binary comparisons, without fitting any model or accessing
EEG. Tests compare the published tables to the saved source and inspect the
nested selection code with AST parsing rather than execution.

Original raw files, participant mappings, cached EEG, and trial-level
predictions are not distributed. Full end-to-end reproduction requires
separately authorized data and compatible software; it is not established
by this saved-score check.
