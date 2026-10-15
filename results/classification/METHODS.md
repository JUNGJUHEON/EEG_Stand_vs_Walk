# Binary nested-CV methods and reporting scope

## Source and unit

The three primary results are the saved `kind=nested` records for RvS, RvW,
and SvW in `results/final_scores.csv`. They are the same thirty participant
estimates summarized in the September 29 source snapshot. No full raw-data
extraction, nested model fitting, or prediction replay was performed to
publish this report. The new export removes unrelated tasks and profile rows.

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

## Nested pipeline selection

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
transforms are inside scikit-learn pipelines. The published candidate pool
does not include the separately rejected external FBCSP evaluation path.
No test-answer filter reconstruction is executed or used in this report.

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
python -m unittest discover -s code/classification -p 'test_*.py'
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
