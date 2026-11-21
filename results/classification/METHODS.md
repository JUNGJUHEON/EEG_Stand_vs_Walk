# Binary nested-CV methods

## Inputs

The three comparisons are Rest-Stand, Rest-Walk, and Stand-Walk. Each trial
contributes eight channels from 0.5 to 4.5 s after task onset, sampled at
256 Hz. The analysis uses the recorded reference (`ref=none`).

Both models use 4–8, 8–12, 12–16, 16–20, 20–24, 24–28, 28–32, and
32–40 Hz subbands. [The loader](../../code/steam_io.py) applies the same
trial mask to all bands. After the recording keep rules, it excludes clipped
and extreme-variance epochs, leaving 94–100 trials per binary comparison.
The median and median absolute deviation for this label-free rejection are
estimated across each participant's epochs before cross-validation.

## Models

Both models estimate an 8 × 8 shrinkage covariance matrix in each band.

- **Filter-bank tangent space with logistic regression (`fb_ts_lr`).**
  The diagonal records channel variances and the off-diagonal entries record
  how channels vary together. The model expresses each trial's covariance
  relative to the corresponding band's Riemannian mean from the training
  trials. The resulting 36 tangent-space features per band form a
  288-feature vector. The pipeline standardizes the features using the
  training set and fits a logistic regression with `C=0.1`.
- **Filter-bank CSP with LDA (`fbcsp_mibif_lda`).** This model looks for
  weighted channel combinations whose variance differs between conditions.
  It learns four CSP combinations per band from the training labels and
  computes the log variance of each combination. A filter ranks the
  resulting 32 features by mutual information with the labels. The top
  eight feed a shrinkage linear discriminant analysis classifier.

These features describe patterns in the recorded scalp EEG. The power analyses
examine condition differences in the recordings and discuss possible sources
of those differences.

## Evaluation

Each participant and contrast uses five shuffled stratified outer folds
(seed 0). Four shuffled stratified inner folds (seed 1) compare the two
models on mean validation accuracy within each outer training set. The
order `fb_ts_lr`, then `fbcsp_mibif_lda`, breaks exact ties. The selected
model is refit on the outer training set and predicts the held-out trials.
All learned transforms and the classifier are fitted on the training side
of each split. The test labels are used for stratification and scoring.

Preprocessing and evaluation choices were informed by the same recordings.
These scores describe exploratory, within-recording prediction. New-person,
new-day, and online performance require separate evaluation.

## Reading the results

- **Accuracy, balanced accuracy, per-class recall and precision, Macro-F1,
  and MCC** use the model's default class decisions. Balanced accuracy gives
  both recalls equal weight. An empty precision cell indicates that the model
  made zero predictions for that class. Precision averages use defined folds.
- **ROC-AUC, PR-AUC, and average precision (AP)** use continuous decision
  scores within each fold. A higher score favors the second condition.
  Reversing the score allows first-condition PR-AUC and AP. PR-AUC uses
  trapezoidal integration. AP weights precision by each increase in recall.
  The macro versions give both classes equal weight.
- Metrics are averaged across outer folds within each participant. Group
  results give each of the ten participants equal weight. Confusion counts
  are summed, with the second condition treated as positive.

| File | Unit |
|---|---|
| [Fold metrics](two_model/binary_fold_metrics.csv) | One row per participant, contrast, and outer fold, with the selected model and test metrics (150 rows). |
| [Participant metrics](two_model/binary_participant_metrics.csv) | One row per participant and contrast (30 rows). |
| [Group metrics](two_model/binary_group_metrics.csv) | One row per contrast (3 rows). |

## Reproduction

Run [the evaluation script](../../code/classification/evaluate_binary.py)
with authorized recordings. This run used NumPy 2.1.3, SciPy 1.15.1,
scikit-learn 1.8.0, pyriemann 0.9, and MNE 1.10.2.

```bash
python code/classification/evaluate_binary.py \
  --data /path/to/authorized-recordings --out cache/binary_evaluation
```
