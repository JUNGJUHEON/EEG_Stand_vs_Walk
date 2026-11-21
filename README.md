# 8-Channel EEG: Binary Classification and Power Differences

This project examines two questions: **can EEG distinguish the condition
labels, and what power differences are visible in the recorded signals?**
Ten participants contributed recordings of Rest, standing-up imagery (Stand),
and walking imagery (Walk). Recording-level selection retained fifty trials
per condition. The classification analysis additionally rejected epochs,
leaving 94-100 trials per binary comparison; the power analyses below use
fifty trials per condition. Stand and Walk refer to requested mental tasks,
not actual movements.

Eight channels (FC3, FC4, C3, C4, Cz, CP3, CP4, CPz) were recorded at 256 Hz.
Each trial included 3 s fixation, a 2 s condition-specific visual cue, and
a 5 s task. Raw EEG and participant identifiers are not distributed.

## 1. Three Binary Classifiers

We evaluate **Rest-Stand, Rest-Walk, and Stand-Walk** within each participant
using two models that represent the same EEG trial in different ways.
Both use the eight recorded channels from **0.5–4.5 s after task onset** and
estimate an 8 × 8 channel covariance matrix in each of eight frequency bands.
Each prediction assigns one of the two condition labels to a single trial.

### What each model captures

- **`fb_ts_lr` uses how individual channels vary and how pairs of channels
  vary together within each band.** For each band, the model expresses the
  trial's covariance relative to a reference estimated from the training
  trials. That tangent-space transformation gives 36 features per band.
  Logistic regression combines the 288 standardized features across bands.
- **`fbcsp_mibif_lda` uses the variance of channel combinations that help
  distinguish the conditions.** Common spatial patterns (CSP) learns four
  weighted combinations of channels in each band from the training labels.
  The model computes the log variance for each combination, ranks the 32
  features by mutual information with the training labels, and uses the
  top eight for linear discriminant analysis.

### How the models are evaluated

For each of five outer folds, we compare the two models on four inner
validation folds and choose the one with higher mean validation accuracy.
We then refit it on the outer training set and score its predictions on
the held-out trials.

### Held-out performance

Each metric is calculated per outer fold, averaged within each participant,
then averaged across ten participants. PR-AUC and AP give equal weight to the
two classes. Recall reports the percentage of each true condition recovered.

| Contrast | Accuracy | Balanced accuracy | ROC-AUC | Macro PR-AUC | Macro AP | Recall by class |
|---|---:|---:|---:|---:|---:|---|
| Rest-Stand | 67.2% | 67.3% | 0.727 | 0.731 | 0.752 | Rest 67.6%, Stand 66.9% |
| Rest-Walk | 65.3% | 65.3% | 0.704 | 0.715 | 0.736 | Rest 64.5%, Walk 66.1% |
| Stand-Walk | 54.8% | 54.8% | 0.563 | 0.591 | 0.622 | Stand 55.9%, Walk 53.7% |

Rest-versus-imagery comparisons had higher mean accuracy and ranking scores
than Stand-Walk. The Stand-Walk result shows limited single-trial separation
in these recordings.

[Per-fold metrics and confusion counts](results/classification/two_model/binary_fold_metrics.csv)
| [Participant metrics](results/classification/two_model/binary_participant_metrics.csv)
| [Group metrics](results/classification/two_model/binary_group_metrics.csv)
| [Detailed methods](results/classification/METHODS.md)

### Evaluation scope

Input signals use the recorded reference. All learned transforms and model
selection use the training side of each split. Label-free epoch rejection
uses participant-wide signal statistics before cross-validation. The same
recordings informed the analysis choices, so the scores describe exploratory,
within-recording prediction.
New-person, new-day, and online performance require separate evaluation.

[Binary evaluation code](code/classification/evaluate_binary.py)
| [Pipeline definitions](code/decoders.py)

Run the two-model evaluation with authorized recordings.

```bash
python code/classification/evaluate_binary.py \
  --data /path/to/authorized-recordings --out cache/binary_evaluation
```

## 2. EEG Power Differences and Feature Diagnostics

The primary nested evaluation asks whether labels can be predicted. This second topic
examines **how recorded power changes within and between conditions**, then
uses separate fixed linear classifiers to characterize condition information
across **time, frequency bands, and channel patterns**. The dB measurements
themselves do not use a classifier.

### Measurement

Band-limited power is estimated from equal-length **1 s windows**. Each
window is expressed relative to the same trial's pre-cue fixation baseline:

```text
change (dB) = 10 * log10(window power / fixation reference power)
```

The fixation reference is the geometric mean of two 1 s power estimates
from -4.5 to -2.5 s relative to task onset. The summaries below average the
four task windows from **1-5 s**, the two central channels **Cz and CPz**,
and fifty trials per condition, before averaging the ten participant means.
Positive dB means an increase from
fixation; negative dB means a decrease. These are averages of log power ratios,
not a ratio of power pooled across trials or participants.

Two references are compared: the recorded **earlobe reference** and
**8-channel common average reference (CAR)**, which subtracts the average
across the eight channels at each time point. This reference comparison is
not the T/CT/C input normalization used in other model profiles.

### Changes Within Each Condition

Values are mean changes from each condition's own fixation, in dB.

| Condition | 8-13 Hz, earlobe | 8-13 Hz, CAR | 13-30 Hz, earlobe | 13-30 Hz, CAR |
|---|---:|---:|---:|---:|
| Rest | +0.46 | +0.23 | -0.23 | -0.23 |
| Stand | +1.16 | +0.46 | -0.13 | -0.27 |
| Walk | +1.61 | +0.92 | -0.10 | -0.32 |

Rest also changes, so an imagery-versus-fixation change alone is not enough
to establish an imagery-specific effect. Comparing the changes between
conditions gives:

| Difference between fixation-relative changes | 8-13 Hz, earlobe | 8-13 Hz, CAR | 13-30 Hz, earlobe | 13-30 Hz, CAR |
|---|---:|---:|---:|---:|
| Stand minus Rest | +0.70 | +0.24 | +0.10 | -0.04 |
| Walk minus Rest | +1.16 | +0.70 | +0.13 | -0.10 |
| Walk minus Stand | +0.46 | +0.46 | +0.03 | -0.06 |

Each contrast is calculated within participant before averaging; the
Walk-minus-Stand row is derived from the participant condition summaries.
For 8-13 Hz, Walk exceeded Rest in **9/10 participants** with the earlobe
reference and **8/10** with CAR. The difference was positive in both the
first and last 25 trials in **8/10** and **7/10**, respectively. Stand-minus-Rest
was more reference-sensitive: **8/10** positive with earlobe, **4/10** with CAR.
The 13-30 Hz group mean contrasts were smaller than the 8-13 Hz contrasts;
this does not rule out participant-specific or spatial beta-band information.

![Central EEG power changes through the cue and task, for both references](results/reanalysis/within_class_timecourse.png)

The curves show all five 1 s task bins and the first cue bin. Shaded bands
are participant-level 95% t intervals. The 1-5 s summary window was chosen
after inspecting the time course; these are descriptive results, not
confirmatory significance tests.

[Participant power summaries](results/reanalysis/within_class_late_participant.csv)
| [Group power summaries](results/reanalysis/within_class_late_group.csv)
| [Full time-course summaries](results/reanalysis/within_class_group.csv)
| [Power calculation](code/reanalysis/within_class_checks.py)
| [Task-window aggregation](code/reanalysis/within_class_late_summary.py)

### Time, Frequency Band, and Spatial Pattern

These auxiliary analyses ask **where condition information appears in the
recorded features**. They are not an explanation of the selected pipelines'
decisions or replacement estimates of their accuracy. All values below are
**mean balanced accuracy (%) across ten participants**, with 50% as the
chance reference. Balanced accuracy averages recall for the two labels.

Each probe uses a fixed logistic regression with C=0.3. Feature standardization
and classifier fitting occur only within the training fold. The probes use
their own cross-validation folds and retained trial sets, **not the outer-fold
predictions of the primary nested evaluation**, and are not independent
confirmation of its scores. The values are taken from saved aggregate tables.

#### Time: When Is Condition Information Present?

Eight-channel 8-13 and 13-30 Hz log power is compared across stages, using
five contiguous held-out trial blocks within each participant. The fixation,
cue, early-task, and late-task stages each average two matched 1 s estimates.
The pooled task 1-5 s feature averages four. Task-minus-fixation subtracts
each trial's own fixation log power before classification.

| Contrast | Reference | Fixation -4.5 to -2.5 s | Cue -2 to 0 s | Task 0-2 s | Task 3-5 s | Task 1-5 s | Task 1-5 s minus fixation |
|---|---|---:|---:|---:|---:|---:|---:|
| Rest-Stand | Earlobe | 46.8 | 52.3 | 57.5 | 60.8 | 62.5 | 56.5 |
| Rest-Stand | CAR | 48.4 | 51.8 | 60.0 | 59.1 | 63.4 | 58.2 |
| Rest-Walk | Earlobe | 49.7 | 55.0 | 59.4 | 55.8 | 59.1 | 56.5 |
| Rest-Walk | CAR | 48.6 | 51.1 | 60.1 | 57.4 | 62.8 | 61.0 |

Before-cue group means were close to 50%; task features were more predictive
under these probes. Equal-duration early and late task windows do **not** show
a consistent late advantage across contrasts and references. The pooled 4 s
window averages more data, so its higher scores cannot be called a late-only
effect. Subtracting fixation still leaves some predictive information,
especially for Rest-Walk under CAR, but does not exclude delayed cue effects
or ocular/muscle contributions.

[Stage results, earlobe](results/reanalysis/rest_mi_stage_group.csv)
| [Stage results, CAR](results/reanalysis/rest_mi_stage_car_group.csv)
| [Participant scores, earlobe](results/reanalysis/rest_mi_stage_participant.csv)
| [Participant scores, CAR](results/reanalysis/rest_mi_stage_car_participant.csv)
| [Stage feature and evaluation code](code/reanalysis/rest_mi_stage_diagnostics.py)

#### Frequency Band: Which Band Carries Condition Information?

Using the same task 1-5 s windows and contiguous-fold evaluation, the probes
are fitted separately to the eight channel powers in each band:

| Contrast | Reference | 8-13 Hz only | 13-30 Hz only |
|---|---|---:|---:|
| Rest-Stand | Earlobe | 57.0 | 61.3 |
| Rest-Stand | CAR | 57.7 | 60.5 |
| Rest-Walk | Earlobe | 61.0 | 58.4 |
| Rest-Walk | CAR | 60.7 | 57.0 |

The beta-band probe was more predictive for Rest-Stand, while the 8-13 Hz
probe was more predictive for Rest-Walk, in both references. These are
descriptions of label information in band-power features, **not estimates of
each band's physiological contribution** or proof of a motor-cortical source.
The source tables and calculation code are the same as for the time probes.

#### Spatial Pattern: Overall Power or Relative Channel Distribution?

A separate decomposition uses task **0.5-4.5 s** power with the earlobe
reference. For each trial and band, the eight-channel mean of log power gives
the global level; subtracting that mean from each channel's log power gives the relative
spatial pattern. Both bands are retained. This channel-pattern transformation
is not CAR rereferencing: it operates on log-power features, not time-domain
voltages.

Fixed, class-balanced logistic probes use five repetitions of stratified
five-fold cross-validation. **Pooled imagery combines Stand and Walk into
one label** (100 imagery trials versus 50 Rest trials). It is an auxiliary
binary contrast for feature characterization, not one of the three primary
comparisons.

| Feature set | Rest vs pooled imagery | Stand-Walk |
|---|---:|---:|
| Global band levels | 59.7 | 51.5 |
| Relative channel pattern | 61.6 | 50.6 |
| Both combined | 63.2 | 50.7 |

Both global levels and relative channel patterns carry some information for
Rest versus pooled imagery in this evaluation. Stand-Walk group means remain
close to 50% for all three feature sets; this is not proof of equivalence to chance
or an upper limit on another model. The feature sets are related transformations
of the same powers, not independent physiological sources. Neither this
decomposition nor the band probes identify which features the selected primary
pipelines relied on to make a particular prediction.
Different windows, power estimates, and validation folds also prevent a direct
ranking of these scores against the time/band probes or primary nested estimates.

[Feature-decomposition group scores](results/reanalysis/physiology_pattern_group.csv)
| [Participant scores](results/reanalysis/physiology_pattern_participant.csv)
| [Feature-decomposition code](code/reanalysis/physiology_pattern_ablation.py)
| [Cross-validation helper](code/reanalysis/contrast_logic_checks.py)
| [Task-power feature extraction](code/reanalysis/reanalyse_cached_eeg.py)

All probes are exploratory comparisons of the same recordings. Multiple
windows and feature sets were inspected without confirmatory significance
tests. Continuous offline filtering and within-recording splits do not remove
all temporal dependence or establish new-session performance.

### Interpretation

The clearest group pattern is an **8-13 Hz increase in Walk relative to Rest**,
not a consistent decrease across the imagery conditions. A positive
Walk-minus-Stand mean is a descriptive difference, not evidence that this
single feature reliably separates individual trials.

These measurements support a **condition-associated scalp EEG power
difference**, not an identified motor-imagery-specific cortical response.
The 8-13 Hz band is not treated as confirmed sensorimotor mu activity merely
because of its frequency. Condition-specific cues, attention, eye or muscle
activity, and imagery performance cannot be separated without the missing
EOG, EMG, gaze, and trial-level compliance measurements.

The dB summaries use task 1-5 s and fifty trials per condition; the primary
nested evaluation uses task 0.5-4.5 s and its own epoch selection and folds.
The linear probes use separately specified windows and folds. Power summaries
and probes describe differences and predictive features in the recordings;
they do **not** establish why a selected primary pipeline classified a
particular participant well.
