# Fixed Filterbank CNN: methods and release scope

Recorded training: October 13, 2026. Model-focused release: October 14, 2026.
This release packages already-completed runs; it is not a new training experiment.

## Data and task

Ten participants contributed fifty retained trials each of Rest, standing-up
imagery, and walking imagery. The channels were FC3, FC4, C3, C4, Cz, CP3,
CP4, and CPz at 256 Hz. Trials used 3 s fixation, a 2 s condition-specific
visual cue, and a 5 s task. Rest-Stand and Rest-Walk are separate binary tasks,
not Rest versus pooled imagery or actual movement.

The authorized local extraction used the existing retained MI/MI2 rules,
continuous 1-40 Hz fourth-order Butterworth filtering and a 60 Hz notch, then
task-onset epochs from 0 to 5 s inclusive: 1,281 samples. Each channel's task
onset sample was subtracted, corresponding to baseline=(0,0). The original
XDF numerical scale was retained; physical-unit calibration was not newly
established. There was no dataset-wide z-scoring or per-trial T/CT/C input
normalization. This is not the MD2DEEGNet normalization experiment.

## Model and features

The user-provided `FilterbankTripleShallowCNN` definition came from
`TSCNN_3CLS_new_trigger_file_v4_Startified_k_fold.py`. Its class and
`butter_bandpass_filter` helper are retained in the portable model module.
The class is called with five bands, eight channels, two output classes, and
dropout .3. The model is an adaptation of the supplied TSCNN implementation,
not an independently invented architecture or a claim of paper replication.

Fixed trial-wise Butterworth bands are 1-60, 1-4, 4-8, 8-13, and 14-30 Hz.
The 1-60 label does not restore frequencies removed by the earlier 1-40 Hz
filter. A magnitude STFT of the first bank uses nperseg=nfft=128,
noverlap=64, and boundary=None. The first two branches summarize each band's
features and average across bands; the third summarizes the spectrogram.
Their 120 concatenated features feed the two-class output.

Internal BatchNorm acts on convolution features. It is not one of the
seven MD2D raw-input normalization options and is not updated in evaluation.

## Fixed evaluation

For every participant and pair, original trials are ordered in first-class
then second-class blocks before stratified splitting. Split seed42 gives
train60/validation20/test20, with 30/10/10 trials per class. The same trial
membership is retained for training seeds42,43,44. These training seeds affect
initial weights and minibatch order, not which trials are tested.

There is no augmentation, learned CSP, supervised feature selection, or
hyperparameter search in this profile. AdamW uses lr=.001 and
weight_decay=.0001; StepLR uses step_size=10 and gamma=.5. Batch size is32,
cross entropy is used, maximum epochs100, and patience5. An improvement in
validation loss must exceed .0001. The validation-loss-selected state is
restored and saved before frozen test prediction. The training function takes
no test arguments; prediction takes no truth labels. Test labels are used for
stratified assignment and final scoring, never fitting or checkpoint selection.

All ninety participant/pair/seed scores are included. No best seed is selected.
Each accuracy counts twenty test trials, so one trial is five percentage
points. Class balance makes ordinary and balanced accuracy equal here.

This profile uses a binary output, one fixed setting, common 60/20/20 splits,
validation-loss stopping, and explicit best-weight restoration. It is a
model-side adaptation, not a reproduction of the supplied source's full
search procedure. The Filterbank profile is highlighted after inspecting
broader model comparisons; its results are not an unbiased estimate of a
procedure that selects it over other models.

## Result files

- `participant_scores.csv`: all 90 aggregate accuracy/AUC records and correct
  counts, without trial identities or trial-level predictions.
- `participant_summary.csv`: three-seed mean/min/max within each person/pair.
- `group_summary.csv`: ten-participant means and ranges for each seed/pair.
  These are not cross-subject prediction results.
- `seed_sensitivity.png`: all participants and all three seeds, not selected
  high-scoring individuals only.
- `release_provenance.json`: source/release hashes and recorded software,
  without raw-file names, participant keys, or raw-data hashes.
- `VERIFICATION.md`: checked code/numbers and explicitly unverified scope.

## Reproduction boundary

`code/classification/run.py` is a portable extraction of the
fixed model/evaluation path. It accepts authorized, already-preprocessed
`S01.npz` through `S10.npz`, each with `X=(150,8,1281)` and integer
`y=(150,)`, using class IDs0=Rest,1=Stand,2=Walk in retained recording order.
The channel order and exact preprocessing above are part of the contract.
Shape checks alone cannot prove that an arbitrary supplied array satisfies it.

The public entry point does not implement private XDF discovery, participant
mapping, or retained-trial extraction. Raw recordings, those mappings, the
cached inputs, weights, and trial-level predictions are not distributed.
Consequently it reproduces the model-side calculation with authorized matching
inputs, not an unrestricted end-to-end reconstruction from public raw data.

Recorded real-data training used PyTorch 2.5.1 on Apple MPS, Python 3.12.4,
NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.4.2, and two torch threads. Different
devices or software can change numerical outcomes; bitwise cross-platform
reproduction is not promised. The original local helper precomputed the
fixed per-trial features for all 150 trials in seeds 43/44, whereas this release
computes train/validation first and test after checkpoint saving. These
transforms use no cross-trial estimates or labels, so this change does not
alter their values.

Run from the repository root in an isolated Python 3.12 environment:

```bash
python -m pip install -r code/classification/requirements.txt
python -m unittest discover -s code/classification -p 'test_*.py'
python code/classification/run.py \
  --data /authorized/path/to/preprocessed_trials \
  --out cache/filterbank_new_run --device mps
```

The tests use synthetic arrays and aggregate public scores. They do not retrain
the reported real-data models. New outputs default to ignored cache storage;
use a new directory rather than overwrite published results.

## Interpretation and accountability

This is a post-hoc analysis of the same ten-person cohort, not new subjects,
new sessions, or an online BCI trial. The three seeds reuse the same held-out
twenty trials; they are sensitivity checks, not three biological replications.
Random within-recording splits and continuous offline filtering do not remove
all temporal dependence. A high participant score is neither a signal-quality
diagnosis nor evidence that the classifier decoded MI-specific cortex.

Condition-specific cues, attention, ocular/muscle activity, and imagery
performance cannot be uniquely separated without the missing reference and
compliance measurements. The fixation-relative 1-5 s power summaries address
a separate question: they are not the CNN's 0-5 s features or a causal
explanation of its decisions.

Codex assisted code adaptation, execution, checking, and documentation. The
data/model were user-provided. Automated tests and independent AI review are
not human authorship, scientific approval, or data-publication permission.
The researcher remains responsible for source-data review and interpretation.
