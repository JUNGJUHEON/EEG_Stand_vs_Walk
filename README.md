# Standing and walking motor imagery with 8-channel EEG

## Fixed Filterbank CNN: Participant-Level Revalidation

This model-focused report presents **freshly trained, fixed-setting Filterbank
CNN results on the original ten participants' recordings**. Rest-Stand and
Rest-Walk are separate binary comparisons. Stand means imagery of standing up,
not actual movement or static standing. The original objective was to distinguish
standing-up from walking imagery; all three pairwise results remain visible.

The model is the supplied TSCNN `FilterbankTripleShallowCNN`, adapted to a
two-class output and a fixed evaluation protocol. Codex assisted code adaptation,
execution, checking, and documentation. The recorded real-data training took
place on October 13; this October 14 update packages those completed results.
It is not a new dataset or a newly invented model.

### Main Observation

**Some participants showed repeatable Rest-versus-imagery prediction across
three training seeds on the same held-out trials.** For example:

- **S03:** Rest-Stand **85-90%** and Rest-Walk **70-75%**.
- **S06:** Rest-Stand and Rest-Walk each **80-85%**.

These are the complete three-seed ranges for two illustrative participants,
identified after inspecting the results. They are **not a cohort-wide 70-90%
accuracy**. Other participants performed less well or varied with training seed.
Changing the seed repeats training on the same split; it does not add new test
trials, sessions, or people. High accuracy does not establish MI-specific cortex.

### All Participants and Training Seeds

Each cell lists accuracy (%) for **training seeds 42 / 43 / 44**. The split seed
stays 42, with train 60 / validation 20 / test 20 original trials for each binary
comparison. The two test classes have ten trials each; one error changes accuracy
by five percentage points. No best seed was selected.

| Participant | Rest-Stand | Rest-Walk | Stand-Walk |
|---|---:|---:|---:|
| S01 | 40 / 75 / 50 | 45 / 45 / 45 | 55 / 50 / 55 |
| S02 | 50 / 55 / 30 | 50 / 50 / 60 | 50 / 50 / 50 |
| S03 | 90 / 90 / 85 | 75 / 75 / 70 | 50 / 50 / 40 |
| S04 | 85 / 90 / 55 | 75 / 75 / 75 | 50 / 45 / 65 |
| S05 | 45 / 55 / 55 | 50 / 55 / 60 | 50 / 55 / 45 |
| S06 | 80 / 85 / 85 | 85 / 80 / 80 | 65 / 65 / 65 |
| S07 | 50 / 50 / 35 | 60 / 60 / 50 | 50 / 50 / 55 |
| S08 | 50 / 50 / 65 | 75 / 50 / 55 | 45 / 65 / 50 |
| S09 | 65 / 55 / 60 | 60 / 60 / 60 | 70 / 65 / 75 |
| S10 | 50 / 50 / 55 | 50 / 55 / 50 | 55 / 55 / 55 |

![Fixed Filterbank accuracy for all participants and three training seeds](results/reanalysis/filterbank_fixed_2026_10_14/seed_sensitivity.png)

[All 90 recorded scores and correct counts](results/reanalysis/filterbank_fixed_2026_10_14/participant_scores.csv)
· [Within-participant seed summaries](results/reanalysis/filterbank_fixed_2026_10_14/participant_summary.csv)
· [Ten-participant summaries by seed](results/reanalysis/filterbank_fixed_2026_10_14/group_summary.csv)

### Model and Evaluation

Eight channels (FC3, FC4, C3, C4, Cz, CP3, CP4, CPz) were recorded at 256 Hz,
with fifty retained trials per condition. Each trial had 3 s fixation, a 2 s
condition-specific cue, and a 5 s task. Model inputs cover task 0-5 s inclusive
(1,281 samples), following continuous 1-40 Hz filtering, 60 Hz notch, and onset
sample subtraction. No MD2D T/CT/C input normalization was applied.

The five fixed filterbank bands and magnitude STFT feed three CNN branches.
One setting was held fixed: dropout=0.3, AdamW lr=0.001/weight decay=0.0001,
StepLR step=10/gamma=0.5, batch size=32, at most 100 epochs, patience=5. Weights were selected by
**validation loss**, saved, and then used for frozen test prediction. Test
answers were not used for fitting, checkpoint selection, or choosing a seed.

This adapted protocol differs from the supplied three-class grid-search script.
It does not claim full reproduction of that original search. The model is
highlighted after a broader set of reruns, not selected and validated on a new
independent dataset. Other evaluated model profiles did not all perform similarly.

[Detailed methods and differences from the original](results/reanalysis/filterbank_fixed_2026_10_14/METHODS.md)
· [Model and fixed runner](code/reanalysis/filterbank_fixed/)
· [Verification scope](results/reanalysis/filterbank_fixed_2026_10_14/VERIFICATION.md)

### What the Scores Do and Do Not Mean

The results support participant-specific condition prediction in this recording
set, with substantial between-person and training-seed variability. They do
not demonstrate stable 70%+ accuracy across all ten participants, prediction
for a new person/day, or online BCI performance. They also do not establish
that the predictive signal originated specifically from motor imagery rather
than visual cues, attention, ocular/muscle activity, or a mixture.

The September 29 signal reanalysis is retained below as **separate evidence**.
Its power changes do not causally explain the CNN's decisions, and its CV
scores must not be substituted for this model's fixed-split scores.

### Reproduce the Model-Side Evaluation

With authorized matching preprocessed trial arrays, use an isolated Python 3.12
environment and run from the repository root:

```bash
python -m pip install -r code/reanalysis/filterbank_fixed/requirements.txt
python -m unittest discover -s code/reanalysis/filterbank_fixed -p 'test_*.py'
python code/reanalysis/filterbank_fixed/run.py \
  --data /authorized/path/to/preprocessed_trials \
  --out cache/filterbank_new_run --device mps
```

Raw XDF extraction, private retained-trial mappings, cached EEG, checkpoints,
and trial-level predictions are not distributed. The public runner starts at
the documented preprocessed-input boundary; exact cross-platform results are
not guaranteed. The tests use synthetic inputs and published aggregate scores.

## September 29 Reanalysis: Separate Signal Evidence

The following preserves the earlier evidence-bounded reanalysis. Its original
overview is also available at the [September 29 commit](https://github.com/JUNGJUHEON/EEG_Stand_vs_Walk/blob/4a4ac8885d31377b32fb342fe9084cd3ea0f04d3/README.md).
It uses the same cohort but different features and evaluation procedures.

### 1. Earlier Classification Results

The original nested cross-validation results below were **recalculated from saved scores**, not obtained by rerunning the entire original model search. The interval is across the ten participant scores and does not account for analysis choices made on these data.

| Contrast | Mean accuracy | Participant-level 95% t interval | Chance |
|---|---:|---:|---:|
| Rest vs Stand | 65.8% | 57.1–74.5% | 50% |
| Rest vs Walk | 64.5% | 57.1–71.9% | 50% |
| Stand vs Walk | 53.1% | 47.5–58.7% | 50% |
| Three classes | 45.8% | 39.5–52.0% | 33.3% |

[Saved-score audit](results/reanalysis/nested_score_summary.csv) · [Original participant scores](results/final_scores.csv)

A separate fixed analysis used 8–30 Hz log power from all eight channels, a fixed logistic regression, and repeated within-participant five-fold cross-validation. Its mean balanced accuracy was **60.6%** for Rest–Stand, **59.5%** for Rest–Walk, and **51.6%** for Stand–Walk. These values are from a different feature and evaluation procedure, so they are a qualitative check of the original ordering rather than a replication of the nested estimates. The Stand–Walk results do not establish equivalence to chance or an upper limit on what another measurement or model could recover.

[Fixed-model participant scores](results/reanalysis/fixed_model_scores.csv)

### 2. Does Each Condition Show an EEG Change?

We compared equal-length 1 s power estimates during fixation with the task bins from 1–5 s after task onset. These are participant means for **8–13 Hz at Cz and CPz**:

| Condition | Change from its own fixation, earlobe | Change from its own fixation, CAR |
|---|---:|---:|
| Rest | +0.46 dB | +0.23 dB |
| Stand | +1.16 dB | +0.46 dB |
| Walk | +1.61 dB | +0.92 dB |

Subtracting the rest change within each participant gives the following contrasts. The 1–5 s summary was selected after viewing the time course, so its intervals are descriptive and do not provide confirmatory significance tests.

| Extra change relative to Rest | Earlobe reference, dB (95% interval) | 8-channel CAR, dB (95% interval) | Participants with a positive value, earlobe/CAR |
|---|---:|---:|---:|
| Stand | +0.70 (+0.05, +1.35) | +0.24 (−0.43, +0.90) | 8/10, 4/10 |
| Walk | +1.16 (+0.35, +1.97) | +0.70 (+0.01, +1.38) | 9/10, 8/10 |

The walking condition has the more consistent rest-exceeding increase. Its direction persisted in both trial halves for 8/10 participants with the earlobe reference and 7/10 with CAR. Rest itself also changed over the task period, and the standing result was reference-sensitive. The walking CAR interval is close to zero at its lower bound. The repeated direction across halves supports a **within-recording, task-associated scalp power change**; it does not identify the source of the 8–13 Hz signal as sensorimotor cortex or verify imagery performance.

![Fixation-relative EEG power by condition](results/reanalysis/within_class_timecourse.png)

[Time-course summaries](results/reanalysis/within_class_group.csv) · [Selected 1–5 s summaries](results/reanalysis/within_class_late_group.csv)

### 3. What Separates Rest from the Imagery Conditions?

For each participant, a fixed classifier used the 8–13 and 13–30 Hz log power of eight channels. The table gives mean **balanced accuracy** with contiguous five-fold validation. The first four windows each cover 2 s; the pooled 1–5 s window covers 4 s and therefore averages more data. Chance is 50%.

| Contrast | Reference | Before cue | Cue | Task 0–2 s | Task 3–5 s | Task 1–5 s | Task 1–5 s minus same-trial fixation |
|---|---|---:|---:|---:|---:|---:|---:|
| Rest–Stand | Earlobe | 46.8% | 52.3% | 57.5% | 60.8% | 62.5% | 56.5% |
| Rest–Stand | CAR | 48.4% | 51.8% | 60.0% | 59.1% | 63.4% | 58.2% |
| Rest–Walk | Earlobe | 49.7% | 55.0% | 59.4% | 55.8% | 59.1% | 56.5% |
| Rest–Walk | CAR | 48.6% | 51.1% | 60.1% | 57.4% | 62.8% | 61.0% |

Before the cue, future condition labels were not predictably represented by these features at group level. During the task, both references carried label information. Comparing equal-duration task windows does **not** show a consistent advantage for the end of the trial; the 4 s pooled window's higher score should not be described as a late-only effect. Removing same-trial fixation power still left some information, especially for Rest–Walk under CAR. The cue could also have delayed effects that these comparisons cannot exclude.

In the pooled 1–5 s window, beta-only features reached 61.3%/60.5% (earlobe/CAR) for Rest–Stand; 8–13 Hz-only features reached 61.0%/60.7% for Rest–Walk. A separate decomposition of 0.5–4.5 s power found Rest–pooled-imagery information in both global band levels (59.7%) and relative channel pattern (61.6%). These are **predictive descriptions of recorded signals**, not estimates of the physiological contribution of each source. The walking 8–13 Hz increase above cannot by itself explain the original Rest–Walk classifier: the first 2 s of the task also carried information.

[Stage-by-stage results, earlobe](results/reanalysis/rest_mi_stage_group.csv) · [Stage-by-stage results, CAR](results/reanalysis/rest_mi_stage_car_group.csv) · [Power-pattern decomposition](results/reanalysis/physiology_pattern_group.csv)

### 4. Interpretation and Limits

Condition labels changed together with the visual cue and the requested mental task. No simultaneous EOG, EMG, gaze, or trial-level imagery-compliance measure is available. Cortical motor imagery, visual attention, other mental state changes, and ocular or muscle contributions therefore cannot be separated causally in these recordings. The data support a **condition-associated EEG power difference** and partial Rest-versus-imagery label prediction; they do not establish why the physiological signal arose or why Stand–Walk prediction was weak.

The original ERD figures in the top-level `results/` directory should not be used as evidence for sustained 8–13 Hz suppression. The estimator used there gave **−2.32 dB for a stationary simulated signal** with no task change. A comparison using matched 1 s power estimates instead found a later 8–13 Hz increase, most consistently in Walk. Likewise, exploratory channel curves, model sweeps, execution controls, and artifact-removal attempts do not establish a device-wide information ceiling, a peripheral cause, or absence of neural differences.

This reanalysis was developed after inspecting the original results. The participant intervals, window comparisons, and alternative models are descriptive; they are not independent validation. A follow-up study would need balanced cues, a matched nonmotor control, direct artifact measurements, trial-level performance checks, and a separately held-out sample or session to test a motor-specific explanation.

[Stationary-signal estimator check](results/reanalysis/erd_estimator_stationary_null.json)

### Reproduce the September 29 Reanalysis

The new scripts are in `code/reanalysis/`; their selected anonymized outputs are in `results/reanalysis/`. The original scripts and outputs remain in the top-level `code/` and `results/` directories as a record of the exploratory work. Interpret older figures and logs using the revised limits above.

With authorized access to the recordings, set `STEAM_DATA_DIR` to the raw-data directory and build the continuous cache using `code/steam_io.py`. The reanalysis scripts take that cache as input. Python 3.11 with NumPy, SciPy, pandas, scikit-learn, matplotlib, MNE, and pyxdf covers the scripts below.

```bash
export STEAM_DATA_DIR=/authorized/path/to/BBRaw
PYTHONPATH=code python -c 'import steam_io; steam_io.build_cache()'
python code/reanalysis/audit_existing_results.py --source . --out results/reanalysis
python code/reanalysis/check_xdf_cache.py --raw "$STEAM_DATA_DIR" --cache cache --out results/reanalysis
python code/reanalysis/reanalyse_cached_eeg.py --cache cache --out results/reanalysis
python code/reanalysis/contrast_logic_checks.py --cache cache --out results/reanalysis
python code/reanalysis/physiology_pattern_ablation.py --cache cache --out results/reanalysis
python code/reanalysis/within_class_checks.py --cache cache --out results/reanalysis
python code/reanalysis/within_class_late_summary.py --results results/reanalysis
python code/reanalysis/within_class_figure.py --results results/reanalysis
python code/reanalysis/rest_mi_stage_diagnostics.py --cache cache --out results/reanalysis --reference ear
python code/reanalysis/rest_mi_stage_diagnostics.py --cache cache --out results/reanalysis --reference car
python code/reanalysis/erd_estimator_null.py --out results/reanalysis
```

`check_xdf_cache.py` requires the original XDF files and writes local file hashes; those hashes are not included in this repository. The cache and raw recordings are not distributed. Public result tables use pseudonyms S01–S10 and do not contain trial-level signals or direct identifiers. The published files are sufficient to inspect the numerical summaries; full signal-level reproduction requires separately authorized data access.
