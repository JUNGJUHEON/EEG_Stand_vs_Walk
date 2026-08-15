# Bounding Lower-Limb Motor-Imagery Decodability on an 8-Channel Wearable EEG Montage: Rest Is Detectable, Standing and Walking Are Not Separable

**Draft v1.** Every number below traces to a script in this folder; the source
is named in square brackets after each result so it can be regenerated or
checked. Text in `⟨angle brackets⟩` is a placeholder you must fill.

Target: IEEE TNSRE (matches the reference paradigm, Ma *et al.* 2023).

---

## Abstract

**Objective.** Brain–computer interfaces for lower-limb exoskeletons must
distinguish rest, standing-up and walking intentions. Whether a low-cost
8-channel wearable montage can support that distinction is unknown; existing
demonstrations use 56–64 electrode research systems. **Methods.** Ten
participants performed motor imagery of rest, standing up, and walking
(50 trials each) with a Bitbrain Versatile 8-channel montage at 256 Hz.
Decoding used filter-bank Riemannian tangent-space features with nested
cross-validation, benchmarked against 18 further pipelines including FBCSP
and EEGNet. Claims were tested with per-participant permutation nulls, a
chronological split, a corrected resampled *t*-test, and topography-based
physiological screens. **Results.** Rest was detectable against both imagery
conditions — 64.5 % versus walking (significant in 8/10 participants) and
65.8 % versus standing (5/10); both are significant at the participant level
(*p* ≤ .0026, *n* = 10). Standing versus walking reached only 53.1 % (1/10)
and is not significantly above chance across participants (*p* = .246, 95 % CI
[47.5, 58.7]), while being significantly *below* the ~70 % usability threshold
(*p* = 4.1 × 10⁻⁵). *That positive result carries one caveat: against the visually matched
rest condition, only 1/10 participants showed the expected mu
desynchronisation, so the rest-versus-imagery separation cannot be attributed
specifically to motor imagery as opposed to differences in visual attention.*
The negative is bounded rather than merely unachieved: an 840-configuration
search scored optimistically on the reported folds averaged 68.4 % and left
seven of ten participants below 70 %; the learning curve yields +1.7 points
for a doubled session; accuracy saturates at three electrodes; and the
standing and walking topographies differ no more than two random halves of a
single condition do (ratio 1.07–1.12, *p* > .46). One participant reached 67 % but the effect
proved to be a spatially uniform arousal-and-muscle-tone difference, not
sensorimotor rhythm. A motor-execution positive control on the same hardware
separates two *executed* movements at **91.1 %** in every participant, so no
result here reflects a broken recording; but that separation is peripheral —
beta rises during movement and shows no post-movement rebound — and the
cortical component is a mu-specific trough only 1.56 dB deep. The ordering of
every accuracy we report follows the peripheral signal available in the
contrast, not the cortical difference the paradigm intended to elicit. **Conclusion.** The compound-limbs element cannot help as
implemented, but for a more specific reason than the absence of an asymmetry.
The benefit reported by Ma *et al.* requires the mirrored configurations to sit
in *different classes*; the walking instruction here places both inside one
trial, where a trial-averaged laterality contrast cancels them (+0.026 dB,
*p* = .73). Six timing-free tests of the within-trial alternation and a
peak-locked analysis were run; the strongest reached +1.02 dB with 56.1 %
decoding (*p* = .0084) and does not clear correction for the ten hypotheses
tested, so that question is left open rather than answered. **Significance.** We provide a bounded null for a
clinically motivated contrast, and a worked demonstration that accuracy plus a
permutation *p*-value is insufficient evidence for a BCI claim on a
low-channel montage.

**Index terms** — motor imagery, lower limb, low-channel EEG, Riemannian
geometry, negative results, electromyographic contamination.

---

## I. Introduction

⟨Two paragraphs of standard framing: exoskeleton BCIs, why lower-limb MI
matters, why low-cost wearable montages matter for deployment. Cite the
existing lower-limb MI literature you already have in the lab folder.⟩

Ma *et al.* [1] recently showed that adding a contralateral upper-limb swing
to lower-limb motor imagery raises imagery decoding accuracy to
73.70 ± 12.47 %, which they report as **11.02 points above** their
single-lower-limb paradigm — a compound-limbs design intended to match natural
gait. That result was obtained with a 56-electrode head cap on a 256-channel
g.HIamp amplifier at 1200 Hz, four electrooculography and four
surface-electromyography channels for artifact rejection, and an
electromagnetically shielded environment.

A walking exoskeleton does not need to know which leg is stepping; it needs to
know whether the wearer intends to stay still, to stand up, or to walk. Those
three states, not the left-versus-right contrasts of [1], are the class set a
device would be built around, and no study has established whether they are
separable on hardware a person could actually wear.

Two questions follow, and they are the subject of this paper.

**Q1. Can the clinically required class set — rest, standing-up and walking
imagery — be separated on a consumer-grade 8-channel montage?**

**Q2. Does incorporating the compound-limbs element of [1] — pairing the
imagined step with a contralateral arm swing — help on that class set?**

**The answer to Q1 is *partly*.** Rest is detectable against both imagery
conditions; standing and walking imagery are not separable from one another.

**Q2 could not be answered as posed, and that is itself the finding.** The
mechanism the compound-limbs manipulation exploits in [1] is a contralateral
C3/C4 asymmetry that arises only when the two classes are mirror images of
each other. Here the mirrored pair exists *within* a walking trial — the
instruction specifies two steps, so both configurations occur in sequence —
rather than *between* classes, and a trial-averaged contrast cancels them by
construction. We therefore report neither that the benefit failed to transfer
to eight channels nor that the asymmetry is absent, but that this class set
places it where a between-class contrast cannot reach it. Section IV-A
quantifies what remains, and Section III-J reports the strongest attempt to
recover the alternation directly.

**Contributions.**
1. A nested-cross-validated benchmark of the three clinically motivated
   classes on an 8-channel wearable montage, reported per participant rather
   than as a group mean.
2. A *bounded* negative result for standing-versus-walking imagery,
   established by exhaustive search over preprocessing and decoder families
   rather than asserted from a single pipeline.
3. A physiological account of that null rather than an assertion of it: the
   two conditions produce scalp topographies no more different than two random
   halves of a single condition; the montage spans ~1.5 effective spatial
   dimensions; and a spatial constraint deliberately pointed at the wrong
   anatomy costs 4.5 points on a contrast that does carry localised signal
   while costing nothing on this one.
4. A screen — spatial gradient, rhythm specificity, band ablation and
   stability across a within-session gap — that identifies an apparently
   significant decoding result as arousal and muscle artifact, together with a
   quantification of the visual-cue leakage that a commonly used epoch window
   introduces (52–56 % from the cue period alone).
5. A motor-execution positive control that validates the screens rather than
   assuming them: the same criteria classify executed movement as
   non-cortical, and an independent post-movement beta-rebound test agrees.
   The control also excludes recording failure as an explanation of the null,
   and yields a single mechanism — dominance by peripheral and global signals —
   that accounts for every accuracy reported here.

---

## II. Methods

### A. Participants and ethics

Ten participants ⟨age, sex, handedness⟩. ⟨Ethics approval body and number.⟩
All gave written informed consent.

### B. Paradigm

Three conditions were cued in randomised order: **Rest** (fixate the monitor,
imagine nothing), **Stand** (imagine standing up), and **Walk** (imagine two
steps, beginning with a right-foot step and a left-arm swing — the
compound-limbs element adopted from [1]).

Each trial comprised a 3.0 s fixation screen, a 2.0 s class-specific cue, and
a 5.0 s imagery period on a blank white screen initiated by a 1 s beep. Fifty
trials per class were collected per participant (150 total). Marker structure
and timing were verified identical across all thirteen recording files
(three participants' recordings were interrupted and restarted, giving two
files 34–46 min apart on the same day with the cap left in place;
trials were selected by pre-specified keep-rules and are class-balanced
within every run) [`steam_io.py`].

Critically, the **Rest condition is visually identical to the imagery
conditions** — same duration, same blank screen, same beep, same preceding
cue — with only the imagery removed. This permits a manipulation check that
the reference paradigm, which has no rest class, cannot perform.

![Fig. 1 — trial structure](results/fig_paradigm.png)

**Fig. 1.** Trial structure. 3.00 s fixation → 2.00 s class-specific cue →
5.00 s imagery on a blank screen. Redrawn from the PsychoPy builder flow and
confirmed against the recorded timing (±0.02 s across thirteen files). Only
the cue differs between classes; the imagery screen is identical for all
three, which is what makes Rest a visually matched baseline. Time is
referenced to imagery onset.

### C. Recording

Bitbrain Versatile 8-channel, water-based semi-dry sensors with active
shielding, at FC3, FC4, C3, C4, Cz, CP3, CP4 and CPz, referenced to the left
earlobe (A1), 256 Hz, recorded via LSL to XDF alongside the PsychoPy marker
stream. No EOG or EMG channels were recorded and no electromagnetic shielding
was used.

Two instrument properties bear on the analysis. First, the 8-channel
configuration has an analogue bandwidth of DC–40 Hz with a third-order
low-pass, so bands extending past 40 Hz are measured through instrumental
attenuation; this is not stated in the manufacturer's reference guide, and we
confirmed it in the recordings, where spectral power falls at −7.6 dB/octave
between 20 and 40 Hz but −15.8 dB/octave between 40 and 80 Hz
[Fig. 16]. Only the M4 artifact index of
Section II-G reaches into that region. Second, the manufacturer defines no
numeric impedance threshold for a water-based system — sensor quality is
verified qualitatively before recording using the device's own signal-quality
indicator — so we cannot report impedances in kΩ as a gel-based montage would,
and note this as a reproducibility limitation rather than an omitted criterion.

### D. Preprocessing

A 60/120 Hz notch removed mains interference. This is a necessary step, not a
precaution: in the unfiltered recordings the 60 Hz line sits **22 dB above its
spectral neighbours** and 2.5 dB above the mu-band peak itself, and the 120 Hz
harmonic is still visible ~10 dB above its neighbours. Data were band-pass
filtered
8–30 Hz (FIR, firwin) and epoched from **+0.5 to +4.5 s** relative to imagery
onset.

Two choices require justification. First, the epoch **excludes the cue
period**: because the cue is a class-specific visual stimulus, any window
beginning before imagery onset imports a visual evoked response. We quantify
this in Results. Second, **no re-referencing was applied**. Common average
referencing was tested and was 1.1 points *worse*; with eight electrodes all
positioned over sensorimotor cortex, the common average contains signal
rather than only noise [`run_sweep.py`].

Trials were rejected if any channel showed clipping at the amplifier rails or
a log-variance exceeding 5 robust standard deviations from that channel's
median (0–8 trials rejected per participant). Amplifier saturation was
present in several recordings but fell entirely within the between-block
rest periods; no clipped sample occurred inside a task epoch
[`01_diagnose.py`].

### E. Decoding pipeline, and why it takes this form

The pipeline is derived from properties of the recording rather than selected
by benchmark performance. Fig. 2 summarises it, with the data shape and the
justification at each stage, and marks the boundary of what is refit inside
the cross-validation training fold.

Motor imagery does not produce a time-locked evoked potential; it modulates
the *amplitude* of ongoing sensorimotor rhythms [4]. The discriminative
information therefore lies in how band-limited power is distributed across
electrodes — that is, in the **spatial covariance of band-filtered signals**.
Per-channel band power discards the cross-channel structure that carries
source orientation and depth; raw time-series models must learn the spatial
and spectral filtering from data, which at ~50 trials per class means more
parameters than independent examples.

Covariance matrices are symmetric positive-definite and lie on a curved
manifold. Euclidean operations on them distort distances, most severely where
the matrices are poorly conditioned — precisely the small-trial regime here.
We therefore project to the **tangent space at the Riemannian mean** [6],
after which a linear classifier is geometrically valid.

Single-participant channel × frequency maps show the discriminative ridge to
be approximately 2 Hz wide with a centre varying between 9 and 16 Hz across
participants. A single 8–30 Hz covariance averages this away, so the signal
is decomposed into **ten overlapping 4 Hz bands (2 Hz hop, 8–30 Hz)** and the
band weighting is learned per participant inside the training fold.

The upper limit of 30 Hz reduces, but does not eliminate, muscle
contamination. Goncharova *et al.* [9] report that the topographic spread of
scalp EMG "grows as frequency rises, and is greatest above 40 Hz", so a 30 Hz
ceiling excludes the range where contamination is most widespread. It does not
make the passband clean: the same study finds frontalis EMG amplitude
"greatest at 20–30 Hz frontally", identifies a 20–30 Hz component they call
the EMG beta rhythm, and concludes that "even weak contractions can produce
EMG that obscures or mimics EEG alpha, mu, or beta rhythms over the entire
scalp". The analysis band therefore overlaps a genuine EMG component, which is
precisely why decoding accuracy inside 8–30 Hz cannot by itself be taken as
cortical, and why the physiological screens of Section II-G are necessary
rather than optional. We demonstrate EMG presence in this dataset in Results.

The resulting 360 tangent-space features against ~100 trials are classified
by **L2-regularised logistic regression**. A linear model keeps weights
traceable to specific electrode pairs and frequencies, which is what makes
the physiological screens in Section III-E possible.

This pipeline is denoted `fb_ts_lr`. Eighteen further pipelines — CSP+LDA,
CSP+SVM, MDM, FgMDM, tangent-space variants, time-delay (CSSP-style)
covariances, windowed covariances, three FBCSP variants [5] (per-band CSP
followed by mutual-information-based best-individual-feature selection, with
LDA, SVM and random-forest classifiers), crop ensembles [10], and
EEGNet-8,2 [7] — were implemented for comparison [`decoders.py`,
`eegnet.py`].

![Fig. 2 — decoding pipeline](results/fig_pipeline.png)

**Fig. 2.** Decoding pipeline. Each stage is shown with its data shape and
the property of the recording that motivates it. The shaded region marks
everything refit inside the cross-validation training fold — band weighting,
covariance shrinkage, tangent-space reference, scaler and classifier.

### F. Evaluation protocol

**Unit of analysis.** Every inferential test in this paper treats the
**participant** as the independent unit (*n* = 10). Trials are nested within
participants and are not independent; trial-pooled tests are reported only
where explicitly labelled as descriptive. Group-level claims use one-sample
or paired *t*-tests over the ten per-participant values, and per-participant
claims use permutation nulls over that participant's own trials.

Three numbers are reported per participant and contrast.

1. **Repeated stratified 5-fold CV** (5 repeats), the conventional figure.
2. **Nested accuracy**: an inner 4-fold CV selects the pipeline within each
   outer training fold, so the reported fold never influences model choice.
   With ~100 trials, selecting the best of many pipelines on the reported CV
   inflates accuracy by several points; the nested figure is the one we
   quote.
3. **Chronological accuracy**: trained on the first 60 % of trials in
   acquisition order and tested on the last 40 %, to detect within-session
   drift being learned as class information.

All spatial filters, tangent-space references, scalers and feature selectors
are fitted inside the training fold only [`02_final_eval.py`].

Per-participant significance used 1000-permutation label nulls with
Holm correction across the ten participants. Pipeline comparisons used the
Nadeau–Bengio corrected resampled *t*-test [8], which corrects the variance of
the cross-validation estimate for "the variability due to the choice of
training sets" that overlapping resamples introduce; the uncorrected paired test is
anticonservative here and is reported only to show the size of that bias
[`09_stats.py`].

### G. Physiological validation

An accuracy and a permutation *p*-value establish only that *something*
separates the conditions. To test whether that something is sensorimotor
cortex, we defined four metrics from motor-imagery physiology **before**
examining any accuracy value [`11_neuro_criterion.py`]:

- **M1, spatial gradient**: (max|*r*| − min|*r*|)/max|*r*| across electrodes,
  8–30 Hz. Cortical sources project unevenly; global states do not.
- **M2, rhythm specificity**: mean *r*² in 8–30 Hz divided by mean *r*² in
  2–8 and 30–45 Hz. There is no sensorimotor rhythm outside mu/beta.
- **M3, central focus**: mean|*r*| over {Cz, CPz, C3, C4, CP3, CP4} divided by
  mean|*r*| over {FC3, FC4}. Foot representation is medial.
- **M4, EMG ratio**: mean|*r*| in 30–45 Hz over mean|*r*| in 8–30 Hz. Two
  qualifications apply. Because EMG also has a 20–30 Hz component [9], a low
  M4 does not certify an EMG-free signal, only a spectrum weighted away from
  the range where muscle dominates. And because the amplifier's analogue
  bandwidth is DC–40 Hz (Section II-C), the 30–45 Hz numerator lies partly in
  the third-order roll-off and is attenuated by the instrument. M4 is
  therefore valid as a *relative* index across participants, who share the
  same hardware, but its absolute magnitude understates the true high-frequency
  content — which makes any positive M4 finding conservative.

Prespecified prediction: if separability is neural, accuracy correlates
positively with M1–M3 and negatively with M4.

We additionally used the visually matched Rest condition as a baseline for a
**manipulation check** — motor imagery should desynchronise mu/beta relative
to it [`12_manipulation_check.py`] — and tested the compound-limbs mechanism
directly via a laterality index (right-hemisphere {FC4,C4,CP4} over
left-hemisphere {FC3,C3,CP3} band power).

---

## III. Results

### A. Protocol verification

Marker structure was identical across all thirteen files: 150 fixation
markers and 50 each of the nine trial markers per session, with screen
durations of 3.00, 2.00 and 5.00 s (inter-file range ±0.02 s). Class balance
was exact within every run, including the three participants whose trials
span two recording files [`steam_io.py`].

### B. Decoding accuracy

Fig. 3 summarises the four results that follow.

![Fig. 3 — summary of results](results/fig_summary.png)

**Fig. 3.** (A) Nested accuracy per contrast; bars are group means, open
circles individual participants, dotted lines chance, and *n*/10 the number
of participants individually significant against a permutation null.
(B) Stand vs Walk per participant: nested accuracy (solid) inside the
best-of-840-configurations bound obtained by scoring each participant's best
configuration on the folds being reported (pale). Even under that optimistic
procedure the bound remains close to chance. (C) Ceiling curves —
accuracy against training trials per class, with electrodes inset.
(D) Band ablation for the single individually significant participant:
delta/theta and gamma, which contain no sensorimotor rhythm, decode as well
as mu and beta, indicating a broadband state difference rather than SMR.

**TABLE I — Nested cross-validated accuracy, mean ± SD across ten
participants** [`02_final_eval.py`]

| Contrast | Accuracy | Chance | Range | 95 % CI | *t*(9) vs chance | *d* |
|---|---|---|---|---|---|---|
| Rest vs Stand | 65.8 ± 12.1 % | 50 | 48–88 | [57.1, 74.5] | +4.11, *p* = .0026 | 1.30 |
| Rest vs Walk | 64.5 ± 10.3 % | 50 | 42–76 | [57.1, 71.9] | +4.42, *p* = .0017 | 1.40 |
| **Stand vs Walk** | 53.1 ± 7.9 % | 50 | 43–70 | **[47.5, 58.7]** | **+1.24, *p* = .246** | 0.39 |
| Three-class | 45.8 ± 8.7 % | 33 | 35–64 | [39.5, 52.0] | +4.51, *p* = .0015 | 1.43 |

**The independent unit is the participant** (*n* = 10), not the trial. Trials
are nested within participants, so a binomial test pooling ~1000 trials across
the group treats correlated observations as independent and returns
implausibly small *p*-values; we report the participant-level one-sample
*t*-test instead. The distinction is decisive for the central contrast: pooled
over trials Stand vs Walk appears significant (*p* = .026), whereas at the
participant level it is **not significantly above chance and its confidence
interval includes chance**.

Per-participant optimisation over the fine filter bank with a soft-vote
ensemble raised Stand-vs-Walk to 54.5 % and three-class to 48.2 %
[`05_optimize.py`].

**Bounding the negative formally.** A non-significant result is not by itself
evidence of absence, so we test the bound directly rather than inferring it
from *p* > .05. Against the ~70 % accuracy commonly cited as the minimum for
practical BCI control, Stand-vs-Walk accuracy is significantly *below*
threshold (one-sided *t*(9) = −6.78, *p* = 4.1 × 10⁻⁵) — a positive
demonstration, not a failure to reject. Two one-sided tests for equivalence to
chance, however, do **not** support a claim of exact equivalence at a ±5-point
margin (*p* = .232); equivalence is supported only at ±8 points (*p* = .040).
The defensible statement is therefore bounded on both sides: the true
Stand-vs-Walk accuracy lies within roughly eight points of chance and is
demonstrably below the level at which it could drive a device. We do not claim
it is exactly chance.

The pooled three-class confusion matrix localises the entire error
[Fig. 2, `fig_confusion_3cls.png`]: Rest is recovered at 53 %, while Stand
and Walk exchange 36 % and 34 % of their mass with each other. Rest is
recovered above chance in every participant. **The three-class problem and
the Stand-versus-Walk problem are the same problem.**

`fb_ts_lr` achieved the highest mean accuracy on all four contrasts. However,
under the corrected resampled *t*-test none of sixteen pipeline contrasts
reached significance (all *p* ≥ .247), although the uncorrected paired test
gave *p* = .046 for `fb_ts_lr` versus CSP+LDA on Rest-vs-Stand. We therefore
describe `fb_ts_lr` as best-performing, not significantly better.
EEGNet-8,2 underperformed the classical pipelines (Stand vs Walk 51.8 %,
three-class 42.1 %) [`04_deep.py`].

**Chronological validation.** Training on the first 60 % of trials and
testing on the last 40 % cost at most 1.4 points (Rest vs Stand 65.6 %,
Rest vs Walk 65.1 %, three-class 46.0 %). Within-session drift is therefore
not contributing to the reported accuracies.

### C. Per-participant significance

**TABLE II — Participants individually exceeding chance (1000-permutation
null, Holm-corrected)** [`09_stats.py`, Fig. 3]

| Contrast | Significant |
|---|---|
| Three-class | 8 / 10 |
| Rest vs Walk | 8 / 10 |
| Rest vs Stand | 5 / 10 |
| Stand vs Walk | 1 / 10 |

Note that Rest-vs-Stand has the highest group mean but the fewest
individually significant participants, its spread being widest (± 12.1).
Group means alone invert the ordering of these two contrasts.

**TABLE IIb — The two rest-versus-imagery contrasts, side by side**

| | Rest vs **Stand** | Rest vs **Walk** |
|---|---|---|
| Nested accuracy | 65.8 ± 12.1 % | 64.5 ± 10.3 % |
| Range across participants | 48–88 % | 42–76 % |
| **Individually significant (Holm)** | **5 / 10** | **8 / 10** |
| Chronological-split cost | −1.4 pts | −0.7 pts |
| M2 rhythm specificity | 1.60 | **1.83** |
| M4 EMG ratio (lower is cleaner) | 0.98 | **0.74** |
| M3 (central focus) vs accuracy | ρ = +0.28, *p* = .43 | **ρ = +0.55, *p* = .097** |

**Rest-vs-Walk is the stronger result despite the lower group mean**, on four
independent grounds: it is significant in 8 of 10 participants rather than 5;
its spread is narrower; it is the cleaner contrast physiologically (less
gamma-range contamination, more of its discriminability confined to mu/beta);
and it is the only contrast in the dataset for which the prespecified
physiological criterion predicts decoding performance at all.

There is also a specific problem with the Rest-vs-Stand ranking. Its highest
scorer is the participant whose Rest-vs-Stand effect is shown in Section III-E
to peak at 36 Hz with a flat topography (M4 = 1.78) — i.e. the top of that
leaderboard is muscle artifact. Rest-vs-Walk has no comparable dependence on
a contaminated participant.

We therefore report **Rest vs Walk as the primary positive finding** and
Rest vs Stand as a secondary one.

### D. Bounding the Stand-versus-Walk negative

Three independent analyses converge.

**(1) Exhaustive configuration search.** Fifteen epoch windows × seven bands
× two references × four pipelines = 840 configurations per participant. Each
participant was then allowed to select its single best configuration *scored
on the same folds being reported* — a deliberately optimistic procedure that
would be indefensible as a result. Per-participant maxima ranged
**61.5–75.0 % (mean 68.4 %)**. Seven of ten participants remain below the
~70 % accuracy commonly cited as the minimum for practical BCI control
[2], [3], and the three that clear it (75.0, 72.0, 71.7 %)
do so only under a procedure that chooses the configuration on the very folds
it is scored on — these are not accuracies obtainable prospectively
[`07_svw_focus.py`].

**(2) No temporal structure.** Standing-up imagery is a single ~2 s postural
transition whereas walking imagery spans the full cue, so the classes might
diverge late in the trial. Marginal accuracy across all fifteen windows
ranged 51.4–52.9 %; a late [3.5, 5.0] s window gave 52.4 %, no better than
the full period.

**(3) Effect size.** The grand-mean peak channel × frequency *r*² was 0.034
for Stand vs Walk, against 0.062 (Rest vs Stand) and 0.074 (Rest vs Walk).
The effect is physiologically coherent — a ~13 Hz ridge over C3/C4/CP4 with
higher power during Walk — but roughly an order of magnitude too small for
high-accuracy single-trial decoding [Fig. 4, `01_diagnose.py`].

### E. Screening an individual participant: an apparently significant result that is not neural

One participant (P06) reached 67.0 % on Stand vs Walk (permutation
*p* = .004, Holm-corrected *p* = .040), the only individually significant
case. Note that with ten participants at chance and *n* = 100 trials the
expected best-of-ten is ≈ 58 %, so this exceeds what selection alone
produces.

Four physiological checks nevertheless reject it [`10_s06_physiology.py`]:

**TABLE III — Physiological screen, participant P06, Stand vs Walk**

| Check | Result | Interpretation |
|---|---|---|
| Peak of *r*² map | FC4, **36 Hz**, *r* = .45 | EMG range |
| mu (8–13) topography | +.20 to +.26 on all eight electrodes, max/min = 1.27 | spatially uniform |
| gamma (30–45) topography | −.23 to −.32 on all eight, max/min = 1.41 | spatially uniform |
| Band ablation | mu 64 %, beta 67 %, gamma 63 %, delta/theta 62 % | every band equal |
| Peak *r*² by band | gamma 30–45 Hz **0.083** > mu 8–13 Hz 0.072 > beta 0.026 | maximal where SMR is absent |
| Train file 1 → test file 2 (46 min later, same cap) | 76.2 % → **60.0 %** (*p* = .25) | not stable across the gap |

Genuine lower-limb sensorimotor rhythm is focal at Cz/CPz with a clear
spatial gradient and confined to mu/beta. These criteria are validated
independently in Section III-I, where the same screens classify executed
movement as non-cortical and a post-movement beta-rebound test — which the
screens do not use — reaches the same verdict. What P06 shows is a globally
uniform alpha increase plus a globally uniform 30–45 Hz decrease during Walk,
decodable equally well from delta/theta — where no sensorimotor rhythm exists
— and not reproducible 46 minutes later in the same recording appointment.
The gamma result is conservative: 30–45 Hz lies in the amplifier's roll-off
(Section II-C), so the 63 % obtained there was reached *through* instrumental
attenuation, and the underlying muscle activity is correspondingly stronger
than the band-limited figure suggests.
That last point is worth stating precisely: because the cap was never removed
between the two files, the collapse from 76.2 % to 60.0 % cannot be attributed
to a change in electrode position or impedance, and must reflect the signal
itself. This is a broadband state difference: the participant was globally
more relaxed while imagining walking than while imagining standing.

**This generalises.** Applying the prespecified criterion to all ten
participants, none of the four metrics predicted Stand-vs-Walk accuracy
(M1 ρ = +.13, M2 ρ = −.01, M3 ρ = +.19, M4 ρ = +.16; all *p* > .60), and M4
ran *backwards*: greater EMG contamination accompanied higher accuracy. The
two best decoders were the two most broadband participants, while the
participant with the cleanest sensorimotor signature sat at chance
[`11_neuro_criterion.py`]. For Rest vs Walk the criterion behaved better
(three of four metrics in the predicted direction; central focus ρ = +.55,
*p* = .097).

*With n = 10 and twelve correlations none reaches p < .05; these are
descriptive. The load-bearing claim is the absence of support together with
the sign of M4.*

### F. Manipulation check

Relative to the pre-trial fixation, all three conditions showed −1.3 to
−2.9 dB, replicating the standard ERD phenomenon [4]. However, against the
**visually matched Rest condition**, motor imagery produced a mu power
*increase* of +0.55 dB (Stand, *p* = .014) and +0.77 dB (Walk, *p* = .008),
uniform across electrodes, with no beta effect; only 1/10 participants showed
the expected desynchronisation [`12_manipulation_check.py`].

Direction, topography and band are all inconsistent with sensorimotor
activation. We attribute this to the Rest instruction, which demanded active
visual fixation while the imagery conditions implicitly invited defocusing —
a manipulation that raises alpha globally.

This bears on Q1 beyond the attribution question. A decoder that separates
rest from imagery by visual engagement has no path to deployment: an
exoskeleton user is not fixating a monitor, and the confound that carries the
65 % here would be absent. The Rest-versus-imagery result should therefore be
read as an **upper bound obtained under a favourable confound**, not as an
estimate of what the same montage would achieve in use. Section IV-E states
the paradigm change this requires.

### G. Ceiling analyses

**Trials.** Accuracy rose +5.5 points per decade of training trials for Stand
vs Walk (95 % CI +2.1 to +8.8) and +9.0 for three-class (CI +6.5 to +11.5).
Doubling the session is therefore worth ≈ +1.7 and +2.7 points respectively.
Extrapolating log-linearly from the largest measured point, reaching the
~70 % usability threshold would require **≈ 10 200 trials per class** for
Stand vs Walk and **≈ 13 800** for three-class [`06_ceiling.py`].

*These two figures extrapolate roughly 200-fold beyond the observed range
(10–50 trials per class) and should be read as an order-of-magnitude
statement, not an estimate. Log-linear growth is assumed, not verified, and
empirical learning curves typically saturate rather than continue — which
would make the true requirement larger, not smaller. The measured quantity,
and the one the argument rests on, is the slope itself: +5.5 points per decade
(CI +2.1 to +8.8), against a 17-point gap to threshold.*

**Electrodes.** Under greedy forward selection with the subset chosen inside
the training fold, accuracy saturated at three electrodes (Stand vs Walk:
52.0 → 53.3 → 51.5 % for one, three and eight electrodes; three-class:
41.7 → 47.1 → 45.8 %). The most frequently selected first electrodes were
CP3/CP4/CPz. The remaining five electrodes of this montage add nothing — they
lie within one spatial correlation blob.

### H. Physiological account of the null: why these recordings contain nothing to decode

Sections III-B to III-D establish that Stand and Walk cannot be *decoded*.
This section addresses the prior question — do they produce different patterns
on the scalp at all? — and then asks what about the montage prevents the
difference from being resolved.

**H.1 The recorded topographies are not distinguishable from the noise floor.**
The tempting comparison, correlating the Stand-minus-Rest topography with the
Walk-minus-Rest topography, is invalid: both contain the term −Rest and are
correlated by construction. Measured here, that comparison returns r = +0.67,
but re-referencing the same data moves it to +0.47 (Stand as baseline) and
+0.19 (Walk as baseline), so it is dominated by the shared baseline.

Instead each contrast is compared against that participant's own noise floor
with matched sample size. Trials of one condition are split at random into two
halves; the distance between those two half-topographies is pure noise. The
between-class distance uses halves of the same size, so the noise contribution
is identical by construction and no baseline is shared. Ratio = 1.00 means the
two conditions differ no more than two random halves of a single condition do
(200 splits per participant) [`19_topography_floor.py`].

**TABLE IIIa — Topographic distance relative to the within-condition noise floor**

| Contrast | mu 8–13 Hz, ratio [95 % CI] | *p* | beta 13–30 Hz, ratio [95 % CI] | *p* |
|---|---|---|---|---|
| Rest vs Walk | **1.78** [1.18, 2.38] | **.023** | 1.29 [1.01, 1.57] | .079 |
| Rest vs Stand | 1.51 [0.85, 2.18] | .118 | 1.39 [0.80, 1.98] | .205 |
| **Stand vs Walk** | **1.12** [0.70, 1.55] | .463 | **1.07** [0.83, 1.31] | .683 |

*Intervals are 95 % CIs on the mean ratio across the ten participants; the
*p*-values come from a paired test on the underlying between- and
within-condition distances. The two address slightly different nulls (ratio
= 1 versus difference = 0), which is why the Rest-vs-Walk beta interval
excludes 1.0 while its paired *p* is .079; we treat that cell as
inconclusive. Only the Stand-vs-Walk intervals are centred on the noise floor
in both bands, and only Rest-vs-Walk in mu is unambiguous by both criteria.*

The ordering reproduces the decoding results exactly (8/10, 5/10 and 1/10
individually significant), from a measurement that uses no classifier and no
cross-validation. Six of ten participants have a Stand-vs-Walk ratio below
1.0, i.e. the between-class distance is *smaller* than their own noise. The
single participant above 2.0 is the one whose effect Section III-E identifies
as arousal and muscle tone.

**This locates the failure.** In these recordings, standing and walking
imagery do not produce different scalp patterns in the first place. The
failure is therefore not a limitation of the decoder or of the trial count at
the decoding stage; there is no pattern difference for a decoder to find.

**Rival explanations, and which of them survive.** "No pattern difference was
recorded" is initially consistent with three claims: that the recording was
unusable, that standing and walking imagery are not separable on this montage,
and that these ten participants did not generate separable imagery.

The first is excluded by the positive control of Section III-I: on the same
electrodes, amplifier and decoder, executed movements separate at 91 % in every
participant, so no result here can be attributed to a broken pipeline or an
unusable recording.

The third cannot be excluded. Participants were not screened for imagery
ability, received no feedback training, and completed no imagery-vividness
questionnaire, so task compliance was never measured; the manipulation check
in Section III-F is the only available probe and only 1/10 showed the expected
desynchronisation. Motor execution demonstrates that a participant can perform,
not that they imagined. We therefore cannot separate a montage that does not
resolve the difference from participants who did not produce one, and Section
IV-E lists the design changes that would.

**H.2 The montage is nonetheless pointed at the right place.** Single-electrode
decoding ranks the medial sites above the frontolateral ones in every contrast
(Cz 58.1 % and CPz 57.3 % highest, C4 54.8 % and FC4 54.4 % lowest, averaged
across contrasts) — the ordering expected of a medial foot representation, not
of a lateral generator.

**H.3 Adding explicit anatomical constraints does not help, but misdirecting
them costs.** We compared the generic pipeline against four physiologically
constrained variants: a foot-area surface Laplacian (Cz − mean(C3, C4, CPz)
and the corresponding derivations at CPz, C3, C4), individually estimated mu
and beta bands (label-free spectral peaks; mu 9.0–13.0 Hz and beta
15.0–28.5 Hz across participants, against a textbook 8–13 / 13–30 for
everyone), CSP restricted to the four medial derivations, and explicit named
band-power features. A fifth variant is the control: the identical machinery
with the Laplacian ring centred on the *frontolateral* electrodes instead,
where lower-limb imagery has no anatomical reason to be
[`18_physio_classifier.py`].

**TABLE IIIb — Physiologically constrained variants (5-fold, identical folds)**

| Model | Rest vs Stand | Rest vs Walk | Stand vs Walk |
|---|---|---|---|
| A generic (as used throughout) | 62.7 ± 9.8 % | 63.5 ± 9.7 % | 54.3 ± 5.2 % |
| B + foot-area Laplacian | 62.8 ± 11.6 | 65.2 ± 10.7 | 55.3 ± 7.5 |
| C + individual SMR bands | 62.2 ± 12.8 | 62.5 ± 9.4 | 53.3 ± 5.3 |
| D full physiological constraint | 63.7 ± 11.4 | 62.8 ± 8.0 | 53.0 ± 8.6 |
| **E same machinery, wrong anatomy** | **59.6 ± 10.5** | **59.0 ± 10.3** | **54.6 ± 6.3** |

*Mean ± SD across the ten participants, identical folds for every model, so
the model contrasts are paired; the between-model tests quoted below are
paired *t*-tests over those ten values. The standard deviations are large
relative to the between-model differences — no model is distinguishable from
another on the group means alone, which is the point of using the paired
tests.*

No constraint improves significantly (B–D vs A, all *p* > .37); individually
estimated bands slightly *hurt*, the fine filter bank having already absorbed
that adaptation. Two things follow.

First, the generic pipeline is not arbitrary with respect to physiology — it
already extracts what the foot-area constraint would impose, which is why
adding the constraint changes nothing.

Second, the wrong-anatomy control costs **4.5 points on Rest-vs-Walk
(*p* = .019)** and **3.1 points on Rest-vs-Stand** (*p* = .267), but nothing at
all on Stand-vs-Walk (**−0.4 points, *p* = .804** — nominally an
*improvement*). The same misdirection is expensive for the contrasts that
carry signal and free for the one that does not.

*This comparison is suggestive rather than probative, for two reasons. First,
a floor effect: Rest-vs-Walk sits 13.5 points above chance and Stand-vs-Walk
only 4.3, so there is roughly three times as much signal available to destroy
in the former, and part of the asymmetry follows from that alone. Taken by
itself the argument would be circular — a spatial constraint can only be
misdirected away from something, so "misdirection is free" and "there is no
localised signal" are close to the same statement rather than independent
evidence for each other. Second, this montage places all eight electrodes over
sensorimotor cortex, so the "wrong" ring is only mildly wrong; no occipital or
prefrontal site is available. We report the control because it is consistent
with H.1 and H.4, not because it independently establishes them.*

**H.4 The montage measures far fewer spatial dimensions than it has
electrodes.** H.1 shows that no distinguishable topographic difference is
present, but on its own it cannot separate two readings: that no difference
exists in the brain, or that one exists and this montage cannot resolve it.
The second reading is addressable directly.

Eigendecomposing each participant's 8×8 band-limited covariance:

| | mean across participants |
|---|---|
| Components for 95 % of variance | **3.4** of 8 |
| Effective dimensionality (participation ratio) | **1.49** |
| Mean inter-electrode correlation | **0.783** |

Eight electrodes span roughly **one and a half** effective spatial dimensions.
Neighbouring sites correlate at 0.78, i.e. they largely report the same
quantity. This is an independent route to the same conclusion as the greedy
channel curve in Section III-G, which saturated at three electrodes: one
measurement is classification accuracy, the other is the eigenvalue spectrum
of the covariance, and they agree.

The montage is therefore spatially saturated — adding more electrodes *at this
spacing* cannot add information, which is what Section III-G measured. Whether
a denser medial ring at ~2 cm would help is a different question that this
dataset cannot answer, because those electrodes were not recorded; see the
spacing argument in Section IV-A and the limitation in Section IV-D.

**H.5 Removing the muscle does not recover a signal — and trying it with a
decomposition destroys the signals that exist.** H.1 shows no topographic
difference and H.4 shows the montage is spatially saturated, but neither rules
out a third possibility: that a real difference is present and buried under
electromyographic noise. Because the analysis band overlaps a genuine EMG
component (Section II-E), this had to be tested rather than argued.

Independent component analysis is not the tool. It decomposes an *N*-channel
recording into *N* components, and H.4 measures the effective dimensionality
of this montage at 1.49; a one-and-a-half dimensional space cannot be split
into separable neural and muscular subspaces. We therefore used two methods
that do not require a high channel count [`22_emg_removal.py`]. The first
projects a per-trial 30–45 Hz nuisance direction out of the tangent-space
features, with the regression fitted inside the training fold. The second is
BSS-CCA: canonical correlation between the signal and its one-sample lag
orders components by autocorrelation, and the two least autocorrelated — the
most muscle-like — are zeroed before reconstruction. Every outcome was
assigned an interpretation before running.

**TABLE IIIc — Effect of EMG removal (single-band pipeline, identical folds)**

| Contrast | raw | nuisance regression | BSS-CCA |
|---|---|---|---|
| Stand vs Walk | 50.9 % | 50.8 % (Δ −0.12, *p* = .938) | 47.1 % (Δ −3.88, *p* = .180) |
| Rest vs Walk | 62.9 % | 62.8 % (Δ −0.10, *p* = .938) | **53.8 % (Δ −9.08, *p* < .0001)** |
| Rest vs Stand | 62.9 % | 61.4 % (Δ −1.57, *p* = .427) | **53.6 % (Δ −9.29, *p* = .055)** |
| Rest-vs-Walk topographic ratio | 1.79 | — | **1.25 (*p* = .032)** |
| Participant P06, Stand vs Walk | 66.0 % | **61.0 %** | **47.0 %** |

*This stage uses a single 8–30 Hz covariance rather than the filter bank, so
its absolute values are a few points below Table I. Only the within-stage
raw-versus-cleaned comparison, which uses identical folds, is interpreted.*

Three results follow, and they fall on three different branches of the
pre-registered table.

**Nuisance regression changes nothing, which is informative.** Stand-vs-Walk
moves by −0.12 points (*p* = .938) and neither rest contrast moves either. Had
the null been an SNR problem, removing a variance source correlated with
muscle should have improved it. It does not. The binding constraint is
information content, not noise — the same conclusion the flat learning curve
of Section III-G and the three-electrode saturation reach by other routes.

**P06 falls under both methods**, from 66.0 % to 61.0 % and 47.0 %. That
participant is the one Section III-E rejects as arousal and muscle tone, and
this is the prediction that rejection makes: clean the muscle and the effect
should go. It does.

**BSS-CCA over-removes, and the fact that it does is itself a result.**
It costs Rest-vs-Walk 9.1 points (*p* < .0001) and Rest-vs-Stand 9.3, and it
collapses the Rest-vs-Walk topographic ratio from 1.79 to 1.25 (*p* = .032) —
that is, it destroys the one contrast this montage demonstrably resolves.
Its Stand-vs-Walk output therefore cannot be interpreted. The reason is the
dimensionality of H.4: discarding two of eight components removes a quarter of
the recorded space, and when that space spans 1.49 effective dimensions there
is no artifact subspace to discard that does not also contain the signal.
**This is a measured demonstration that decomposition-based artifact removal
is not available at this channel count**, which the ICA literature asserts on
theoretical grounds and which practitioners of low-channel systems should
treat as a hard constraint rather than a preference.

### I. Positive control: what this montage registers when the signal is unambiguous

Every claim above is conditional on a premise nothing so far has tested — that
this montage can register lower-limb sensorimotor activity at all, and that the
M1–M4 screens recognise it when present. If it cannot, the null of Sections
III-D and III-H is an instrument result rather than a physiological one. The
motor-execution runs supply the control: participants actually moved, so
genuine sensorimotor activity is present by construction. Nine of ten runs
passed a pre-set trial-retention gate of 70 % [`20_me_positive_control.py`,
`21_me_discriminate.py`].

Decoding accuracy would not answer the question. Real movement produces
cortical ERD, limb and trunk EMG, and motion artifact, and all three separate
task from rest, so a high accuracy would prove nothing about cortex — the trap
Section III-E documents. We therefore tested the *signature* rather than the
separability, on three properties that distinguish a cortical from a peripheral
source, with every outcome specified in advance.

**TABLE V — Motor-execution positive control**

| Test | Prediction if cortical | Measured | Verdict |
|---|---|---|---|
| Task vs rest, mu 8–13 Hz | decrease | **−2.83 dB**, *p* = .0015 (−3.69 under CAR, *p* = .0003) | consistent |
| Confined to mu/beta | gamma unaffected | gamma 30–45 Hz **−2.23 dB**, *p* = .0017 | **fails** |
| Medial focus (foot area) | Cz/CPz > FC3/FC4 | medial − lateral **+0.56 dB**, *p* = .299 | **fails** |
| **T1** mu-specific trough | trough at individual mu | **−1.56 dB** below the gamma level, *t* = −4.16, *p* = .0032 | consistent |
| **T2** post-movement beta rebound | rise after movement offset | beta **+2.70 dB during** movement (*p* = .0046) and **−1.67 dB after** (*p* = .0119) — the opposite | **fails** |
| **T3** executed TaskA vs TaskB | — | **91.1 %** (range 71.9–100), *t* = +14.27, *p* < .0001 | see below |

Three findings follow, and together they resolve the interpretation of the
whole paper.

**The recording chain works.** Executed movements separate at 91 % in every
participant. Electrodes, amplifier, synchronisation, epoching and decoder are
all functional, so no result in this paper can be attributed to a broken
pipeline or an unusable recording. This is the single most useful thing the
control establishes.

**What separates them is peripheral, not cortical.** T2 is decisive. Genuine
motor execution produces beta desynchronisation *during* movement and a beta
rebound *after* offset; the rebound is a cortical signature that muscle cannot
imitate, because muscle stops when movement stops rather than overshooting
upward. We measured the reverse: beta rises 2.70 dB during movement and decays
afterwards, which is the time course of electromyogram. The 91 % of T3 is
therefore a peripheral result, and the failure of the medial-focus and
band-confinement tests follows from the same source.

**A weak cortical component is nonetheless present.** T1 finds a mu-specific
trough of 1.56 dB *below* the broadband shift (*p* = .0032), so the change is
not spectrally flat. Cortical activity is measurable on this montage; it is
simply small relative to whatever peripheral signal accompanies it.

*One quantity had to be reconciled. Against the rest condition the executed
task shows a beta* decrease *(−2.10 dB), but against the within-trial pre-cue
baseline it shows an* increase *(+2.70 dB). Both are correct: relative to their
own pre-cue baselines the rest condition rises +3.78 dB (p = .0023) and the
task conditions +2.70 dB (p = .0046), so the ME rest condition is not a quiet
baseline and the negative difference reflects rest rising more. We report the
decomposition rather than the difference alone.*

**This validates the M1–M4 screens rather than undermining them.** The screens
classified the execution data as non-cortical, and T2 independently confirms
that classification by a criterion the screens do not use. The concern that
Section III-E rejected participant P06 on criteria too strict to admit genuine
motor activity is therefore not supported: the criteria were correct, and the
signature they demanded is genuinely absent even when movement is real.

### J. Exploratory analysis — peak laterality in the beta band

*This subsection is exploratory and is reported as such. It does not meet the
significance criterion once the search that produced it is accounted for
(below), and no claim in the Abstract or Conclusion rests on it.*

**Motivation.** The compound-limbs manipulation adopted from [1] pairs a
right-foot step with a LEFT-arm swing, which predicts a right-hemisphere
dominant response. Section III-D showed no such asymmetry in the trial
average, but a trial average cannot see it: the instruction specifies two
steps, so both compound configurations occur inside one trial and cancel.

**Method.** For each trial independently, and without reference to its label,
we located the instant of maximum laterality — max |LI(t)| where
LI = 10·log10(right-hemisphere over left-hemisphere band power, over
{FC4,C4,CP4} vs {FC3,C3,CP3}) — and measured the signed laterality there.
The identical procedure was applied to all three conditions.

**Observations** (beta 13–30 Hz; the same analysis in mu did not survive the
jackknife below and is not reported) [`14_peak_compound.py`,
`15_peak_locked.py`, `16_p1b_robustness.py`]:

| | Walk − Stand | *p* |
|---|---|---|
| Signed laterality at the peak instant | +1.02 dB | .083 |
| FC3/FC4 pair, own peak selection | +0.32 dB | .262 |
| C3/C4 pair | +0.66 dB | .063 |
| CP3/CP4 pair | +1.03 dB | .083 |
| Decoding from peak-derived features | 56.1 ± 5.5 % | **.0084** vs chance |

Two properties are worth noting. All three symmetric electrode pairs move in
the predicted direction with a monotonic posterior gradient, which random
variation does not usually produce. And the decoder survives removal of any
single participant (*p* = .003–.021 across the ten jackknife samples),
including removal of the participant whose Stand-vs-Walk effect was shown in
Section III-E to be arousal and muscle tone (*p* = .020 without them).

**Why this is nonetheless not reported as a result.** Three reasons.

1. **The search.** Approximately ten distinct hypotheses were tested against
   the Stand-vs-Walk contrast in this dataset — envelope-modulation cadence,
   late-window divergence, an 840-cell preprocessing grid, participant
   experience, a physiological-plausibility criterion, alternating laterality
   (twelve statistics), peak-laterality magnitude, peak-laterality direction,
   adaptive-window decoding, and peak-locked decoding — and each was tried
   only after the previous one failed. Under a Bonferroni correction for ten
   comparisons the criterion is *p* < .005; the strongest result above,
   *p* = .0084, does not clear it.
2. **No improvement over the existing pipeline.** 56.1 % versus 52.2 % for
   the standard decoder on the same trials, paired *p* = .085, and part of
   that margin comes from one participant whose baseline decoder failed
   pathologically (32 %).
3. **No participant is individually significant** after Holm correction
   (0/10; 4/10 uncorrected).

**Sample size required.** From the observed effect size (Cohen's *d* ≈ 0.62),
a pre-registered replication would need approximately **21 participants** for
80 % power, or ~33 for the weaker mu-band estimate. Because *d* is estimated
from the analysis that selected this hypothesis, it is biased upward by the
winner's curse, so 21 is a **lower bound** on the required sample rather than
a target; a replication should be powered on the lower confidence limit of
*d*, not its point estimate. The present *n* = 10 is in either case below the
threshold at which this question can be settled.

---

## IV. Discussion

### A. Why standing versus walking fails

Every classification task in Ma *et al.* [1] is a **left-versus-right**
contrast: left arm vs right arm, left leg vs right leg, and left-leg +
right-arm vs right-leg + left-arm. The decoded signal is contralateral ERD
asymmetry between C3 and C4. The authors report explicitly that lower limbs
alone barely separate — "no obvious difference between tasks of the single
left leg and right leg" — and that fewer channels exceeded their
discriminability threshold in the single-lower-limb paradigm than in either
the single-upper-limb or the compound-limbs paradigm, for most subjects. The
compound-limbs manipulation exists precisely to *import* the arm's
lateralisation into a lower-limb task, which is what yields their reported
+11.02 points.

**The present design does contain the mirrored pair — but inside a trial
rather than between classes.** This distinction is the whole of Section IV-A
and it is easy to state imprecisely. Ma *et al.* place the two mirrored
configurations in *separate classes*, so the C3/C4 contrast is one sign in
class A and the other in class B and survives trial averaging. The walking
instruction here specifies **two steps** — right foot with left arm, then left
foot with right arm — so both configurations occur **within a single Walk
trial**, in sequence. Stand, being bilateral and armless, has no such
alternation.

The consequence is arithmetic. A laterality index averaged over the analysis
window cancels the two phases against each other by construction. The measured
Walk-minus-Stand difference of **+0.026 dB in mu (*p* = .729)** and
+0.133 dB in beta (*p* = .142) is therefore consistent with *"the asymmetry
alternates and cancels"* as much as with *"there is no asymmetry"*, and cannot
distinguish them.

**We therefore tested the alternation directly**, with six statistics that
relax the timing assumption in stages — a fixed mid-trial switch, the
within-trial standard deviation of the laterality index, decoding from its
trajectory, the largest step-like transition at any instant, the number of
crossings of the trial mean, and the power of the laterality fluctuation in
0.2–1.0 Hz. The last four assume nothing about when or how often the switch
occurs [`13_time_laterality.py`]. Of the twelve tests (six statistics × two
bands), one survives Holm correction: the beta-band crossing count, at
−0.242 (*p*_Holm = .039) — **in the direction opposite to the registered
prediction**. The closest result in the predicted direction is the largest
step-like transition in mu, +0.131 (*p* = .056, not significant after
correction). The peak-locked analysis of Section III-J is the strongest
version of this family and reaches +1.02 dB with 56.1 % decoding
(*p* = .0084), which does not clear the multiple-comparison threshold either.

**What this establishes is narrower than a null.** The benefit reported by [1]
depends on a *between-class* mirrored contrast, and that this design does not
contain; that part is settled. Whether the *within-trial* alternation carries
decodable information is not settled: the largest laterality effect measured
anywhere in this dataset is the +1.02 dB of Section III-J, roughly forty times
the trial-averaged value, and it fails on multiple-comparison and sample-size
grounds rather than on effect size. A study that pre-registered a single
alternation statistic and recruited the ~21 participants of Section III-J
could settle it; the present one cannot.

### B. One mechanism accounts for every result

Taken together the results do not require four separate explanations. A single
statement covers them:

> **This montage is dominated by peripheral and global signals wherever they
> exist, and its cortical component is weak.**

| Result | Peripheral or global signal available? | Outcome |
|---|---|---|
| Executed TaskA vs TaskB (§III-I) | large — limb and trunk EMG, motion | **91.1 %** |
| Rest vs imagery (§III-B, §III-F) | moderate — a global visual-attention difference | **64–66 %** |
| Participant P06, Stand vs Walk (§III-E) | moderate — broadband arousal and muscle tone | **67.0 %** |
| Stand vs Walk, group (§III-B) | none — both classes are motionless and visually identical | **53.1 %** |

The ordering of the four accuracies is the ordering of the peripheral signal
available in each contrast, not the ordering of the cortical difference the
task was designed to elicit. Section III-I shows the cortical component is
measurable but small — a mu-specific trough 1.56 dB below the broadband shift
during real movement — which is consistent with the effective dimensionality of
about 1.5 reported in Section III-H.4.

This is the paper's central practical claim, and it is the reason a permutation
*p*-value is not sufficient evidence for a low-channel BCI result. Three of the
four rows above are statistically significant; only the first two would
normally be reported; and none of them is decoding what the paradigm intended.

### C. Instrumentation

**TABLE IV — Recording comparison**

| | Ma *et al.* [1] | Present study |
|---|---|---|
| Electrodes | 56 | 8 |
| EOG / sEMG | 4 / 4 | none |
| Artifact handling | ASR → CAR → ICA guided by EOG/sEMG → ICLabel | amplitude and variance rejection |
| Environment | shielded room | ordinary laboratory |
| Sampling | 1200 Hz | 256 Hz |

The artifact row bears directly on Sections III-E, III-F and III-I: [1] could
identify and remove ocular and muscular components because they were recorded
on dedicated channels. With eight EEG channels and no EOG or EMG reference,
the present dataset offers no such route, which is why the confounds
documented here are uncontrolled rather than removed.

### D. Methodological implications

Three cautions, each with a worked example in this dataset.

1. **Epoch windows that precede imagery onset import the cue.** The
   commonly used −1 s pre-onset baseline overlaps a class-specific visual
   stimulus here; the cue period alone decoded at 52–56 %.
2. **Uncorrected paired tests over CV folds overstate pipeline
   differences.** The naive test gave *p* = .046 where the corrected
   resampled test gave *p* = .247.
3. **Accuracy plus a permutation *p*-value is insufficient evidence for a
   BCI claim on a low-channel montage.** Section III-E is a case in which a
   Holm-corrected significant result proved to be arousal and muscle tone.
   We recommend reporting spatial gradient, rhythm specificity, and
   stability across recording gaps alongside accuracy.

### E. Limitations

- Ten participants; all correlational analyses in Section III-E are
  descriptive, not inferential.
- **Number of hypotheses tested.** Roughly ten distinct hypotheses were
  evaluated against the Stand-vs-Walk contrast (enumerated in Section III-J),
  each attempted after the preceding one failed. That asymmetric stopping
  rule inflates the false-positive rate of any single reported comparison,
  and readers should discount Section III-J accordingly. It does not affect
  the confirmatory results in Sections III-B to III-D, whose protocol was
  fixed in advance; if anything an unsuccessful search of this size
  strengthens a null rather than weakening it.
- **No non-compound walking condition.** Every walking trial carried the arm
  swing, so the contribution of the compound-limbs instruction cannot be
  isolated even in principle. Section IV-A argues from the *absence of the
  mechanism* — a measured laterality difference of +0.026 dB where the
  mechanism requires a large one — not from a compound-versus-simple
  comparison, which this design cannot support.
- **Task compliance was never measured.** Participants were not screened
  for imagery ability, given feedback training, or asked to rate imagery
  vividness. The null in Section III-H is therefore consistent with an
  untrainable montage, with genuinely inseparable imagery, and with
  participants who did not perform the task; the present design cannot
  distinguish these.
- No EOG or EMG channels, so artifact contributions are *identified* by
  topography and spectrum but cannot be *removed*.
- Class-to-marker assignment follows the experimenters' protocol
  documentation. An independent reading of the within-trial marker ordering
  is possible; every per-class statement is conditional on the assignment
  used here.
- The M1–M4 thresholds distinguishing "cortical" from "artifact" are
  criteria we defined from motor-imagery physiology, not established
  standards.
- The manipulation check indicates that the Rest-versus-imagery contrast is
  confounded with visual attention. The ~65 % results are valid as decoding
  but should not be attributed specifically to motor imagery.

### F. Recommendations

For a follow-up with the same hardware, the highest-value change is to the
paradigm rather than the equipment. The present montage contains three
symmetric lateral pairs (FC3/FC4, C3/C4, CP3/CP4) and is therefore
reasonably suited to a *lateralised* contrast; it was simply applied to a
non-lateralised one. Replacing standing-versus-walking with the mirrored
compound-limbs pair of [1] would test the demonstrated mechanism on
low-cost hardware — a stronger result than the original design could yield,
since the question becomes whether the effect survives a 56→8 channel
reduction.

Further: equalise cognitive and visual demands between rest and imagery
conditions, record at least one EOG derivation, and include feedback
training, since these participants performed open-loop imagery without
screening.

---

## V. Conclusion

An 8-channel wearable montage detects rest against lower-limb motor imagery at
approximately 65 % within participant, individually significant in 8 of 10
participants for Rest versus Walk — though a manipulation check against the
visually matched rest condition means this separation cannot be attributed
specifically to motor imagery rather than to visual attention, and would not
be expected to survive deployment outside a fixation task.

It does not separate standing from walking imagery. At the participant level
that contrast is not significantly above chance (53.1 %, *p* = .246, 95 % CI
[47.5, 58.7]) and is significantly below the accuracy a device would require
(*p* = 4.1 × 10⁻⁵). The bound is established four ways — an 840-configuration
search, a learning curve worth +1.7 points per session doubling, a channel
curve saturating at three electrodes, and topographies indistinguishable from
each participant's own noise floor — and the single apparently significant
participant proved to be decoding arousal and muscle tone.

Two boundaries on that conclusion should be carried forward. The compound-limbs
paradigm's benefit requires the mirrored configurations to occupy different
classes; this design places both inside a single walking trial, so a
trial-averaged contrast cancels them and this study does not test whether that
benefit survives a channel reduction. Whether the within-trial alternation is
itself decodable is left open: it produced the largest laterality effect
measured here (+1.02 dB) but fails multiple-comparison correction at *n* = 10. And because task compliance was never measured,
the null is consistent with an unresolvable montage, with genuinely
inseparable imagery, and with participants who did not perform the task; the
present design cannot distinguish them. What the study establishes is that
this class set, on this hardware, with untrained participants, does not
support the discrimination an exoskeleton would need — and it identifies the
paradigm and screening changes that a study capable of settling the question
would require.

A positive control on the same hardware sharpens that statement. Executed
movements separate at 91.1 %, so the recording chain is sound; but the
separation is peripheral rather than cortical, and every accuracy in this paper
orders by the peripheral signal a contrast contains. A low-channel wearable
montage of this kind is best understood not as a weak cortical sensor but as
one that reports whatever non-cortical difference is available — which is why
significance testing alone cannot establish a BCI claim on such hardware.

---

## Figures

Three figures are drawn specifically for the paper and are **vector**
(`.pdf`, Type-42 editable text): `fig_paradigm`, `fig_pipeline`,
`fig_summary`. Regenerate any of them with `python fig_<name>.py`. The
remaining figures are 200 dpi PNG from the analysis stages; change the
`savefig` extension in their scripts if the journal requires vector
throughout.

| # | File | Caption |
|---|---|---|
| 1 | **`fig_paradigm.pdf`** | **Trial structure.** 3.00 s fixation → 2.00 s class-specific cue → 5.00 s imagery on a blank screen, redrawn from the PsychoPy builder flow and confirmed against the recorded marker stream (±0.02 s across 13 files). Only the cue differs between classes; the imagery screen is identical, which is what makes Rest a visually matched baseline. Time axis is referenced to imagery onset. |
| 2 | **`fig_pipeline.pdf`** | **Decoding pipeline.** Each stage with its data shape and the property of the recording that motivates it. The shaded region marks everything refit inside the cross-validation training fold. |
| 3 | **`fig_summary.pdf`** | **Results in one panel.** (A) nested accuracy per contrast with individual participants and the number individually significant; (B) Stand-vs-Walk, nested versus the deliberately optimistic best-of-840 search; (C) trial and electrode ceiling curves; (D) band ablation showing the one significant participant decodes equally well from bands containing no sensorimotor rhythm. |
| 4 | `fig_confusion_3cls.png` | Per-participant and pooled three-class confusion matrices. All error mass lies in the Stand–Walk block. |
| 5 | `fig_subject_significance.png` | Per-participant accuracy against a 1000-permutation label null, four contrasts |
| 6 | `fig_rsq_Stand_vs_Walk.png` | Channel × frequency discriminability for the hard contrast |
| 7 | `fig_svw_window_scan.png` | 840-configuration window × band scan (exploration bound) |
| 8 | `fig_s06_physiology.png` | Full physiological screen rejecting the one significant participant |
| 9 | `fig_manipulation_check.png` | MI versus the visually matched Rest condition |
| 10 | `fig_ceiling_trials.png`, `fig_ceiling_channels.png` | Trial and electrode ceiling curves, full detail |
| 11 | `fig_erd_bands.png`, `fig_erds_grand.png` | ERD/ERS against the pre-trial fixation baseline |
| 12 | `fig_quality.png` | Per-channel amplitude and amplifier saturation |
| 13 | `fig_neuro_criterion.png` | Prespecified physiological criterion versus decodability |
| 14 | `fig_me_positive_control.png` | Motor-execution positive control: is the task-vs-rest change a cortical ERD (negative, band-limited, medial) or muscle (positive, broadband, peripheral)? |
| 15b | `fig_emg_removal.png` | Effect of EMG removal by nuisance regression and BSS-CCA on all three contrasts |
| 15 | `fig_me_discriminate.png` | The three discriminating tests — spectral shape, post-movement beta rebound, and separability of two *executed* movements |
| 16 | `fig_hardware_spectrum.png` | Unfiltered spectrum of all ten recordings to Nyquist: the 60 Hz mains line motivating the notch, and the slope change near 40 Hz identifying the amplifier's analogue low-pass |
| 17 | `fig_peak_locked.png` | Exploratory (§III-J): peak-locked laterality, topography and decoding |
| 18 | `fig_p1b_robustness.png` | Exploratory (§III-J): electrode-pair replication and jackknife |

For a four-page-limited submission, Figs. 1–3 alone carry the argument;
Figs. 4–18 are supplementary.

## References

[1] R. Ma, Y.-F. Chen, Y.-C. Jiang, and M. Zhang, "A new compound-limbs
paradigm: Integrating upper-limb swing improves lower-limb stepping intention
decoding from EEG," *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 31,
pp. 3823–3834, 2023, doi: 10.1109/TNSRE.2023.3315717.

[2] J. Perelmouter and N. Birbaumer, "A binary spelling interface with random
errors," *IEEE Trans. Rehabil. Eng.*, vol. 8, no. 2, pp. 227–232, 2000,
doi: 10.1109/86.847824.

[3] F. Nijboer, A. Furdea, I. Gunst, J. Mellinger, D. J. McFarland,
N. Birbaumer, and A. Kübler, "An auditory brain–computer interface (BCI),"
*J. Neurosci. Methods*, vol. 167, no. 1, pp. 43–50, 2008,
doi: 10.1016/j.jneumeth.2007.02.009.

[4] G. Pfurtscheller and F. H. Lopes da Silva, "Event-related EEG/MEG
synchronization and desynchronization: basic principles," *Clin.
Neurophysiol.*, vol. 110, no. 11, pp. 1842–1857, 1999,
doi: 10.1016/S1388-2457(99)00141-8.

[5] K. K. Ang, Z. Y. Chin, C. Wang, C. Guan, and H. Zhang, "Filter bank common
spatial pattern algorithm on BCI Competition IV Datasets 2a and 2b," *Front.
Neurosci.*, vol. 6, art. 39, 2012, doi: 10.3389/fnins.2012.00039.

[6] A. Barachant, S. Bonnet, M. Congedo, and C. Jutten, "Multiclass
brain–computer interface classification by Riemannian geometry," *IEEE Trans.
Biomed. Eng.*, vol. 59, no. 4, pp. 920–928, 2012,
doi: 10.1109/TBME.2011.2172210.

[7] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and
B. J. Lance, "EEGNet: a compact convolutional neural network for EEG-based
brain–computer interfaces," *J. Neural Eng.*, vol. 15, no. 5, p. 056013, 2018,
doi: 10.1088/1741-2552/aace8c.

[8] C. Nadeau and Y. Bengio, "Inference for the generalization error,"
*Mach. Learn.*, vol. 52, no. 3, pp. 239–281, 2003,
doi: 10.1023/A:1024068626366.

[9] I. I. Goncharova, D. J. McFarland, T. M. Vaughan, and J. R. Wolpaw, "EMG
contamination of EEG: spectral and topographical characteristics," *Clin.
Neurophysiol.*, vol. 114, no. 9, pp. 1580–1593, 2003,
doi: 10.1016/S1388-2457(03)00093-2.

[10] R. T. Schirrmeister *et al.*, "Deep learning with convolutional neural
networks for EEG decoding and visualization," *Hum. Brain Mapp.*, vol. 38,
no. 11, pp. 5391–5420, 2017, doi: 10.1002/hbm.23730.
