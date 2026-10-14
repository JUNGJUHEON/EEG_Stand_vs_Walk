# Independent code and artifact verification

Reviewed October 14, 2026 by a separate agent from the release implementer.
The review followed the installed claim-source-verification and
scientific-critical-thinking procedures. It is a bounded implementation and
artifact review, not an independent experiment, human scientific approval,
or a full raw-recording replay.

## Confirmed

- The released `FilterbankTripleShallowCNN` and `butter_bandpass_filter`
  definitions have ASTs exactly equal to the supplied original definitions.
  The fixed revalidation caller's five bands, eight channels, binary output,
  and dropout .3 are retained. The binary model has 38,002 parameters.
- Independent synthetic comparisons produced exactly equal float32
  filterbank and magnitude-STFT arrays, including the original per-channel
  STFT and public vectorized calculation. Two input trials gave shapes
  `(2,5,8,1281)` and `(2,8,65,20)`. Equal-weight original/public forward
  outputs were exactly equal and had shape `(2,2)`.
- The fixed split helper is AST-identical to the revalidation core. All
  three pairs reconstructed disjoint 60/20/20 original-trial sets with
  30/10/10 trials per class. Split seed 42 and class-block ordering are
  retained.
- Source review confirmed the fixed Filterbank optimizer, scheduler,
  batching, loss, early-stopping settings, and validation-selected state
  restoration. The training function accepts training and validation
  arrays/labels only. Frozen prediction accepts no truth labels and sets
  evaluation mode with gradients disabled. The checkpoint is saved before
  test inference; test truth is then used for final scoring. Stratified
  assignment's label use is distinct from fitting.
- The supplied original source specifies a five-fold 32-configuration grid,
  a final 72/8/20 split, and validation-accuracy stopping. The fixed release
  is explicitly narrower. This checks the source specification, not whether
  a historical full grid was executed.
- The original aggregate CSV contains all 90 unique participant/pair/seed
  keys. Independent arithmetic reproduced the participant-mean accuracy
  percentages for seeds 42/43/44: Rest-Stand 60.5/65.5/57.5,
  Rest-Walk 62.5/60.5/60.5, and Stand-Walk 54.0/55.0/55.5.
- One existing frozen checkpoint, S01/Rest-Stand/seed43, strictly loaded
  into the public model and was run on CPU without fitting. All twenty class
  predictions matched the saved MPS predictions. The maximum probability
  difference was `1.1920928955078125e-07`. Independently counted accuracy
  was .75 and pairwise tie-corrected AUC was .73. These are compatibility
  checks, not a selected representative result.
- On the matching S01 preprocessed cache, calculating fixed features on
  all 150 trials and then selecting a split was exactly equal to calculating
  them separately for each train/validation/test split. The transforms use
  no labels or cross-trial fitted statistics.
- All 90 public score rows agree with the original aggregate accuracy/AUC
  records, including derived correct counts and balanced accuracies. All
  30 participant summaries and 9 group summaries match independent
  calculations. The README's 90 participant accuracy cells and four
  highlighted S03/S06 ranges match the same source. The figure was visually
  checked for all ten panels, three contrasts, three seeds, and unclipped
  labels; exact numerical comparisons were made against the CSVs and README.

## Partial and blocked scope

- **PARTIAL:** Raw XDF extraction, retained-trial selection, continuous
  preprocessing, physical-unit calibration, and all other participants'
  signal transformations were not replayed in this release review.
  Shape/class-count validation cannot establish channel/sample-rate/trial
  provenance. The input contract in `METHODS.md` remains necessary.
- **PARTIAL:** This review replayed one of ninety checkpoint inferences and
  independently summarized the ninety-row source CSV. It did not repeat
  the other eighty-nine inferences or all ninety prediction rescoring
  operations. The earlier independent numerical review's broader checks
  are separate evidence and are not recast as newly executed checks here.
- **BLOCKED:** These checks do not identify an MI-specific neural mechanism
  or resolve cue, attention, ocular/muscle, and imagery-performance effects.
  They do not establish new-person, new-session, online, or prospective
  model-selection performance.

No actual-data training, supervised filter fitting, feature selection,
hyperparameter search, or rejected auxiliary evaluation was executed by
this verifier. The original model module was parsed rather than executed
at top level. The implementation's bounded synthetic training test was not
part of this independent inference-only replay. Recorded real-data
training and the October 14 release are separate events.

The complete local review records fourteen explicit scopes: eleven CONFIRMED,
two PARTIAL, one BLOCKED, and zero REJECTED. These are review outcomes,
not a quality score. NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.4.2, and
PyTorch 2.5.1 matched the pinned requirements; CPU inference used two
torch threads.

## Procedural reference

The scientific-critical-thinking procedure's requested attribution is
Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026),
*Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
The current arXiv metadata was checked on the review date. This is a
procedural reference, not an external source for the EEG findings or an
endorsement of this study.
