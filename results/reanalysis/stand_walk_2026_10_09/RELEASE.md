# October 10 publication record

This folder publishes the October 9 exploratory analysis of the same previously inspected recordings. Original scripts and results outside the new subfolders are preserved. The only existing tracked file edited for this release is the root README.

## Included

- Six analysis scripts and pinned computational dependencies under `code/reanalysis/stand_walk_2026_10_09/`.
- Participant-pseudonym aggregate scores, nested choices, five-seed stability summaries, signal summaries/participant means, matched controls, transfer sensitivities and 199 null maxima.
- Two figures, the Korean report, protocol/hypothesis provenance, analysis-plan history, literature claim verification, and this release audit.

Participant-level differences are averages over trials, not recorded trial-level EEG. IDs are S01–S10. Protocol slides are described but are not redistributed.

## Excluded

Raw recordings, continuous caches, trial-level features/predictions, trial-level high-band diagnostics, raw-input hashes, local package installations, personal statements, contact emails, and the private teaching PDFs are not part of this release. Data access authorization is still required for signal-level reproduction. This curation is not a substitute for institutional data-sharing approval.

## Portable code and reproducibility checks

The three `manifest_*.json` files retain the **October 9 computation-source** SHA-256 `d63fb40366e7e3763a643b510d0e4ed7706872b6a79ffc4881db55f0b2019499`. They do not claim to hash the ported release file. `release_manifest.json` separately hashes the portable scripts and public outputs.

Porting changes set the cache/work paths through `EEG_CACHE_DIR` and `EEG_RECHECK_WORK`, import the existing repository loader, and replace a local normality-diagnostic helper with the same SciPy Shapiro calculation. The diagnostic's explanatory text changed; the test statistic and p-value did not. Signal, feature, model, seed and split definitions were not changed.

Local release checks completed:

- All six scripts compiled and ran successfully. The primary extraction, within-person, LOSO and four-extra-seed stages were rerun using the public code.
- **22 aggregate tables** matched the October 9 outputs to tolerance `1e-12`, covering extraction/signal summaries, participant/fixed scores, choices, repeated-split summaries, channel/amplitude controls and transfer calibration.
- **290 feature arrays** across ten participants matched exactly after fresh extraction with the public loader.
- All four synthetic/boundary self-tests passed. They test feature sensitivity and implementation boundaries, not biological detection of imagined steps.
- The first **five** maximum-search permutations were rerun as a portable-code smoke/parity check and matched the original first five null rows. The original full 199-permutation result is preserved; the full 199 were **not rerun during packaging**. The five-permutation smoke result is not substituted for the published p=.015.
- The 30–45/8–30 Hz participant table was rerun and matched the original. That diagnostic is not EMG identification.
- Published numeric claims and literature scope received a separate fresh-context AI-agent verification. This is not a human independent replication or final authorship approval.

## Interpretation corrections before publication

The first local report incorrectly said no Stand–Walk signal metric survived Holm correction. Actual `signal_summary.csv` has two **parametric t-test Holm** survivors under CAR:

| Metric | Walk minus Stand | Adjusted p | Across people |
|---|---:|---:|---|
| mu / CPz / slow modulation | +0.0585665, log10 modulation units | .0487763 | 10/10 positive; both trial halves positive in 4/10 |
| low beta / pooled laterality / range | -0.3058111 dB | .00723998 | 10/10 negative; both halves negative in 5/10 |

No exact sign-flip Holm result survives, but with ten participants the minimum two-sided exact p is `2/1024`, above the first `0.05/480` Holm threshold. Non-survival is therefore structurally inevitable and is not negative evidence against an effect. The public report and README now preserve the two observed differences and explain the resolution limit. The original computation CSVs are unchanged.

The negative-change counts 3/10 (earlobe) and 4/10 (CAR) describe the **mean 1–5 s Cz/CPz fixation-relative change** per imagery condition. They do not establish continuously sustained suppression throughout those four seconds.

The overall conclusion is narrowed accordingly: weak label information and some exploratory signal differences are present, but stable high-accuracy generalization, the hypothesized robust walking-alternation signature, and an MI-specific physiological source are not established.
