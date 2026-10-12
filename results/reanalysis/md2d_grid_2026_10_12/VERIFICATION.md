# Claim-to-evidence verification, 2026-10-12

This is an independent agent review, not human author approval. A fresh-context skeptic
read the original logs and source and recomputed the claims without importing the proposed
audit script. The full local review retains execution evidence; this public record removes
private paths and original participant identifiers.

## Results

| Claim group | Verdict | Source evidence and scope |
|---|---|---|
| Complete binary grid | CONFIRMED | Original files: 70 per contrast, 210 total; 18 unique configuration rows each; 1,260 per contrast, 3,780 total. Ten people each have seven normalizations and 126 candidates. Three-class files are excluded. |
| Best-of-126 means/ranges | CONFIRMED | Original `test_acc_fused`: sums of ten maxima = 820%, 790%, 785%; means = 82.0%, 79.0%, 78.5%. Ranges = 70–100%, 65–90%, 70–90%. |
| CT-only means | CONFIRMED | Sum of participant-specific maxima over the 18 CT configurations = 715%, 660%, 720%; means = 71.5%, 66.0%, 72.0%. |
| Public candidate and winner equivalence | CONFIRMED | All 3,780 public keys and 30,240 numeric fields match original rows under Decimal parsing. Each of 30 selected rows attains the stated maximum. |
| Current source selects using test accuracy | CONFIRMED | Original function: “pick best by fused TEST accuracy.” Condition: `acc_fused > best["acc"]`, with `np.isclose` plus higher validation accuracy for ties. The across-normalization maximum is this audit's aggregation. |
| Maxima are descriptive, not independent performance | CONFIRMED | `audit_scope.json`: “descriptive post-test-selected maximum; not independent generalization accuracy”. Trial-level predictions or new held-out performance cannot be inferred from the maxima. |
| Saved nested score summaries | CONFIRMED | `results/final_scores.csv`, `kind=nested`: means .65778947, .64458480, .53095906, .45763875; ten rows per contrast. Displayed t intervals also recompute. This is not a full rerun. |
| Fixed-model means | CONFIRMED | `fixed_model_scores.csv`, `task_8_30`/`random`: balanced-accuracy means .6058, .5948, .5156; ten rows per contrast. |
| October 9 nested/fixed/LOSO summaries | CONFIRMED | Participant CSVs: initial nested .525, time blocks .516, five-seed nested .5212, LOSO .48, five-seed fixed trajectory .5526. Average within people before group summarization. |
| October 9 permutation arithmetic/family | CONFIRMED | `blocked_search_test.json`: “23 fixed candidates x 2 evaluation modes; group mean across 10 participants”. Two of 199 null maxima ≥ .559; (2+1)/(199+1)=.015. Not a correction for the historical neural-network search. |
| Scoped direct-identifier removal | CONFIRMED, LIMITED | The candidate/winner tables contain S01–S10 and model scores. Scanned proposed files contained no original participant keys or literal local user-directory paths. This is not proof of irreversible anonymization or a consent audit. |
| Exact historical source/preprocessing/RNG identity | PARTIAL | The log schema contains configuration/score columns but no source revision, split indices or RNG record. Matching the current grid does not establish execution parity. The report retains this limitation. |
| Max-selected Rest scores versus nested Stand–Walk proves success/failure contrast | REJECTED | The same maximum rule gives Stand–Walk 78.5%. Changing the evaluation rule across contrasts does not support this inference. The nested and fixed analyses support a narrower ordering within their own procedures. |
| Independent ~80% decoding, pure MI, no physiological signal, or universal ceiling | REJECTED | The grid scores do not measure an untouched external evaluation or identify a physiological source. Weak performance is not evidence of equivalence or impossibility. |
| Final human source-data/scientific approval | BLOCKED | `audit_scope.json`: `"author_approval": "PENDING"`. Agent verification cannot complete the responsible author's final review. The user's request to update repository files is separate from scientific sign-off. |

Fifteen claim groups: **11 CONFIRMED, 1 PARTIAL, 2 REJECTED, 1 BLOCKED**.
No checked numerical table claim failed. The rejections concern proposed inferences, not
false log values. Prior physiological sections and consent/release terms were outside this
bounded independent audit; their prior verification record is not replaced.

## Remaining Actions and Limits

- Do not promote log aggregation to a source/RNG-matched reproduction or a full new grid run.
- Do not describe all score differences as the amount of selection bias; models and evaluation sets also differ.
- Do not treat the October 9 permutation p-value as validating this historical grid.
- Keep human review PENDING until the responsible author completes it.
- Only pseudonymized configuration scores and summary documents are included in this update. Raw EEG, cache arrays, trial predictions, direct identifiers and private file hashes remain local.

## Local Checks

The audit's three synthetic tests passed: test-score maximum with validation tie-break,
rejection of incomplete logs, and full-grid aggregation with identifier removal and a
missing-file failure. Independent Decimal calculations additionally checked all real rows.
These implementation checks do not validate the biological source or historical provenance.

The two revised local Korean study books received a separate docs-only check: historical
tables, six restricted neural-network rerun means, validation-loss-based learning-rate
selection, test denominators, presentation examples and interpretation limits matched the
stored sources. One wording item was clarified to identify the October 9 Stand–Walk
23-candidate bank rather than suggesting that every analysis used 23 candidates. Those
local books and neural-network rerun files are not part of the public grid-log dataset.

## Procedural Tool Attribution

The installed statistical-analysis, scientific-critical-thinking, scientific-writing and
claim-source-verification procedures informed the separation of participant units,
exploration, prediction and causal interpretation. This is tool attribution, not EEG evidence.

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
[arXiv record](https://arxiv.org/abs/2609.00065),
[DOI](https://doi.org/10.48550/arXiv.2609.00065).
The author list/title/year and September 2 revision were checked on October 12, 2026.
