# Independent publication claim check

2026-10-10. A separate fresh-context AI skeptic checked the README's new section and Korean report against actual CSV/JSON values and primary papers. The verifier did not author or edit those documents. This is a claim-to-source audit, not an independent human EEG replication.

| Claim group | Outcome and evidence |
|---|---|
| Fixed 55.9%, five-seed 55.3% and nested 52.1% | CONFIRMED from fixed_candidate_summary.csv and split_stability_summary.csv; these evaluate different procedures |
| Time blocks 51.6%, uncalibrated LOSO 48.0% | CONFIRMED from within_person_summary.csv and cross_person_summary.csv |
| Maximum-search p=.015 | CONFIRMED from blocked_search_test.json and two exceedances in 199 null rows, with +1 correction and the defined search scope |
| No Holm signal survivors | Initially REJECTED; public text corrected to two CAR parametric survivors, now CONFIRMED against signal_summary.csv |
| Negative mean changes interpreted as sustained suppression | Initially PARTIAL; narrowed to mean 1–5 s change, now CONFIRMED against within_condition_change.csv |
| Exact sign-flip Holm resolution | CONFIRMED arithmetically: two-sided minimum 2/1024 exceeds .05/480 |
| Reference and trial-half sensitivity | CONFIRMED from signal_summary.csv and signal_participant_differences.csv |
| Session whitening uses test-person statistics | CONFIRMED from code and alignment_scope.json: all 150 unlabeled trials, transductive |
| Ma 2023 compound classes differ from our comparison | CONFIRMED in Methods/Fig. 1 and classification definition of the author-accessible full-text copy |
| 2006 hand-area mu ERS and midcentral foot ERD | CONFIRMED in author-hosted full-text Abstract/Methods/Discussion |
| 2013 early beta ERD in nogo | CONFIRMED in primary full-text Methods/Results/Discussion; not a claim about every mu/beta response |
| 1997 hand-MI contralateral ERD | CONFIRMED at abstract level only |
| Absence of a specificity test throughout the 1997 full text | BLOCKED: full text not obtained; do not make this absence claim |

The detailed private audit examined 15 claim connections: initial CONFIRMED 12 / PARTIAL 1 / REJECTED 1 / BLOCKED 1; after the two wording corrections, CONFIRMED 14 / BLOCKED 1. Seven classification confidence-interval rows and six laterality examples were recomputed without rounding discrepancies. Both original findings and their corrections are retained in [RELEASE.md](RELEASE.md).

Primary sources and access-level restrictions are listed in [source_verification.md](source_verification.md). For the 2013 recheck, the verifier also accessed [Europe PMC's official full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3726939/fullTextXML). No public manuscript should turn partial/abstract-only access into a claim of full-paper verification.

Still unresolved: block exchangeability as a substantive assumption, actual imagined step timing/compliance, cortical source attribution, ocular/muscle contributions, and generalization to an independent session or sample. The audit does not correct all historical data-dependent choices.
