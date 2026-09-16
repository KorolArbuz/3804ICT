# Finalization status

This document tracks release work; it is not a substitute for the generated validation
report or the selected run manifest.

| Checkpoint | Current evidence/status |
|---|---|
| Inventory and archive | PASS: 244 original files, 23,132,161 bytes; raw SHA-256 manifest checked before migration and again after packaging work |
| Frozen configurations | Baseline k19/uniform and final k101/distance/balanced_low_fp carried into separate semantic configuration files |
| Migration characterization | PASS: matrices, feature order and all 6,000 scores/labels match; baseline source hash unchanged |
| Historical unit suite | PASS: 64 tests in restored disposable tree; archived bytes unchanged before/after |
| Manual C++ all-row correctness | PASS for scalar, portable, native and persistent paths; see cpp_numerical_validation.json |
| Native sanitizer | BLOCKED: installed MSVC AddressSanitizer runtime initialization fails before test code; Debug CTest passed |
| Toolkit comparability | Native score differences retained and diagnosed; no probability correction or relaxed tolerance |
| Reporting tests | PASS: four tests for saved metrics, stale configs, warmups and deterministic saved-only regeneration |
| Packaging tests | PASS: eleven tests, including current-byte allowlist export, omissions, archive retention, unsafe ZIP paths and all-five clean-reference comparison |
| Fresh headline benchmark | PASS: run 20260916T205945.220601_0000_b410e8a7; all five models, 20 measured resident passes plus 3 warmups and 5 fresh-process passes each; 140 raw records |
| Clean install and extracted ZIP run | PASS: fresh environments, pinned installs, pip check, C++/Weka builds, all five implementations, 62 active tests with no skips and saved-only report regeneration; both runs reproduced all 30,000 scores and labels exactly |
| Full validation before cleanup | complete_with_documented_differences: all six mandatory completion scopes PASS; evidence in verification/release_20260916T213602_19415e/validation.json |
| Archive-backed cleanup | PASS: 198 superseded active files removed only after full validation and ZIP execution; original archive bytes verified again; exact paths and hashes in active_cleanup_manifest.json |
| Final cleaned-tree validation | The same full coordinator is rerun after cleanup; the selected run's validation.json and external release_validation_external.json are authoritative for the final package |

History is preserved in `archive/research/pre_final_df1a2c3f67e4/`. Its original manifest,
snapshot and documents are immutable. The source cleanup is restricted to duplicates
whose originals were verified in the snapshot. User data and unknown local files are retained.
The removed files are historical search/report/benchmark modules, four historical test
modules, old generated research results and local IDE tasks. The 64 historical tests
remain in the snapshot and were run separately. The two unsuccessful development run
directories were moved intact to ignored verification/rejected_runs; only the completed
selected comparison remains under results/final.

The active comparison reports five independently executed implementations. Baseline
and final are different model workloads; only the four final implementations enter
direct implementation speedup comparisons. Historical measurement summaries are not
substituted for new timings.

The selected experiment has status `complete_with_documented_differences`: manual
Python/C++ labels agree on all 6,000 rows; scikit-learn labels also agree but ten scores
exceed the declared `1e-12` tolerance; Weka has one label disagreement and retains its
native score semantics. The experiment's completion status is separate from full
release and package validation. Interrupted earlier runs remain diagnostic evidence.

The final status must distinguish build correctness, manual numerical agreement,
toolkit comparability, real-data reproduction, runtime evidence and packaging completeness.
The existing sanitizer blocker and any subsequent FAIL/SKIP/BLOCKED records must remain
visible even if the other release checks pass.
