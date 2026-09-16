# Inference and release provenance

A final run keeps two explicit source snapshots. `run_manifest.json.sources`
records the raw source hashes captured for the measured experiment. It is never
replaced by hashes from later documentation, validation or packaging edits.
`release_sources` records current working source hashes when validated evidence
is prepared for submission.

Similarly, `resolved_configs.json` remains the configuration saved at inference.
`release_resolved_configs.json` may add descriptions outside `model`, such as a
model identifier, an exact threshold-source JSON pointer, or per-implementation
score semantics. The entire `model` dictionary, its semantic `config_hash`, the
schema version and the model group must remain equal. No predictions, timings,
metrics or thresholds are changed by this metadata supplement.

Benchmark reuse checks the dataset hash, model semantic hashes, source hashes of
prediction/preprocessing/native/build/benchmark code, and complete saved evidence.
Raw hash changes of exactly `configs/baseline.json` and `configs/final.json` are
permitted only because their full model dictionaries are independently checked.
The original raw config hashes stay in `sources`; current bytes appear in
`release_sources`. Unknown configuration files remain guarded by raw hashes.
Test, report and validation edits do not trigger another expensive benchmark.

After strict saved-run validation, `run_manifest.json.outputs` records SHA-256
for predictions, metrics, agreement, benchmark files, diagnostics and fixed
configuration/data/environment manifests. Subsequent validation rejects changes
to those sealed scientific files. The release-config supplement alone may gain
more descriptive metadata after its unchanged model has been checked again.

The run manifest does not hash itself, validation reports, or generated report
files. This avoids a circular chain in which writing validation changes the
manifest that validation claims to verify. Prepared matrices have hashes in
`data_manifest.json`; all files actually included in the submission ZIP have
exact-byte hashes in `submission_manifest.json`, including reports and figures.
The actual clean execution of the exported ZIP and its checksum are recorded in
the external release-validation sidecar, since a ZIP cannot contain its own
final checksum.
