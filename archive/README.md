# Research history

The preserved research tree is `research/pre_final_df1a2c3f67e4/`. Its `manifest.json`
records original relative paths, byte sizes, raw SHA-256 hashes, Git tracking state,
purpose and reproduction limits. The snapshot includes the locally available model,
threshold, operating-point and runtime studies, earlier source and tests. It does not
invent missing historical experiments or contain raw data, virtual environments,
compiler installations or native build products.

Original documents and evidence are unchanged. Historical caveats are recorded
separately in `research/pre_final_df1a2c3f67e4/ARCHIVE_NOTES.md`. The active experiment
does not import or execute the snapshot. Previously inspected test results remain
continuity evidence, rather than new independent statistical validation.

Verify preserved bytes from the repository root:

```text
python scripts/verify_archive.py
```

Restore and run the historical unit suite in a new disposable tree with an interpreter
that has the snapshot's pinned requirements installed:

```text
python scripts/reproduce_research.py --work-root verification/historical_check
```

The helper copies the snapshot with original relative paths, makes copied evidence
read-only, runs only historical unit tests and verifies original archive bytes again.
It writes logs and `historical_validation.json` in the disposable directory. It never
runs expensive historical search or writes into the archive or the active results.
Use a fresh directory on each invocation; the helper does not remove old evidence.
For a full historical experiment, work exclusively in the restored tree, install
its `requirements.txt` in a separate environment, and supply the dataset and native
tools independently. Earlier documents describe the commands and known limitations.

The compact submission includes this explanation. The separate
`dist/3804ICT-research-history.zip` contains the preserved history and manifest;
`scripts/export_submission.py` generates both packages and checks every archived
byte before exporting history. The main submission and generated source appendix
contain active code, with research source supplied in the history package.

One clearly labelled **new retention overlay**, `snapshot/results/.gitignore`, was
added during finalization. Its `!**` rule makes preserved result evidence visible
to Git despite the original snapshot's own ignore rules. It is not an original
research artifact: the original `manifest.json`, parent `.gitignore` and all 244
manifest-indexed files remain byte-for-byte unchanged. The verifier checks the
overlay's exact declared content separately, and the history ZIP indexes it explicitly.
No files were force-added or staged to bypass ignore rules. The root `.gitattributes`
marks `archive/research/** -text` to preserve these raw bytes across future Git
checkouts without converting historical line endings.
