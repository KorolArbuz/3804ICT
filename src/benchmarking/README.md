# Current five-implementation benchmark

This package measures the frozen V5.1 baseline and four final implementations through
the same workflow as `run_all.py`. The original k19 benchmark, scaling studies and
optimization history are preserved in the separate research snapshot; they do not
supply measurements to the current reports.

```text
python run_all.py --data data/raw/UCI_Credit_Card.csv --build --benchmark --runs 20 --warmups 3
```

The equivalent standalone entry point is:

```text
python -m src.benchmarking --data data/raw/UCI_Credit_Card.csv --build --runs 20 --warmups 3
```

Each invocation creates a new `results/final/<run_id>/`. It prepares the fixed split,
executes all five implementations independently, checks prediction identities, then
collects fresh measurements. A completed run contains:

- `benchmark_raw.csv`: every warmup and measured observation with run/configuration,
  trial, sequence, implementation, model group, shape, k, process and checksum evidence;
- `benchmark_summary.csv`: count, median, mean, range, standard deviation, quartiles,
  IQR, p10/p90 and final-group paired ratios;
- `benchmark_protocol.json`: precise timing scopes and randomization settings;
- `benchmark_setup.json`: resident model/worker setup evidence;
- `environment.json`: recorded versions and thread limits;
- prediction CSVs and numerical comparison diagnostics from the same run.

## Two scopes and two model groups

Resident `prediction` starts from a ready model and prepared queries. Every trial
performs a fresh exact neighbour search and creates scores and thresholded labels.
The timer includes query validation and output allocation/materialization, and stops
before checksum calculation, IPC serialization or file output. Training, loading,
process startup and preprocessing are excluded. Resident Python models and the native
C++/Weka workers remain alive across trials. Three warmups and twenty measured trials
are recorded separately by default.

`prepared_pipeline` starts a new process for each of five trials and includes process
startup, prepared-file loading, fitting, prediction and CSV output. Python/C++ use
canonical CSV; Weka uses canonical ARFF. Raw-data preprocessing and compilation are
excluded. This is a separate process lifecycle; differences between scope medians
must not be interpreted as an isolated estimate of startup or fitting cost.

The k19 uniform V5.1 baseline uses 33 features and is a different workload. Its timing
is reported separately. The final implementations share k101, 60 features, inverse
distance and the frozen `balanced_low_fp` threshold. Only this final group enters
direct implementation speedup comparisons. Ratios pair identical run, phase, trial
and workload keys and are summarized as medians of paired ratios.

Prediction rounds are sequential and implementation order is randomized with seed
20260916. Numerical pools use one compute worker; the C++ KNN and Weka query loop are
sequential. JVM service/JIT/GC threads can still exist. The workflow does not alter
affinity, priority, power settings or system protections. Run without unrelated heavy
workload; record actual environment limits instead of claiming universal hardware speed.
Checksums cover complete materialized score/label vectors outside the measured interval.
The raw adapter batch-size field is 64 for the baseline and 32 for C++. A recorded
scikit-learn value of 1 is the adapter-level/default reporting unit, not a claim
about the library's internal batching or blocking.
All measured samples are retained, including outliers. Quantiles use NumPy's linear
method. Correctness probes and diagnostic smoke runs are distinct from headline timings.

## Completed run

The selected run is
[`20260916T205945.220601_0000_b410e8a7`](../../results/final/20260916T205945.220601_0000_b410e8a7/run_manifest.json).
It contains 115 resident observations (15 warmups plus 100 measured) and 25
fresh-process observations, for 140 raw records. Resident prediction medians/IQRs:

| Final implementation | Median seconds | IQR seconds |
|---|---:|---:|
| Custom Python | 35.661570 | 0.352449 |
| scikit-learn | 1.403104 | 0.001843 |
| C++20 portable Release | 3.677404 | 0.006887 |
| Genuine Java/Weka IBk | 13.359275 | 0.047886 |

The different baseline workload is 1.256276 s median and 0.005059 s IQR. The measured
paired Custom-Python/implementation ratios are 25.397863 for scikit-learn, 9.686462
for C++ and 2.667060 for Weka. These results describe one machine and one fixed workload.
They do not claim cross-machine or independent-session reproducibility.

The C++ fresh-process external median was 2.140769 s, with internal prediction median
1.774813 s, despite the identical artifact, configuration and full score/label vectors
used by the 3.677404 s resident samples. The
[timing audit](../../docs/benchmark_timing_audit.json) found consistent scopes; the
cause of the process-mode difference was not measured. No scheduling, cache, thermal
or frequency explanation is asserted. Both scopes and every original sample are retained.

## Saved-evidence reporting and checks

```text
python -m src.reporting --run-dir results/final/<run_id>
python -m src.validation --data data/raw/UCI_Credit_Card.csv --full
```

Reporting reads only saved metrics, predictions and timings. It checks identities and
arithmetic before producing figures; it performs no inference, dataset loading, JVM
launch or compilation. Missing repeated measurements cause explicit runtime-figure
omissions. Full validation can reuse a successfully verified current-source benchmark
instead of repeating it, while still building/testing and running fresh clean-tree and
extracted-submission comparisons. Benchmark completion alone does not prove those
clean-install and packaging checks have passed.
