# Exact alignment acceleration

The experimental hybrid tracker uses a compiled batched DTW recurrence to
reduce correction latency. Candidate selection, graph decisions, acceptance
thresholds and the established asphalt engine are unchanged. The original
NumPy `batch_dtw` remains callable as an independent numerical reference.

Numba 0.67.0 is pinned in the application dependencies. The kernel uses float64
accumulation, no fast-math, the original diagonal/up/left tie precedence and
independent forward/reverse alignments. NumPy computes the final exponential.
It changes computation, not the radar observations or their validity masks.

All 720 measured packet pairs from the fixed Mandiali/Gujrat operating-seed
neighbourhoods match bit for bit, including signed displacement and reciprocal
results. Kernel timings improve by 22–35 times on those packet batches; this
is not a claimed whole-application speedup. Numba's BSD-2-Clause license is
preserved in the benchmark evidence; no compiler source is copied into the
tracker. The numerical implementation is original project code.

The complete accelerated Mandiali application workflow passed all 11 checks:
GUI/CLI parity, both requested corrections, exact observations, preserved
unrelated layers, save/reopen and unresolved TWTT withholding. Four complete
path stages and all pick coordinates, identities and statuses exactly match
the earlier workflow. Queries and answers are identical. The only allowed
TWTT difference is the independently verified −0.01171875 ns time-origin
repair that preceded this acceleration.

Measured application times in this run:

| Operation | Elapsed seconds | Process CPU seconds |
|---|---:|---:|
| Initial GUI fit | 57.5 | 56.3 |
| First local correction | 66.2 | 65.8 |
| Second local correction | 60.8 | 60.4 |
| Reopen and reconstruct | 168.9 | 168.6 |

These are actual executions under concurrent workload. Older wall timings
include a documented Windows suspension in one correction, so they must not
be used as a controlled speedup ratio. The first attempted accelerated
verification was interrupted; its partial outputs are preserved separately.
The completed run is `exports/seeded-tracker/workflow-verification-compiled-v2`.

Tracking accuracy is unchanged: Mandiali base remains 4/4 agreeing accepted
automatic observations, and subbase changes from 42/44 initially to 45/48
after the same two requested observations. Faster inference does not solve
the deep-reflector identity and useful-coverage failures.

Reproduce using fresh output paths:

```powershell
.venv/Scripts/python.exe scripts/benchmark_waveform_matching_fast.py --output exports/dtw-benchmark-NEW
.venv/Scripts/python.exe scripts/verify_processed_workflow.py --output exports/workflow-NEW
.venv/Scripts/python.exe scripts/compare_processed_workflows.py --before exports/seeded-tracker/workflow-verification-20260909-v3 --after exports/workflow-NEW --twtt-shift-ns -0.01171875 --output exports/workflow-parity-NEW.json
```

`exports/seeded-tracker/dtw-acceleration/benchmark.json` records packet, source,
dependency and license hashes. `application-parity.json` records the complete
before/after comparison. The completed integration suite has 461 passing
tests and one optional-upstream-environment skip; 47 differential cases cover
the compiled recurrence, including ties and invalid/nonfinite input cases.
