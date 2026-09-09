# Exact DTW acceleration: keep for controlled application integration

The batch DTW kernel is a reproduced correction-latency bottleneck: a root-owned
10-second sample during a Gujrat correction found 164 of 236 successful stack
samples in `batch_dtw`. This describes that phase, not a whole-run CPU fraction.

The new `processing/waveform_matching_fast.py` retains float64 accumulation,
diagonal/up/left tie precedence, exact centre-displacement backtracing, and
independent reciprocal alignments. NumPy computes the final exponential outside
Numba, with `fastmath=False`. Candidate eligibility, validity masks, thresholds,
graph structure, and scalar DTW are unchanged. The reference NumPy `batch_dtw`
remains independently callable.

Numba 0.67.0 was already installed in the isolated upstream environment. Its
installed BSD-2-Clause license was inspected and is preserved as
`NUMBA-LICENSE.txt`; hashes appear in `benchmark.json`. No compiler source was
copied or adapted. The benchmark environment uses NumPy 2.5.3. After the root's
dependency/dispatch integration, all 47 differential tests also passed in the
production environment with NumPy 2.5.2 and Numba 0.67.0. Full workflow parity
remains the integration owner's separate check.

## Measured reciprocal kernel results

| Radar packet crop | Batch / length / band | NumPy median | Compiled median | Speedup |
|---|---:|---:|---:|---:|
| Mandiali short | 128 / 31 / 2 | 10.011 ms | 0.454 ms | 22.1x |
| Mandiali short | 32 / 89 / 7 | 36.089 ms | 1.031 ms | 35.0x |
| Gujrat second | 128 / 31 / 2 | 8.391 ms | 0.363 ms | 23.1x |
| Gujrat second | 32 / 89 / 7 | 24.774 ms | 0.950 ms | 26.1x |

Each median uses five timing samples. Each reference timing sample repeats five
calls; each compiled timing sample repeats 100 calls. The report retains all
wall and process-CPU timings. Other long-running jobs were present, so these
numbers are kernel measurements, not a guaranteed application speedup.

The first call with a fresh Numba cache, including compilation, took 0.643 s;
imports took another 1.076 s. Warm calls use cached compiled code.

For a 32-pair, 89-sample call the named cost/parent buffers decrease from
2,327,072 bytes to 9,361 bytes. The compiled function also allocates 512 bytes
for costs/shifts; converting float32 inputs requires 45,568 bytes. These are
analytical array sizes, not process peak RSS. Workspace changes from
`O(pairs*n*n)` to `O(n*n+n+pairs)`, with pairs processed sequentially.

## Numerical evidence and selection contract

- All 720 measured packet pairs (180 for each road/length combination) match
  bitwise for agreement and signed shift, in both independently computed
  directions and in the reciprocal aggregate. Maximum absolute errors are zero.
- Radar coordinates were fixed offsets around the six operating base/subbase
  seeds. Both original radar and seed hashes are verified before loading.
  DZX and withheld references are never read. Every packet sample must pass
  the current processed boundary mask; no interpolation fills gaps.
- 47 differential tests cover lengths 0/1/2/7/32/89; multiple bands; randomized,
  tied, zero and negative values; float32/float64/integer/noncontiguous inputs;
  empty batches; unequal shapes; and nonfinite values. No input is mutated.
- The nonfinite tests explicitly preserve the reference's first-NaN argmin and
  NaN cost propagation; negative/NaN bands preserve the reference's unreachable
  terminal state and default backtrace behavior. These do not confer validity
  on any measurement; production eligibility remains the caller's responsibility.

Run:

```powershell
exports/seeded-tracker/upstream/venv/Scripts/python.exe -m pytest tests/test_waveform_matching_fast.py
exports/seeded-tracker/upstream/venv/Scripts/python.exe scripts/benchmark_waveform_matching_fast.py --output exports/seeded-tracker/dtw-acceleration-NEW
```

Keep the compiled batch kernel for a production dependency and dispatch patch
owned by integration. Require actual application before/after pick, status,
TWTT, query and correction parity before claiming integration complete. No
reflector-selection improvement is claimed: an exact numerical acceleration
cannot repair the demonstrated interpretation errors.
