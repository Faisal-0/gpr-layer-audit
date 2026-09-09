# Dense state and measured-event experiment

Reject application promotion. The explicit basin representation improves short
Mandiali subbase replay, but useful reliable tracking does not transfer to Gujrat.
Application source and the established asphalt path remain unchanged.

## Mechanism and diagnosis

The existing unit-weight endpoint guide was first tested on Jamshoro with exact
native seeds, identical radar-only candidate masks and unchanged classical NCC.
Every array of the unguided control reproduces the frozen run exactly. Correct
proposals improve from 422 to 855/4,115 for Layer 2 and from 6 to 343/2,242 for
Layer 3. Both arms accept zero automatic observations. This is a development
comparison, not successful coverage or a physical-thickness result.

An isolated capture of unchanged hidden states shows that 688 Layer 2 and 286
Layer 3 hidden states have measured basin peaks agreeing with reviewed points.
These projections diagnose a reporting/representation loss; no confidence or
accepted measurement is inherited from them.

The tested implementation carries the original waveform-member position for
template matching, geometry and guide costs, while each observable member emits
its unique supported amplitude peak. Ambiguous valleys and unsupported peaks
remain gaps. Original seeds must already coincide with supported peaks; the
experiment never snaps a seed. Exact competing-route costs compare emitted peak
classes, including separate peaks within one signed lobe. Member states are
never pooled into a single score. No provisional base path prunes subbase.

All acceptance thresholds, physical motion bounds and the frozen signed-lobe/time
scorer remain unchanged. The old interval-normalized margin is still uncalibrated
and causes excessive abstention on long brackets. A local waveform basin is not
proof of physical reflector identity.

## Actual four-request replay

Each run starts with three native observations per deep interface and uses the
real local merger and ordering guard. Both policies receive the same four-request
budget. An unavailable exact reviewed answer consumes a request without inventing
a label. All supplied answers are excluded from automatic credit.

| Road/policy | Initial base | Final base | Initial subbase | Final subbase |
|---|---:|---:|---:|---:|
| Mandiali active | 20/20 | 20/20 | 23/23 | 84/85 |
| Mandiali midpoint | 20/20 | 39/42 | 23/23 | 73/74 |
| Gujrat active | 0/0 | 53/62 | 0/0 | 0/0 |
| Gujrat midpoint | 0/0 | 0/0 | 0/0 | 0/0 |

Cells show agreeing/accepted automatic observations; 0/0 has undefined precision.
Fixed initial reviewed pools are 443/404 for Mandiali base/subbase and 1,586/1,067
for Gujrat. Final active correct coverage is 4.51%/20.79% on Mandiali and
3.34%/0% on Gujrat. Wrong accepted coverage is 0%/0.25% and 0.57%/0%, respectively.
Mandiali is 11.45 m: four requests mean 349.34 requests/km and the +/-25 m window
covers the entire short road. Gujrat is 287.4 m: 13.92 requests/km. Every action's
outside-window and other-layer arrays are verified unchanged on both roads.

Mandiali active has one same-lobe timing error, with a 0.025 m wrong observation
footprint. Gujrat active has nine wrong-signed-lobe observations and a longest
contiguous wrong footprint of 0.4 m. The frozen field named `reflector_switches`
counts wrong-signed-lobe observations, **not semantic switch events**. The new
readout names it accurately. Unknown labels are unscored, and missing legacy
correspondence instrumentation is unavailable rather than a measured zero.

The separate confidence-weight experiment removes the implicit horizontal-motion
preference when geometry support is zero. Exact tests cover zero support, partial
weights, direction and physical unit conversion. Correct accepted Mandiali
observations fall from 88 to 69 across the two layers; that variant is rejected
without a weight sweep or Gujrat transfer.

## Reproduce and inspect

Use the frozen source/input setup in `SEEDED_TRACKER_WORKFLOW.md`. Exact execution
sources, hashes, every replay step, full-road overlays and intervention curves are
indexed in `benchmarks/seeded-basin-results/index.json`. The six new numerical
regressions and existing dense/replay tests pass. The short Mandiali case is one
spatial block; these sample counts do not establish independent-road reliability.

```powershell
.venv/Scripts/python.exe scripts/experiment_jamshoro_guide.py predict --output exports/guide-reproduction
.venv/Scripts/python.exe scripts/experiment_jamshoro_guide.py evaluate --output exports/guide-reproduction
.venv/Scripts/python.exe scripts/experiment_dense_basin_emissions.py --case mandiali-short --output exports/basin-mandiali
.venv/Scripts/python.exe scripts/experiment_dense_basin_emissions.py --case gujrat-second --output exports/basin-gujrat
.venv/Scripts/python.exe scripts/experiment_basin_replay.py --case mandiali-short --output exports/basin-replay-mandiali
.venv/Scripts/python.exe scripts/experiment_basin_replay.py --case gujrat-second --output exports/basin-replay-gujrat
.venv/Scripts/python.exe scripts/experiment_dense_geometry_confidence.py --case mandiali-short --output exports/geometry-reproduction
.venv/Scripts/python.exe scripts/package_basin_results.py --verify --originals
```

Jamshoro guide replay requires the frozen `patchnet-v2-mask` prediction artifacts,
and basin replay requires the preserved first dense replay worker/configuration.
Both are local development dependencies explicitly recorded by their drivers.
Omit `--originals` to verify packaged bytes without those original exports.
