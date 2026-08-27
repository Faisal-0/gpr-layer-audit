# Benchmark evidence

`talagang-baseline.json` is the fixed research manifest for the supplied Talagang road. It chooses three station locations from radar evidence, replays simulated seed samples without consulting the workbook, and evaluates the primary tracker plus four reproducible baselines/ablations using the same blocked-span and leave-one-road-out splits.

`talagang-baseline-result.json` intentionally records a failed run. No configuration meets the accuracy and review gates, so `selected_configuration` is `null` and the benchmark command exits with status 2.

The failure is informative:

- The seed-adaptive method declines to automatically accept paths from the crude radar-only seed simulation rather than presenting a confident layer.
- The current-style and no-deconvolution baselines accept an asphalt-looking reflector with very low held-out accuracy, demonstrating why review coverage cannot be optimized independently of accuracy.
- Runtime remains well below 60 seconds per kilometre and the recorded peak working set remains below 1 GiB.

This workbook is provisional interpretation evidence, not independent core truth. Do not tune seed locations, preprocessing, thresholds, or tracker weights after inspecting a held-out reference block.

`talagang-design.json` freezes the new 50.8 mm asphalt / 101.6 mm base design-corridor workflows: design-only and design plus two radar-ranked stations selected before opening the reference workbook. `talagang-design-result.json` is deliberately preserved as a failed research result. The seeded asphalt path is accurate where accepted (95.96% within 12.7 mm in leave-one-road-out evaluation), but review coverage is 28.93%. The provisional base comparison reaches only 20% within 25.4 mm and 89.54% review/unresolved coverage. Runtime (24.33 s/km) and peak memory (989.6 MiB) pass their gates. The software therefore asks for base corrections instead of lowering confidence thresholds or manufacturing continuity.

## Phase-locked event-family development checkpoint

The event-family replacement is recorded as a development checkpoint rather
than overwriting the older frozen results. With automatic fine retracking
disabled, the current phase-complete radar seeds produce:

- Talagang: 46.4% visible asphalt and 81.6% visible base, five grouped review
  regions, and 23.0 s/km.  On manually entered, non-interpolated workbook
  points that overlap visible output, asphalt agreement is 86.8% within
  12.7 mm and base agreement is 29.6% within 25.4 mm.
- Pattoki-Jhoru: 99.8% visible asphalt and 55.2% visible base, with runtime
  about 21.1 s/km at the corrected 0.4 m global grid.  Its manual workbook is
  retained only as disagreement evidence because the shallow workbook family
  and the deeper 2 in + 6 in seeded family conflict over substantial spans.

The reliability upgrade now separates the selected display lobe from canonical
event time, uses polarity-gated original/residual evidence, locally adaptive
layer stripping, signed-slope tracklets, robust seed corridors, and joint
selection over second-order path hypotheses. A no-pick row can no longer erase
event-family memory and restart on an unrelated ringing cycle.

The base release gate is still not met. In the latest diagnostic run, Talagang
base has 51.0% visible coverage and 46.6% of overlapping manual comparison
points within 25.4 mm; Pattoki base has 92.6% visible coverage but only 14.6%
within 25.4 mm. The figures are disagreement diagnostics, not field accuracy.
Pattoki's current base seeds are 10–25 samples deeper than the nearby workbook
family, while Talagang's 672.6 m base seed is a robust gap outlier. The software
now isolates that Talagang seed as a local regime conflict and requests review
instead of widening the whole-road corridor or manufacturing agreement. A
user-confirmed candidate from the intended base family (or independent core
evidence) is required before the base gate can honestly pass.
