# Benchmark evidence

`talagang-baseline.json` is the fixed research manifest for the supplied Talagang road. It chooses three station locations from radar evidence, replays simulated seed samples without consulting the workbook, and evaluates the primary tracker plus four reproducible baselines/ablations using the same blocked-span and leave-one-road-out splits.

`talagang-baseline-result.json` intentionally records a failed run. No configuration meets the accuracy and review gates, so `selected_configuration` is `null` and the benchmark command exits with status 2.

The failure is informative:

- The seed-adaptive method declines to automatically accept paths from the crude radar-only seed simulation rather than presenting a confident layer.
- The current-style and no-deconvolution baselines accept an asphalt-looking reflector with very low held-out accuracy, demonstrating why review coverage cannot be optimized independently of accuracy.
- Runtime remains well below 60 seconds per kilometre and the recorded peak working set remains below 1 GiB.

This workbook is provisional interpretation evidence, not independent core truth. Do not tune seed locations, preprocessing, thresholds, or tracker weights after inspecting a held-out reference block.
