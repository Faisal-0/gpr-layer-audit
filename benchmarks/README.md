# Current benchmark inputs

The conventional repair/evaluation protocol, frozen-comparator hashes and
development measurements are in `conventional-*.json`. See
[`../docs/CONVENTIONAL_TRACING_VALIDATION.md`](../docs/CONVENTIONAL_TRACING_VALIDATION.md)
for commands, input-coordinate rules, overlays and the failed promotion gates.
The repaired backend remains experimental; held-out accuracy scoring is gated on
successful calibration and matching frozen configuration/backend fingerprints.

For resolution comparisons, use `conventional evaluate --seed-source prior-run.json`
to retain exactly the same native processed-radar observations. Both source
fingerprints and layer labels must match; incompatible grids fail instead of
moving a seed. New evaluations export a `seed_support` manifest; older reports
recover only their recorded seed rows from the matching DZX.
`scripts/evaluate_conventional_development.py` accepts the same option for current
backend experiments. Use a new output root when changing settings or support.

After both runs finish, compare their common non-seed reference rows with:

```powershell
uv run python scripts/compare_conventional_resolution.py coarse.json fine.json --output comparison.json
```

This operation verifies matching backend, configuration, time origin, seed
observations and scoring tolerance. It reports accepted agreement, correct
coverage and signed-lobe switches on the same native reference observations.
The findings and remaining limitations are recorded in
[`../docs/CONVENTIONAL_CONTINUATION.md`](../docs/CONVENTIONAL_CONTINUATION.md).

This directory contains only reproducible inputs for the current
`joint_seed_adaptive` implementation:

- `talagang-design.json`: Talagang 50.8 mm asphalt / 101.6 mm base workflow.
- `talagang-development-seeds.json`: provisional radar-only development seeds;
  these are not independent validation checkpoints.
- `pattoki-jhoru-design.json`: untouched Pattoki transfer manifest.
- `pattoki-development-seeds.json`: provisional Pattoki development seeds.

Generated results, plots and diagnostic arrays belong under `exports/` and are
not versioned as implementation files. Workbooks remain fallible secondary
disagreement diagnostics. Release claims require independently confirmed,
radar-only checkpoints and must preserve failed acceptance-gate results.
