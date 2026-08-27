# Current benchmark inputs

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
