# Processed candidate correspondence experiment

Reject this model for production tracking. Keep the experiment as a reproducible negative result. The trained evidence improves neighboring-lobe ranking slightly when added to the classical local score, but it does not deliver useful fixed-graph tracking improvement.

| Gujrat second portion | Classical correct / wrong / missing proposals | Learned addition correct / wrong / missing proposals | Neighboring-lobe AUC, classical → combined |
| --- | --- | --- | --- |
| Base, 1,586 reviewed nonseed rows | 397 / 556 / 633 | 396 / 557 / 633 | 0.87945 → 0.88904 |
| Subbase, 1,067 reviewed nonseed rows | 97 / 269 / 701 | 98 / 266 / 703 | 0.80713 → 0.82613 |

The learned evidence alone gives neighboring-lobe AUC 0.86684 for base and 0.81873 for subbase. AUC is a candidate-ranking diagnostic, not empirical identity confidence. The longest contiguous reviewed wrong-proposal span stays 2.2 m for base and 1.2 m for subbase. These are selection results, not accepted measurement metrics; no acceptance gate or interactive policy is changed.

The one fixed L2 logistic model uses 345 signed, gain-normalized seed/candidate patch comparisons. Patches use 33 native time observations over ±0.46875 ns and five native lateral observations over ±0.1 m; nearest native offsets are explicit and no radar interpolation occurs. Three immutable operating seed patches are blended by bracketing position. The graph score change is prospectively fixed at `classical_score + 0.55 * (learned_evidence - 0.5)`. There is no architecture or parameter sweep.

Training uses only processed Mandiali acquisitions P11 and P21, layers 2 and 3. From 3,780 reviewed nonseed rows, 43,319 candidate targets generate 78,862 weighted seed/candidate pairs. Unknown rows and same-lobe timing misses never become negatives; eight nearest wrong-lobe candidate timings at a reviewed row supply hard negatives. Support rows are excluded. All paired recorded amplitudes match and header times agree within the existing audit tolerance. The raw-coordinate U-Net loader is unchanged.

Both Gujrat predictions are saved and hashed before its evaluation references are opened. The original retained graphs, edges, support, anchors and configuration remain identical between comparators. Exact frozen seed coordinates are preserved. Scoring uses the retained graph's seed-only pulse width and unchanged same-signed-lobe plus `max(2, pulse_width / 4)` tolerance. Labels are interpretation references; neither road is unused development data.

The final result is `verified/result.json`; `verified/training.json` has the complete per-source coordinate audit and training counts. The first run and final verification run have identical model hashes and prediction arrays.

- Model SHA-256: `224dcd67bcb7825839f3ded19d37c64cef515f4e3ab6856446460c07b6c49b5c`
- Final source SHA-256: `6ba8c5f8f9ced6d2765562503735a5e1aea6eb353ec6ca0c62ac650b370cd9d7`
- Dependencies: existing NumPy 2.x and SciPy 1.18.1; no copied upstream tracker or new package dependency.

Reproduce from the preserved source snapshot, selecting a fresh output directory:

```powershell
.venv/Scripts/python.exe exports/seeded-tracker/learning/verified/source-6ba8c5f8f9ced6d2/scripts/experiment_processed_correspondence.py --workspace-root C:/Users/faisa/Desktop/automate_gpr --output exports/seeded-tracker/learning/reproduction
```

Verification: all five tests in `tests/test_processed_correspondence.py` pass; Ruff passes on the new module, runner and tests. Tests cover unknown/support label masking, signed-lobe hard negatives, native-coordinate preservation, signal gaps, seed weighting, gain/polarity behavior, convergence and model save/reopen.

Limitations: training has one physical road; nearby rows are correlated; the retained graph already has structural route losses; patch features do not encode persistent path-level identity; probabilities are class-balanced uncalibrated classifier outputs. This experiment supports neither generalization to untouched roads nor calibrated physical thickness. It is rejected for integration rather than promoted on its improved AUC.
