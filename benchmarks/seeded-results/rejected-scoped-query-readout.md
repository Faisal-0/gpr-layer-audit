# Scoped query ranking: one prospective development experiment

Decision: **reject performance promotion**. No reliable 95% deep coverage claim is supported.

One frozen variant counts observable radar/model pair disagreements only inside the declared ±25 m local correction window. It collapses routes identical inside that window so remote-only route splits cannot multiply local benefit. Original graph frontier priorities, distance weighting, tracker acceptance and correction behavior are unchanged.

All compared runs use three fixed native seeds per layer, four requests, the same configuration, frozen initial pulse, signed-lobe and timing scoring, and common initial denominators (base 1586, subbase 1067). Revealed answers receive no automatic accuracy credit and remain in these denominators. All roads are development data.

| Run | Step | Layer | Correct / accepted | Wrong | Correct coverage | New correct | Lost correct (excluding revealed) | New wrong |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| gujrat-fixed-active | 0 | Base | 369/547 | 178 | 23.266% | 0 | 0 | 0 |
| gujrat-fixed-active | 0 | Subbase | 4/63 | 59 | 0.375% | 0 | 0 | 0 |
| gujrat-fixed-active | 1 | Base | 369/547 | 178 | 23.266% | 0 | 0 | 0 |
| gujrat-fixed-active | 1 | Subbase | 4/63 | 59 | 0.375% | 0 | 0 | 0 |
| gujrat-fixed-active | 2 | Base | 369/547 | 178 | 23.266% | 0 | 0 | 0 |
| gujrat-fixed-active | 2 | Subbase | 26/88 | 62 | 2.437% | 22 | 0 | 3 |
| gujrat-fixed-active | 3 | Base | 369/547 | 178 | 23.266% | 0 | 0 | 0 |
| gujrat-fixed-active | 3 | Subbase | 26/88 | 62 | 2.437% | 0 | 0 | 0 |
| gujrat-fixed-active | 4 | Base | 369/547 | 178 | 23.266% | 0 | 0 | 0 |
| gujrat-fixed-active | 4 | Subbase | 25/60 | 35 | 2.343% | 0 | 1 | 0 |
| gujrat-fixed-midpoint | 0 | Base | 369/547 | 178 | 23.266% | 0 | 0 | 0 |
| gujrat-fixed-midpoint | 0 | Subbase | 4/63 | 59 | 0.375% | 0 | 0 | 0 |
| gujrat-fixed-midpoint | 1 | Base | 369/547 | 178 | 23.266% | 0 | 0 | 0 |
| gujrat-fixed-midpoint | 1 | Subbase | 4/63 | 59 | 0.375% | 0 | 0 | 0 |
| gujrat-fixed-midpoint | 2 | Base | 369/547 | 178 | 23.266% | 0 | 0 | 0 |
| gujrat-fixed-midpoint | 2 | Subbase | 4/63 | 59 | 0.375% | 0 | 0 | 0 |
| gujrat-fixed-midpoint | 3 | Base | 445/634 | 189 | 28.058% | 76 | 0 | 11 |
| gujrat-fixed-midpoint | 3 | Subbase | 4/63 | 59 | 0.375% | 0 | 0 | 0 |
| gujrat-fixed-midpoint | 4 | Base | 457/641 | 184 | 28.815% | 24 | 11 | 4 |
| gujrat-fixed-midpoint | 4 | Subbase | 4/63 | 59 | 0.375% | 0 | 0 | 0 |
| gujrat-scoped-active | 0 | Base | 369/547 | 178 | 23.266% | 0 | 0 | 0 |
| gujrat-scoped-active | 0 | Subbase | 4/63 | 59 | 0.375% | 0 | 0 | 0 |
| gujrat-scoped-active | 1 | Base | 369/547 | 178 | 23.266% | 0 | 0 | 0 |
| gujrat-scoped-active | 1 | Subbase | 6/92 | 86 | 0.562% | 2 | 0 | 27 |
| gujrat-scoped-active | 2 | Base | 369/547 | 178 | 23.266% | 0 | 0 | 0 |
| gujrat-scoped-active | 2 | Subbase | 6/92 | 86 | 0.562% | 0 | 0 | 0 |
| gujrat-scoped-active | 3 | Base | 369/547 | 178 | 23.266% | 0 | 0 | 0 |
| gujrat-scoped-active | 3 | Subbase | 6/92 | 86 | 0.562% | 0 | 0 | 0 |
| gujrat-scoped-active | 4 | Base | 369/547 | 178 | 23.266% | 0 | 0 | 0 |
| gujrat-scoped-active | 4 | Subbase | 6/92 | 86 | 0.562% | 0 | 0 | 0 |

| Run | Request | Layer | Native trace | Chainage (m) | Answer outcome |
|---|---:|---:|---:|---:|---|
| gujrat-fixed-active | 1 | 3 | 6724 | 168.10 | correction |
| gujrat-fixed-active | 2 | 3 | 6492 | 162.30 | correction |
| gujrat-fixed-active | 3 | 3 | 692 | 17.30 | unavailable_reviewed_answer |
| gujrat-fixed-active | 4 | 3 | 296 | 7.40 | correction |
| gujrat-fixed-midpoint | 1 | 3 | 3360 | 84.00 | unavailable_reviewed_answer |
| gujrat-fixed-midpoint | 2 | 3 | 7776 | 194.40 | unavailable_reviewed_answer |
| gujrat-fixed-midpoint | 3 | 2 | 3432 | 85.80 | correction |
| gujrat-fixed-midpoint | 4 | 2 | 7736 | 193.40 | correction |
| gujrat-scoped-active | 1 | 3 | 2272 | 56.80 | correction |
| gujrat-scoped-active | 2 | 3 | 692 | 17.30 | unavailable_reviewed_answer |
| gujrat-scoped-active | 3 | 3 | 10608 | 265.20 | unavailable_reviewed_answer |
| gujrat-scoped-active | 4 | 3 | 1552 | 38.80 | unavailable_reviewed_answer |

Mandiali spans 11.45 m, so every 25 m radius query window covers the entire road. Both saved final checkpoints produced exactly identical next requests under the original and scoped policies (including priority and candidate samples). This verifies those saved states; earlier complete states were not retained by the original replay.

Reproduce with `.venv/Scripts/python.exe scripts/experiment_scoped_queries.py prepare`, then `run`, `mandiali`, and `report`. Preparation deliberately refuses to overwrite the frozen experiment. `run-command.json` records the exact replay command; `ranking-only.patch` contains the source change. `experiment.json`, `actions.json`, `gujrat-comparison.json`, `regression.json`, and `mandiali-checkpoint-equivalence.json` retain source hashes, decisions, metrics and exact checks.
