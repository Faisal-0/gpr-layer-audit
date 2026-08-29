# Cross-road tracking progress — 2026-08-29

## Current decision

The program is materially safer and more auditable, but reliable pavement-layer
accuracy is **not yet proven**. The remaining release blocker is independent
radar-only validation, not another confidence threshold. Available development
seeds cover only Talagang and Pattoki-Jhoru; they were used during development
and cannot serve as blind accuracy evidence.

## Changes made

- Preserved a dewow-only, surface-flattened road measurement before metal-plate
  subtraction. `interpretation_input_radargram` and the UI Raw view now expose
  that actual acquisition branch rather than a plate-subtracted copy.
- Added packet-scale, dip-steered lateral support computed only from that
  pre-subtraction branch. It has zero weight in candidate ranking and therefore
  cannot select a reflector.
- Added a low fail-closed automatic-pick veto for sustained absence of the raw
  packet. Manual seeds remain authoritative and are restored regardless of the
  automatic gate.
- Fixed coarse segment retracking so it no longer reapplies gain, filtering,
  and denoising to the already enhanced output. The cropped matched template
  is also no longer passed where a full-length plate waveform is required.
- Fixed metal-plate candidate ranking. Gain mismatch still keeps all plates
  below amplitude-calibration suitability, while road/plate name and zone
  evidence now survives for waveform pairing.
- Ambiguous plate pairings are no longer auto-selected in either the UI or CLI.

## Bounded seed-identity checks

The production graph family did not change because raw measurement support is
an acceptance gate, not a selector.

| Section | Retained before raw gate | Retained after raw gate | Explicit gaps after | Seed samples |
|---|---:|---:|---:|---|
| Talagang 950–1250 m, exploratory deeper seeds | 30.13% | 29.33% | 70.67% | exact |
| Pattoki-Jhoru 300–750 m, confirmed base seeds | 62.22% | 60.80% | 39.20% | exact |

The Talagang exploratory seeds are analyst hypotheses, not field truth. The
Pattoki check confirms that a user-seeded persistent horizontal event is not
globally blacklisted.

## Cross-road acquisition readiness

`scripts/audit_cross_road_readiness.py` inspected a bounded centre window from
all 17 road acquisitions under `GPR Data`.

- Pre-subtraction support was finite on 17/17 roads.
- The low raw gate left 87.7–92.9% of below-surface locations eligible before
  the tracker's other identity, continuity, and confidence checks.
- 11/17 plate pairings had a distinct name/zone winner; six remain ambiguous
  and now require analyst selection.
- 0/17 winning plate candidates are gain compatible. Plate waveform timing may
  be used, but amplitude-derived dielectric must remain disabled.

This proves acquisition-path robustness, not pavement-layer accuracy.

## Sparse-seed recovery diagnostic

`scripts/evaluate_sparse_seed_recovery.py` withheld each provisional Talagang
or Pattoki development station and ran the full radar-only graph with the other
stations.

- All remaining training seeds were preserved exactly.
- 9/10 comparable withheld interfaces were graph-selected within one declared
  pulse width.
- 6/10 were automatically visible; the others failed closed.
- The sole event-identity miss was Pattoki base at 325.8 m: training seeds were
  both more than 400 m away, candidates near the withheld 278-sample lobe were
  present, but none was seed-reachable. The graph selected the adjacent
  296-sample cycle and correctly hid it.

This demonstrates useful sparse-seed transfer and a remaining long-range
extrapolation limit. Because the seeds are development data, the result is not
an independent accuracy rate.

## What still blocks a reliability claim

1. Capture at least 30 blinded radar-only checkpoints per released layer,
   distributed across at least three roads and across visible, absent, and
   ambiguous spans. The UI checkpoint mode and retention audit already support
   this workflow.
2. Keep checkpoint identities completely out of seeds, design corridors,
   parameter tuning, and confidence calibration until the configuration is
   frozen.
3. Require candidate retention, correct graph family, accepted accuracy, and
   review coverage to pass separately. A hidden wrong cycle is safer than a
   false thickness but still fails automation coverage.
4. Add another seed when the requested layer must be extrapolated hundreds of
   metres beyond the nearest confirmed event or when a new construction/phase
   regime appears.
5. Do not release automated subbase tracking until the same gates pass for
   asphalt and base on independent roads.

## Reproducible evidence

- `exports/cross-road-readiness-20260829.json`
- `exports/sparse-seed-recovery-20260829.json`
- `exports/pattoki-holdout-a-diagnostic-v2-20260829.json`
- `exports/measurement-support-20260829/seed-identity-sections-gated-v2/summary.json`

Generated exports are intentionally unversioned. The scripts, tests, and this
assessment are versioned so the evidence can be recreated from the source
acquisitions.
