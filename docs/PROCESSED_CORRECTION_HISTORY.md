# Processed correction replay

Processed projects persist `local_correction_actions` in run parameters. Each
successful local retrack appends one immutable JSON snapshot containing:

- Exact start/end chainages, affected interfaces and accepted-pick preservation mode.
- The operating station samples, visibility and waveform metadata used in that fit.
- Generic anchors, structural-break context, DZT hash and processed trace stride.

Reopening replays these actions in order. Editing another interface at the same
station, or editing one interface again, does not replace an earlier action's
observation or widen its scope. Saved current stations remain available for the
workbench; their final values do not enter earlier historical correction fits.
Snapshots preserve the original model-seed context used by each local action.
An explicit new local refit is required to replace a historical correction window
with changed model-seed context. Replay assumes the same tracking configuration;
it is not a cache of accepted physical truth.

GUI correction samples are committed after the worker succeeds. Cancellation or
failure restores the prior station state and leaves the completed action history
unchanged. Undoing a correction retracks its window and records the resulting
removal as another action. Generic review confirmations retain their separate,
proposal-specific review-event contract.

Undo removes the most recently created station, regardless of road distance.
Saving an existing station preserves its creation time. Repeated edits to one
station remain distinct correction actions, while station undo removes that
station as a whole.

Projects predating this log have `local_correction_actions = null` or no field.
Their first reconstruction uses the historical final-station/25 m-radius behavior
and records `legacy final-station fallback` in `local_correction_replay.mode`.
It freezes those inferred operations for subsequent reopens. Missing earlier
samples and per-layer action order cannot be recovered from old final stations.
An empty action list means no completed correction actions and does not trigger
legacy replay.

Focused checks:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_processed_action_history.py tests/test_processed_workflow_verification.py -q
```
