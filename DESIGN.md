# GPR Layer Audit Design System

## Product character

The workbench is a scientific inspection instrument, not a dashboard. The radar evidence owns the central workspace; project scope and dielectric state stay narrow at left, while exceptions and compact numeric results stay at right. The interface direction is an oscilloscope and continuous strip-chart bench (concept seed `6dd06bf2`).

## Visual system

- Backgrounds: ink-black radar field `#071016`, workbench `#0d1720`, instrument chrome `#13222d` and `#1a303d`.
- Primary text: `#dbe7ed`; secondary text: `#9eb3bd`; dividers: `#2b4350`.
- Interface 1 / asphalt: cyan `#28d7e5`, solid line.
- Interface 2 / base: amber `#ffc857`, dashed line.
- Interface 3 / sub-base: coral `#ff6b6b`, dotted line.
- Typography: Segoe UI Variable/Segoe UI, dense 13 px default, uppercase tracked section labels.
- Corners remain tight (3–4 px). Hairline borders and compact controls preserve the instrument feel.

Colour never carries layer identity alone: every layer also has a fixed order, label, and line style. The palette is selected for useful red/cyan/amber separation on dark and grayscale radargrams.

## Interaction conventions

- Automatic analysis is the primary action (`Ctrl+R`); export is available only after a result exists (`Ctrl+E`).
- Two-way time increases downward. Radargram and A-scan are synchronized.
- `Enhanced view` toggles the automatic interpretation branch against its amplitude-preserving input.
- `Ctrl+click` adds an anchor for the selected layer. Re-tracking is bounded to ±25 m with 10 m context; picks outside the segment are preserved bit-for-bit.
- `Ctrl+Z` removes the last anchor and repeats the same bounded re-track.
- Double-clicking an exception zooms to its chainage range.
- Design comparison is imported separately and never changes interface times or thickness measurements.
- Yellow text indicates an explicit dielectric assumption or calibration limitation, never a successful physical calibration.

## Scientific state language

- `high_confidence`: automatically accepted evidence above the configured gate.
- `review`: a plausible path that needs analyst attention.
- `unresolved`: travel-time evidence is retained, but physical thickness is not manufactured.
- `accepted`: a reviewed measurement explicitly accepted by an analyst.

Automatic preprocessing is recorded and reversible for viewing. Dewow, zero-phase band-pass, bounded time gain, trace normalization, denoising, and matched filtering operate only on the interpretation branch. Reflection-amplitude dielectric uses the isolated calibration branch.
