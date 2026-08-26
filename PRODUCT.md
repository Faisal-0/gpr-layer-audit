# Product

<!-- impeccable:product-schema 1 -->

## Platform

Windows desktop

## Stack

Python 3.11+ with PySide6, PyQtGraph, NumPy, SciPy, SQLite, openpyxl, and Matplotlib. Development runs from source through the BAT/CMD launcher or `uv`; packaging is deferred while the tracker remains experimental.

## Users

Internal pavement and GPR analysts reviewing long GSSI road surveys. They work primarily on Windows workstations and need to inspect exceptions without manually tracing every metre of a road.

## Product Purpose

GPR Layer Audit imports GSSI road, GPS, metadata, and metal-plate files; detects pavement interfaces; converts travel time to explicitly sourced thickness estimates; requests human input where evidence is weak; and compares accepted measurements with construction design thickness.

## Positioning

The product separates directly observed interface travel time from dielectric assumptions and physical thickness. It automates routine tracking without presenting assumed dielectric values or interpolated analyst labels as measured truth.

## Operating Context

Projects contain `.DZT`, `.DZG`, and `.DZX` acquisition files, a metal-plate calibration survey, optional RADAN-derived reference workbooks, and optional chainage-based design schedules. Analysts work from a radar-first review bench and export an auditable Excel/CSV/GeoJSON evidence package.

## Capabilities and Constraints

- Raw survey files are read-only.
- Asphalt, base, and sub-base are the default ordered layers; project scope remains configurable.
- The signal-only path is immutable. Design thickness may influence only a separately retained second pass through a soft prior capped at 20%; disagreements are sent to review.
- Physical thickness requires a dielectric source. Reflection-derived, analyst-supplied, explicitly assumed, and unresolved values remain distinguishable.
- The available data contain no independent core truth, so reports describe estimated rather than independently verified physical thickness.
- Processing is CPU-only, cancellable, deterministic, and suitable for long files through memory mapping and adaptive horizontal stacking.
- A path may contain explicit no-pick spans. Gaps are not drawn or exported as confident continuous interfaces.
- A maximum of five seed stations identifies the interface families. After the first three are complete, uncertain regions are retracked from raw traces at finer horizontal resolution; later corrections retrack only their bounded segment with continuity constraints at both ends.

## Evidence on Hand

- Real GSSI surveys and reference workbooks under `GPR Data/`.
- Talagang contains a 75,436-trace road file, matching GPS and DZX metadata, a metal-plate survey, and a reviewed/interpolated reference workbook.
- `scan_settings_used.jpeg` records the 15 ns range and manually entered dielectric constant of 7.0.
- No core measurements are currently available.

## Product Principles

- Preserve measurement provenance before optimizing convenience.
- Automate the ordinary and focus analyst attention on exceptions.
- Fail visibly when calibration or dielectric evidence is insufficient.
- Keep design compliance separate from measurement.
- Make every exported value reproducible from source hashes, configuration, and edit history.

## Accessibility & Inclusion

The workbench supports keyboard operation, high-DPI displays, strong focus visibility, and layer identification by both colour and line style.
