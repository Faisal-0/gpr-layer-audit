"""Native processed-input tracking shared by the application and interaction replay.

Processed samples are already interpreted acquisition coordinates. This explicit
mode neither stacks nor estimates a new surface alignment. DZX picks are never
read by this module. A correction recomputes evidence with the original seed
context, then replaces only its declared road window and layer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np

from gpr_layer_audit.io.dzt import DZTFile, fingerprint_file
from gpr_layer_audit.models import (
    AnalysisResult,
    CalibrationDiagnostics,
    LayerSpec,
    SeedRequest,
    TrackingProvenance,
    VisibilityState,
)
from gpr_layer_audit.seeds import correction_seed_stations, model_seed_stations
from gpr_layer_audit.time_coordinates import measurement_zero_sample, sample_twtt_ns

from .active_queries import request_observation
from .conventional_config import resolve_config
from .conventional_signal import numerical_extension, processed_boundary_mask
from .tracker import pick_interfaces


def native_anchors(stations, chainage, extra=None):
    """Reject grid-incompatible observations; never snap a seed to another trace."""
    anchors = {}
    step = float(np.median(np.diff(chainage))) if len(chainage) > 1 else 1.0
    for station in stations:
        row = int(round((station.chainage_m - chainage[0]) / step))
        if not 0 <= row < len(chainage) or abs(chainage[row] - station.chainage_m) > 1e-6:
            raise ValueError(
                "Processed seed must name an exact native trace; snapping is prohibited"
            )
        for order in station.samples:
            value = station.visible_sample(order)
            if value is None:
                continue
            if not np.isfinite(value) or abs(value - round(value)) > 1e-6:
                raise ValueError("Processed seed must name an exact native sample")
            anchors.setdefault(order, {})[row] = int(round(value))
    for order, observations in (extra or {}).items():
        for distance, sample in observations:
            row = int(round((distance - chainage[0]) / step))
            if (
                not 0 <= row < len(chainage)
                or abs(chainage[row] - distance) > 1e-6
                or abs(sample - round(sample)) > 1e-6
            ):
                raise ValueError("Processed anchor must name exact native trace/sample coordinates")
            anchors.setdefault(order, {})[row] = int(round(sample))
    return anchors


def fit_processed(
    measurement,
    valid,
    surface,
    dt,
    step,
    layers,
    anchors,
    *,
    method="seed_hybrid",
    config=None,
    seed_metadata=None,
    pulse_anchors=None,
    breaks=(),
    cancel=None,
):
    """Run the public tracker with the frozen processed-radar numerical contract."""
    config = resolve_config(config)
    if pulse_anchors is not None:
        seed_metadata = deepcopy(seed_metadata or {})
        for order, values in anchors.items():
            for row, sample in values.items():
                seed_metadata.setdefault(order, {}).setdefault(row, {})["pulse_estimation_use"] = (
                    pulse_anchors.get(order, {}).get(row) == sample
                )
    working = (
        numerical_extension(measurement, valid) if config.signal_validity else measurement.copy()
    )
    branches = {"hybrid_measurement": working}
    if config.signal_validity:
        branches["sample_validity"] = valid
    return pick_interfaces(
        working,
        surface,
        layers,
        anchor_samples=anchors,
        seed_metadata=seed_metadata,
        feature_branches=branches,
        method=method,
        horizontal_step_m=step,
        sample_interval_ns=dt,
        ml_policy="off",
        conventional_config=asdict(config),
        break_rows=set(breaks),
        cancel=cancel,
    )


def merge_local_paths(previous, regenerated, *, orders, start_row, stop_row):
    """Replace a declared correction window; proposals/evidence outside stay frozen."""
    result = deepcopy(previous)
    for order in orders:
        if order not in result or order not in regenerated:
            continue
        old, new = result[order], regenerated[order]
        use = slice(start_row, stop_row + 1)
        for name in old.__dataclass_fields__:
            a, b = getattr(old, name), getattr(new, name)
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
                a[use] = b[use]
            elif name in {"evidence", "candidate_components"}:
                for key, values in b.items():
                    if key in a and isinstance(a[key], np.ndarray) and a[key].shape == values.shape:
                        a[key][use] = values[use]
        old.provenance = deepcopy(old.provenance)
        # Full-fit hypotheses cannot be attached to a partially replaced display.
        base = old.provisional_samples if old.provisional_samples is not None else old.samples
        proposed = new.provisional_samples if new.provisional_samples is not None else new.samples
        hypotheses = []
        for h in old.provenance.get("hypothesis_samples", []):
            merged = np.asarray(h, dtype=np.int32).copy()
            merged[use] = proposed[use]
            hypotheses.append(merged.tolist())
        for h in new.provenance.get("hypothesis_samples", []):
            merged = base.copy()
            merged[use] = np.asarray(h)[use]
            if not any(np.array_equal(merged, old_h) for old_h in hypotheses):
                hypotheses.append(merged.tolist())
        old.provenance["hypothesis_samples"] = hypotheses
        retained = [
            q
            for q in old.provenance.get("suggested_observations", [])
            if not start_row <= q["row"] <= stop_row
        ]
        retained.extend(
            q
            for q in new.provenance.get("suggested_observations", [])
            if start_row <= q["row"] <= stop_row
        )
        old.provenance["suggested_observations"] = retained
        old.provenance.setdefault("local_updates", []).append(
            {
                "start_row": start_row,
                "stop_row": stop_row,
                "outside_preserved": True,
                "seed_context": "full original interval",
            }
        )
    return result


def guard_local_order(paths, previous, layers, anchors, orders, start_row, stop_row):
    """Withhold changed automation conflicting with a preserved neighboring layer."""
    gaps = {layer.order: layer.min_gap_samples for layer in layers}
    for order in orders:
        if order not in paths:
            continue
        path = paths[order]
        for other, neighbor in previous.items():
            if other == order:
                continue
            use = np.arange(start_row, stop_row + 1)
            measured = (path.samples[use] >= 0) & path.visible[use]
            measured &= (neighbor.samples[use] >= 0) & neighbor.visible[use]
            if other < order:
                conflict = measured & (path.samples[use] < neighbor.samples[use] + gaps[order])
            else:
                conflict = measured & (neighbor.samples[use] < path.samples[use] + gaps[other])
            bad = use[conflict]
            if any(row in anchors.get(order, {}) for row in bad):
                raise ValueError(
                    "Correction contradicts a preserved neighboring interface; "
                    "review both interface observations"
                )
            path.samples[bad] = -1
            path.visible[bad] = False
            path.confidence[bad] = 0
            if "hybrid_accepted" in path.evidence:
                path.evidence["hybrid_accepted"][bad] = 0
    return paths


def _refresh_result(result, options):
    from .pipeline import (
        _aggregate_results,
        _dielectric_from_result,
        _profile_points,
        _review_issues,
    )

    result.thickness = _aggregate_results(
        result.picks,
        options.layer_specs,
        result.header,
        options.report_interval_m,
        _dielectric_from_result(result),
        measurement_zero_sample(result),
    )
    result.review_issues = _review_issues(result.picks)
    result.profile = _profile_points(
        result.thickness,
        options.layer_specs,
        options.layer_designs,
        options.design_segments,
        result.anomaly_regions,
    )


def _refresh_paths_and_requests(result, options):
    from .pipeline import _attach_candidate_metadata, _candidate_events

    paths = result.processed_paths
    result.parameters["hybrid_provenance"] = {str(o): p.provenance for o, p in paths.items()}
    result.provisional_paths = {
        o: p.provisional_samples.copy()
        for o, p in paths.items()
        if p.provisional_samples is not None
    }
    result.signal_only_paths = {o: p.samples.copy() for o, p in paths.items()}
    result.candidate_events = _candidate_events(
        paths, result.interpretation_input_radargram, result.chainage_m, {}
    )
    _attach_candidate_metadata(result.picks, result.candidate_events)
    anchors = native_anchors(options.seed_stations, result.chainage_m, options.anchors)
    step = result.header.distance_per_trace_m * result.parameters["processed_stride"]
    visited = {
        (order, int(round(station.chainage_m / step)))
        for station in options.seed_stations
        for order, confirmed in station.user_confirmed.items()
        if confirmed
    }
    visited.update(
        (pick.layer_order, int(round(pick.chainage_m / step)))
        for pick in result.picks
        if pick.provenance == TrackingProvenance.MANUAL_CORRECTION
    )
    query = request_observation(
        {o: p for o, p in paths.items() if o in options.query_layer_orders},
        result.interpretation_input_radargram,
        result.sample_validity,
        anchors,
        step,
        visited=visited,
    )
    result.proposed_seed_requests = []
    result.proposed_seed_chainages = []
    if query:
        result.proposed_seed_requests = [
            SeedRequest(
                chainage_m=float(query["row"] * step),
                layer_orders=[query["layer_order"]],
                reason=query["reason"],
                priority=query["priority"],
            )
        ]
        result.proposed_seed_chainages = [float(query["row"] * step)]


def _paths_to_picks(result, options, paths, anchors):
    from .pipeline import _interface_picks

    picks = _interface_picks(
        paths,
        options.layer_specs,
        result.interpretation_input_radargram,
        result.reference_surface_sample,
        result.header.sample_interval_ns,
        np.arange(len(result.chainage_m)) * result.parameters["processed_stride"],
        result.chainage_m,
        anchors,
        options.confidence_threshold,
        options.layer_confidence_thresholds,
    )
    # A proposal is never an accepted TWTT, including in the older shallow path.
    from gpr_layer_audit.models import PickStatus

    for pick in picks:
        if pick.status not in {PickStatus.ACCEPTED, PickStatus.HIGH_CONFIDENCE}:
            pick.twtt_ns = float("nan")
        elif np.isfinite(pick.twtt_ns):
            # Integer graph surface indexing is not the exact DZT time origin.
            pick.twtt_ns = sample_twtt_ns(result, pick.sample_index)
    return picks


def sync_processed_review(result, options, issue):
    """Keep the active-query representation consistent with an explicit review."""
    step = result.header.distance_per_trace_m * result.parameters["processed_stride"]
    path = result.processed_paths[issue.layer_order]
    for pick in result.picks:
        if (
            pick.layer_order != issue.layer_order
            or not issue.start_chainage_m <= pick.chainage_m <= issue.end_chainage_m
        ):
            continue
        row = int(round(pick.chainage_m / step))
        sample = int(round(pick.selected_lobe_sample)) if pick.is_accepted_measurement else -1
        path.samples[row] = sample
        path.visible[row] = sample >= 0
        path.confidence[row] = 1.0 if sample >= 0 else 0.0
        if "hybrid_accepted" in path.evidence:
            path.evidence["hybrid_accepted"][row] = float(sample >= 0)
    _refresh_paths_and_requests(result, options)


def _run(result, options, stations, *, cancel=None):
    from .pipeline import _seed_metadata_rows

    chainage = result.chainage_m
    step = result.header.distance_per_trace_m * result.parameters["processed_stride"]
    anchors = native_anchors(stations, chainage, options.anchors)
    metadata = _seed_metadata_rows(stations, chainage, include_corrections=True)
    # Selected-lobe coordinates remain canonical in this explicit processed mode.
    # No packet-centre conversion is asserted to be a separate measurement.
    for order, values in anchors.items():
        for row, sample in values.items():
            metadata.setdefault(order, {}).setdefault(row, {})["canonical_sample_index"] = sample
    breaks = set()
    for x in options.structural_breaks_m:
        row = int(round(x / step))
        if 0 <= row < len(chainage):
            breaks.add(row)
    paths = fit_processed(
        result.interpretation_input_radargram,
        result.sample_validity,
        result.reference_surface_sample,
        result.header.sample_interval_ns,
        step,
        options.layer_specs,
        anchors,
        method=options.tracker_method,
        config=options.conventional_config,
        seed_metadata=metadata,
        pulse_anchors=native_anchors(
            model_seed_stations(options.seed_stations), chainage, options.anchors
        ),
        breaks=breaks,
        cancel=cancel,
    )
    return paths, anchors


def _apply_path_visibility(paths, stations, chainage):
    """An explicit negative answer invalidates every measurement representation."""
    step = float(np.median(np.diff(chainage))) if len(chainage) > 1 else 1.0
    for station in stations:
        row = int(round((station.chainage_m - chainage[0]) / step))
        if not 0 <= row < len(chainage):
            continue
        for order, visibility in station.visibility.items():
            if (
                order not in paths
                or not station.user_confirmed.get(order, False)
                or visibility not in {VisibilityState.NOT_VISIBLE, VisibilityState.ABSENT}
            ):
                continue
            path = paths[order]
            for name in (
                "samples",
                "provisional_samples",
                "alternate_samples",
                "signal_only_samples",
                "design_guided_samples",
            ):
                values = getattr(path, name)
                if values is not None:
                    values[row] = -1
            path.visible[row] = False
            path.confidence[row] = 0
            for name in ("hybrid_accepted", "hybrid_path_alternate"):
                if name in path.evidence:
                    path.evidence[name][row] = -1 if name.endswith("alternate") else 0
            for hypothesis in path.provenance.get("hypothesis_samples", []):
                hypothesis[row] = -1


def _restore_path_row(path, previous, row):
    for name in path.__dataclass_fields__:
        current, old = getattr(path, name), getattr(previous, name)
        if isinstance(current, np.ndarray) and isinstance(old, np.ndarray):
            current[row] = old[row]
        elif name in {"evidence", "candidate_components"}:
            for key in current.keys() & old.keys():
                current[key][row] = old[key][row]
    for hypothesis in path.provenance.get("hypothesis_samples", []):
        hypothesis[row] = int(previous.samples[row])


def analyze_processed(source, options, *, progress=None, cancel=None):
    from .dielectric import resolve_dielectric
    from .pipeline import _apply_seed_visibility

    if cancel and cancel():
        raise InterruptedError("Analysis cancelled")
    if progress:
        progress(5, "Reading native processed coordinates")
    road = DZTFile(source.dzt_path)
    step, dt = road.header.distance_per_trace_m, road.header.sample_interval_ns
    if step is None or step <= 0 or dt <= 0:
        raise ValueError("Processed tracking needs positive trace and sample intervals")
    if options.stack_size not in (0, 1):
        raise ValueError("Processed input preserves native traces; set stack size to 1")
    stride = options.processed_stride
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("Processed trace stride must be a positive integer")
    measurement = np.asarray(road.channel()[::stride], dtype=np.float32).copy()
    step *= stride
    valid = processed_boundary_mask(measurement)
    surface = int(np.clip(round(-road.header.position_ns / dt), 0, measurement.shape[1] - 1))
    source.fingerprint = fingerprint_file(source.dzt_path)
    # Broad radar bounds are shared with the processed evaluator; no design depths.
    layers = [
        replace(
            layer,
            min_offset_samples=1,
            max_offset_samples=measurement.shape[1] - 1 - surface,
            min_gap_samples=1,
        )
        for layer in options.layer_specs
        if layer.analysis_enabled
    ]
    options = replace(options, layer_specs=layers)
    dielectric = {}
    for layer in layers:
        design = next((d for d in options.layer_designs if d.layer_order == layer.order), None)
        value, origin = resolve_dielectric(
            None,
            options.analyst_dielectric.get(
                layer.order, design.dielectric if design else layer.dielectric
            ),
            road.header.dielectric,
            options.accept_scan_dielectric,
        )
        dielectric[str(layer.order)] = {"value": value, "source": origin.value}
    result = AnalysisResult(
        source=source,
        header=road.header,
        stack_size=1,
        chainage_m=np.arange(len(measurement)) * step,
        calibrated_radargram=measurement.copy(),
        surface_samples_raw=np.full(len(measurement), surface),
        reference_surface_sample=surface,
        picks=[],
        thickness=[],
        review_issues=[],
        diagnostics=CalibrationDiagnostics(
            False,
            messages=[
                "Processed samples retained without stacking, surface refitting "
                "or amplitude calibration.",
                "Interpretation agreement does not establish physically calibrated thickness.",
            ],
        ),
        sample_validity=valid,
        interpretation_input_radargram=measurement,
        seed_stations=list(options.seed_stations),
        display_radargrams={"Raw": measurement, "Clean": measurement},
        parameters={
            "input_mode": "processed",
            "processed_stride": stride,
            "query_layer_orders": list(options.query_layer_orders),
            "tracker_method": options.tracker_method,
            "conventional_config": asdict(resolve_config(options.conventional_config)),
            "stack_size": 1,
            "ml_policy": "off",
            "required_seed_orders": [],
            "required_seed_count": 0,
            "report_interval_m": options.report_interval_m,
            "coordinate_provenance": {
                "mode": "processed",
                "trace_origin": 0,
                "sample_origin_ns": road.header.position_ns,
                "dt_ns": dt,
                "graph_surface_sample": surface,
                "measurement_zero_sample": -road.header.position_ns / dt,
                "twtt_formula": "sample_origin_ns + canonical_sample * dt_ns",
                "trace_spacing_m": step,
                "orientation": "stored acquisition order",
                "transform": "native sample; row times declared stride",
                "trace_stride": stride,
                "dzt_sha256": source.fingerprint,
            },
            "dielectric_by_layer": dielectric,
            "accept_scan_dielectric": options.accept_scan_dielectric,
            "analyst_dielectric": dict(options.analyst_dielectric),
            "processed_layers": [asdict(layer) for layer in layers],
            "local_correction_replay": {"count": 0, "scope": "saved 25 m radius"},
            "local_correction_order": list(options.local_correction_order),
            "structural_breaks_m": list(options.structural_breaks_m),
        },
    )
    if progress:
        progress(25, "Tracking seeded reflectors in native processed coordinates")
    paths, anchors = _run(
        result, options, model_seed_stations(options.seed_stations), cancel=cancel
    )
    if cancel and cancel():
        raise InterruptedError("Analysis cancelled")
    _apply_path_visibility(paths, model_seed_stations(options.seed_stations), result.chainage_m)
    result.picks = _paths_to_picks(result, options, paths, anchors)
    result.processed_paths = paths
    _apply_seed_visibility(
        result.picks, model_seed_stations(options.seed_stations), result.chainage_m
    )
    result.parameters["hybrid_provenance"] = {str(o): p.provenance for o, p in paths.items()}
    corrections = correction_seed_stations(options.seed_stations)
    ordering = {value: index for index, value in enumerate(options.local_correction_order)}
    corrections.sort(key=lambda station: ordering.get(station.station_id, -1))
    replayed = list(model_seed_stations(options.seed_stations))
    for station in corrections:
        replayed.append(station)
        retrack_processed(
            result,
            replace(options, seed_stations=list(replayed)),
            max(0, station.chainage_m - 25),
            station.chainage_m + 25,
            cancel=cancel,
            layer_orders={o for o, confirmed in station.user_confirmed.items() if confirmed},
        )
    result.seed_stations = list(options.seed_stations)
    result.parameters["local_correction_replay"]["count"] = len(
        correction_seed_stations(options.seed_stations)
    )
    _refresh_paths_and_requests(result, options)
    _refresh_result(result, options)
    if progress:
        progress(100, "Native processed tracking complete")
    return result


def retrack_processed(
    result, options, start, stop, *, preserve_accepted=False, cancel=None, layer_orders=None
):
    from gpr_layer_audit.models import PickStatus

    from .pipeline import _apply_seed_visibility

    layers = [LayerSpec(**item) for item in result.parameters["processed_layers"]]
    options = replace(options, layer_specs=layers)
    stations = [
        s for s in options.seed_stations if s.role != "correction" or start <= s.chainage_m <= stop
    ]
    corrected_orders = {
        o
        for s in stations
        if s.role == "correction"
        for o, confirmed in s.user_confirmed.items()
        if confirmed
    }
    orders = set(layer_orders) if layer_orders is not None else corrected_orders
    if layer_orders is None and options.local_correction_order:
        latest = next(
            (
                s
                for identifier in reversed(options.local_correction_order)
                for s in stations
                if s.station_id == identifier
            ),
            None,
        )
        if latest:
            orders = {o for o, confirmed in latest.user_confirmed.items() if confirmed}
    orders = orders or {layer.order for layer in layers}
    paths, anchors = _run(result, options, stations, cancel=cancel)
    if cancel and cancel():
        raise InterruptedError("Analysis cancelled")
    step = result.header.distance_per_trace_m * result.parameters["processed_stride"]
    lo = max(0, int(np.ceil(start / step)))
    hi = min(len(result.chainage_m) - 1, int(np.floor(stop / step)))
    merged = merge_local_paths(
        result.processed_paths, paths, orders=orders, start_row=lo, stop_row=hi
    )
    retained = set()
    confirmed_rows = {
        (o, int(round(s.chainage_m / step)))
        for s in stations
        if s.role == "correction"
        for o, confirmed in s.user_confirmed.items()
        if confirmed
    }
    if preserve_accepted:
        for pick in result.picks:
            row = int(round(pick.chainage_m / step))
            key = (pick.layer_order, row)
            if (
                pick.layer_order in orders
                and lo <= row <= hi
                and key not in confirmed_rows
                and pick.status in {PickStatus.ACCEPTED, PickStatus.HIGH_CONFIDENCE}
            ):
                _restore_path_row(
                    merged[pick.layer_order], result.processed_paths[pick.layer_order], row
                )
                retained.add((pick.layer_order, pick.trace_index))
    _apply_path_visibility(merged, stations, result.chainage_m)
    guard_local_order(merged, result.processed_paths, layers, anchors, orders, lo, hi)
    replacements = {
        (p.layer_order, p.trace_index): p
        for p in _paths_to_picks(result, options, merged, anchors)
        if p.layer_order in orders and start <= p.chainage_m <= stop
    }
    _apply_seed_visibility(
        list(replacements.values()), stations, result.chainage_m, include_corrections=True
    )
    if cancel and cancel():
        raise InterruptedError("Analysis cancelled")
    # Mutation is delayed until the complete fit succeeds, including cancellation.
    result.picks = [
        p
        if (p.layer_order, p.trace_index) in retained
        else replacements.get((p.layer_order, p.trace_index), p)
        for p in result.picks
    ]
    result.processed_paths = merged
    result.seed_stations = list(options.seed_stations)
    result.parameters["local_correction_order"] = list(options.local_correction_order)
    result.parameters["query_layer_orders"] = list(options.query_layer_orders)
    result.parameters["structural_breaks_m"] = list(options.structural_breaks_m)
    result.parameters.setdefault("processed_local_retracks", []).append(
        {
            "start_chainage_m": start,
            "end_chainage_m": stop,
            "layer_orders": sorted(orders),
            "outside_preserved": True,
            "operation": "local_correction",
            "compute_scope": "full original seed context; replace declared window only",
        }
    )
    _refresh_result(result, options)
    _refresh_paths_and_requests(result, options)
    return result
