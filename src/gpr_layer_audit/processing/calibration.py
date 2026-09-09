from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter, uniform_filter1d

from gpr_layer_audit.io.dzt import DZTFile, DZTHeader
from gpr_layer_audit.models import CalibrationDiagnostics


@dataclass(slots=True)
class CalibratedData:
    radargram: NDArray[np.float32]
    measurement_radargram: NDArray[np.float32]
    raw_stacks: NDArray[np.float32]
    surface_samples: NDArray[np.int32]
    reference_surface_sample: int
    plate_template: NDArray[np.float32]
    surface_amplitudes: NDArray[np.float64]
    diagnostics: CalibrationDiagnostics
    trace_centres: NDArray[np.float64]
    sample_validity: NDArray[np.bool_] | None = None
    coordinate_provenance: dict = field(default_factory=dict)


def horizontal_stack(
    traces: NDArray[np.number], stack_size: int
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    if stack_size < 1:
        raise ValueError("stack_size must be at least 1")
    count, samples = traces.shape
    bins = (count + stack_size - 1) // stack_size
    output = np.empty((bins, samples), dtype=np.float32)
    centres = np.empty(bins, dtype=float)
    for index in range(bins):
        start = index * stack_size
        stop = min(start + stack_size, count)
        output[index] = np.median(np.asarray(traces[start:stop], dtype=np.float32), axis=0)
        centres[index] = (start + stop - 1) / 2.0
    return output, centres


def dewow(
    data: NDArray[np.floating], window_samples: int = 31, *, valid: NDArray[np.bool_] | None = None
) -> NDArray[np.float32]:
    if valid is not None:
        from .conventional_signal import numerical_extension

        data = numerical_extension(data, valid)
    window = max(3, int(window_samples) | 1)
    baseline = uniform_filter1d(data, size=window, axis=1, mode="nearest")
    return np.asarray(data - baseline, dtype=np.float32)


def _surface_search_bounds(samples: int) -> tuple[int, int]:
    return max(4, int(samples * 0.22)), min(samples - 4, int(samples * 0.43))


def _plate_template(plate: DZTFile, conventional=False) -> tuple[NDArray[np.float32], int, float]:
    plate_data = np.asarray(plate.channel(0), dtype=np.float32)
    mask = None
    if conventional:
        from .conventional_signal import processed_boundary_mask, source_layout

        layout = source_layout(plate)
        mask = layout.mask(plate_data)
        if layout.kind == "processed":
            mask = processed_boundary_mask(plate_data, mask)
    plate_clean = dewow(plate_data, valid=mask)
    template = np.median(plate_clean, axis=0).astype(np.float32)
    start, stop = _surface_search_bounds(template.size)
    peak = start + int(np.argmax(np.abs(template[start:stop])))
    return template, peak, float(template[peak])


def _shift_template(template: NDArray[np.float32], shift: int) -> NDArray[np.float32]:
    shifted = np.zeros_like(template)
    if shift >= 0:
        shifted[shift:] = template[: len(template) - shift]
    else:
        shifted[:shift] = template[-shift:]
    return shifted


def calibrate(
    road: DZTFile,
    plate: DZTFile | None,
    *,
    stack_size: int = 10,
    start_trace: int = 0,
    stop_trace: int | None = None,
    cancel: Callable[[], bool] | None = None,
    conventional: bool = False,
    alignment_before_stacking: bool = False,
) -> CalibratedData:
    stop_trace = road.header.trace_count if stop_trace is None else stop_trace
    source_values = road.channel(0, start_trace=start_trace, stop_trace=stop_trace)
    mask, provenance = None, {}
    if conventional:
        from dataclasses import asdict

        from .conventional_signal import (
            numerical_extension,
            processed_boundary_mask,
            shift_validity,
            source_layout,
            stack_valid,
        )

        layout = source_layout(road)
        raw_mask = layout.mask(source_values)
        if layout.kind == "processed":
            raw_mask = processed_boundary_mask(source_values, raw_mask)
        if alignment_before_stacking and stack_size > 1:
            # Explicit ablation: keep per-trace shifts for source coordinate provenance.
            individual = dewow(source_values, valid=raw_mask)
            lo, hi = _surface_search_bounds(individual.shape[1])
            raw_surface = lo + np.argmax(abs(individual[:, lo:hi]), axis=1)
            raw_reference = int(np.median(raw_surface))
            shifts = raw_reference - raw_surface
            shifted_mask = shift_validity(raw_mask, shifts)
            shifted = np.array(
                [_shift_template(t, int(s)) for t, s in zip(individual, shifts, strict=True)]
            )
            stacks, mask, centres = stack_valid(shifted, shifted_mask, stack_size)
            provenance["pre_stack_shifts_samples"] = shifts.tolist()
        else:
            stacks, mask, centres = stack_valid(source_values, raw_mask, stack_size)
        provenance.update(
            signal_layout=asdict(layout),
            source_start_trace=start_trace,
            stacking_order="align_then_stack" if alignment_before_stacking else "stack_then_align",
            invalid_source_samples=int(np.count_nonzero(~raw_mask)),
        )
    else:
        stacks, centres = horizontal_stack(source_values, stack_size)
    centres += start_trace
    clean = (
        stacks
        if conventional and alignment_before_stacking and stack_size > 1
        else dewow(stacks, valid=mask)
    )
    samples = clean.shape[1]
    start, stop = _surface_search_bounds(samples)
    median_trace = np.median(clean, axis=0)
    median_peak = start + int(np.argmax(np.abs(median_trace[start:stop])))
    polarity = -1.0 if median_trace[median_peak] < 0 else 1.0
    local_start = max(start, median_peak - 24)
    local_stop = min(stop, median_peak + 25)
    signed_window = clean[:, local_start:local_stop] * polarity
    surface = local_start + np.argmax(signed_window, axis=1)
    surface_amplitudes = clean[np.arange(len(clean)), surface].astype(float)
    reference = int(np.median(surface))

    diagnostics = CalibrationDiagnostics(
        valid_for_dielectric=False,
        reference_surface_sample=reference,
        surface_amplitude_median=float(np.median(surface_amplitudes)),
    )

    aligned = np.zeros_like(clean)
    for index, trace in enumerate(clean):
        if cancel and cancel():
            raise InterruptedError("Analysis cancelled")
        aligned[index] = _shift_template(trace, reference - int(surface[index]))
    if mask is not None:
        mask = shift_validity(mask, reference - surface)
        aligned = numerical_extension(aligned, mask)
        provenance.update(
            trace_centres=centres.tolist(),
            alignment_shifts_samples=(reference - surface).tolist(),
            stored_sample_coordinates_preserved=True,
        )

    plate_waveform = np.zeros(samples, dtype=np.float32)
    if plate is None:
        diagnostics.messages.append(
            "No metal-plate file supplied; bounce correction used surface alignment only."
        )
        return CalibratedData(
            aligned,
            aligned.copy(),
            stacks,
            surface.astype(np.int32),
            reference,
            plate_waveform,
            surface_amplitudes,
            diagnostics,
            centres,
            mask,
            provenance,
        )

    if plate.header.samples_per_trace != road.header.samples_per_trace:
        diagnostics.messages.append("Road and plate sample counts differ.")
        return CalibratedData(
            aligned,
            aligned.copy(),
            stacks,
            surface.astype(np.int32),
            reference,
            plate_waveform,
            surface_amplitudes,
            diagnostics,
            centres,
            mask,
            provenance,
        )

    plate_waveform, plate_peak, plate_amplitude = _plate_template(plate, conventional)
    plate_aligned = _shift_template(plate_waveform, reference - plate_peak)
    diagnostics.plate_peak_sample = plate_peak
    diagnostics.plate_peak_amplitude = plate_amplitude
    diagnostics.gain_compatible = (
        road.header.range_gain_bytes == plate.header.range_gain_bytes
        and road.header.bits_per_sample == plate.header.bits_per_sample
    )

    raw_plate = np.asarray(plate.channel(0))
    if np.issubdtype(raw_plate.dtype, np.integer):
        limits = np.iinfo(raw_plate.dtype)
        clipped = np.count_nonzero((raw_plate == limits.min) | (raw_plate == limits.max))
        diagnostics.clipping_fraction = clipped / raw_plate.size

    compatible = (
        diagnostics.gain_compatible
        and diagnostics.clipping_fraction < 1e-5
        and road.header.antenna == plate.header.antenna
        and np.isclose(road.header.range_ns, plate.header.range_ns, rtol=0, atol=1e-4)
    )
    denominator = float(plate_aligned[reference])
    corrected = aligned.copy()
    # Full waveform subtraction assumes that plate and road amplitudes share
    # the same acquisition scale.  When gain, antenna, time range, or clipping
    # checks fail, scaling the plate to the road surface can rewrite genuine
    # deep events.  Keep the surface-flattened road unchanged in that case;
    # the aligned plate template remains available to downstream matched-filter
    # and candidate branches as non-measurement evidence.
    subtraction_applied = bool(compatible and abs(denominator) > 1e-9)
    diagnostics.plate_subtraction_applied = subtraction_applied
    if subtraction_applied:
        scales = aligned[:, reference].astype(float) / denominator
        corrected[:, reference:] -= scales[:, None] * plate_aligned[None, reference:]
        # Remove isolated subtraction spikes while retaining thin-layer pulses.
        corrected = median_filter(corrected, size=(1, 3), mode="nearest").astype(np.float32)
    elif abs(denominator) <= 1e-9:
        diagnostics.messages.append("Metal-plate reference peak is zero after preprocessing.")

    diagnostics.valid_for_dielectric = bool(compatible and abs(denominator) > 1e-9)
    if not diagnostics.gain_compatible:
        diagnostics.messages.append(
            "Road and plate range-gain records differ; amplitude dielectric inversion is disabled."
        )
    if road.header.antenna != plate.header.antenna:
        diagnostics.messages.append("Road and plate antenna identifiers differ.")
    if diagnostics.clipping_fraction >= 1e-5:
        diagnostics.messages.append("Metal-plate signal contains clipped samples.")
    if diagnostics.valid_for_dielectric:
        diagnostics.messages.append("Plate amplitude calibration passed compatibility checks.")
    if subtraction_applied:
        diagnostics.messages.append(
            "Metal-plate waveform subtraction and surface flattening applied."
        )
    else:
        diagnostics.messages.append(
            "Metal-plate waveform subtraction skipped because compatibility checks failed; "
            "surface flattening and the road-only measurement branch were retained."
        )

    return CalibratedData(
        corrected,
        aligned,
        stacks,
        surface.astype(np.int32),
        reference,
        plate_aligned,
        surface_amplitudes,
        diagnostics,
        centres,
        mask,
        provenance,
    )


def headers_compatible_for_processing(road: DZTHeader, plate: DZTHeader) -> list[str]:
    problems: list[str] = []
    if road.samples_per_trace != plate.samples_per_trace:
        problems.append("sample count")
    if road.bits_per_sample != plate.bits_per_sample:
        problems.append("sample width")
    if road.antenna != plate.antenna:
        problems.append("antenna")
    if not np.isclose(road.range_ns, plate.range_ns, atol=1e-4):
        problems.append("time range")
    return problems
