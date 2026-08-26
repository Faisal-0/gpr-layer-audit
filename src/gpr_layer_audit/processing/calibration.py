from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter, uniform_filter1d

from gpr_layer_audit.io.dzt import DZTFile, DZTHeader
from gpr_layer_audit.models import CalibrationDiagnostics


@dataclass(slots=True)
class CalibratedData:
    radargram: NDArray[np.float32]
    raw_stacks: NDArray[np.float32]
    surface_samples: NDArray[np.int32]
    reference_surface_sample: int
    plate_template: NDArray[np.float32]
    surface_amplitudes: NDArray[np.float64]
    diagnostics: CalibrationDiagnostics
    trace_centres: NDArray[np.float64]


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


def dewow(data: NDArray[np.floating], window_samples: int = 31) -> NDArray[np.float32]:
    window = max(3, int(window_samples) | 1)
    baseline = uniform_filter1d(data, size=window, axis=1, mode="nearest")
    return np.asarray(data - baseline, dtype=np.float32)


def _surface_search_bounds(samples: int) -> tuple[int, int]:
    return max(4, int(samples * 0.22)), min(samples - 4, int(samples * 0.43))


def _plate_template(plate: DZTFile) -> tuple[NDArray[np.float32], int, float]:
    plate_data = np.asarray(plate.channel(0), dtype=np.float32)
    plate_clean = dewow(plate_data)
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
) -> CalibratedData:
    stop_trace = road.header.trace_count if stop_trace is None else stop_trace
    stacks, centres = horizontal_stack(
        road.channel(0, start_trace=start_trace, stop_trace=stop_trace), stack_size
    )
    centres += start_trace
    clean = dewow(stacks)
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

    plate_waveform = np.zeros(samples, dtype=np.float32)
    if plate is None:
        diagnostics.messages.append(
            "No metal-plate file supplied; bounce correction used surface alignment only."
        )
        return CalibratedData(
            aligned,
            stacks,
            surface.astype(np.int32),
            reference,
            plate_waveform,
            surface_amplitudes,
            diagnostics,
            centres,
        )

    if plate.header.samples_per_trace != road.header.samples_per_trace:
        diagnostics.messages.append("Road and plate sample counts differ.")
        return CalibratedData(
            aligned,
            stacks,
            surface.astype(np.int32),
            reference,
            plate_waveform,
            surface_amplitudes,
            diagnostics,
            centres,
        )

    plate_waveform, plate_peak, plate_amplitude = _plate_template(plate)
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

    denominator = float(plate_aligned[reference])
    corrected = aligned.copy()
    if abs(denominator) > 1e-9:
        scales = aligned[:, reference].astype(float) / denominator
        corrected[:, reference:] -= scales[:, None] * plate_aligned[None, reference:]
        # Remove isolated subtraction spikes while retaining thin-layer pulses.
        corrected = median_filter(corrected, size=(1, 3), mode="nearest").astype(np.float32)
    else:
        diagnostics.messages.append("Metal-plate reference peak is zero after preprocessing.")

    compatible = (
        diagnostics.gain_compatible
        and diagnostics.clipping_fraction < 1e-5
        and road.header.antenna == plate.header.antenna
        and np.isclose(road.header.range_ns, plate.header.range_ns, rtol=0, atol=1e-4)
    )
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
    diagnostics.messages.append("Metal-plate waveform subtraction and surface flattening applied.")

    return CalibratedData(
        corrected,
        stacks,
        surface.astype(np.int32),
        reference,
        plate_aligned,
        surface_amplitudes,
        diagnostics,
        centres,
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
