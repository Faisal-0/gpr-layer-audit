from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import butter, sosfiltfilt


@dataclass(slots=True)
class PreprocessingOptions:
    enabled: bool = True
    automatic_bandpass: bool = True
    time_varying_gain: bool = True
    trace_normalisation: bool = True
    light_denoise: bool = True
    maximum_time_gain: float = 5.0


@dataclass(slots=True)
class PreprocessingResult:
    radargram: NDArray[np.float32]
    matched_template: NDArray[np.float32] | None
    steps: list[str]
    metrics: dict[str, float | bool]


def _automatic_band(
    data: NDArray[np.float32], reference_surface_sample: int
) -> tuple[float, float]:
    start = max(0, reference_surface_sample - 12)
    spectrum = np.median(np.abs(np.fft.rfft(data[:, start:], axis=1)) ** 2, axis=0)
    frequencies = np.fft.rfftfreq(data.shape[1] - start)
    usable = (frequencies >= 0.01) & (frequencies <= 0.45)
    energy = np.where(usable, spectrum, 0.0)
    cumulative = np.cumsum(energy)
    if not len(cumulative) or cumulative[-1] <= 0:
        return 0.02, 0.40
    cumulative /= cumulative[-1]
    low_index = min(int(np.searchsorted(cumulative, 0.015)), len(frequencies) - 1)
    high_index = min(int(np.searchsorted(cumulative, 0.985)), len(frequencies) - 1)
    low = float(frequencies[low_index])
    high = float(frequencies[high_index])
    low = float(np.clip(low, 0.01, 0.30))
    high = float(np.clip(high, low + 0.04, 0.46))
    return low, high


def _bandpass(
    data: NDArray[np.float32], low_cycles: float, high_cycles: float
) -> NDArray[np.float32]:
    sos = butter(4, [low_cycles / 0.5, high_cycles / 0.5], btype="bandpass", output="sos")
    return np.asarray(sosfiltfilt(sos, data, axis=1), dtype=np.float32)


def _time_gain(
    data: NDArray[np.float32], reference_surface_sample: int, maximum_gain: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    rms = np.sqrt(np.median(np.square(data, dtype=np.float64), axis=0) + 1e-12)
    rms = gaussian_filter1d(rms, sigma=5, mode="nearest")
    start = min(data.shape[1] - 1, reference_surface_sample + 8)
    target = float(np.median(rms[start:])) or 1.0
    gain = np.clip(target / np.maximum(rms, target / maximum_gain), 0.5, maximum_gain)
    gain[: max(0, reference_surface_sample - 3)] = 1.0
    transition_stop = min(data.shape[1], reference_surface_sample + 12)
    if transition_stop > reference_surface_sample - 3:
        left = max(0, reference_surface_sample - 3)
        gain[left:transition_stop] = np.linspace(
            1.0, gain[transition_stop - 1], transition_stop - left
        )
    return np.asarray(data * gain[None, :], dtype=np.float32), gain.astype(np.float32)


def preprocess_for_interpretation(
    radargram: NDArray[np.floating],
    reference_surface_sample: int,
    plate_template: NDArray[np.floating] | None = None,
    options: PreprocessingOptions | None = None,
) -> PreprocessingResult:
    options = options or PreprocessingOptions()
    data = np.asarray(radargram, dtype=np.float32).copy()
    template = (
        np.asarray(plate_template, dtype=np.float32).copy()
        if plate_template is not None and np.any(plate_template)
        else None
    )
    steps = ["surface flattening", "dewow", "metal-plate ringdown subtraction"]
    metrics: dict[str, float | bool] = {"enabled": options.enabled}
    if not options.enabled:
        return PreprocessingResult(data, template, steps, metrics)

    if options.automatic_bandpass:
        low, high = _automatic_band(data, reference_surface_sample)
        data = _bandpass(data, low, high)
        if template is not None:
            template = _bandpass(template[None, :], low, high)[0]
        metrics.update(
            {"bandpass_low_cycles_per_sample": low, "bandpass_high_cycles_per_sample": high}
        )
        steps.append("zero-phase automatic spectral band-pass")

    if options.time_varying_gain:
        data, gain = _time_gain(data, reference_surface_sample, options.maximum_time_gain)
        metrics["time_gain_maximum_applied"] = float(np.max(gain))
        steps.append("bounded time-varying RMS gain (interpretation branch only)")

    if options.trace_normalisation:
        scale = np.median(np.abs(data[:, reference_surface_sample + 4 :]), axis=1)
        target = float(np.median(scale)) or 1.0
        factors = np.clip(target / np.maximum(scale, target / 2.5), 0.5, 2.5)
        data *= factors[:, None]
        metrics["trace_normalisation_maximum"] = float(np.max(factors))
        steps.append("robust per-stack amplitude normalisation")

    if options.light_denoise:
        data = gaussian_filter(data, sigma=(0.55, 0.35), mode="nearest").astype(np.float32)
        steps.append("light anisotropic denoising")

    if template is not None:
        centre = reference_surface_sample
        radius = min(18, centre, len(template) - centre - 1)
        cropped = template[centre - radius : centre + radius + 1]
        cropped -= np.mean(cropped)
        norm = float(np.linalg.norm(cropped))
        template = np.asarray(cropped / norm, dtype=np.float32) if norm > 0 else None
        steps.append("plate-wavelet matched-filter feature")

    metrics["options"] = asdict(options)
    return PreprocessingResult(data, template, steps, metrics)
