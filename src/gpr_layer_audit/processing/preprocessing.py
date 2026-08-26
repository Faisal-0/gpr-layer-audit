from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt


@dataclass(slots=True)
class PreprocessingOptions:
    enabled: bool = True
    automatic_bandpass: bool = True
    time_varying_gain: bool = True
    trace_normalisation: bool = True
    light_denoise: bool = True
    rolling_background_removal: bool = True
    background_window_traces: int = 75
    phase_features: bool = True
    regularized_deconvolution: bool = True
    deconvolution_regularization: float = 0.08
    display_gain: float = 3.0
    maximum_time_gain: float = 5.0


@dataclass(slots=True)
class PreprocessingResult:
    radargram: NDArray[np.float32]
    matched_template: NDArray[np.float32] | None
    steps: list[str]
    metrics: dict[str, float | bool]
    feature_branches: dict[str, NDArray[np.float32]]
    display_views: dict[str, NDArray[np.float32]]


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


def _normalise_feature(data: NDArray[np.floating]) -> NDArray[np.float32]:
    values = np.asarray(data, dtype=np.float32)
    low, high = np.percentile(values, [2.0, 98.0])
    scale = max(float(high - low), 1e-6)
    return np.asarray(np.clip((values - low) / scale, 0.0, 1.0), dtype=np.float32)


def _background_branches(
    data: NDArray[np.float32], window_traces: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    full = data - np.mean(data, axis=0, keepdims=True)
    window = max(3, int(window_traces) | 1)
    rolling = data - uniform_filter1d(data, size=window, axis=0, mode="nearest")
    return np.asarray(full, dtype=np.float32), np.asarray(rolling, dtype=np.float32)


def _regularized_deconvolution(
    data: NDArray[np.float32], template: NDArray[np.float32], regularization: float
) -> NDArray[np.float32]:
    kernel = np.zeros(data.shape[1], dtype=np.float32)
    count = min(len(template), len(kernel))
    kernel[:count] = template[:count]
    kernel = np.roll(kernel, -(count // 2))
    transfer = np.fft.rfft(kernel)
    denominator = np.abs(transfer) ** 2
    floor = max(float(np.max(denominator)) * max(regularization, 1e-4), 1e-8)
    inverse = np.conj(transfer) / (denominator + floor)
    output = np.fft.irfft(np.fft.rfft(data, axis=1) * inverse[None, :], n=data.shape[1], axis=1)
    return np.asarray(output, dtype=np.float32)


def _phase_and_coherence(
    data: NDArray[np.float32], window_traces: int = 9
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    analytic = hilbert(data, axis=1)
    phase = np.cos(np.angle(analytic)).astype(np.float32)
    coherent = np.abs(
        uniform_filter1d(analytic, size=max(3, window_traces | 1), axis=0, mode="nearest")
    )
    amplitude = uniform_filter1d(
        np.abs(analytic), size=max(3, window_traces | 1), axis=0, mode="nearest"
    )
    coherence = np.asarray(coherent / np.maximum(amplitude, 1e-6), dtype=np.float32)
    return phase, np.clip(coherence, 0.0, 1.0)


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
        gradient = np.abs(np.gradient(data, axis=1)).astype(np.float32)
        features = {"amplitude": data, "gradient": _normalise_feature(gradient)}
        return PreprocessingResult(
            data,
            template,
            steps,
            metrics,
            features,
            {"Raw": data, "Clean": data, "Gradient": gradient},
        )

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

    full_background, rolling_background = _background_branches(
        data, options.background_window_traces
    )
    if options.rolling_background_removal:
        # Keep the horizontally coherent signal in the main branch while using
        # background removal as a supporting feature. This avoids erasing a
        # genuinely flat pavement interface.
        data = np.asarray(0.82 * data + 0.18 * rolling_background, dtype=np.float32)
        metrics["background_window_traces"] = int(options.background_window_traces)
        steps.append("rolling horizontal-background feature (18% interpretation blend)")

    if template is not None:
        centre = reference_surface_sample
        radius = min(18, centre, len(template) - centre - 1)
        cropped = template[centre - radius : centre + radius + 1]
        cropped -= np.mean(cropped)
        norm = float(np.linalg.norm(cropped))
        template = np.asarray(cropped / norm, dtype=np.float32) if norm > 0 else None
        steps.append("plate-wavelet matched-filter feature")

    deconvolved = data
    if options.regularized_deconvolution and template is not None:
        deconvolved = _regularized_deconvolution(
            data, template, options.deconvolution_regularization
        )
        steps.append("regularized wavelet deconvolution candidate branch")

    phase, coherence = _phase_and_coherence(data)
    ensemble_phase, ensemble_coherence = _phase_and_coherence(full_background)
    gradient = np.abs(np.gradient(data, axis=1)).astype(np.float32)
    envelope = np.abs(hilbert(data, axis=1)).astype(np.float32)
    candidate = (
        0.30 * _normalise_feature(envelope)
        + 0.22 * _normalise_feature(gradient)
        + 0.20 * _normalise_feature(np.abs(phase))
        + 0.18 * coherence
        + 0.10 * _normalise_feature(np.abs(deconvolved))
    ).astype(np.float32)
    feature_branches = {
        "amplitude": data,
        "envelope": envelope,
        "gradient": gradient,
        "phase": phase,
        "coherence": coherence,
        "background_full": full_background,
        "background_rolling": rolling_background,
        "ensemble_amplitude": full_background,
        "ensemble_phase": ensemble_phase,
        "ensemble_coherence": ensemble_coherence,
        "deconvolved": deconvolved,
        "candidate": candidate,
    }
    display_views = {
        "Raw": np.asarray(radargram, dtype=np.float32),
        "Clean": data,
        "Phase": phase,
        "Gradient": gradient,
        "Candidates": candidate,
    }

    metrics["options"] = asdict(options)
    return PreprocessingResult(
        data, template, steps, metrics, feature_branches, display_views
    )
