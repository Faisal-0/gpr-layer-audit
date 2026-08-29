from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    gaussian_filter,
    gaussian_filter1d,
    maximum_filter1d,
    uniform_filter1d,
)
from scipy.signal import butter, hilbert, sosfiltfilt

try:
    import pywt
except ModuleNotFoundError:  # Optional fallback for bare system-Python checks.
    pywt = None


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
    stationary_wavelet_denoising: bool = True
    wavelet_name: str = "sym4"
    wavelet_level: int = 2
    mixed_phase_reflectivity: bool = True
    oriented_semblance: bool = True


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


def measurement_packet_support(
    radargram: NDArray[np.floating],
    *,
    pulse_width_samples: float = 7.0,
    lateral_window_traces: int = 7,
) -> NDArray[np.float32]:
    """Return packet-scale support from the pre-subtraction measurement signal.

    This branch is deliberately not a selector.  It measures whether a local
    wave packet has both observable energy and lateral phase consistency in
    the dewow-only, surface-flattened acquisition.  The small vertical maximum
    tolerates dipping layers without allowing a remote ringing cycle to lend
    support to the selected event.
    """
    data = np.asarray(radargram, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("radargram must be a two-dimensional array")
    if not data.size:
        return np.zeros_like(data, dtype=np.float32)
    analytic = hilbert(data, axis=1)
    envelope = np.abs(analytic).astype(np.float32)
    background_window = max(15, int(round(5.0 * pulse_width_samples)) | 1)
    local_background = uniform_filter1d(
        envelope, size=background_window, axis=1, mode="nearest"
    )
    local_ratio = envelope / np.maximum(local_background, 1e-7)
    strength = np.asarray(
        np.clip((local_ratio - 0.75) / 2.25, 0.0, 1.0),
        dtype=np.float32,
    )
    lateral_window = max(3, int(lateral_window_traces) | 1)
    half_window = max(1, lateral_window // 2)
    phase_consistency, _ = _oriented_coherence(data, half_window=half_window)
    local_strength = uniform_filter1d(
        strength, size=lateral_window, axis=0, mode="nearest"
    )
    support = np.sqrt(
        np.clip(strength * local_strength * phase_consistency, 0.0, 1.0)
    ).astype(np.float32)
    packet_window = max(3, int(round(0.75 * pulse_width_samples)) | 1)
    return np.asarray(
        maximum_filter1d(support, size=packet_window, axis=1, mode="nearest"),
        dtype=np.float32,
    )


def _minimum_phase_wavelet(template: NDArray[np.float32]) -> NDArray[np.float32]:
    """Estimate a centred minimum-phase wavelet by homomorphic factorisation."""
    values = np.asarray(template, dtype=np.float64)
    size = 1
    while size < max(64, 4 * len(values)):
        size *= 2
    padded = np.zeros(size, dtype=float)
    padded[: len(values)] = values
    magnitude = np.maximum(np.abs(np.fft.rfft(padded)), 1e-8)
    cepstrum = np.fft.irfft(np.log(magnitude), n=size)
    causal = np.zeros_like(cepstrum)
    causal[0] = cepstrum[0]
    causal[1 : size // 2] = 2.0 * cepstrum[1 : size // 2]
    if size % 2 == 0:
        causal[size // 2] = cepstrum[size // 2]
    spectrum = np.exp(np.fft.rfft(causal))
    wavelet = np.fft.irfft(spectrum, n=size)[: len(values)]
    wavelet -= np.mean(wavelet)
    wavelet = np.roll(wavelet, len(values) // 2 - int(np.argmax(np.abs(wavelet))))
    norm = float(np.linalg.norm(wavelet))
    if norm <= 1e-8:
        return np.asarray(template, dtype=np.float32)
    return np.asarray(wavelet / norm, dtype=np.float32)


def _stationary_wavelet_denoise(
    data: NDArray[np.float32], wavelet: str, level: int
) -> NDArray[np.float32]:
    if pywt is None:
        return np.asarray(gaussian_filter1d(data, sigma=0.65, axis=1), dtype=np.float32)
    level = max(1, int(level))
    multiple = 2**level
    padding = (-data.shape[1]) % multiple
    padded = np.pad(data, ((0, 0), (0, padding)), mode="edge") if padding else data
    coefficients = pywt.swt(padded, wavelet, level=level, axis=1)
    finest = coefficients[-1][1]
    sigma = float(np.median(np.abs(finest))) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(max(2, padded.shape[1])))
    shrunk = [
        (approximation, pywt.threshold(detail, threshold, mode="soft"))
        for approximation, detail in coefficients
    ]
    reconstructed = pywt.iswt(shrunk, wavelet, axis=1)
    return np.asarray(reconstructed[:, : data.shape[1]], dtype=np.float32)


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


def _mixed_phase_reflectivity(
    data: NDArray[np.float32], reference_surface_sample: int
) -> tuple[NDArray[np.float32], float]:
    """Rotate a deconvolved analytic signal to its sparsest deterministic phase."""
    analytic = hilbert(data, axis=1)
    start = min(data.shape[1] - 1, reference_surface_sample + 6)
    row_step = max(1, data.shape[0] // 256)
    sample_step = max(1, max(data.shape[1] - start, 1) // 256)
    training = analytic[::row_step, start::sample_step]
    best_angle = 0.0
    best_sparsity = -np.inf
    for angle in np.linspace(0.0, np.pi, 36, endpoint=False):
        rotated = np.real(training * np.exp(1j * angle))
        magnitude = np.abs(rotated)
        median = float(np.median(magnitude)) + 1e-7
        sparsity = float(np.percentile(magnitude, 99.0) / median)
        if sparsity > best_sparsity:
            best_sparsity = sparsity
            best_angle = float(angle)
    rotated = np.real(analytic * np.exp(1j * best_angle)).astype(np.float32)
    noise = np.median(np.abs(rotated[:, start:]), axis=1, keepdims=True) / 0.6745
    threshold = 0.65 * np.maximum(noise, 1e-7)
    sparse = np.sign(rotated) * np.maximum(np.abs(rotated) - threshold, 0.0)
    return np.asarray(sparse, dtype=np.float32), float(np.degrees(best_angle))


def _oriented_coherence(
    data: NDArray[np.float32], half_window: int = 3
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return the strongest locally dip-steered analytic coherence and its dip."""
    analytic = hilbert(data, axis=1).astype(np.complex64)
    rows, samples = data.shape
    best = np.zeros((rows, samples), dtype=np.float32)
    best_slope = np.zeros((rows, samples), dtype=np.float32)
    offsets = range(-half_window, half_window + 1)
    for slope in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
        total = np.zeros_like(analytic)
        magnitude = np.zeros((rows, samples), dtype=np.float32)
        count = np.zeros((rows, samples), dtype=np.float32)
        for offset in offsets:
            shifted = np.roll(analytic, shift=-offset, axis=0)
            sample_shift = int(round(slope * offset))
            if sample_shift:
                shifted = np.roll(shifted, shift=-sample_shift, axis=1)
            valid = np.ones((rows, samples), dtype=bool)
            if offset < 0:
                valid[: -offset] = False
            elif offset > 0:
                valid[rows - offset :] = False
            if sample_shift < 0:
                valid[:, : -sample_shift] = False
            elif sample_shift > 0:
                valid[:, samples - sample_shift :] = False
            total[valid] += shifted[valid]
            magnitude[valid] += np.abs(shifted[valid])
            count[valid] += 1.0
        coherence = np.abs(total) / np.maximum(magnitude, 1e-6)
        coherence[count < max(3, half_window + 1)] = 0.0
        replace = coherence > best
        best[replace] = coherence[replace]
        best_slope[replace] = slope
    return np.clip(best, 0.0, 1.0), best_slope


def subtract_tracked_reflection(
    data: NDArray[np.floating],
    path: NDArray[np.integer],
    *,
    pulse_width_samples: float = 7.0,
    maximum_shift_samples: int = 2,
    ridge_penalty: float = 0.08,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None]:
    """Fit a local mixed-phase wavelet at a tracked interface and subtract it per trace.

    The fit uses real and quadrature components, so amplitude and phase may vary
    without shifting the interface to a neighboring wavelet cycle.
    """
    values = np.asarray(data, dtype=np.float32)
    selected = np.asarray(path, dtype=int)
    radius = max(5, int(round(1.6 * pulse_width_samples)))
    width = 2 * radius + 1
    snippets: list[np.ndarray] = []
    snippet_rows: list[int] = []
    maximum_shift = max(0, int(maximum_shift_samples))
    regularization = max(0.0, float(ridge_penalty))
    for row, sample in enumerate(selected):
        if sample - maximum_shift < radius or sample + maximum_shift + radius >= values.shape[1]:
            continue
        snippet = values[row, sample - radius : sample + radius + 1].astype(float)
        snippet -= np.median(snippet)
        norm = np.linalg.norm(snippet)
        if norm > 1e-7:
            snippets.append(snippet / norm)
            snippet_rows.append(row)
    if len(snippets) < 3:
        return values.copy(), np.zeros_like(values), None
    template = np.median(np.stack(snippets), axis=0)
    template -= np.mean(template)
    norm = float(np.linalg.norm(template))
    if norm <= 1e-8:
        return values.copy(), np.zeros_like(values), None
    template = template / norm
    taper = np.hanning(width)
    residual = values.copy()
    improvement = np.zeros_like(values)
    snippet_stack = np.stack(snippets)
    snippet_row_array = np.asarray(snippet_rows, dtype=int)
    for row, sample in enumerate(selected):
        if sample < radius or sample + radius >= values.shape[1]:
            continue
        local = np.abs(snippet_row_array - row) <= 25
        local_template = (
            np.median(snippet_stack[local], axis=0)
            if np.count_nonzero(local) >= 5
            else template
        )
        local_template = local_template - np.mean(local_template)
        local_norm = float(np.linalg.norm(local_template))
        if local_norm <= 1e-8:
            continue
        local_template = local_template / local_norm
        quadrature = np.imag(hilbert(local_template)).astype(float)
        quadrature -= np.mean(quadrature)
        quadrature /= max(float(np.linalg.norm(quadrature)), 1e-8)
        basis = np.column_stack((local_template, quadrature, np.ones(width)))
        penalty = np.diag((regularization, regularization, regularization * 0.1))
        normal = basis.T @ basis + penalty
        best: tuple[float, slice, np.ndarray, float] | None = None
        for shift in range(-maximum_shift, maximum_shift + 1):
            centre = sample + shift
            region = slice(centre - radius, centre + radius + 1)
            observed = values[row, region].astype(float)
            coefficients = np.linalg.solve(normal, basis.T @ observed)
            model = (basis[:, :2] @ coefficients[:2]) * taper
            before = float(np.sum(np.square(observed))) + 1e-9
            after = float(np.sum(np.square(observed - model)))
            # Select timing from the upper-interface fit itself. Looking in an
            # expected base window would reward erasing the target reflector.
            objective = after / before + 0.004 * abs(shift)
            if best is None or objective < best[0]:
                best = (objective, region, model, np.clip((before - after) / before, 0.0, 1.0))
        if best is None:
            continue
        _, region, model, gain = best
        residual[row, region] -= np.asarray(model, dtype=np.float32)
        improvement[row, region] = gain
    return (
        np.asarray(residual, dtype=np.float32),
        improvement,
        np.asarray(template, dtype=np.float32),
    )


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
        # Keep clean, full-width and rolling-background branches independent.
        # A global blend can erase a real continuous layer or inject local
        # disturbance energy into every downstream feature.
        metrics["background_window_traces"] = int(options.background_window_traces)
        steps.append("independent rolling and full-width background evidence branches")

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
        minimum_phase_template = _minimum_phase_wavelet(template)
        deconvolved = _regularized_deconvolution(
            data, minimum_phase_template, options.deconvolution_regularization
        )
        steps.append(
            "homomorphic minimum-phase, regularized wavelet deconvolution candidate branch"
        )

    reflectivity = deconvolved
    phase_rotation_degrees = 0.0
    if options.mixed_phase_reflectivity:
        reflectivity, phase_rotation_degrees = _mixed_phase_reflectivity(
            deconvolved, reference_surface_sample
        )
        metrics["mixed_phase_rotation_degrees"] = phase_rotation_degrees
        steps.append("sparsity-selected mixed-phase reflectivity branch")

    wavelet_denoised = data
    if options.stationary_wavelet_denoising:
        wavelet_denoised = _stationary_wavelet_denoise(
            data, options.wavelet_name, options.wavelet_level
        )
        metrics["stationary_wavelet_available"] = pywt is not None
        steps.append(
            "shift-invariant stationary-wavelet denoising candidate branch"
            if pywt is not None
            else "Gaussian denoising fallback (PyWavelets unavailable)"
        )

    phase, coherence = _phase_and_coherence(data)
    ensemble_phase, ensemble_coherence = _phase_and_coherence(full_background)
    wavelet_phase, wavelet_coherence = _phase_and_coherence(wavelet_denoised)
    deconvolved_phase, deconvolved_coherence = _phase_and_coherence(deconvolved)
    if options.oriented_semblance:
        oriented_coherence, oriented_slope = _oriented_coherence(data)
        steps.append("dip-steered lateral coherence branch")
    else:
        oriented_coherence = coherence
        oriented_slope = np.zeros_like(coherence)
    gradient = np.abs(np.gradient(data, axis=1)).astype(np.float32)
    envelope = np.abs(hilbert(data, axis=1)).astype(np.float32)
    candidate = (
        0.30 * _normalise_feature(envelope)
        + 0.22 * _normalise_feature(gradient)
        + 0.20 * _normalise_feature(np.abs(phase))
        + 0.18 * coherence
        + 0.10 * _normalise_feature(np.abs(deconvolved))
    ).astype(np.float32)
    start = min(data.shape[1] - 1, reference_surface_sample + 8)
    lateral_gradient = np.abs(np.gradient(data[:, start:], axis=0))
    vertical_gradient = np.abs(np.gradient(data[:, start:], axis=1))
    orientation = np.median(
        lateral_gradient / np.maximum(lateral_gradient + vertical_gradient, 1e-6),
        axis=1,
    )
    trace_energy = np.median(envelope[:, start:], axis=1)
    coherence_loss = 1.0 - np.median(coherence[:, start:], axis=1)

    def robust_excess(values: NDArray[np.floating], onset_z: float) -> NDArray[np.float32]:
        vector = np.asarray(values, dtype=float)
        median = float(np.median(vector))
        mad = 1.4826 * float(np.median(np.abs(vector - median)))
        if mad <= 1e-8:
            return np.zeros_like(vector, dtype=np.float32)
        z_score = (vector - median) / mad
        return np.asarray(np.clip((z_score - onset_z) / 4.0, 0.0, 1.0), dtype=np.float32)

    orientation_excess = robust_excess(orientation, 2.5)
    energy_excess = robust_excess(trace_energy, 3.0)
    coherence_excess = robust_excess(coherence_loss, 2.5)
    anomaly_support = (
        (orientation_excess > 0.08).astype(np.float32)
        + (energy_excess > 0.08).astype(np.float32)
        + (coherence_excess > 0.08).astype(np.float32)
    ) / 3.0
    anomaly_score = gaussian_filter1d(
        (
            0.45 * orientation_excess
            + 0.35 * energy_excess
            + 0.20 * coherence_excess
        )
        * np.clip(0.25 + anomaly_support, 0.0, 1.0),
        sigma=1.2,
        mode="nearest",
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
        "wavelet_amplitude": wavelet_denoised,
        "wavelet_phase": wavelet_phase,
        "wavelet_coherence": wavelet_coherence,
        "deconvolved": deconvolved,
        "deconvolved_phase": deconvolved_phase,
        "deconvolved_coherence": deconvolved_coherence,
        "reflectivity": reflectivity,
        "oriented_coherence": oriented_coherence,
        "oriented_slope": oriented_slope,
        "candidate": candidate,
        "anomaly_score": anomaly_score,
        "anomaly_support": anomaly_support.astype(np.float32),
    }
    display_views = {
        "Raw": np.asarray(radargram, dtype=np.float32),
        "Clean": data,
        "Phase": phase,
        "Gradient": gradient,
        "Candidates": candidate,
        "Reflectivity": reflectivity,
        "Oriented ridge": oriented_coherence,
    }

    metrics["options"] = asdict(options)
    return PreprocessingResult(data, template, steps, metrics, feature_branches, display_views)
