"""Isolated dense backend using the application's actual local merge/replay contract."""

from __future__ import annotations

import inspect
from dataclasses import asdict

from gpr_layer_audit.processing.conventional_config import ConventionalConfig, resolve_pulse
from gpr_layer_audit.processing.seeded_challenger import PacketConfig, pick_seeded_packet


def timing_aware_picker(original=pick_seeded_packet):
    """Preserve selection; include distinguishable timing modes in exact ambiguity."""
    source = inspect.getsource(original)
    previous = "                other[lobes == lobes[sample]] = np.inf\n"
    replacement = (
        "                equivalent = (lobes == lobes[sample]) & (\n"
        "                    abs(np.arange(samples) - sample) <= max(2, pulse_width_samples / 4)\n"
        "                )\n"
        "                other[equivalent] = np.inf\n"
    )
    if source.count(previous) != 1:
        raise ValueError("Review changed dense ambiguity source before instrumentation")
    namespace = dict(original.__globals__)
    exec(
        compile(source.replace(previous, replacement), "<dense-timing-ambiguity>", "exec"),
        namespace,
    )
    return namespace[original.__name__]


def fit_dense(
    measurement,
    valid,
    surface,
    dt,
    step,
    layers,
    anchors,
    *,
    method,
    config,
    pulse_anchors=None,
    breaks=(),
    cancel=None,
    **unused,
):
    if method != "dense_packet_research":
        raise ValueError("This isolated adapter is not application dispatch")
    options = dict(config)
    timing = options.pop("timing_ambiguity")
    packet_config = PacketConfig(**options)
    picker = timing_aware_picker() if timing else pick_seeded_packet
    result = {}
    for layer in layers:
        order = layer.order
        operating = anchors.get(order, {})
        original = (pulse_anchors or anchors).get(order, {})
        pulse = resolve_pulse(measurement, valid, original, {}, dt, ConventionalConfig())
        path = picker(
            measurement,
            operating,
            valid=valid,
            dt_ns=dt,
            dx_m=step,
            pulse_width_samples=pulse.lobe_samples,
            context_radius=pulse.context_radius,
            reference_surface=surface,
            breaks=breaks,
            config=packet_config,
            cancel=cancel,
        )
        path.provenance.update(
            {
                "backend": "dense_packet_research",
                "resolved_pulse": pulse.metadata(),
                "timing_ambiguity": timing,
                "config": asdict(packet_config),
                "alternative_contract": "Pointwise exact constrained alternatives; not a coherent route",
                "acceptance_is_calibrated": False,
            }
        )
        result[order] = path
    return result
