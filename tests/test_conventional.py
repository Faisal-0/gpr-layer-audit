from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.conventional_reference import distributed_seeds, road_partition
from gpr_layer_audit.io.dzx import read_dzx
from gpr_layer_audit.processing.calibration import dewow
from gpr_layer_audit.processing.conventional_config import ConventionalConfig, resolve_pulse
from gpr_layer_audit.processing.conventional_signal import (
    CoordinateTransform,
    SignalLayout,
    numerical_extension,
    processed_boundary_mask,
    resample_valid,
    shift_validity,
    stack_valid,
)
from gpr_layer_audit.processing.directed_support import directed_seed_support
from gpr_layer_audit.processing.hybrid import ReflectorSegment, solve_complete_intervals


def test_radan_layer_trace_import_keeps_labels_and_properties(tmp_path):
    p = tmp_path / "reference.DZX"
    p.write_text("""<DZX xmlns="www.geophysical.com/DZX/1.02"><LayerGroup>
    <groupName>Unusual analyst label</groupName><layerNum>2</layerNum><pickType>0</pickType>
    <LayerWayPt><scanSampChanProp>10,21,0,2</scanSampChanProp>
    <timeAmpDepVel>1.2,-500,3,4</timeAmpDepVel></LayerWayPt></LayerGroup>
    <TargetGroup><LayerWayPt><scanSampChanProp>1,2,0,1</scanSampChanProp></LayerWayPt></TargetGroup></DZX>""")
    result = read_dzx(p)
    assert len(result.layers) == 1
    layer = result.layers[0]
    assert layer.name == "Unusual analyst label" and layer.number == 2
    assert layer.picks[0].recorded_amplitude == -500
    assert layer.picks[0].sample == 21 and layer.picks[0].interpretation_property == 2
    assert len(result.source_sha256) == 64


def test_raw_boundary_values_cannot_contaminate_dewow_or_visibility():
    from gpr_layer_audit.processing.preprocessing import measurement_packet_support

    clean = np.tile(np.sin(np.arange(100) / 3), (5, 1)).astype(np.float32)
    broken = clean.copy()
    broken[:, :2] = 2e9
    valid = SignalLayout("gssi_raw_sir30", 2).mask(broken)
    assert np.array_equal(dewow(broken, valid=valid), dewow(clean, valid=valid))
    support = measurement_packet_support(broken, valid=valid)
    assert np.all(support[:, :2] == 0)
    assert np.array_equal(broken[:, :2], np.full((5, 2), 2e9))


def test_processed_padding_is_separate_and_unknown_layout_is_not_guessed():
    data = np.array([[9.0, 8, 7, 0, 0, 0, 0]])
    valid = processed_boundary_mask(data)
    assert valid[0, 0] and valid[0, 1] and not np.any(valid[0, 3:])
    assert np.all(SignalLayout().mask(data))
    with pytest.raises(ValueError):
        SignalLayout("unresolved", 2).mask(data)


def test_invalid_samples_do_not_enter_stack_or_return_through_resampling():
    data = np.array([[1.0, 1e9, 3], [3, 1e9, 5], [5, 1e9, 7]])
    valid = np.ones_like(data, bool)
    valid[:, 1] = False
    stacked, mask, centres = stack_valid(data, valid, 3)
    assert stacked[0, 0] == 3 and stacked[0, 2] == 5
    assert not mask[0, 1] and centres[0] == 1
    _, sampled = resample_valid(stacked, mask, [0], [0, 0.5, 1, 2])
    assert sampled.tolist() == [[True, False, False, True]]
    shifted = shift_validity(valid, [1, 0, -1])
    assert not shifted[0, 0] and not shifted[2, -1]


def test_coordinate_round_trip_and_scoring_provenance():
    mapping = CoordinateTransform(
        "processed", "raw", 16, 2, 0.5, -4, 0.2, 0.1, True, "radar_registration"
    )
    a, b = mapping.forward(np.array([0, 5, 10]), np.array([40, 50, 60]))
    x, y = mapping.inverse(a, b)
    assert np.allclose(x, [0, 5, 10]) and np.allclose(y, [40, 50, 60])
    mapping.require_scoring("processed", "raw", 2)
    with pytest.raises(ValueError, match="verified radar-only"):
        CoordinateTransform(
            **{**asdict(mapping), "basis": "evaluation_layer_picks"}
        ).require_scoring("processed", "raw", 2)
    with pytest.raises(ValueError):
        mapping.require_scoring("wrong", "raw", 2)


def test_distributed_seeds_and_road_grouping():
    picks = [SimpleNamespace(trace=x) for x in range(100, 201)]
    assert [p.trace for p in distributed_seeds(picks)] == [110, 150, 190]
    assert road_partition("GUJRAT second portion") == road_partition("SOHL KALAN GUJRAT")
    assert road_partition("JHANG local road")[1] == "held_out"


def test_resolution_independent_pulse_and_withholding_isolation():
    config = ConventionalConfig()
    pulses = []
    for dt in (0.02, 0.01):
        t = np.arange(0, 4, dt)
        z = (t - 2) / 0.08
        data = np.tile(-(1 - z * z) * np.exp(-z * z / 2), (3, 1)).astype(np.float32)
        sample = round(2 / dt)
        pulse = resolve_pulse(
            data,
            np.ones_like(data, bool),
            {0: sample},
            {2: {"verified": True, "selected_lobe_width_ns": 500}},
            dt,
            config,
        )
        pulses.append(pulse)
    assert abs(pulses[0].lobe_ns - pulses[1].lobe_ns) <= 0.021
    assert pulses[0].context_ns == pytest.approx(pulses[1].context_ns, abs=0.061)
    assert pulses[0].source == "seed_waveform_zero_crossings"


def test_directed_support_cannot_walk_backwards_through_a_junction():
    samples = np.full((5, 2), 60)
    nodes = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (1, 1)]
    links = [((0, 0), (2, 0), 1.0, 2), ((1, 1), (2, 0), 1.0, 1), ((2, 0), (4, 0), 1.0, 2)]
    support, _, _ = directed_seed_support(nodes, links, {0: 60}, samples)
    assert support[2, 0] == 1 and support[1, 1] == 0


def test_bracketing_seed_support_requires_both_directions():
    samples = np.full((5, 1), 60)
    nodes = [(r, 0) for r in range(5)]
    links = [((0, 0), (1, 0), 1.0, 1), ((1, 0), (2, 0), 1.0, 1), ((3, 0), (4, 0), 1.0, 1)]
    support, _, _ = directed_seed_support(nodes, links, {0: 60, 4: 60}, samples)
    assert support[2, 0] == 0


def test_incompatible_endpoints_preserve_exact_clicks_and_report_interval():
    table = SimpleNamespace(samples=np.full((5, 2), 60))
    segments = [
        ReflectorSegment(0, [(0, 0)], {0}),
        ReflectorSegment(1, [(1, 0)]),
        ReflectorSegment(2, [(4, 0)], {4}),
    ]
    diagnostics = {}
    paths = solve_complete_intervals(
        segments,
        {(0, 1): 1},
        np.ones(3),
        table,
        np.ones((5, 2)),
        {0: 60, 4: 60},
        set(),
        7,
        0.4,
        ConventionalConfig(),
        diagnostics,
    )
    assert paths[0].tolist() == [60, -1, -1, -1, 60]
    assert diagnostics["unresolved_intervals"]


def test_signal_extension_cannot_change_original_values():
    data = np.array([[np.nan, 2, 4, np.inf]], np.float32)
    valid = np.isfinite(data)
    assert numerical_extension(data, valid).tolist() == [[2, 2, 4, 4]]
    assert np.isnan(data[0, 0]) and np.isinf(data[0, -1])


def test_batched_dtw_matches_independent_scalar_reference():
    from gpr_layer_audit.processing.hybrid import constrained_dtw
    from gpr_layer_audit.processing.waveform_matching import batch_dtw

    rng = np.random.default_rng(45)
    left, right = rng.normal(size=(2, 8, 17))
    for band in (1, 3, 5):
        score, shift = batch_dtw(left, right, band)
        expected = np.array([constrained_dtw(a, b, band) for a, b in zip(left, right, strict=True)])
        assert np.allclose(score, expected[:, 0])
        assert np.allclose(shift, expected[:, 1])


def test_calibration_rejects_held_out_observations(tmp_path):
    import json

    from gpr_layer_audit.conventional_validation import calibrate_acceptance

    source = tmp_path / "held-out.json"
    source.write_text(json.dumps({"partition": ["jhang", "held_out"]}))
    with pytest.raises(ValueError, match="Held-out"):
        calibrate_acceptance([source], {})


def test_promotion_requires_independent_groups_correctness_and_useful_gain():
    from gpr_layer_audit.conventional_validation import promotion_gate

    base = {
        "split": "held_out",
        "observations": 40,
        "distributed_observations": True,
        "accepted_pick_agreement": 0.97,
        "baseline_accepted_pick_agreement": 0.96,
        "selected_lobe_identity_checked": True,
        "correct_coverage": 0.7,
        "baseline_correct_coverage": 0.5,
    }
    groups = [{**base, "physical_road_group": name} for name in ("jhang", "pattoki", "daska")]
    assert promotion_gate(groups, layer_order=2)["eligible"]
    assert not promotion_gate(groups[:2], layer_order=2)["eligible"]
    assert not promotion_gate(groups, layer_order=3)["eligible"]
    groups[0]["accepted_pick_agreement"] = 0.94
    assert not promotion_gate(groups, layer_order=2)["eligible"]


def test_uncertain_upper_hypothesis_is_not_a_hard_search_boundary(monkeypatch):
    import gpr_layer_audit.processing.seed_graph as graph
    from gpr_layer_audit.models import LayerSpec
    from gpr_layer_audit.processing.tracker import pick_interfaces

    axis = np.arange(128)
    data = np.tile(
        np.exp(-(((axis - 45) / 2) ** 2)) - np.exp(-(((axis - 70) / 2) ** 2)), (25, 1)
    ).astype(np.float32)
    original = graph.pick_seed_conditioned_interfaces

    def uncertain_upper(*args, **kwargs):
        paths = original(*args, **kwargs)
        upper = paths[1]
        upper.samples[4:20] = -1
        upper.visible[4:20] = False
        upper.provisional_samples = np.full(25, 45, np.int32)
        upper.provisional_samples[4:20] = 90
        upper.provenance["hypothesis_samples"] = [upper.provisional_samples.tolist(), [45] * 25]
        return paths

    monkeypatch.setattr(graph, "pick_seed_conditioned_interfaces", uncertain_upper)
    paths = pick_interfaces(
        data,
        15,
        [LayerSpec(1, "Asphalt", 15, 95, 4), LayerSpec(2, "Base", 20, 105, 4)],
        method="seed_hybrid",
        ml_policy="off",
        anchor_samples={1: {2: 45, 22: 45}, 2: {2: 70, 22: 70}},
    )
    assert np.count_nonzero(paths[2].visible[4:20]) >= 8
    selected = paths[2].samples[4:20]
    assert np.all(abs(selected[selected >= 0] - 70) <= 2)


def test_processed_benchmark_uses_header_origin_without_recalibration(tmp_path, monkeypatch):
    import struct

    from conftest import write_dzt

    import gpr_layer_audit.conventional as evaluation

    source = tmp_path / "processed.DZT"
    write_dzt(source, np.ones((10, 128)), range_ns=12.8)
    with source.open("r+b") as stream:
        stream.seek(22)
        stream.write(struct.pack("<f", -3.0))
    source.with_suffix(".DZX").write_text(
        "<DZX><LayerGroup><layerNum>0</layerNum>"
        "<LayerWayPt><scanSampChanProp>2,40,0,2</scanSampChanProp>"
        "<timeAmpDepVel>1,1,1,1</timeAmpDepVel></LayerWayPt></LayerGroup></DZX>"
    )
    origins = []

    def fake_pick(data, surface, layers, **kwargs):
        origins.append(surface)
        assert np.all(data == 1)
        assert kwargs["ml_policy"] == "off"
        return {}

    monkeypatch.setattr(evaluation, "pick_interfaces", fake_pick)
    monkeypatch.setattr(evaluation, "_overlay", lambda *args: None)
    result = evaluation.evaluate_reference(
        source.with_suffix(".DZX"), output=tmp_path / "score.json"
    )
    assert origins == [30, 30]
    assert result["reference_surface_sample"] == 30


def test_export_order_checks_across_unknown_middle_and_preserves_deep_click():
    from gpr_layer_audit.models import LayerSpec
    from gpr_layer_audit.processing.hybrid import enforce_accepted_order

    def path(sample):
        return SimpleNamespace(
            samples=np.array(sample),
            visible=np.ones(2, bool),
            confidence=np.ones(2),
            evidence={},
            provenance={},
        )

    paths = {1: path([80, 80]), 2: path([-1, -1]), 3: path([70, 70])}
    enforce_accepted_order(paths, LayerSpec.defaults(), {3: {1: 70}})
    assert paths[3].samples.tolist() == [-1, 70]
    assert paths[1].samples.tolist() == [80, -1]
