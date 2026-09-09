import copy
import json
from dataclasses import asdict

import numpy as np
import pytest

from gpr_layer_audit.conventional_seeds import load_support, native_support
from gpr_layer_audit.io.dzx import DZXPick


@pytest.fixture
def seed_case(tmp_path):
    picks = [DZXPick(r, 30 + r % 3, 0, 2, 1.0, 1.0, 0.0, 0.0) for r in range(97)]
    audit = {
        "metadata": {
            "source_sha256": "dzx",
            "layers": [
                {"number": 0, "picks": [asdict(p) for p in picks]},
            ],
        },
        "dzt_sha256": "dzt",
    }
    mapping = {"0": 1, "1": 2, "2": 3}
    support = native_support(audit, mapping, {1: [picks[r] for r in (16, 48, 80)]})
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps(support))
    return path, audit, mapping, support


def test_reused_support_keeps_native_coordinates_across_resolutions(seed_case):
    path, audit, mapping, support = seed_case
    for stride in (16, 4, 1):
        anchors, provenance = load_support(path, audit, mapping, stride, "processed")
        assert anchors == {1: {16: 31, 48: 30, 80: 32}}
        assert len(provenance["sha256"]) == 64
    assert len(support["observations"]["1"]) == 3


@pytest.mark.parametrize(
    "case,reason",
    [
        ("grid", "incompatible"),
        ("source", "fingerprint"),
        ("radar", "fingerprint"),
        ("mapping", "mapping"),
        ("sample", "exact reviewed"),
        ("duplicate", "duplicate"),
        ("missing", "every evaluated"),
        ("raw", "processed-input"),
        ("fractional", "integers"),
    ],
)
def test_incompatible_seed_reuse_is_rejected(seed_case, case, reason):
    path, audit, mapping, support = seed_case
    stride, mode = 4, "processed"
    if case == "grid":
        stride = 3
    elif case == "source":
        support["dzx_sha256"] = "other"
    elif case == "radar":
        support["dzt_sha256"] = "other"
    elif case == "mapping":
        support["reference_label_mapping"] = {"0": 2}
    elif case == "sample":
        support["observations"]["1"][0]["sample"] += 1
    elif case == "duplicate":
        support["observations"]["1"].append(support["observations"]["1"][0])
    elif case == "missing":
        support["observations"] = {}
    elif case == "raw":
        mode = "raw"
    elif case == "fractional":
        support["observations"]["1"][0]["trace"] = 16.0
    path.write_text(json.dumps(support))
    with pytest.raises(ValueError, match=reason):
        load_support(path, audit, mapping, stride, mode)


def test_legacy_reuse_reads_seed_rows_and_reference_never_tracker_paths(seed_case):
    path, audit, mapping, _ = seed_case
    document = {
        "schema": "conventional-evaluation-v2",
        "mode": "processed",
        "audit": audit,
        "stride": 16,
        "reference_label_mapping": mapping,
        "methods": {
            "seed_hybrid": {
                "layers": {
                    "1": {
                        "seed_rows": [1, 3, 5],
                        "accepted_pick_agreement": 0.1,
                        "diagnostics": {"hypothesis_samples": [[999] * 7]},
                    }
                }
            }
        },
    }
    path.write_text(json.dumps(document))
    anchors, provenance = load_support(path, audit, mapping, 4, "processed")
    assert anchors == {1: {16: 31, 48: 30, 80: 32}}
    assert provenance["basis"].startswith("prior_seed_rows")
    document["methods"]["other"] = copy.deepcopy(document["methods"]["seed_hybrid"])
    document["methods"]["other"]["layers"]["1"]["seed_rows"] = [1, 3, 4]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="inconsistent"):
        load_support(path, audit, mapping, 4, "processed")


def test_evaluation_reuses_clicks_pulses_and_excludes_support_from_metrics(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from conftest import write_dzt

    import gpr_layer_audit.conventional as evaluation

    source = tmp_path / "processed.DZT"
    axis = np.arange(128)
    radar = np.tile(1000 * np.sin(axis / 4), (97, 1))
    write_dzt(source, radar, range_ns=12.8)
    dzx = source.with_suffix(".DZX")
    dzx.write_text(
        "<DZX><LayerGroup><layerNum>0</layerNum>"
        + "".join(
            f"<LayerWayPt><scanSampChanProp>{r},{30 + r % 3},0,2</scanSampChanProp>"
            "<timeAmpDepVel>1,1,0,0</timeAmpDepVel></LayerWayPt>"
            for r in range(97)
        )
        + "</LayerGroup></DZX>"
    )
    calls = []

    def fake_pick(data, surface, layers, **kwargs):
        calls.append(copy.deepcopy(kwargs["anchor_samples"]))
        # Tracker input contains seed coordinates only. Return no measured observations.
        assert "references" not in kwargs and "observations" not in kwargs
        assert kwargs["ml_policy"] == "off"
        return {
            1: SimpleNamespace(
                layer_order=1,
                samples=np.full(len(data), -1, np.int32),
                confidence=np.zeros(len(data)),
                visible=np.zeros(len(data), bool),
                provisional_samples=None,
                candidate_components={},
                evidence={},
                provenance={},
            )
        }

    monkeypatch.setattr(evaluation, "pick_interfaces", fake_pick)
    monkeypatch.setattr(evaluation, "_overlay", lambda *args: None)
    coarse = evaluation.evaluate_reference(
        dzx, output=tmp_path / "coarse.json", stride=16, methods=["seed_hybrid"]
    )
    fine = evaluation.evaluate_reference(
        dzx,
        output=tmp_path / "fine.json",
        stride=4,
        seed_source=tmp_path / "coarse.json",
        methods=["seed_hybrid"],
    )
    assert {r * 16: s for r, s in calls[0][1].items()} == {r * 4: s for r, s in calls[1][1].items()}
    assert coarse["seed_support"] == fine["seed_support"]
    assert coarse["scoring_pulses"] == fine["scoring_pulses"]
    assert fine["methods"]["seed_hybrid"]["layers"]["1"]["observations_excluding_seeds"] == 22
    # Withheld labels must not alter reused support or its pulse estimate.
    original_read = evaluation.read_dzx
    from dataclasses import replace

    def changed_withheld(path):
        metadata = original_read(path)
        layer = metadata.layers[0]
        modified = tuple(
            p if p.trace in (16, 48, 80) else replace(p, sample=75) for p in layer.picks
        )
        return replace(metadata, layers=(replace(layer, picks=modified),))

    monkeypatch.setattr(evaluation, "read_dzx", changed_withheld)
    changed = evaluation.evaluate_reference(
        dzx,
        output=tmp_path / "changed.json",
        stride=4,
        seed_source=tmp_path / "coarse.json",
        methods=["seed_hybrid"],
    )
    assert calls[-1] == calls[-2]
    assert changed["scoring_pulses"] == fine["scoring_pulses"]

    # Resolution scoring must use the common native observations, not each
    # run's different denominator. Four common rows remain after seed exclusion.
    import runpy
    from pathlib import Path

    compare = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/compare_conventional_resolution.py")
    )["compare"]
    array_path = tmp_path / "fine-seed_hybrid.npz"
    with np.load(array_path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["layer1_accepted"][64 // 4] = 31
    arrays["layer1_accepted"][32 // 4] = 80
    # A positive stored sample that was not accepted/visible must not be counted.
    arrays["layer1_accepted"][96 // 4] = 30
    np.savez_compressed(array_path, **arrays)
    for observation in fine["methods"]["seed_hybrid"]["layers"]["1"]["evaluation_observations"]:
        observation["accepted"] = observation["row"] in (64 // 4, 32 // 4)
    (tmp_path / "fine.json").write_text(json.dumps(fine))
    comparison = compare(tmp_path / "coarse.json", tmp_path / "fine.json")
    layer = comparison["layers"]["1"]
    assert layer["common_observations_excluding_seeds"] == 4
    assert layer["runs"][1]["accepted"] == 2
    assert layer["runs"][1]["accepted_agree"] == 1
    assert layer["runs"][1]["correct_coverage"] == 0.25
    assert layer["runs"][1]["reflector_switches"] == 1
    fine["seed_support"]["observations"]["1"][0]["trace"] = 4
    # Row 4 has the same reference sample as row 16, so this is valid support
    # individually, but cannot be called an identical-seed resolution comparison.
    (tmp_path / "different.json").write_text(json.dumps(fine))
    with pytest.raises(ValueError, match="identical native seeds"):
        compare(tmp_path / "coarse.json", tmp_path / "different.json")
