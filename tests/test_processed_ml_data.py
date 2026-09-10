"""Research cache coordinate, masking, provenance and split contracts."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from conftest import write_dzt

from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.ml.processed_data import (
    NativeWindow,
    build_processed_dataset,
    load_processed_manifest,
    open_processed_record,
    research_leave_one_group_out,
)


def _pair(root, group="MANDIALI", *, suffix="P_11", picks=None):
    path = root / f"{group}.PRJ" / "Proc" / f"{group}_001 {suffix}.DZT"
    path.parent.mkdir(parents=True, exist_ok=True)
    signal = np.tile(np.array([0, 0, 2, -5, 11, -13, 8, -3, 4, 0, 0, 0]), (8, 1))
    signal += np.arange(8)[:, None] * (signal != 0)
    write_dzt(path, signal, range_ns=12, scans_per_meter=40)
    with path.open("r+b") as stream:
        stream.seek(22)
        stream.write(struct.pack("<f", -3.0))
    picks = picks if picks is not None else [(0, 1, 3, 0.0), (1, 2, 5, 0.0), (2, 3, 8, 0.0)]
    groups = []
    for layer in range(3):
        points = []
        for order, trace, sample, time_error in picks:
            if order == layer:
                points.append(
                    f"<LayerWayPt><scanSampChanProp>{trace},{sample},0,2</scanSampChanProp>"
                    f"<timeAmpDepVel>{sample - 3 + time_error},{signal[trace, sample]},0,0"
                    "</timeAmpDepVel></LayerWayPt>"
                )
        groups.append(
            f"<LayerGroup><layerNum>{layer}</layerNum><groupName>Layer {layer + 1}"
            f"</groupName><pickType>0</pickType><link>1</link>{''.join(points)}</LayerGroup>"
        )
    path.with_suffix(".DZX").write_text("<DZX>" + "".join(groups) + "</DZX>")
    return path, signal


def test_native_cache_preserves_signed_samples_times_masks_and_unknowns(tmp_path):
    root = tmp_path / "source"
    path, signal = _pair(root)
    manifest = build_processed_dataset(root, tmp_path / "cache")
    assert manifest["counts"]["unique_pair_stored_interface_observations"] == 3
    assert manifest["counts"]["processed_dzt_files"] == 1
    data = open_processed_record(
        tmp_path / "cache/manifest.json", manifest["records"][0], verify_hashes=True
    )
    record = data["record"]
    np.testing.assert_array_equal(data["amplitudes"], signal)
    np.testing.assert_array_equal(data["native_trace_indices"], np.arange(8))
    np.testing.assert_array_equal(data["native_sample_indices"], np.arange(12))
    np.testing.assert_array_equal(data["distances_m"], np.arange(8) * 0.025)
    assert record["time_origin_ns"] == -3
    assert record["sample_interval_ns"] == 1
    assert record["source_sha256"] == fingerprint_file(path)
    assert not data["sample_validity"][:, :2].any()
    assert not data["sample_validity"][:, 9:].any()
    assert data["sample_validity"][:, 2:9].all()
    assert int(data["label_valid"].sum()) == 3
    assert np.isnan(data["labels"][~data["label_valid"]]).all()
    assert data["label_signed_lobe"][0, 1] == -1
    assert data["label_signed_lobe"][1, 2] == -1
    assert not record["visibility_absence_labels"]
    assert record["training_use"] == "research_allowed"
    assert "unknown" in record["label_provenance"]
    assert not data["amplitudes"].flags.writeable


def test_invalid_reference_time_and_padding_never_become_training_targets(tmp_path):
    _pair(tmp_path / "source", picks=[(0, 1, 3, 0.3), (1, 2, 9, 0), (2, 3, 8, 0)])
    manifest = build_processed_dataset(tmp_path / "source", tmp_path / "cache")
    record = manifest["records"][0]
    assert record["label_counts"] == {"1": 0, "2": 0, "3": 1}
    assert record["exclusions"]["recorded_time_mismatch"] == 1
    assert record["exclusions"]["numerical_padding_at_label"] == 1


def test_conflicting_duplicates_remain_unknown_not_last_pick_wins(tmp_path):
    _pair(tmp_path / "source", picks=[(0, 1, 3, 0), (0, 1, 4, 0), (0, 1, 3, 0)])
    manifest = build_processed_dataset(tmp_path / "source", tmp_path / "cache")
    data = open_processed_record(tmp_path / "cache/manifest.json", manifest["records"][0])
    assert not data["label_valid"].any()
    assert np.isnan(data["labels"]).all()


def test_explicit_prohibition_and_immutable_output(tmp_path):
    path, _ = _pair(tmp_path / "source")
    manifest = build_processed_dataset(
        tmp_path / "source",
        tmp_path / "cache",
        prohibited_hashes=[fingerprint_file(path.with_suffix(".DZX"))],
    )
    assert not manifest["records"]
    assert manifest["inventory"][0]["label_status"] == "explicitly_prohibited_or_evaluation_only"
    with pytest.raises(FileExistsError, match="immutable"):
        build_processed_dataset(tmp_path / "source", tmp_path / "cache")


def test_two_other_groups_enable_all_three_research_subbase_folds():
    records = []
    for group in ("mandiali", "gujrat", "jamshoro"):
        for variant in ("p11", "p21"):
            records.append(
                {
                    "record_id": group + variant,
                    "physical_road_group": group,
                    "training_use": "research_allowed",
                    "label_counts": {"3": 40},
                }
            )
    records.append(
        {
            "record_id": "prohibited",
            "physical_road_group": "fourth",
            "training_use": "evaluation_only",
            "label_counts": {"3": 40},
        }
    )
    folds = research_leave_one_group_out({"records": records}, 3)
    assert len(folds) == 3
    for fold in folds:
        assert len(fold["training_groups"]) == 2
        assert len(fold["evaluation_record_ids"]) == 2
        assert not any(x.startswith(fold["held_out_group"]) for x in fold["training_record_ids"])
        assert "prohibited" not in fold["training_record_ids"]
        assert not fold["production_eligible"]
    assert not research_leave_one_group_out({"records": records}, 3, min_training_groups=3)


@pytest.mark.parametrize("flip_trace", [False, True])
@pytest.mark.parametrize("flip_sample", [False, True])
def test_crop_padding_decimation_and_augmentation_roundtrip(flip_trace, flip_sample):
    image = (np.arange(40 * 32).reshape(40, 32) - 500).astype(np.float32)
    valid = np.ones_like(image, bool)
    window = NativeWindow(-4, 24, -3, 40, 2, flip_trace, flip_sample)
    tile, mask = window.extract(image, valid)
    native_t, native_s = window.axes()
    yy, xx = np.where(mask)
    np.testing.assert_array_equal(tile[yy, xx], image[native_t[yy], native_s[xx]])
    assert np.all(tile[~mask] == 0)
    # Fractional clicks are inverted without quantization, including distant seeds.
    traces, samples = np.array([0, 10, 25.5, 100]), np.array([3.25, 20, 21.5, -10])
    lt, ls = window.native_to_local(traces, samples)
    rt, rs = window.local_to_native(lt, ls)
    np.testing.assert_array_equal(rt, traces)
    np.testing.assert_array_equal(rs, samples)
    # Signed-lobe identity follows translated/reversed data; no sign inversion.
    assert np.sign(tile[yy[0], xx[0]]) == np.sign(image[native_t[yy[0]], native_s[xx[0]]])


def test_hash_verification_detects_changed_cache(tmp_path):
    _pair(tmp_path / "source")
    manifest = build_processed_dataset(tmp_path / "source", tmp_path / "cache")
    path = tmp_path / "cache" / manifest["records"][0]["arrays"]["labels"]
    values = np.load(path)
    values[0, 1] += 1
    np.save(path, values)
    with pytest.raises(ValueError, match="fingerprint changed"):
        open_processed_record(
            tmp_path / "cache/manifest.json", manifest["records"][0], verify_hashes=True
        )


def test_reject_unsupported_preprocessing(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"schema": "processed-ml-dataset-v1", "preprocessing_version": "raw"})
    )
    with pytest.raises(ValueError, match="preprocessing"):
        load_processed_manifest(path)
