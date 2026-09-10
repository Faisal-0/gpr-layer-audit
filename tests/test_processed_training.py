"""Native coordinates, honest episodic supervision, and grouped fitting guards."""

import copy

import numpy as np
import pytest

from gpr_layer_audit.ml.processed_training import (
    EpisodeSampler,
    TrainingConfig,
    context_rows,
    extract_episode,
    scan_normalization,
    spatial_blocks,
)


def native_scan(traces=4001, samples=96):
    rng = np.random.default_rng(3)
    amplitude = rng.normal(size=(traces, samples)).astype(np.float32)
    return {
        "amplitudes": amplitude,
        "sample_validity": np.ones_like(amplitude, dtype=bool),
        "labels": np.tile(np.array([20.25, 40.5, 60.75])[:, None], (1, traces)).astype(np.float32),
        "label_valid": np.ones((3, traces), dtype=bool),
        "native_trace_indices": np.arange(traces),
        "native_sample_indices": np.arange(samples),
        "distances_m": np.arange(traces) * 0.025,
    }


def synthetic_dataset(monkeypatch, *, forbidden=False):
    from gpr_layer_audit.ml import processed_data

    arrays, records = {}, []
    for group in ("outer", "first", "second"):
        for variant in range(2):
            key = f"{group}-{variant}"
            arrays[key] = native_scan()
            records.append(
                {
                    "record_id": key,
                    "physical_road_group": group,
                    "source_sha256": key,
                    "acquisition_id": group,
                    "shape": [4001, 96],
                    "horizontal_step_m": 0.025,
                    "training_use": "research_allowed",
                    "label_counts": {"1": 4001, "2": 4001, "3": 4001},
                }
            )
    if forbidden:
        records[2]["training_use"] = "evaluation_only"
    monkeypatch.setattr(
        processed_data, "load_processed_manifest", lambda path: {"records": records}
    )
    monkeypatch.setattr(
        processed_data, "open_processed_record", lambda path, rec, **kw: arrays[rec["record_id"]]
    )
    config = TrainingConfig(held_out_group="outer", fine_width=33, patch_depth=25, block_m=16)
    return records, arrays, config


def test_native_grid_is_contiguous_and_exactly_multiscale_centered():
    for width in (33, 65, 129):
        rows = context_rows(500, width)
        wide = context_rows(500, width, 4)
        np.testing.assert_array_equal(np.diff(rows), 1)
        np.testing.assert_array_equal(np.diff(wide), 4)
        assert rows[width // 2] == wide[width // 2] == 500
        assert (rows[0] + rows[-1]) == (wide[0] + wide[-1])


def test_scan_normalization_preserves_signed_observations_and_ignores_invalid():
    data = np.array([[2, -3], [4, 1e20]], dtype=np.float32)
    valid = np.array([[True, True], [True, False]])
    scale = scan_normalization(data, valid)
    np.testing.assert_allclose(scale, np.sqrt(29 / 3))
    assert (data[valid] / scale[np.where(valid)[1]])[1] < 0
    np.testing.assert_array_equal(scan_normalization(data, valid, "none"), [1, 1])
    np.testing.assert_allclose(scan_normalization(data, valid, "depth_rms"), [np.sqrt(10), 3])


def test_label_free_extraction_keeps_fractional_seed_and_native_time():
    data = native_scan(traces=101)
    data.pop("labels")
    data.pop("label_valid")
    config = TrainingConfig(fine_width=33, patch_depth=25)
    episode = extract_episode(
        data,
        center=50,
        support_rows=[5],
        support_samples=[40.75],
        layer=3,
        config=config,
        scale=np.ones(96),
    )
    assert episode["fine"].shape == (1, 96, 33)
    np.testing.assert_array_equal(episode["fine"][0].T, data["amplitudes"][34:67])
    assert episode["relative_depth"][0, 40, 0] == -0.75
    assert episode["relative_dx_m"][0, 16] == pytest.approx((50 - 5) * 0.025)
    assert not episode["coarse_valid"][0, :, 0].any()  # padding is explicitly unknown
    assert not np.any(episode["coarse"][~episode["coarse_valid"]])
    assert not {"target_depth", "labels", "label_valid"} & episode.keys()


def test_block_buffer_covers_every_query_and_support_context():
    data = native_scan()
    config = TrainingConfig(fine_width=65, block_m=16)
    split, block, audit = spatial_blocks(data["distances_m"], config)
    assert audit["context_buffer_m"] >= 128 * 0.025
    for center in np.flatnonzero(split >= 0)[::11]:
        for rows in (context_rows(center, 65, 4), context_rows(center, config.patch_width)):
            assert np.all(block[rows] == block[center])
    assert {0, 1} <= set(split)


def test_research_two_groups_and_forbidden_records(monkeypatch):
    records, _, config = synthetic_dataset(monkeypatch, forbidden=True)
    sampler = EpisodeSampler("unused", config)
    assert {r["physical_road_group"] for r in sampler.records.values()} == {"first", "second"}
    assert records[2]["record_id"] not in sampler.records
    records[4]["training_use"] = records[5]["training_use"] = "prohibited"
    with pytest.raises(ValueError, match="two training groups"):
        EpisodeSampler("unused", config)


def test_source_and_acquisition_overlap_are_rejected(monkeypatch):
    records, _, config = synthetic_dataset(monkeypatch)
    records[2]["source_sha256"] = records[0]["source_sha256"]
    with pytest.raises(ValueError, match="leaked"):
        EpisodeSampler("unused", config)


def test_supports_fit_blocks_query_loss_masks_and_layer_balance(monkeypatch):
    _, arrays, config = synthetic_dataset(monkeypatch)
    # Missing observations must not become a learned target/background label.
    for data in arrays.values():
        data["label_valid"][:, ::3] = False
        data["labels"][:, ::3] = np.nan
    sampler = EpisodeSampler("unused", config)
    rng = np.random.default_rng(5)
    layers, budgets = [], []
    for validation in (False, True):
        for _ in range(30):
            episode = sampler.sample(rng, validation=validation)
            split, block = sampler.splits[episode["record_id"]]
            support = episode["support_rows"]
            assert np.all(split[support] == 0)
            assert np.all(split[episode["rows"]][episode["valid"]] == int(validation))
            assert not episode["support_mask"].any()
            assert not np.any(np.isin(support, episode["wide_rows"]))
            assert np.isfinite(episode["target_depth"][episode["valid"]]).all()
            assert not episode["valid"][episode["rows"] % 3 == 0].any()
            assert len(np.unique(block[episode["wide_rows"]])) == 1
            layers.append(int(episode["layer"]))
            budgets.append(len(support))
    assert set(layers) == {1, 2, 3}
    assert set(budgets) == {1, 2, 3, 4, 5}


def test_seed_extraction_does_not_depend_on_withheld_labels():
    data = native_scan(traces=201)
    changed = copy.copy(data)
    changed["labels"] = data["labels"] + 30
    cfg = TrainingConfig(fine_width=33)
    kwargs = dict(center=100, support_rows=[20], support_samples=[32.25], layer=3, config=cfg)
    first = extract_episode(data, **kwargs)
    second = extract_episode(changed, **kwargs)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_fractional_support_trace_is_rejected_without_silent_snapping():
    with pytest.raises(ValueError, match="integral native"):
        extract_episode(
            native_scan(traces=201),
            center=100,
            support_rows=[20.25],
            support_samples=[32.25],
            layer=3,
            config=TrainingConfig(fine_width=33),
        )
