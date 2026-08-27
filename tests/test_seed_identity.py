from __future__ import annotations

import numpy as np

from gpr_layer_audit.processing.seed_identity import stationary_competitor_mask


def _case(seed_samples: tuple[int, ...], *, break_rows: set[int] | None = None):
    rows, samples = 81, 96
    radar = np.zeros((rows, samples), dtype=np.float32)
    # Persistent horizontal packet plus a seed-selected dipping packet.
    radar[:, 38] = -1.0
    radar[:, 39] = -0.45
    selected = np.rint(np.linspace(56, 68, rows)).astype(int)
    radar[np.arange(rows), selected] = -0.72
    candidates = np.column_stack((np.full(rows, 38), selected, np.full(rows, -1))).astype(np.int32)
    anchors = {
        row: int(sample)
        for row, sample in zip(
            np.linspace(10, 70, len(seed_samples), dtype=int), seed_samples, strict=True
        )
    }
    lower = np.full(rows, 25.0)
    upper = np.full(rows, 80.0)
    blocked, dense = stationary_competitor_mask(
        radar,
        candidates,
        anchors,
        lower,
        upper,
        pulse_width_samples=6.0,
        horizontal_step_m=1.0,
        break_rows=break_rows or set(),
    )
    return blocked, dense, anchors


def test_persistent_competitor_is_excluded_only_between_bracketing_seeds():
    blocked, dense, anchors = _case((58, 66))
    first, last = sorted(anchors)
    assert np.all(blocked[first + 1 : last, 0])
    assert not np.any(blocked[: first + 1, 0])
    assert not np.any(blocked[last:, 0])
    assert not np.any(blocked[:, 1])
    assert np.all(dense[[first, last]] == 0)


def test_horizontal_packet_is_protected_when_user_seeds_it():
    blocked, _dense, _anchors = _case((38, 38))
    assert not np.any(blocked[:, 0])


def test_single_seed_never_creates_a_road_length_exclusion():
    blocked, dense, _anchors = _case((60,))
    assert not np.any(blocked)
    assert not np.any(dense)


def test_structural_break_prevents_seed_pair_from_owning_the_interval():
    blocked, dense, _anchors = _case((58, 66), break_rows={40})
    assert not np.any(blocked)
    assert not np.any(dense)
