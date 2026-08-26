from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpr_layer_audit.io import DZTFile, interpolate_gps, read_dzg, read_dzx
from gpr_layer_audit.io.dzt import DZTFormatError


def test_dzt_header_and_memory_map(synthetic_acquisition):
    road_path, _, _ = synthetic_acquisition
    dzt = DZTFile(road_path)
    assert dzt.header.trace_count == 240
    assert dzt.header.samples_per_trace == 256
    assert dzt.header.bits_per_sample == 32
    assert dzt.header.antenna == "42000S"
    assert dzt.header.data_offset == 1024
    assert dzt.channel().shape == (240, 256)
    assert not dzt.channel().flags.writeable


def test_truncated_dzt_fails(tmp_path: Path):
    path = tmp_path / "bad.DZT"
    path.write_bytes(b"short")
    with pytest.raises(DZTFormatError):
        DZTFile(path)


def test_dzg_parse_and_bounded_interpolation(synthetic_acquisition):
    road_path, _, _ = synthetic_acquisition
    observations = read_dzg(road_path.with_suffix(".DZG"))
    assert len(observations) == 3
    targets = np.asarray([-1, 0, 60, 239, 240], dtype=float)
    lat, lon = interpolate_gps(observations, targets)
    assert np.isnan(lat[0]) and np.isnan(lat[-1])
    assert lat[1] == pytest.approx(33.0)
    assert lon[3] == pytest.approx(73.0002)


def test_dzx_parse(synthetic_acquisition):
    road_path, _, _ = synthetic_acquisition
    metadata = read_dzx(road_path.with_suffix(".DZX"))
    assert metadata.dielectric == 7.0
    assert metadata.units_per_scan == 0.1
    assert metadata.antenna_serial == "734"
    assert metadata.custom_fir == (0.1, -0.2, 0.1)
