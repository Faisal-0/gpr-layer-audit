from __future__ import annotations

from pathlib import Path

import pytest

from gpr_layer_audit.io import DZTFile, read_dzg, read_dzx

TALAGANG = Path("GPR Data/talagang/TALAGANG.PRJ/TALAGANG_001.DZT")


@pytest.mark.data
@pytest.mark.skipif(not TALAGANG.exists(), reason="Talagang source data not available")
def test_talagang_source_contract():
    dzt = DZTFile(TALAGANG)
    assert dzt.header.trace_count == 75_436
    assert dzt.header.samples_per_trace == 512
    assert dzt.header.bits_per_sample == 32
    assert dzt.header.range_ns == pytest.approx(15.0)
    assert dzt.header.scans_per_meter == pytest.approx(40.0)
    assert dzt.header.data_offset == 131_072
    # The file contains 262 GPS epochs; six report an invalid fix and are
    # intentionally excluded from interpolation.
    assert len(read_dzg(TALAGANG.with_suffix(".DZG"))) == 256
    assert read_dzx(TALAGANG.with_suffix(".DZX")).dielectric == 7.0
