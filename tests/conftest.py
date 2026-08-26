from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest


def write_dzt(
    path: Path,
    data: np.ndarray,
    *,
    range_ns: float = 15.0,
    scans_per_meter: float = 10.0,
    dielectric: float = 7.0,
    antenna: str = "42000S",
    gain_record: bytes = b"\x01\x00\x00\x00\x00\x04",
) -> None:
    traces, samples = data.shape
    header = bytearray(1024)
    struct.pack_into("<HHHHh", header, 0, 0x00FF, 1, samples, 32, 0)
    struct.pack_into("<fffff", header, 10, 100.0, scans_per_meter, 1.0, 0.0, range_ns)
    struct.pack_into("<H", header, 30, 0)
    struct.pack_into("<HH", header, 40, 512, len(gain_record))
    struct.pack_into("<H", header, 52, 1)
    struct.pack_into("<fff", header, 54, dielectric, 0.0, 0.0)
    header[98:112] = antenna.encode("ascii").ljust(14, b"\x00")
    header[113] = (9 << 3) | 2
    header[512 : 512 + len(gain_record)] = gain_record
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(np.asarray(data, dtype="<i4").tobytes(order="C"))


def pulse(samples: int, centre: float, width: float = 2.2) -> np.ndarray:
    x = np.arange(samples, dtype=float)
    z = (x - centre) / width
    return (1.0 - z**2) * np.exp(-0.5 * z**2)


@pytest.fixture
def synthetic_acquisition(tmp_path: Path):
    traces, samples = 240, 256
    rng = np.random.default_rng(42)
    plate_wave = -8_000_000 * pulse(samples, 70, 2.5)
    plate_wave += 1_500_000 * pulse(samples, 79, 3.0)
    plate = np.tile(plate_wave, (60, 1)) + rng.normal(0, 8_000, (60, samples))
    road = np.empty((traces, samples), dtype=float)
    expected = {1: [], 2: [], 3: []}
    for index in range(traces):
        surface = 70 + round(1.5 * np.sin(index / 19))
        l1 = surface + 27 + round(2 * np.sin(index / 23))
        l2 = surface + 64 + round(3 * np.sin(index / 31))
        l3 = surface + 105 + round(4 * np.sin(index / 37))
        signal = np.roll(plate_wave * 0.43, surface - 70)
        signal += 720_000 * pulse(samples, l1, 2.4)
        signal -= 540_000 * pulse(samples, l2, 3.2)
        signal += 390_000 * pulse(samples, l3, 4.0)
        signal += rng.normal(0, 20_000, samples)
        road[index] = signal
        expected[1].append(l1)
        expected[2].append(l2)
        expected[3].append(l3)
    road_path = tmp_path / "road.DZT"
    plate_path = tmp_path / "plate.DZT"
    write_dzt(road_path, road)
    write_dzt(plate_path, plate)
    (tmp_path / "road.DZG").write_text(
        "$GSSIS,0,0\n$GPGGA,100000,3300.0000,N,07300.0000,E,1,08,1.0,500.0,M,0,M,,\n"
        "$GSSIS,120,0\n$GPGGA,100010,3300.0060,N,07300.0060,E,1,09,1.0,501.0,M,0,M,,\n"
        "$GSSIS,239,0\n$GPGGA,100020,3300.0120,N,07300.0120,E,1,10,1.0,502.0,M,0,M,,\n",
        encoding="ascii",
    )
    (tmp_path / "road.DZX").write_text(
        """<?xml version="1.0"?>
<DZX><GlobalProperties><verticalUnit>cm</verticalUnit><horizontalUnit>m</horizontalUnit>
<dielectric>7.0</dielectric><unitsPerMark>1.0</unitsPerMark><unitsPerScan>0.1</unitsPerScan>
</GlobalProperties><Macro><CustomFIR><antennaName>42000</antennaName><serialNumber>734</serialNumber>
<sampsPerNs>17.0667</sampsPerNs><doubleArray>0.1,-0.2,0.1</doubleArray></CustomFIR></Macro>
<DataCollection><system>SIR-30</system><softwareVersion>1.0</softwareVersion></DataCollection></DZX>""",
        encoding="utf-8",
    )
    return road_path, plate_path, expected
