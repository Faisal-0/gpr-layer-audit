from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

MIN_HEADER_SIZE = 1024


class DZTFormatError(ValueError):
    """Raised when a DZT file is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class DZTHeader:
    tag: int
    data_field: int
    samples_per_trace: int
    bits_per_sample: int
    zero: int
    scans_per_second: float
    scans_per_meter: float
    meters_per_mark: float
    position_ns: float
    range_ns: float
    passes: int
    range_gain_offset: int
    range_gain_size: int
    range_gain_bytes: bytes
    channels: int
    dielectric: float
    top_m: float
    depth_m: float
    antenna: str
    system_code: int
    version_code: int
    data_offset: int
    dtype: np.dtype
    trace_count: int
    file_size: int
    trailing_bytes: int
    raw_header: bytes

    @property
    def sample_interval_ns(self) -> float:
        return self.range_ns / self.samples_per_trace

    @property
    def distance_per_trace_m(self) -> float | None:
        return 1.0 / self.scans_per_meter if self.scans_per_meter > 0 else None

    @property
    def gain_signature(self) -> str:
        return self.range_gain_bytes.hex()


def _dtype_for(bits: int) -> np.dtype:
    if bits == 8:
        return np.dtype("u1")
    if bits == 16:
        return np.dtype("<u2")
    if bits == 32:
        return np.dtype("<i4")
    raise DZTFormatError(f"Unsupported DZT sample width: {bits} bits")


def _unpack(fmt: str, data: bytes, offset: int):
    try:
        return struct.unpack_from(fmt, data, offset)[0]
    except struct.error as exc:
        raise DZTFormatError(f"DZT header is truncated at byte {offset}") from exc


def read_dzt_header(path: str | Path) -> DZTHeader:
    file_path = Path(path)
    file_size = file_path.stat().st_size
    if file_size < MIN_HEADER_SIZE:
        raise DZTFormatError("DZT file is smaller than the minimum 1024-byte header")

    with file_path.open("rb") as stream:
        first = stream.read(MIN_HEADER_SIZE)

    tag = _unpack("<H", first, 0)
    data_field = _unpack("<H", first, 2)
    samples = _unpack("<H", first, 4)
    bits = _unpack("<H", first, 6)
    zero = _unpack("<h", first, 8)
    scans_per_second = _unpack("<f", first, 10)
    scans_per_meter = _unpack("<f", first, 14)
    meters_per_mark = _unpack("<f", first, 18)
    position_ns = _unpack("<f", first, 22)
    range_ns = _unpack("<f", first, 26)
    passes = _unpack("<H", first, 30)
    range_gain_offset = _unpack("<H", first, 40)
    range_gain_size = _unpack("<H", first, 42)
    channels = max(1, _unpack("<H", first, 52))
    dielectric = _unpack("<f", first, 54)
    top_m = _unpack("<f", first, 58)
    depth_m = _unpack("<f", first, 62)
    antenna = first[98:112].split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
    version_system = first[113]
    system_code = version_system >> 3
    version_code = version_system & 0x07

    if samples <= 0 or channels <= 0:
        raise DZTFormatError(f"Invalid DZT dimensions: {samples} samples, {channels} channels")
    if not np.isfinite(range_ns) or range_ns <= 0:
        raise DZTFormatError(f"Invalid DZT time range: {range_ns}")

    dtype = _dtype_for(bits)
    data_offset = (
        MIN_HEADER_SIZE * data_field
        if 0 < data_field < MIN_HEADER_SIZE
        else MIN_HEADER_SIZE * channels
    )
    if data_offset < MIN_HEADER_SIZE or data_offset >= file_size:
        raise DZTFormatError(f"Invalid DZT data offset {data_offset} for {file_size}-byte file")

    with file_path.open("rb") as stream:
        stream.seek(range_gain_offset)
        gain_bytes = stream.read(range_gain_size) if range_gain_size else b""
        stream.seek(0)
        raw_header = stream.read(data_offset)

    bytes_per_trace = samples * channels * dtype.itemsize
    payload_size = file_size - data_offset
    trace_count, trailing = divmod(payload_size, bytes_per_trace)
    if trace_count <= 0:
        raise DZTFormatError("DZT file contains no complete traces")

    return DZTHeader(
        tag=tag,
        data_field=data_field,
        samples_per_trace=samples,
        bits_per_sample=bits,
        zero=zero,
        scans_per_second=scans_per_second,
        scans_per_meter=scans_per_meter,
        meters_per_mark=meters_per_mark,
        position_ns=position_ns,
        range_ns=range_ns,
        passes=passes,
        range_gain_offset=range_gain_offset,
        range_gain_size=range_gain_size,
        range_gain_bytes=gain_bytes,
        channels=channels,
        dielectric=dielectric,
        top_m=top_m,
        depth_m=depth_m,
        antenna=antenna,
        system_code=system_code,
        version_code=version_code,
        data_offset=data_offset,
        dtype=dtype,
        trace_count=trace_count,
        file_size=file_size,
        trailing_bytes=trailing,
        raw_header=raw_header,
    )


def fingerprint_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class DZTFile:
    """Read-only, memory-mapped access to a GSSI DZT acquisition."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.header = read_dzt_header(self.path)
        self._raw = np.memmap(
            self.path,
            dtype=self.header.dtype,
            mode="r",
            offset=self.header.data_offset,
            shape=(
                self.header.trace_count,
                self.header.channels,
                self.header.samples_per_trace,
            ),
            order="C",
        )

    def channel(
        self,
        channel: int = 0,
        start_trace: int = 0,
        stop_trace: int | None = None,
        *,
        copy: bool = False,
    ) -> NDArray[np.number]:
        if not 0 <= channel < self.header.channels:
            raise IndexError(f"Channel {channel} is outside 0..{self.header.channels - 1}")
        stop = self.header.trace_count if stop_trace is None else stop_trace
        if not 0 <= start_trace <= stop <= self.header.trace_count:
            raise IndexError("Requested trace range is outside the acquisition")
        view = self._raw[start_trace:stop, channel, :]
        return np.array(view) if copy else view

    def iter_channel_chunks(self, channel: int = 0, chunk_traces: int = 8192):
        for start in range(0, self.header.trace_count, chunk_traces):
            stop = min(start + chunk_traces, self.header.trace_count)
            yield start, self.channel(channel, start, stop)
