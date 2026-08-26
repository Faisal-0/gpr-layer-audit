from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes
from pathlib import Path

from gpr_layer_audit.models import AcquisitionFileSet
from gpr_layer_audit.processing import AnalysisOptions, analyze_acquisition


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def working_set_bytes() -> int:
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return int(counters.WorkingSetSize)


def main() -> int:
    road = Path("GPR Data/talagang/TALAGANG.PRJ/TALAGANG_001.DZT")
    plate = Path("GPR Data/talagang/TALAGANG METAL PLATE.PRJ/TALAGANG METAL PLATE_001.DZT")
    peak = [working_set_bytes()]
    finished = threading.Event()

    def sample() -> None:
        while not finished.wait(0.025):
            peak[0] = max(peak[0], working_set_bytes())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    result = analyze_acquisition(
        AcquisitionFileSet(road),
        AcquisitionFileSet(plate),
        AnalysisOptions(stack_size=20, accept_scan_dielectric=True),
    )
    elapsed = time.perf_counter() - started
    finished.set()
    sampler.join()
    peak[0] = max(peak[0], working_set_bytes())
    print(f"Talagang traces: {result.header.trace_count:,}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Peak working set MiB: {peak[0] / 1024**2:.1f}")
    print(f"Review groups: {len(result.review_issues)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
