"""Bounded numerical feature cache; never stores templates, seeds, or correspondence."""

import hashlib
from collections import OrderedDict
from threading import RLock

import numpy as np

_FEATURES = OrderedDict()
_LOCK = RLock()
_MAX_BYTES = 96 * 1024 * 1024


def radar_features(measurement, valid, width, cancel=None):
    from .continuation import local_phase_motion
    from .hybrid import wavelet_ridges

    if cancel and cancel():
        raise InterruptedError("Analysis cancelled")
    digest = hashlib.sha256(np.ascontiguousarray(measurement).view(np.uint8))
    digest.update(np.ascontiguousarray(valid).view(np.uint8))
    key = (digest.hexdigest(), measurement.shape, float(width))
    with _LOCK:
        if key in _FEATURES:
            result = _FEATURES.pop(key)
            _FEATURES[key] = result
            return result[0], result[1], True
    ridges = wavelet_ridges(measurement, width)
    motion = local_phase_motion(measurement, width, cancel)
    size = ridges.nbytes + sum(x.nbytes for x in motion.values())
    if size <= _MAX_BYTES:
        with _LOCK:
            while _FEATURES and sum(x[2] for x in _FEATURES.values()) + size > _MAX_BYTES:
                _FEATURES.popitem(last=False)
            _FEATURES[key] = (ridges, motion, size)
    return ridges, motion, False
