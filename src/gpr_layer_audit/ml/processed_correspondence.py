"""Small processed-radar correspondence experiment; never a calibrated probability.

This module deliberately does not use the raw-coordinate U-Net dataset loader.
Only explicit reviewed rows supply training targets. Production feature extraction
accepts radar, a validity mask, candidate coordinates and immutable operating seeds;
it has no reference-label input. No resampling or gap interpolation is performed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class PatchContract:
    temporal_radius_ns: float = 0.46875
    temporal_points: int = 33
    lateral_radius_m: float = 0.1
    lateral_points: int = 5
    ridge: float = 0.01
    version: str = "processed-pair-logistic-v1"

    def offsets(self, dt_ns, dx_m):
        if min(dt_ns, dx_m) <= 0 or not np.isfinite([dt_ns, dx_m]).all():
            raise ValueError("Positive physical sampling required")
        if (
            min(self.temporal_points, self.lateral_points) < 1
            or self.temporal_points % 2 != 1
            or self.lateral_points % 2 != 1
        ):
            raise ValueError("Patch axes must contain an odd number of points")
        if min(self.temporal_radius_ns, self.lateral_radius_m) < 0:
            raise ValueError("Nonnegative physical patch extents required")
        # Rounding selects native observations, never synthesizes measurements.
        time = np.rint(
            np.linspace(-self.temporal_radius_ns, self.temporal_radius_ns, self.temporal_points)
            / dt_ns
        ).astype(int)
        road = np.rint(
            np.linspace(-self.lateral_radius_m, self.lateral_radius_m, self.lateral_points) / dx_m
        ).astype(int)
        return road, time


DEFAULT_CONTRACT = PatchContract()


def extract_patches(measurement, valid, rows, samples, dt_ns, dx_m, contract=DEFAULT_CONTRACT):
    """Return normalized native patches and explicit support without filling gaps."""
    measurement, valid = np.asarray(measurement), np.asarray(valid)
    rows, samples = np.asarray(rows), np.asarray(samples)
    if measurement.ndim != 2 or valid.shape != measurement.shape or valid.dtype != np.bool_:
        raise ValueError("A boolean measurement-grid mask is required")
    if rows.shape != samples.shape or rows.ndim != 1:
        raise ValueError("One native sample per row coordinate is required")
    if not np.equal(rows, np.rint(rows)).all() or not np.equal(samples, np.rint(samples)).all():
        raise ValueError("Native coordinates must be integral; snapping is prohibited")
    if np.any((rows < 0) | (rows >= len(measurement))):
        raise ValueError("Candidate row outside radar")
    if np.any((samples < 0) | (samples >= measurement.shape[1])):
        raise ValueError("Candidate sample outside radar")
    road, time = contract.offsets(dt_ns, dx_m)
    rr = rows.astype(int)[:, None, None] + road[None, :, None]
    ss = samples.astype(int)[:, None, None] + time[None, None, :]
    inside = (rr >= 0) & (rr < len(measurement)) & (ss >= 0) & (ss < measurement.shape[1])
    rr, ss = np.clip(rr, 0, len(measurement) - 1), np.clip(ss, 0, measurement.shape[1] - 1)
    support = inside & valid[rr, ss] & np.isfinite(measurement[rr, ss])
    values = np.where(support, measurement[rr, ss], 0).astype(np.float32)
    norm = np.sqrt(np.sum(values * values, axis=2, keepdims=True))
    values /= np.maximum(norm, np.finfo(np.float32).tiny)
    centre = (contract.lateral_points // 2, contract.temporal_points // 2)
    usable = support[:, centre[0], centre[1]] & (norm[:, centre[0], 0] > 0)
    return values, support, usable


def pair_features(candidate, candidate_support, seed, seed_support):
    """Gain-invariant signed waveform comparisons; no road id or absolute depth."""
    common = candidate_support & seed_support
    product = np.where(common, candidate * seed, 0)
    difference = np.where(common, (candidate - seed) ** 2, 0)
    count = np.maximum(common.sum(axis=2), 1)
    return np.concatenate(
        [
            product.reshape(len(candidate), -1),
            difference.reshape(len(candidate), -1),
            product.sum(axis=2),
            difference.sum(axis=2) / count,
            common.mean(axis=2),
        ],
        axis=1,
    ).astype(np.float32)


def interval_weights(rows, seed_rows):
    """Blend independent bracketing-seed evidence; tails use their nearest seed."""
    rows, seed_rows = np.asarray(rows, float), np.asarray(seed_rows, float)
    if rows.ndim != 1 or not np.isfinite(rows).all() or not np.isfinite(seed_rows).all():
        raise ValueError("Finite one-dimensional row coordinates required")
    if not len(seed_rows) or np.any(np.diff(seed_rows) <= 0):
        raise ValueError("Unique sorted operating seeds are required")
    weights = np.zeros((len(rows), len(seed_rows)), dtype=np.float32)
    right = np.clip(np.searchsorted(seed_rows, rows), 0, len(seed_rows) - 1)
    left = np.maximum(right - 1, 0)
    left[rows >= seed_rows[-1]] = len(seed_rows) - 1
    denom = np.maximum(seed_rows[right] - seed_rows[left], 1)
    fraction = np.where(left == right, 0, (rows - seed_rows[left]) / denom)
    weights[np.arange(len(rows)), left] += 1 - fraction
    weights[np.arange(len(rows)), right] += fraction
    return weights


def reviewed_candidate_targets(
    trace, candidates, reference_sample, tolerance, *, support_row=False
):
    """1=matching, 0=different lobe, -1=unknown/non-target; never label unknown rows."""
    from ..conventional import _same_lobe

    targets = np.full(len(candidates), -1, dtype=np.int8)
    if support_row or reference_sample is None:
        return targets
    if not np.isfinite(reference_sample) or not 0 <= reference_sample < len(trace):
        raise ValueError("Reviewed native sample outside radar")
    for index, sample in enumerate(candidates):
        if not 0 <= sample < len(trace) or not np.isfinite(trace[int(sample)]):
            continue
        same = _same_lobe(trace, sample, reference_sample)
        if same and abs(sample - reference_sample) <= tolerance:
            targets[index] = 1
        elif not same and trace[int(sample)] != 0:
            targets[index] = 0
        # Same-lobe timing misses are not neighboring-reflector negatives.
    return targets


@dataclass
class PairLogistic:
    contract: PatchContract
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def evidence(self, features):
        return expit((features - self.mean) / self.scale @ self.coefficients + self.intercept)

    def save(self, path):
        np.savez_compressed(
            path,
            mean=self.mean,
            scale=self.scale,
            coefficients=self.coefficients,
            intercept=self.intercept,
            contract=np.array(json.dumps(asdict(self.contract), sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as values:
            contract = PatchContract(**json.loads(str(values["contract"])))
            return cls(
                contract,
                values["mean"],
                values["scale"],
                values["coefficients"],
                float(values["intercept"]),
            )


def fit_pair_logistic(features, target, weights=None, contract=DEFAULT_CONTRACT):
    """One fixed L2 logistic fit, standardization fitted to training pairs only."""
    values, target = np.asarray(features, float), np.asarray(target, float)
    if values.ndim != 2 or target.shape != (len(values),) or not np.isfinite(values).all():
        raise ValueError("Finite feature rows with one explicit binary label each required")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("Both positive and neighboring-lobe negative labels required")
    if not np.isfinite(contract.ridge) or contract.ridge <= 0:
        raise ValueError("Positive finite regularization required")
    weight = np.ones(len(target)) if weights is None else np.asarray(weights, float)
    if weight.shape != target.shape or np.any(weight < 0) or not np.isfinite(weight).all():
        raise ValueError("Finite nonnegative row weights required")
    if weight.sum() <= 0 or any(weight[target == label].sum() <= 0 for label in (0, 1)):
        raise ValueError("Both classes need positive training weight")
    weight /= weight.sum()
    mean = np.sum(values * weight[:, None], axis=0)
    scale = np.sqrt(np.sum((values - mean) ** 2 * weight[:, None], axis=0))
    scale = np.maximum(scale, 1e-3)
    values = (values - mean) / scale

    def objective(parameters):
        score = values @ parameters[:-1] + parameters[-1]
        residual = weight * (expit(score) - target)
        loss = np.sum(weight * (np.logaddexp(0, score) - target * score))
        loss += 0.5 * contract.ridge * np.sum(parameters[:-1] ** 2)
        gradient = np.r_[values.T @ residual + contract.ridge * parameters[:-1], residual.sum()]
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(values.shape[1] + 1),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-6},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Correspondence model did not converge: {result.message}")
    return PairLogistic(contract, mean, scale, result.x[:-1], float(result.x[-1])), {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "weighted_regularized_loss": float(result.fun),
        "message": str(result.message),
    }


def score_candidates(
    model, measurement, valid, rows, samples, anchors, dt_ns, dx_m, *, batch_size=4096
):
    """Score fixed candidates using only immutable operating seed patches.

    Returned evidence is uncalibrated and cannot by itself authorize acceptance.
    Invalid centre measurements return NaN, never a newly invented observation.
    """
    seeds = sorted(anchors.items())
    if not seeds:
        raise ValueError("At least one operating seed required")
    seed_rows, seed_samples = np.asarray(seeds).T
    patches, support, usable = extract_patches(
        measurement, valid, seed_rows, seed_samples, dt_ns, dx_m, model.contract
    )
    if not np.all(usable):
        raise ValueError("Operating seed is not an observable radar measurement")
    rows, samples = np.asarray(rows), np.asarray(samples)
    result = np.full(len(rows), np.nan)
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        candidate, candidate_support, observable = extract_patches(
            measurement, valid, rows[start:stop], samples[start:stop], dt_ns, dx_m, model.contract
        )
        weights = interval_weights(rows[start:stop], seed_rows)
        values = np.zeros(stop - start)
        for index, (seed, mask) in enumerate(zip(patches, support, strict=True)):
            features = pair_features(candidate, candidate_support, seed, mask)
            values += weights[:, index] * model.evidence(features)
        result[start:stop] = np.where(observable, values, np.nan)
    return result
