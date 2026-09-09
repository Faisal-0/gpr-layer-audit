"""Capture unchanged dense latent states; evaluate potential basin emissions separately."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from extremum_state_model import basin_map
from seeded_eval import sha

from gpr_layer_audit.conventional import _same_lobe
from gpr_layer_audit.io.dzt import DZTFile
from gpr_layer_audit.processing import seeded_challenger as dense
from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask


def run(folder):
    destination = folder / "hidden-states"
    if destination.exists():
        raise ValueError("Preserve existing diagnosis")
    destination.mkdir()
    manifest = json.loads((folder / "prediction-manifest.json").read_text())
    entry = manifest["input"]
    assert sha(entry["dzt"]) == entry["dzt_sha256"]
    native = np.asarray(DZTFile(entry["dzt"]).channel(), np.float32)
    data, valid = native[::4], processed_boundary_mask(native)[::4]
    dt, dx = entry["dt_ns"], entry["dx_m"] * 4
    surface = int(np.floor(-entry["origin_ns"] / dt))
    source = inspect.getsource(dense.pick_seeded_packet)
    marker = "    # Latent gaps are not displayed as measured proposals either.\n"
    assert source.count(marker) == 1
    code = source.replace(marker, "    _capture(proposal.copy())\n" + marker)
    captures = []
    namespace = dict(dense.__dict__, _capture=captures.append)
    exec(compile(code, "<unchanged-dense-hidden-audit>", "exec"), namespace)
    picker = namespace["pick_seeded_packet"]
    original = dense._packet_correlations
    mapping = np.array([basin_map(row)[0] for row in data])
    report = {
        "claim": "Diagnostic projection only; no accepted measurements or old confidence reused",
        "script_sha256": sha(__file__),
        "layers": {},
    }
    for order in (2, 3):
        previous = np.load(folder / f"guide1-layer{order}.npz")
        allowed = np.isfinite(previous["candidate"])
        anchors = {int(r): s for r, s in manifest["layers"][str(order)]["seeds"].items()}

        def correlations(measurement, mask, operating, radius, *, candidate=allowed):
            bank, supported = original(measurement, mask, operating, radius)
            return bank, supported & candidate

        namespace["_packet_correlations"] = correlations
        path = picker(
            data,
            anchors,
            valid=valid,
            dt_ns=dt,
            dx_m=dx,
            pulse_width_samples=manifest["layers"][str(order)]["lobe_samples"],
            reference_surface=surface,
            config=dense.PacketConfig(guide_weight=1),
        )
        np.testing.assert_array_equal(path.samples, previous["samples"])
        np.testing.assert_array_equal(path.provisional_samples, previous["provisional"])
        np.testing.assert_array_equal(path.confidence, previous["margin"])
        member = captures[-1]
        peak = mapping[np.arange(len(member)), np.maximum(member, 0)]
        supported = (member >= 0) & (peak >= 0)
        supported &= allowed[np.arange(len(member)), np.maximum(peak, 0)]
        peak[~supported] = -1
        np.savez_compressed(destination / f"layer{order}.npz", member=member, peak=peak)
        # Only after the unchanged solver returns are evaluation records read.
        result = json.loads((folder / "evaluation.json").read_text())["results"][
            f"guide1-layer{order}"
        ]
        stats = dict(
            reviewed_nonseed=result["observations_excluding_seeds"],
            correct_original=result["proposed_agree"],
            projected_correct=0,
            previously_hidden_correct=0,
            hidden_with_peak=0,
        )
        for point in result["evaluation_observations"]:
            r, ref = point["row"], point["reference_sample"]
            hidden = point["proposed_sample"] < 0
            if peak[r] >= 0:
                correct = abs(peak[r] - ref) <= result["tolerance_samples"] and _same_lobe(
                    data[r], peak[r], ref
                )
                stats["projected_correct"] += correct
                stats["previously_hidden_correct"] += hidden and correct
                stats["hidden_with_peak"] += hidden
        report["layers"][str(order)] = {k: int(v) for k, v in stats.items()}
    (destination / "executed-picker.py").write_text(code, encoding="utf-8")
    (destination / "executed-diagnosis.py").write_bytes(Path(__file__).read_bytes())
    (destination / "diagnosis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    run(parser.parse_args().folder)
