"""Verify header times and render stored workflow paths without rerunning inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from gpr_layer_audit.export.audit import _radargram
from gpr_layer_audit.io.dzt import DZTFile, fingerprint_file
from gpr_layer_audit.models import (
    AcquisitionFileSet,
    AnalysisResult,
    CalibrationDiagnostics,
    InterfacePick,
    PickStatus,
    VisibilityState,
)
from gpr_layer_audit.time_coordinates import (
    image_time_bounds_ns,
    measurement_zero_sample,
    sample_from_time_ns,
    sample_time_ns,
    sample_twtt_ns,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="mandiali-short")
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    case = json.loads((root / "benchmarks/seeded-evaluation-inputs.json").read_text())["cases"][
        args.case
    ]
    signature_path = args.workflow / "gui-initial-three-native-seeds-per-layer-pick-signature.json"
    path_file = args.workflow / "gui-initial-three-native-seeds-per-layer-paths.npz"
    signature = json.loads(signature_path.read_text())
    road = DZTFile(root / case["dzt"])
    assert fingerprint_file(road.path) == case["dzt_sha256"]
    radar = np.asarray(road.channel()[:: case["stride"]])
    result = AnalysisResult(
        AcquisitionFileSet(road.path),
        road.header,
        1,
        np.arange(len(radar)) * case["native_dx_m"] * case["stride"],
        radar,
        np.full(len(radar), round(-road.header.position_ns / road.header.sample_interval_ns)),
        round(-road.header.position_ns / road.header.sample_interval_ns),
        [],
        [],
        [],
        CalibrationDiagnostics(False),
        parameters={"input_mode": "processed", "processed_stride": case["stride"]},
    )
    for (
        order,
        trace,
        chainage,
        display,
        canonical,
        sample,
        twtt,
        status,
        family,
        lobe,
        vis,
    ) in signature:
        result.picks.append(
            InterfacePick(
                order,
                {1: "Asphalt", 2: "Base", 3: "Subbase"}[order],
                trace,
                chainage,
                sample,
                twtt,
                float("nan"),
                1.0,
                PickStatus(status),
                visibility=VisibilityState(vis),
                selected_lobe_sample=display,
                canonical_event_sample=canonical,
                event_family_id=family,
                selected_lobe=lobe,
            )
        )
    assert not args.output.exists(), (
        "Use a new output directory; historical artifacts stay immutable"
    )
    args.output.mkdir(parents=True)
    old_audit_path = args.workflow / "source/src/gpr_layer_audit/export/audit.py"
    spec = importlib.util.spec_from_file_location("recorded_audit", old_audit_path)
    old_audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_audit)
    title = "Stored processed radargram: time-origin verification"
    old_audit._radargram(result, args.output / "before-header-time-fix.png", title=title)
    accepted = [pick for pick in result.picks if pick.is_accepted_measurement]
    old_twtt = np.array([pick.twtt_ns for pick in accepted])
    for pick in accepted:
        pick.twtt_ns = sample_twtt_ns(result, pick.sample_index)
    difference = np.array([pick.twtt_ns for pick in accepted]) - old_twtt
    assert np.allclose(difference, -0.01171875)
    assert all(np.isnan(pick.twtt_ns) for pick in result.picks if pick not in accepted)
    _radargram(result, args.output / "after-header-time-fix.png", title=title)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication

    from gpr_layer_audit.ui.radar_view import RadarView

    app = QApplication.instance() or QApplication([])
    view = RadarView()
    view.set_result(result)
    requests = []
    view.anchorRequested.connect(lambda *values: requests.append(values))
    seeds = json.loads((root / case["seed_source"]).read_text())["observations"]
    observations = []
    for order, layer_seeds in seeds.items():
        view.set_active_layer(int(order))
        for seed in layer_seeds:
            time = sample_time_ns(result, seed["sample"])
            assert abs(time - seed["time_ns"]) <= 5.1e-8
            assert sample_from_time_ns(result, time) == seed["sample"]
            view._position = lambda _, s=seed, t=time: (s["trace"] * case["native_dx_m"], t)
            view._mouse_clicked(SimpleEvent())
            assert requests[-1][2] == seed["sample"]
            assert view.image.mapToParent(QPointF(0.5, seed["sample"] + 0.5)).y() == time
            assert np.isclose(
                view.image.mapToParent(
                    QPointF(seed["trace"] / case["stride"] + 0.5, seed["sample"] + 0.5)
                ).x(),
                seed["trace"] * case["native_dx_m"],
            )
            observations.append({"layer": int(order), "sample": seed["sample"], "time_ns": time})
    view.close()
    app.processEvents()
    report = {
        "claim": "Numerical/display coordinate repair only; no inference or scoring rerun",
        "case": args.case,
        "dzt_sha256": case["dzt_sha256"],
        "recorded_signature_sha256": digest(signature_path),
        "recorded_paths_sha256": digest(path_file),
        "source_hashes": {
            str(p.relative_to(root)): digest(p)
            for p in (
                Path(__file__).resolve(),
                root / "src/gpr_layer_audit/time_coordinates.py",
                root / "src/gpr_layer_audit/ui/radar_view.py",
                root / "src/gpr_layer_audit/export/audit.py",
                root / "src/gpr_layer_audit/processing/processed_tracking.py",
                root / "src/gpr_layer_audit/processing/pipeline.py",
            )
        },
        "graph_surface_sample": result.reference_surface_sample,
        "measurement_zero_sample": measurement_zero_sample(result),
        "image_time_edges_ns": image_time_bounds_ns(result),
        "accepted_count_by_layer": {
            str(order): sum(p.layer_order == order for p in accepted) for order in (1, 2, 3)
        },
        "accepted_twtt_change_ns": [float(np.min(difference)), float(np.max(difference))],
        "seed_click_pixel_centre_roundtrips": observations,
        "tracking_samples_statuses_and_lobes_changed": False,
        "limitations": (
            "The stored initial workflow remains pre-fix; this is not a full post-fix retrack."
        ),
    }
    (args.output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


class SimpleEvent:
    def button(self):
        from PySide6.QtCore import Qt

        return Qt.MouseButton.LeftButton

    def modifiers(self):
        from PySide6.QtCore import Qt

        return Qt.KeyboardModifier.ControlModifier

    def scenePos(self):
        from PySide6.QtCore import QPointF

        return QPointF()


if __name__ == "__main__":
    main()
