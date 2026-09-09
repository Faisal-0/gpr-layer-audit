"""Execute the actual processed GUI worker, CLI, corrections and saved-project replay.

The source is copied before imports. Only explicit native operating seeds enter
the initial runs. A reference is opened only after a radar-only query has been
recorded, and only its exact answer becomes a new operating observation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, document):
    path.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")


def _signature(result):
    """Include displayed and canonical samples, gates and selected identities."""
    return json.dumps(
        [
            (
                p.layer_order,
                p.trace_index,
                p.chainage_m,
                p.selected_lobe_sample,
                p.canonical_event_sample,
                p.sample_index,
                p.twtt_ns,
                str(p.status),
                p.event_family_id,
                p.selected_lobe,
                str(p.visibility),
            )
            for p in result.picks
        ],
        sort_keys=True,
    )


def _worker_result(worker):
    received, errors, cancelled = [], [], []
    worker.signals.result.connect(received.append)
    worker.signals.error.connect(errors.append)
    worker.signals.cancelled.connect(lambda: cancelled.append(True))
    worker.run()
    if errors:
        raise RuntimeError(errors[0])
    if cancelled or len(received) != 1:
        raise RuntimeError(f"Unexpected worker completion: {len(received)=}, {cancelled=}")
    return received[0]


def render_verification(output, workspace):
    """Render frozen scoring output; no reference is used to alter inference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = _read(output / "verification.json")
    case = _read(output / "input-case.json")
    metric_key = "application_layers" if "application_layers" in report["metrics"][0] else "layers"
    acceptance = (
        "application accepted observations"
        if metric_key == "application_layers"
        else "tracker path acceptance"
    )
    road_m = (case["dimensions"][0] - 1) * case["native_dx_m"]
    # Unanswerable requests still consume budget and repeat the preceding state.
    state_for_request = [0]
    completed = 0
    for action in report["actions"]:
        completed += int(action.get("outcome") in {"confirmed", "corrected"})
        state_for_request.append(completed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, order, name in zip(axes, (2, 3), ("Base", "Subbase"), strict=True):
        values = [report["metrics"][state][metric_key][str(order)] for state in state_for_request]
        initial_denominator = values[0]["observations_excluding_seeds"]
        x = np.arange(len(values))
        correct = [100 * value["accepted_agree"] / initial_denominator for value in values]
        incorrect = [
            100 * (value["accepted"] - value["accepted_agree"]) / initial_denominator
            for value in values
        ]
        axis.plot(x, correct, "o-", color="#12766e", label="Correct automatically accepted")
        axis.plot(x, incorrect, "x--", color="#bc3434", label="Incorrect automatically accepted")
        for step, value, height in zip(x, values, correct, strict=True):
            axis.annotate(
                f"{value['accepted_agree']}/{value['accepted']}",
                (step, height),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )
        axis.set_title(f"{name}: initial reviewed denominator {initial_denominator}")
        axis.set_xlabel("Additional requested observations, across enabled layers")
        axis.set_xticks(x)
        axis.set_ylim(0, 105)
        axis.grid(alpha=0.2)
        top = axis.secondary_xaxis(
            "top", functions=(lambda v: v * 1000 / road_m, lambda v: v * road_m / 1000)
        )
        top.set_xlabel("Additional observations per kilometre")
    axes[0].set_ylabel("Coverage of initial reviewed observation pool (%)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{report['case']} · three initial seeds per interface · development", y=1.07)
    fig.text(
        0.5,
        -0.015,
        f"{acceptance}; support clicks excluded; annotations = agreeing/accepted. "
        "Unknown labels are unscored.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output / "coverage-versus-intervention.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    sys.path.insert(0, str(output / "source/src"))
    from gpr_layer_audit.io.dzt import DZTFile

    radar = DZTFile(workspace / case["dzt"])
    measurement = np.asarray(radar.channel(), np.float32)
    native = _read(output / "initial-native-seeds.json")["observations"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    final_index = len(report["metrics"]) - 1
    for column, stage in enumerate((0, final_index)):
        archive = np.load(output / f"state-{stage}.npz")
        stage_metrics = report["metrics"][stage][metric_key]
        signature_path = output / "gui-initial-three-native-seeds-per-layer-pick-signature.json"
        if stage:
            signatures = list(output.glob("gui-correction-*-pick-signature.json"))
            if signatures:
                signature_path = max(signatures, key=lambda p: int(p.name.split("-")[2]))
        signature = _read(signature_path) if signature_path.exists() else None
        for row, order in enumerate((2, 3)):
            axis = axes[row, column]
            supplied = list(native[str(order)])
            if stage:
                for answer in sorted(output.glob("answer-*.json")):
                    supplied.extend(_read(answer)["observations"].get(str(order), []))
            seed_samples = [point["sample"] for point in native[str(order)]]
            lo, hi = (
                max(0, min(seed_samples) - 70),
                min(measurement.shape[1] - 1, max(seed_samples) + 70),
            )
            bound = max(float(np.percentile(abs(measurement[:, lo : hi + 1]), 97)), 1)
            view = np.arcsinh(measurement[:, lo : hi + 1] / (bound / 5))
            axis.imshow(
                view.T,
                cmap="gray",
                aspect="auto",
                interpolation="nearest",
                extent=(
                    -case["native_dx_m"] / 2,
                    road_m + case["native_dx_m"] / 2,
                    radar.header.position_ns + (hi + 0.5) * radar.header.sample_interval_ns,
                    radar.header.position_ns + (lo - 0.5) * radar.header.sample_interval_ns,
                ),
                vmin=-np.arcsinh(5),
                vmax=np.arcsinh(5),
            )
            points = stage_metrics[str(order)]["evaluation_observations"]
            provisional = archive[f"layer{order}_provisional_samples"]
            accepted = archive[f"layer{order}_samples"] >= 0
            if signature is not None:
                accepted = np.zeros(len(measurement), bool)
                for item in signature:
                    if (
                        item[0] == order
                        and item[7] in {"accepted", "high_confidence"}
                        and np.isfinite(item[6])
                    ):
                        accepted[item[1]] = True
            unresolved = np.flatnonzero((provisional >= 0) & ~accepted)
            axis.scatter(
                unresolved * case["native_dx_m"],
                radar.header.position_ns
                + provisional[unresolved] * radar.header.sample_interval_ns,
                s=4,
                color="#8e97a4",
                label="Unresolved proposal",
            )
            for correct, colour, label in (
                (True, "#04b79c", "Accepted agreement"),
                (False, "#ed443a", "Accepted disagreement"),
            ):
                subset = [p for p in points if p["accepted"] and p["accepted_correct"] == correct]
                axis.scatter(
                    [p["row"] * case["native_dx_m"] for p in subset],
                    [
                        radar.header.position_ns
                        + archive[f"layer{order}_samples"][p["row"]]
                        * radar.header.sample_interval_ns
                        for p in subset
                    ],
                    s=9,
                    color=colour,
                    label=label,
                )
            known = {p["row"] for p in points} | {p["trace"] for p in supplied}
            unknown = [r for r in np.flatnonzero(accepted) if r not in known]
            axis.scatter(
                np.asarray(unknown) * case["native_dx_m"],
                radar.header.position_ns
                + archive[f"layer{order}_samples"][unknown] * radar.header.sample_interval_ns,
                s=8,
                color="#478cd4",
                label="Accepted, unscored",
            )
            axis.scatter(
                [p["trace"] * case["native_dx_m"] for p in supplied],
                [
                    radar.header.position_ns + p["sample"] * radar.header.sample_interval_ns
                    for p in supplied
                ],
                marker="^",
                s=40,
                color="#ffcf4c",
                edgecolor="black",
                linewidth=0.4,
                label="Supplied observation",
                zorder=8,
            )
            name = "Base" if order == 2 else "Subbase"
            axis.set_title(f"{name} · {'initial' if stage == 0 else f'after {stage} actions'}")
            axis.set_ylabel("Native processed time (ns)")
            axis.set_xlim(0, road_m)
            if row == 1:
                axis.set_xlabel("Stored chainage (m)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(f"{report['case']} · entire processed acquisition · reviewed interpretation")
    fig.text(
        0.5,
        0.045,
        "Display uses asinh amplitude gain; no signal-gap interpolation. "
        "Time and interpretation agreement do not establish physical thickness.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(output / "before-after-deep-radar.png", dpi=175)
    plt.close(fig)
    _write(
        output / "render-manifest.json",
        {
            "renderer_sha256": _sha(__file__),
            "verification_sha256": _sha(output / "verification.json"),
            "source_dzt_sha256": _sha(workspace / case["dzt"]),
            "acceptance_scope": acceptance,
            "curve_denominator": "fixed initial reviewed pool, excluding initial support",
            "curve_budget": (
                "all requested observations; GUI warning confirmations logged separately"
            ),
            "images": {
                name: _sha(output / name)
                for name in (
                    "coverage-versus-intervention.png",
                    "before-after-deep-radar.png",
                )
            },
        },
    )


def verify(args):
    if args.output.exists():
        raise FileExistsError("Choose a fresh output directory; previous artifacts stay frozen")
    args.output.mkdir(parents=True)
    snapshot = args.output / "source" / "src"
    shutil.copytree(args.source, snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, args.output / Path(__file__).name)
    frozen_config = args.output / "configuration.json"
    shutil.copy2(args.config, frozen_config)
    manifest = _read(args.workspace / "benchmarks/seeded-evaluation-inputs.json")
    case = manifest["cases"][args.case]
    _write(args.output / "input-case.json", case)
    source_hashes = {str(p.relative_to(snapshot)): _sha(p) for p in snapshot.rglob("*.py")}
    _write(args.output / "source-hashes.json", source_hashes)
    frozen_scoring = (
        args.workspace / manifest["baseline_source"] / "gpr_layer_audit/conventional.py"
    )
    scoring_copy = snapshot / "gpr_layer_audit/conventional.py"
    if frozen_scoring.read_bytes().replace(b"\r\n", b"\n") != scoring_copy.read_bytes().replace(
        b"\r\n", b"\n"
    ):
        raise ValueError("Scoring code differs from frozen baseline")
    sys.path.insert(0, str(snapshot))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from gpr_layer_audit import cli
    from gpr_layer_audit.conventional import backend_fingerprint, layer_metrics, peak_process_memory
    from gpr_layer_audit.export import export_audit_package
    from gpr_layer_audit.io.dzt import fingerprint_file
    from gpr_layer_audit.io.dzx import read_dzx
    from gpr_layer_audit.models import AcquisitionFileSet, LayerSpec, PickStatus, SeedRequest
    from gpr_layer_audit.native_seed_io import load_native_observations
    from gpr_layer_audit.processing.active_queries import request_observation
    from gpr_layer_audit.processing.conventional_config import ConventionalConfig, resolve_pulse
    from gpr_layer_audit.processing.pipeline import AnalysisOptions
    from gpr_layer_audit.processing.processed_tracking import native_anchors
    from gpr_layer_audit.project import ProjectStore
    from gpr_layer_audit.ui import main_window as ui

    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    source = AcquisitionFileSet(args.workspace / case["dzt"])
    for name, digest in (
        ("dzt", "dzt_sha256"),
        ("dzx", "dzx_sha256"),
        ("seed_source", "seed_sha256"),
    ):
        if fingerprint_file(args.workspace / case[name]) != case[digest]:
            raise ValueError(f"Frozen input changed: {name}")
    frozen_seeds = args.output / "initial-native-seeds.json"
    shutil.copy2(args.workspace / case["seed_source"], frozen_seeds)
    options = AnalysisOptions(
        input_mode="processed",
        processed_stride=1,
        stack_size=1,
        tracker_method="seed_hybrid",
        ml_policy="off",
        report_interval_m=1.0,
        conventional_config=_read(frozen_config),
        layer_specs=LayerSpec.defaults(),
        seed_stations=load_native_observations(frozen_seeds, source.dzt_path),
    )
    log = {
        "schema": "processed-workflow-verification-v1",
        "case": args.case,
        "claim": "processed interpretation development; no physical thickness claim",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.workspace, text=True
        ).strip(),
        "source_snapshot": str(snapshot.resolve()),
        "source_hashes": source_hashes,
        "frozen_scoring_sha256": _sha(frozen_scoring),
        "scoring_equivalence": "byte identical after CRLF normalization",
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "PySide6", "pyqtgraph")
        },
        "backend_sha256": backend_fingerprint(),
        "script_sha256": _sha(__file__),
        "input_sha256": case["dzt_sha256"],
        "reference_sha256": case["dzx_sha256"],
        "initial_seed_sha256": case["seed_sha256"],
        "config_sha256": _sha(frozen_config),
        "configuration": options.conventional_config,
        "query_layer_orders": options.query_layer_orders,
        "gui_execution": (
            "actual AnalysisWorker, requested-seed control, add_seed_pick, local correction "
            "dispatch, ProcessedCorrectionWorker and completion slot; synchronous scheduling"
        ),
        "initial_operating_input": "DZT and native seeds; DZX opened only after query fixed",
        "checks": {},
        "runs": [],
        "actions": [],
        "limitations": [
            "Mandiali short is 11.45 m; the 25 m GUI correction radius covers the entire input",
            "Separate mechanism tests check outside-window and outside-layer preservation",
            "Reference answers simulate observations; no claim of expert interaction speed",
        ],
    }

    def save():
        log["peak_process_memory_bytes"] = peak_process_memory()
        _write(args.output / "verification.json", log)

    def timed(label, function):
        print(f"START {label}", flush=True)
        start = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - start
        log["runs"].append({"operation": label, "runtime_s": elapsed})
        if hasattr(result, "processed_paths"):
            (args.output / f"{label}-pick-signature.json").write_text(
                _signature(result), encoding="utf-8"
            )
            np.savez_compressed(
                args.output / f"{label}-paths.npz",
                **{
                    f"layer{o}_{name}": getattr(path, name)
                    for o, path in result.processed_paths.items()
                    for name in ("samples", "visible", "provisional_samples")
                },
            )
        print(f"DONE {label} {elapsed:.3f}s", flush=True)
        save()
        return result

    try:
        gui = timed(
            "gui-initial-three-native-seeds-per-layer",
            lambda: _worker_result(ui.AnalysisWorker(source, None, options)),
        )
        captured = []

        def export_cli(result, output):
            captured.append(result)
            return export_audit_package(result, output)

        command = [
            "analyze",
            str(source.dzt_path),
            "--input-mode",
            "processed",
            "--stack",
            "1",
            "--method",
            "seed_hybrid",
            "--ml",
            "off",
            "--track-subbase",
            "--conventional-config",
            str(frozen_config),
            "--native-seeds",
            str(frozen_seeds),
            "--output",
            str(args.output / "cli-export"),
        ]
        log["cli_arguments"] = command
        with patch.object(cli, "export_audit_package", export_cli):
            timed("cli-initial-and-export", lambda: cli.main(command))
        log["checks"]["gui_cli_pick_identity_status_twtt_parity"] = _signature(gui) == _signature(
            captured[0]
        )
        anchors = native_anchors(options.seed_stations, gui.chainage_m)
        log["checks"]["exact_initial_seeds"] = all(
            gui.processed_paths[o].samples[row] == sample
            for o, points in anchors.items()
            for row, sample in points.items()
        )
        initial_anchors = deepcopy(anchors)
        pulse = {
            o: resolve_pulse(
                gui.interpretation_input_radargram,
                gui.sample_validity,
                points,
                {},
                gui.header.sample_interval_ns,
                ConventionalConfig(),
            ).lobe_samples
            for o, points in initial_anchors.items()
        }
        log["initial_pulse_samples"] = pulse
        references = None
        states = [deepcopy(gui)]
        excluded = [deepcopy(anchors)]
        visited = set()
        current = gui
        store = ProjectStore.create(args.output / "workflow.gprproj", "processed-workflow")
        store.set_layers(options.layer_specs)
        for station in options.seed_stations:
            store.save_seed_station(station)
        store.save_analysis(current)
        interaction_window = ui.MainWindow()
        interaction_window.road = source
        interaction_window.options = options
        interaction_window.project_store = store
        for action_index in range(args.actions):
            anchors = native_anchors(options.seed_stations, current.chainage_m)
            query_paths = {
                o: path
                for o, path in current.processed_paths.items()
                if o in options.query_layer_orders
            }
            query = request_observation(
                query_paths,
                current.interpretation_input_radargram,
                current.sample_validity,
                anchors,
                current.header.distance_per_trace_m,
                visited=visited,
            )
            if query is None:
                break
            action = {
                "index": action_index + 1,
                "query": query,
                "selection_reads_hidden_answer": False,
            }
            log["actions"].append(action)
            save()  # Freeze the requested coordinate before opening any reference.
            if references is None:
                references = {
                    case["reference_label_mapping"].get(str(layer.number), layer.number + 1): [
                        p for p in layer.picks if p.channel == 0
                    ]
                    for layer in read_dzx(args.workspace / case["dzx"]).layers
                }
            order, row = query["layer_order"], query["row"]
            visited.add((order, row))
            matching = [p for p in references[order] if p.trace == row]
            if len(matching) != 1:
                action["outcome"] = "no_unique_reviewed_answer_at_exact_requested_coordinate"
                save()
                continue
            point = matching[0]
            answer = {
                "schema": "conventional-native-seeds-v1",
                "mode": "processed",
                "dzt_sha256": case["dzt_sha256"],
                "observations": {
                    str(order): [{"trace": row, "sample": int(point.sample), "channel": 0}]
                },
            }
            answer_path = args.output / f"answer-{action_index + 1}.json"
            _write(answer_path, answer)
            previous = current
            answer_chainage = float(row * current.header.distance_per_trace_m)
            unchanged_layer_signatures = {
                o: p.samples.copy() for o, p in previous.processed_paths.items() if o != order
            }

            def answer_in_gui(
                previous=previous,
                order=order,
                row=row,
                point=point,
                query=query,
                action=action,
            ):
                interaction_window.result = previous
                previous.proposed_seed_requests = [
                    SeedRequest(
                        float(previous.chainage_m[row]),
                        [order],
                        query["reason"],
                        query["priority"],
                    )
                ]
                previous.proposed_seed_chainages = [float(previous.chainage_m[row])]
                interaction_window._populate_seed_controls()
                interaction_window.seed_combo.setCurrentIndex(
                    interaction_window.seed_combo.findData(float(previous.chainage_m[row]))
                )
                received = []
                scheduler = SimpleNamespace(
                    start=lambda worker: received.append(_worker_result(worker))
                )
                with (
                    patch.object(interaction_window, "thread_pool", scheduler),
                    patch.object(
                        ui.QMessageBox,
                        "question",
                        return_value=ui.QMessageBox.StandardButton.Save,
                    ) as confirmation,
                ):
                    interaction_window.add_seed_pick(
                        order, float(previous.chainage_m[row]), point.sample
                    )
                action["unusual_seed_confirmation_count"] = confirmation.call_count
                if len(received) != 1:
                    raise RuntimeError("Requested GUI answer did not dispatch one local retracker")
                return received[0]

            current = timed(
                f"gui-correction-{action_index + 1}",
                answer_in_gui,
            )
            station = next(s for s in options.seed_stations if s.chainage_m == answer_chainage)
            action["saved_station_role"] = station.role
            action["outcome"] = (
                "confirmed"
                if previous.processed_paths[order].provisional_samples[row] == point.sample
                else "corrected"
            )
            action["revealed_sample"] = int(point.sample)
            action["exact_answer_preserved"] = int(
                current.processed_paths[order].samples[row]
            ) == int(point.sample)
            action["unrelated_layers_preserved"] = all(
                np.array_equal(values, current.processed_paths[o].samples)
                for o, values in unchanged_layer_signatures.items()
            )
            states.append(deepcopy(current))
            excluded.append(native_anchors(options.seed_stations, current.chainage_m))
            save()

        interaction_window.close()
        for station in options.seed_stations:
            store.save_seed_station(station)
        store.save_analysis(current)
        window = ui.MainWindow()
        try:
            with patch.object(
                ui.QFileDialog, "getOpenFileName", return_value=(str(store.path), "Project")
            ):
                window.open_project()
            replayed = timed(
                "gui-save-reopen-reconstruct",
                lambda: _worker_result(
                    ui.AnalysisWorker(window.road, window.plate, window.options)
                ),
            )
            log["checks"]["saved_correction_order_restored"] = (
                window.options.local_correction_order == options.local_correction_order
            )
            log["checks"]["query_layer_scope_restored"] = (
                window.options.query_layer_orders == options.query_layer_orders
            )
        finally:
            window.close()
        log["checks"]["save_reopen_pick_identity_status_twtt_parity"] = _signature(
            current
        ) == _signature(replayed)
        log["checks"]["unresolved_twtt_withheld"] = all(
            not np.isfinite(p.twtt_ns)
            for result in states
            for p in result.picks
            if p.status not in {PickStatus.ACCEPTED, PickStatus.HIGH_CONFIDENCE}
        )
        log["checks"]["no_uncalibrated_physical_thickness"] = all(
            item.thickness_mm is None for item in current.thickness
        )
        # The evaluator is invoked after all operating calls; seeds are excluded per state.
        if references is None:
            references = {
                case["reference_label_mapping"].get(str(layer.number), layer.number + 1): [
                    p for p in layer.picks if p.channel == 0
                ]
                for layer in read_dzx(args.workspace / case["dzx"]).layers
            }
        log["metrics"] = []
        for index, (state, used) in enumerate(zip(states, excluded, strict=True)):
            metrics = {
                str(o): layer_metrics(
                    path,
                    references[o],
                    used[o],
                    state.interpretation_input_radargram,
                    pulse[o],
                    state.header.distance_per_trace_m,
                )
                for o, path in state.processed_paths.items()
            }
            application_paths = deepcopy(state.processed_paths)
            for order, path in application_paths.items():
                accepted = {
                    p.trace_index: p.is_accepted_measurement
                    for p in state.picks
                    if p.layer_order == order
                }
                use = np.array([accepted[row] for row in range(len(path.samples))])
                path.samples[~use] = -1
                path.visible[~use] = False
            application_metrics = {
                str(o): layer_metrics(
                    path,
                    references[o],
                    used[o],
                    state.interpretation_input_radargram,
                    pulse[o],
                    state.header.distance_per_trace_m,
                )
                for o, path in application_paths.items()
            }
            log["metrics"].append(
                {
                    "step": index,
                    "layers": metrics,
                    "application_layers": application_metrics,
                    "acceptance_contract": (
                        "layers=tracker path; application_layers=finite accepted picks"
                    ),
                }
            )
            np.savez_compressed(
                args.output / f"state-{index}.npz",
                **{
                    f"layer{o}_{name}": getattr(path, name)
                    for o, path in state.processed_paths.items()
                    for name in ("samples", "provisional_samples", "visible")
                },
            )
        timed(
            "corrected-export",
            lambda: export_audit_package(current, args.output / "corrected-export"),
        )
        log["checks"]["all_answer_coordinates_preserved"] = all(
            a.get("exact_answer_preserved", True) for a in log["actions"]
        )
        log["checks"]["gui_answers_use_local_correction_role"] = all(
            a.get("saved_station_role", "correction") == "correction" for a in log["actions"]
        )
        log["review_events"] = store.review_events()
        log["checks"]["default_requests_target_deep_interfaces"] = all(
            a["query"]["layer_order"] in options.query_layer_orders for a in log["actions"]
        )
        log["checks"]["operating_source_unchanged"] = (
            fingerprint_file(source.dzt_path) == case["dzt_sha256"]
        )
        log["success"] = all(log["checks"].values())
        save()
        print(
            json.dumps({"checks": log["checks"], "success": log["success"]}, indent=2), flush=True
        )
        return 0 if log["success"] else 1
    except Exception:
        log["exception"] = traceback.format_exc()
        log["success"] = False
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="mandiali-short")
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, default=ROOT / "src")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "benchmarks/conventional-motion-calibrated-development.json",
    )
    parser.add_argument("--actions", type=int, default=2)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.render_only:
        render_verification(args.output, args.workspace)
        return 0
    return verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
