"""Frozen processed-input real-road comparisons; references are scorer-only.

Run ``python scripts/seeded_eval.py freeze`` once, then ``run --case ...``.
Every run imports an explicit source snapshot in a fresh process. No threshold
selection, reference interpolation or raw/processed registration occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/seeded-evaluation-inputs.json"
BASELINE = ROOT / "exports/seeded-tracker/baseline-source/src"
# Verified against Git blobs at c774cf97572e889fa6cbff690a490e9ae4d2e951.
# An editable baseline directory cannot establish its own scoring integrity.
FROZEN_HELPER_LF_SHA256 = {
    "conventional.py": "fc5b2093d7a1efe8abd549e00ea52b290040237ea519225ab20b8fde8f2a307b",
    "conventional_reference.py": "9a13370b41b3dddd92540ecd47d758b976356fffd4746b3069d99846ab3fd8a0",
    "conventional_seeds.py": "4ba68eaf749189dc23ce779462f0682502dbffbe90dd04504473a462c9bb6813",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(source):
    """Hash the whole Python package, including non-processing coordinate contracts."""
    source = Path(source).resolve()
    files = {
        p.relative_to(source).as_posix(): sha(p)
        for p in sorted((source / "gpr_layer_audit").rglob("*.py"))
    }
    if not files:
        raise ValueError(f"No package source found: {source}")
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"source": str(source), "sha256": digest, "python_files": files}


def runtime_note(path):
    """Surface recorded suspension intervals without adjusting measured runtimes."""
    note_path = Path(path).parent / "runtime-environment-note.json"
    if note_path.exists():
        note = read(note_path)
        if Path(path).stem in note.get("affected_running_jobs", []):
            return {"artifact": str(note_path), "sha256": sha(note_path), **note}
    return None


def validate_frozen_helpers(source, *, baseline=BASELINE):
    """Allow Git's Windows line endings, while rejecting all other scorer edits."""
    helpers = {}
    for module, expected in FROZEN_HELPER_LF_SHA256.items():
        current_path = Path(source) / "gpr_layer_audit" / module
        frozen_path = Path(baseline) / "gpr_layer_audit" / module
        current, frozen = current_path.read_bytes(), frozen_path.read_bytes()
        canonical_current = current.replace(b"\r\n", b"\n")
        canonical_frozen = frozen.replace(b"\r\n", b"\n")
        if hashlib.sha256(canonical_frozen).hexdigest() != expected:
            raise ValueError(f"Frozen baseline helper differs from pinned c774cf9: {module}")
        if canonical_current != canonical_frozen:
            raise ValueError(f"Frozen evaluation helper changed: {module}")
        helpers[module] = {
            "source_sha256": sha(current_path),
            "frozen_sha256": sha(frozen_path),
            "normalized_lf_sha256": hashlib.sha256(canonical_frozen).hexdigest(),
            "byte_identical": current == frozen,
            "only_permitted_normalization": "CRLF to LF; no code or whitespace edits",
        }
    return helpers


def freeze():
    sys.path.insert(0, str(BASELINE))
    from gpr_layer_audit.conventional import backend_fingerprint
    from gpr_layer_audit.conventional_reference import audit_reference, road_partition
    from gpr_layer_audit.conventional_seeds import load_support
    from gpr_layer_audit.io.dzx import read_dzx

    if MANIFEST.exists():
        raise ValueError("Manifest exists; preserve it and use the frozen inputs")
    cases = {}
    for name, prior_dir, stride in (
        ("mandiali-short", "continuation-calibrated", 1),
        ("gujrat-second", "continuation-gujrat-fixed-seeds-0p4", 4),
    ):
        prior_path = next((ROOT / "exports/conventional" / prior_dir / "repaired").glob("*.json"))
        prior = read(prior_path)
        source = Path(prior["audit"]["dzt_path"])
        audit = audit_reference(source.with_suffix(".DZX"))
        mapping = prior["reference_label_mapping"]
        anchors, _ = load_support(prior_path, audit, mapping, stride, "processed")
        observations = {}
        for group in audit["metadata"]["layers"]:
            order = mapping.get(str(group["number"]), group["number"] + 1)
            if order in anchors:
                observations[str(order)] = [
                    p for p in group["picks"] if p["channel"] == 0 and p["trace"] in anchors[order]
                ]
        support = {
            "schema": "conventional-native-seeds-v1",
            "mode": "processed",
            "dzx_sha256": audit["metadata"]["source_sha256"],
            "dzt_sha256": audit["dzt_sha256"],
            "reference_label_mapping": mapping,
            "observations": observations,
        }
        seeds = ROOT / "benchmarks" / f"seeded-{name}-seeds.json"
        write(seeds, support)
        cases[name] = {
            "dzt": str(source.relative_to(ROOT)),
            "dzx": str(source.with_suffix(".DZX").relative_to(ROOT)),
            "dzt_sha256": audit["dzt_sha256"],
            "dzx_sha256": audit["metadata"]["source_sha256"],
            "seed_source": str(seeds.relative_to(ROOT)),
            "seed_sha256": sha(seeds),
            "seed_lineage": str(prior_path.relative_to(ROOT)),
            "seed_lineage_sha256": sha(prior_path),
            "physical_road_group": audit["physical_road_group"],
            "usage": "previously used development road",
            "dimensions": audit["dimensions"],
            "dt_ns": audit["dt_ns"],
            "native_dx_m": audit["dx_m"],
            "header_time_origin_ns": audit["header_time_origin_ns"],
            "stride": stride,
            "orientation": (
                "unchanged stored trace order; physical travel direction not independently verified"
            ),
            "reference_label_mapping": mapping,
            "layer_mapping_status": (
                "historical RADAN number-to-interface convention; "
                "semantic field confirmation not independent"
            ),
            "layer_audit": audit["layers"],
            "issues": audit["issues"],
        }
    inventory = []
    for path in sorted((ROOT / "gpr including proc files").rglob("*.DZX")):
        metadata = read_dzx(path)
        layers = [
            {"number": layer.number, "name": layer.name, "observations": len(layer.picks)}
            for layer in metadata.layers
            if layer.picks
        ]
        if layers:
            inventory.append(
                {
                    "dzx": str(path.relative_to(ROOT)),
                    "sha256": sha(path),
                    "paired_dzt_exists": path.with_suffix(".DZT").exists(),
                    "physical_road_group": road_partition(path)[0],
                    "historical_partition": road_partition(path)[1],
                    "layers": layers,
                }
            )
    write(
        MANIFEST,
        {
            "schema": "seeded-evaluation-inputs-v1",
            "source_commit": "c774cf97572e889fa6cbff690a490e9ae4d2e951",
            "baseline_source": str(BASELINE.relative_to(ROOT)),
            "baseline_backend_sha256": backend_fingerprint(),
            "frozen_scoring_sha256": sha(BASELINE / "gpr_layer_audit/conventional.py"),
            "cases": cases,
            "reviewed_inventory": inventory,
            "coordinate_contract": (
                "processed DZT stored zero-based trace/sample; working row = native trace/stride, "
                "exact divisibility required; no resampling, stacking or reference-fitted alignment"
            ),
            "mask_contract": (
                "processed_boundary_mask; invalid numerical padding cannot be a measurement"
            ),
            "scoring_contract": (
                "initial seed-only max(2 samples,lobe width/4); same signed lobe without "
                "zero/sign crossing; initial and later supplied observations excluded; "
                "unknown labels ignored"
            ),
            "learning_audit": (
                "Processed reviewed interpretation labels exist. Raw mapping and per-label "
                "training permission/provenance are not established by this inventory; "
                "existing raw-coordinate U-Net loader must not silently consume processed labels."
            ),
            "independence": (
                "Mandiali acquisitions are one road, both Gujrat portions are one road. "
                "These cases are development only. Historical Daska/Pattoki use forbids "
                "untouched-road claims without lineage audit."
            ),
        },
    )
    print(MANIFEST)


def worker(args):
    wrapper_sha256 = sha(Path(__file__))
    # Production may change; the established evaluation/coordinate helpers may not.
    frozen_helpers = validate_frozen_helpers(args.source)
    sys.path.insert(0, str(args.source.resolve()))
    from gpr_layer_audit.conventional import evaluate_reference

    manifest = args.manifest.resolve()
    case = read(manifest)["cases"][args.case]
    for field, hash_field in (
        ("dzt", "dzt_sha256"),
        ("dzx", "dzx_sha256"),
        ("seed_source", "seed_sha256"),
    ):
        if sha(ROOT / case[field]) != case[hash_field]:
            raise ValueError(f"Frozen input changed: {field}")
    config = read(args.config) if args.config else None
    source_before = source_manifest(args.source)
    result = evaluate_reference(
        ROOT / case["dzx"],
        output=args.output,
        stride=args.stride or case["stride"],
        methods=[args.method],
        seed_source=ROOT / case["seed_source"],
        config=config,
    )
    if source_manifest(args.source) != source_before:
        raise RuntimeError("Source changed during evaluation; this run cannot support comparison")
    result["seeded_evaluation"] = {
        "manifest": str(manifest),
        "manifest_sha256": sha(manifest),
        "case": args.case,
        "seed_file_sha256": case["seed_sha256"],
        "config_file_sha256": sha(args.config) if args.config else None,
        "config_canonical_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "source_snapshot": source_before,
        "frozen_helpers": frozen_helpers,
        "evaluation_script_sha256": wrapper_sha256,
        "evaluation_script_fingerprint_timing": "before inference; loaded wrapper",
        "evaluation_script_unchanged_at_completion": wrapper_sha256 == sha(Path(__file__)),
        "python_version": sys.version,
    }
    write(args.output, result)
    print(
        json.dumps(
            {
                order: {
                    key: value[key]
                    for key in (
                        "observations_excluding_seeds",
                        "accepted",
                        "accepted_agree",
                        "reflector_switches",
                        "correct_coverage",
                    )
                }
                for order, value in result["methods"][args.method]["layers"].items()
            }
        ),
        flush=True,
    )


def overlay(paths, output, manifest=MANIFEST):
    """Scorer-only truth colors; no overlay arrays are inputs to tracking."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(BASELINE))
    from gpr_layer_audit.io.dzt import DZTFile

    rows = []
    for path in paths:
        run = read(path)
        if run.get("schema") == "seeded-interaction-replay-v1":
            case = read(manifest)["cases"][run["case"]]
            audit = {
                "dzt_path": str(ROOT / case["dzt"]),
                "dzt_sha256": run["input_sha256"],
                "dx_m": case["native_dx_m"],
                "dt_ns": run["dt_ns"],
                "header_time_origin_ns": case["header_time_origin_ns"],
            }
            selected = [run["steps"][0]]
            if run["steps"][-1]["step"] != 0:
                selected.append(run["steps"][-1])
            for step in selected:
                observations = {
                    order: {int(row): sample for row, sample in anchors.items()}
                    for order, anchors in run["initial_anchors"].items()
                }
                for action in run["actions"][: step["requests"]]:
                    if action.get("answer_sample") is not None:
                        observations[str(action["layer_order"])][action["row"]] = action[
                            "answer_sample"
                        ]
                adapted = {
                    "audit": audit,
                    "partition": [run["physical_road_group"], "development"],
                    "stride": run["stride"],
                    "seed_support": {
                        "observations": {
                            order: [
                                {"trace": row * run["stride"], "sample": sample}
                                for row, sample in anchors.items()
                            ]
                            for order, anchors in observations.items()
                        }
                    },
                }
                layers = {
                    order: {
                        **layer,
                        "correct_coverage": layer["accepted_agree"]
                        / run["steps"][0]["layers"][order]["observations_excluding_seeds"],
                    }
                    for order, layer in step["layers"].items()
                }
                rows.append(
                    (
                        path,
                        adapted,
                        run["method"],
                        {"layers": layers},
                        path.with_name(f"{path.stem}-step{step['step']}.npz"),
                        f"{path.stem}: {step['requests']} requests",
                    )
                )
        else:
            rows.extend(
                (
                    path,
                    run,
                    method,
                    result,
                    path.with_name(path.stem + "-" + method + ".npz"),
                    path.stem,
                )
                for method, result in run["methods"].items()
                if result.get("layers")
            )
    fingerprints = {run["audit"]["dzt_sha256"] for _, run, *_ in rows}
    if len(fingerprints) != 1:
        raise ValueError("Comparison overlay requires the same processed input")
    # A shared viewport includes every scored accepted error, even a large
    # reflector jump. Seed-centered windows must not conceal wrong picks.
    display_samples = {"2": [], "3": []}
    for _, run, _, result, arrays_path, _ in rows:
        with np.load(arrays_path) as arrays:
            for order in display_samples:
                if order not in result["layers"]:
                    continue
                reference_points = result["layers"][order]["evaluation_observations"]
                display_samples[order].extend(p["reference_sample"] for p in reference_points)
                display_samples[order].extend(
                    int(arrays[f"layer{order}_accepted"][p["row"]])
                    for p in reference_points
                    if p["accepted"]
                )
                display_samples[order].extend(
                    p["sample"] for p in run["seed_support"]["observations"][order]
                )
    fig, axes = plt.subplots(
        len(rows), 2, figsize=(16, 3.5 * len(rows)), squeeze=False, sharex=True
    )
    for axes_row, (_path, run, _method, result, arrays_path, title) in zip(axes, rows, strict=True):
        if sha(run["audit"]["dzt_path"]) != run["audit"]["dzt_sha256"]:
            raise ValueError("Overlay radar fingerprint differs from inference")
        radar = DZTFile(run["audit"]["dzt_path"]).channel()[:: run["stride"]]
        step = run["audit"]["dx_m"] * run["stride"]
        dt = run["audit"]["dt_ns"]
        origin = run["audit"]["header_time_origin_ns"]
        arrays = np.load(arrays_path)
        for ax, order in zip(axes_row, ("2", "3"), strict=True):
            layer = result["layers"].get(order)
            if layer is None:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "No reviewed observations for this interface", ha="center")
                continue
            refs = layer["evaluation_observations"]
            seed_obs = run["seed_support"]["observations"][order]
            lo = max(0, min(display_samples[order]) - 20)
            hi = min(radar.shape[1], max(display_samples[order]) + 21)
            limit = max(float(np.percentile(abs(radar[:, lo:hi]), 97)), 1e-9)
            ax.imshow(
                radar[:, lo:hi].T,
                cmap="gray",
                aspect="auto",
                vmin=-limit,
                vmax=limit,
                extent=(0, len(radar) * step, origin + hi * dt, origin + lo * dt),
            )
            proposal = arrays[f"layer{order}_provisional"]
            ax.plot(
                np.arange(len(radar)) * step,
                np.where(proposal >= 0, origin + proposal * dt, np.nan),
                color="#ffb000",
                lw=0.6,
                alpha=0.8,
                label="Proposal; unresolved allowed",
            )
            ax.scatter(
                [r["row"] * step for r in refs],
                [origin + r["reference_sample"] * dt for r in refs],
                color="#55cfff",
                s=2,
                label="Reviewed reference",
            )
            accepted = arrays[f"layer{order}_accepted"]
            unresolved = [r for r in refs if not r["accepted"]]
            ax.scatter(
                [r["row"] * step for r in unresolved],
                [origin + r["reference_sample"] * dt for r in unresolved],
                facecolors="none",
                edgecolors="#ffcf70",
                linewidths=0.45,
                s=8,
                label="Unresolved at reviewed location",
            )
            for correct, color, label in (
                (True, "#1bea5a", "Accepted agreeing"),
                (False, "#ff3159", "Accepted wrong"),
            ):
                chosen = [r for r in refs if r["accepted"] and r["accepted_correct"] == correct]
                ax.scatter(
                    [r["row"] * step for r in chosen],
                    [origin + accepted[r["row"]] * dt for r in chosen],
                    color=color,
                    s=7,
                    zorder=4,
                    label=label,
                )
            ax.scatter(
                [p["trace"] * run["audit"]["dx_m"] for p in seed_obs],
                [origin + p["sample"] * dt for p in seed_obs],
                marker="x",
                c="#ed53ff",
                s=48,
                linewidths=1.5,
                zorder=5,
                label="Supplied observations; excluded",
            )
            ax.set_ylim(origin + hi * dt, origin + lo * dt)
            layer_name = "Base" if order == "2" else "Subbase"
            if run.get("partition", [""])[0] == "bahawalpur":
                layer_name = f"Layer {order} ({layer_name.lower()} convention)"
            ax.set(
                title=(
                    f"{title} / {layer_name}: "
                    f"{layer['accepted_agree']}/{layer['accepted']} agreeing, "
                    f"{100 * layer['correct_coverage']:.1f}% correct coverage"
                ),
                xlabel="Stored road distance (m)",
                ylabel="Header-relative time (ns)",
            )
            if ax is axes[0, 0]:
                ax.legend(loc="upper left", fontsize=7, ncol=2)
        arrays.close()
    fig.suptitle(
        "Processed radar interpretation references; development roads; unknown labels unscored",
        fontsize=13,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


def summarize(paths, output):
    import numpy as np

    sys.path.insert(0, str(BASELINE))
    from gpr_layer_audit.conventional import _same_lobe
    from gpr_layer_audit.io.dzt import DZTFile

    reports = []
    for path in paths:
        run = read(path)
        if sha(run["audit"]["dzt_path"]) != run["audit"]["dzt_sha256"]:
            raise ValueError("Processed radar fingerprint differs from the scored run")
        radar = DZTFile(run["audit"]["dzt_path"]).channel()[:: run["stride"]]
        step = run["audit"]["dx_m"] * run["stride"]
        for method, result in run["methods"].items():
            if not result.get("layers"):
                # A failed tracker does not erase eligible observations from the denominator.
                for group in run["audit"]["metadata"]["layers"]:
                    order = run["reference_label_mapping"].get(
                        str(group["number"]), group["number"] + 1
                    )
                    support = run.get("seed_support", {}).get("observations", {}).get(str(order))
                    if support is None:
                        continue
                    seed_rows = {p["trace"] for p in support}
                    n = sum(
                        p["channel"] == 0
                        and p["trace"] % run["stride"] == 0
                        and p["trace"] not in seed_rows
                        for p in group["picks"]
                    )
                    reports.append(
                        {
                            "artifact": str(path.relative_to(ROOT)),
                            "method": method,
                            "failure": result,
                            "layer": order,
                            "segment": "whole",
                            "scored_observations": n,
                            "accepted": 0,
                            "accepted_agree": 0,
                            "accepted_wrong": 0,
                            "correct_coverage": 0.0 if n else None,
                            "incorrect_coverage": 0.0 if n else None,
                            "unresolved_coverage": 1.0 if n else None,
                            "scored_observation_footprint_m": n * step,
                            "physical_road_group": run["partition"][0],
                        }
                    )
                continue
            arrays = np.load(path.with_name(path.stem + "-" + method + ".npz"))
            for order, layer in result["layers"].items():
                samples = arrays[f"layer{order}_accepted"]
                all_rows = layer["evaluation_observations"]
                first, last = min(layer["seed_rows"]), max(layer["seed_rows"])
                for segment, rows in (
                    ("whole", all_rows),
                    ("bracketed", [r for r in all_rows if first < r["row"] < last]),
                    ("left_tail", [r for r in all_rows if r["row"] < first]),
                    ("right_tail", [r for r in all_rows if r["row"] > last]),
                ):
                    n, accepted = len(rows), [r for r in rows if r["accepted"]]
                    correct = sum(r["accepted_correct"] for r in rows)
                    incorrect = len(accepted) - correct
                    wrong_lobe = [
                        r
                        for r in accepted
                        if not _same_lobe(radar[r["row"]], samples[r["row"]], r["reference_sample"])
                    ]
                    wrong_lobe_rows = sorted(r["row"] for r in wrong_lobe)
                    wrong_lobe_runs = sum(
                        i == 0 or row != wrong_lobe_rows[i - 1] + 1
                        for i, row in enumerate(wrong_lobe_rows)
                    )
                    errors = [
                        abs(float(samples[r["row"]]) - r["reference_sample"]) for r in accepted
                    ]
                    wrong = sorted(r["row"] for r in accepted if not r["accepted_correct"])
                    spans, start, previous = [], None, None
                    for row in wrong:
                        if previous is None or row != previous + 1:
                            if start is not None:
                                spans.append((previous - start + 1) * step)
                            start = row
                        previous = row
                    if start is not None:
                        spans.append((previous - start + 1) * step)
                    reports.append(
                        {
                            "artifact": str(path.relative_to(ROOT)),
                            "artifact_sha256": sha(path),
                            "method": method,
                            "physical_road_group": run["partition"][0],
                            "source_dzt_sha256": run["audit"]["dzt_sha256"],
                            "backend_source_sha256": run["backend_source_sha256"],
                            "source_dzx_sha256": run["audit"]["metadata"]["source_sha256"],
                            "seeded_evaluation": run.get("seeded_evaluation"),
                            "config": run["config"],
                            "config_canonical_sha256": hashlib.sha256(
                                json.dumps(
                                    run["config"], sort_keys=True, separators=(",", ":")
                                ).encode("utf-8")
                            ).hexdigest(),
                            "working_step_m": step,
                            "dt_ns": run["audit"]["dt_ns"],
                            "tolerance_samples": layer["tolerance_samples"],
                            "layer": int(order),
                            "segment": segment,
                            "scored_observations": n,
                            "accepted": len(accepted),
                            "accepted_agree": correct,
                            "accepted_wrong": incorrect,
                            "accepted_agreement": correct / len(accepted) if accepted else None,
                            "correct_coverage": correct / n if n else None,
                            "incorrect_coverage": incorrect / n if n else None,
                            "unresolved_coverage": (n - len(accepted)) / n if n else None,
                            "scored_observation_footprint_m": n * step,
                            "correct_accepted_footprint_m": correct * step,
                            "wrong_accepted_footprint_m": incorrect * step,
                            "median_accepted_timing_error_ns": float(np.median(errors))
                            * run["audit"]["dt_ns"]
                            if errors
                            else None,
                            "wrong_signed_lobe_observations": len(wrong_lobe),
                            "wrong_signed_lobe_observed_runs": wrong_lobe_runs,
                            "wrong_signed_lobe_runs_per_scored_km": (
                                wrong_lobe_runs / (n * step / 1000) if n else None
                            ),
                            "wrong_accepted_observed_runs": len(spans),
                            "longest_contiguous_wrong_accepted_observed_span_m": max(
                                spans, default=0
                            ),
                            "candidate_retention": layer["candidate_retention"]
                            if segment == "whole"
                            else None,
                            "correspondence_survival": layer["correspondence_survival"]
                            if segment == "whole"
                            else None,
                            "proposed_correct": sum(r["proposed_correct"] for r in rows),
                            "final_gate_correct_proposal_losses": sum(
                                r["proposed_correct"] and not r["accepted_correct"] for r in rows
                            ),
                            "runtime_s": result["runtime_s"],
                            "runtime_environment_note": runtime_note(path),
                            "process_peak_memory_bytes": result["process_peak_memory_bytes"],
                            "initial_seeds": len(layer["seed_rows"]),
                            "additional_analyst_actions": 0,
                        }
                    )
            arrays.close()
    write(
        output,
        {
            "schema": "seeded-metrics-v1",
            "claim": "development interpretation agreement only",
            "denominators": (
                "Coverage uses eligible reviewed observations excluding seeds. Metres are each "
                "scored row times working spacing, a labelled footprint, never interpolation "
                "through unknown labels. Contiguous wrong spans require consecutive scored "
                "working rows. Wrong signed-lobe observations are not independent semantic "
                "switch events."
            ),
            "uncertainty": (
                "Previously used physical road groups; neighboring observations are "
                "correlated. No IID pointwise confidence interval or physical-thickness "
                "claim is supplied."
            ),
            "results": reports,
        },
    )
    print(output)


def summarize_interaction(paths, output, manifest=MANIFEST):
    """Score completed replay steps on each layer's unchanged initial observation pool.

    A revealed observation becomes analyst supplied and receives no automatic
    accuracy credit. Its footprint remains in the denominator and is reported
    separately, so additional answers cannot inflate coverage by shrinking it.
    """
    import numpy as np

    sys.path.insert(0, str(BASELINE))
    from gpr_layer_audit.conventional import _same_lobe
    from gpr_layer_audit.io.dzt import DZTFile

    inputs = read(manifest)["cases"]
    reports, provenance = [], []
    for path in paths:
        run = read(path)
        case = inputs[run["case"]]
        if (
            sha(ROOT / case["dzt"]) != run["input_sha256"]
            or sha(ROOT / case["dzx"]) != run["reference_sha256"]
        ):
            raise ValueError("Replay input/reference fingerprint mismatch")
        if not run["steps"] or run["steps"][0]["step"] != 0:
            raise ValueError("Interaction summary requires an initial step")
        radar = DZTFile(ROOT / case["dzt"]).channel()[:: run["stride"]]
        original = {
            order: {r["row"]: r for r in layer["evaluation_observations"]}
            for order, layer in run["steps"][0]["layers"].items()
        }
        provenance.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "artifact_sha256": sha(path),
                "case": run["case"],
                "policy": run["policy"],
                "freeze_initial_pulse": run.get("freeze_initial_pulse"),
                "source_snapshot": run.get("source_snapshot"),
                "backend_sha256": run["backend_sha256"],
                "configuration_sha256": run["configuration_sha256"],
                "script_sha256": run["script_sha256"],
                "initial_anchors": run["initial_anchors"],
                "initial_pulse_samples": run["initial_pulse_samples"],
                "observed_steps": len(run["steps"]),
                "stop_reason": run.get("stop_reason"),
                "completion_not_inferred_from_file_presence": True,
            }
        )
        previous, previous_wrong = {}, {}
        for step in run["steps"]:
            arrays_path = path.with_name(f"{path.stem}-step{step['step']}.npz")
            with np.load(arrays_path) as arrays:
                for order, layer in step["layers"].items():
                    initial = original[order]
                    current = {r["row"]: r for r in layer["evaluation_observations"]}
                    if not set(current).issubset(initial):
                        raise ValueError("Replay added scoring rows outside its initial cohort")
                    if any(
                        r["reference_sample"] != initial[row]["reference_sample"]
                        for row, r in current.items()
                    ):
                        raise ValueError("Reference sample changed between replay steps")
                    if (
                        layer["tolerance_samples"]
                        != run["steps"][0]["layers"][order]["tolerance_samples"]
                    ):
                        raise ValueError("Replay scoring tolerance changed after a request")
                    supplied = set(initial) - set(current)
                    observed_answers = {
                        a["row"]
                        for a in run["actions"][: step["requests"]]
                        if str(a["layer_order"]) == order and a.get("answer_sample") is not None
                    }
                    if not supplied.issubset(observed_answers):
                        raise ValueError("Scoring rows disappeared without a revealed answer")
                    samples = arrays[f"layer{order}_accepted"]
                    visible = arrays[f"layer{order}_visible"]
                    for row, expected_sample in run["initial_anchors"][order].items():
                        if samples[int(row)] != expected_sample or not visible[int(row)]:
                            raise ValueError(
                                "Initial native seed was changed or hidden by retracking"
                            )
                    for row, record in current.items():
                        accepted_now = bool(samples[row] >= 0 and visible[row])
                        correct_now = bool(
                            accepted_now
                            and abs(samples[row] - record["reference_sample"])
                            <= layer["tolerance_samples"]
                            and _same_lobe(radar[row], samples[row], record["reference_sample"])
                        )
                        if (
                            accepted_now != record["accepted"]
                            or correct_now != record["accepted_correct"]
                        ):
                            raise ValueError(
                                "Persisted replay metrics disagree with saved inference arrays"
                            )
                    anchor_rows = list(map(int, run["initial_anchors"][order]))
                    first, last = min(anchor_rows), max(anchor_rows)
                    for segment, eligible in (
                        ("whole", set(initial)),
                        ("bracketed", {r for r in initial if first < r < last}),
                        ("left_tail", {r for r in initial if r < first}),
                        ("right_tail", {r for r in initial if r > last}),
                    ):
                        rows = [r for row, r in current.items() if row in eligible]
                        accepted = {r["row"] for r in rows if r["accepted"]}
                        correct = {r["row"] for r in rows if r["accepted_correct"]}
                        wrong = accepted - correct
                        supplied_here = supplied & eligible
                        wrong_lobes = sorted(
                            row
                            for row in wrong
                            if not _same_lobe(
                                radar[row], samples[row], initial[row]["reference_sample"]
                            )
                        )
                        wrong_sorted = sorted(wrong)
                        groups = np.split(
                            wrong_sorted, np.flatnonzero(np.diff(wrong_sorted) > 1) + 1
                        )
                        state_key = (order, segment)
                        old_correct = previous.get(state_key, correct)
                        old_wrong = previous_wrong.get(state_key, wrong)
                        n, dx = len(eligible), run["step_m"]
                        layer_actions = [
                            a
                            for a in run["actions"][: step["requests"]]
                            if str(a["layer_order"]) == order
                        ]
                        reports.append(
                            {
                                "artifact": str(path.relative_to(ROOT)),
                                "case": run["case"],
                                "physical_road_group": run["physical_road_group"],
                                "policy": run["policy"],
                                "scope": run["scope"],
                                "seeding": run["seeding"],
                                "layer": int(order),
                                "segment": segment,
                                "step": step["step"],
                                "requests_all_layers": step["requests"],
                                "requests_this_layer": len(layer_actions),
                                "requests_all_layers_per_road_km": step["requests"]
                                / (run["road_length_m"] / 1000),
                                "initial_seed_count": len(run["initial_anchors"][order]),
                                "initial_observations_all_layers": sum(
                                    len(a) for a in run["initial_anchors"].values()
                                ),
                                "initial_observations_deep_layers": sum(
                                    len(a)
                                    for o, a in run["initial_anchors"].items()
                                    if int(o) in (2, 3)
                                ),
                                "initial_plus_requests_all_layers_per_road_km": (
                                    sum(len(a) for a in run["initial_anchors"].values())
                                    + step["requests"]
                                )
                                / (run["road_length_m"] / 1000),
                                "fixed_initial_observation_denominator": n,
                                "remaining_automatic_observations": len(rows),
                                "revealed_observations_excluded_from_automatic_scoring": len(
                                    supplied_here
                                ),
                                "accepted": len(accepted),
                                "accepted_agree": len(correct),
                                "accepted_wrong": len(wrong),
                                "accepted_agreement": len(correct) / len(accepted)
                                if accepted
                                else None,
                                "correct_automatic_coverage_initial_pool": len(correct) / n
                                if n
                                else None,
                                "incorrect_automatic_coverage_initial_pool": len(wrong) / n
                                if n
                                else None,
                                "analyst_supplied_coverage_initial_pool": len(supplied_here) / n
                                if n
                                else None,
                                "unresolved_coverage_initial_pool": (
                                    n - len(accepted) - len(supplied_here)
                                )
                                / n
                                if n
                                else None,
                                "new_correct_automatic_observations": len(correct - old_correct),
                                "lost_correct_automatic_observations_not_revealed": len(
                                    old_correct - correct - supplied_here
                                ),
                                "previous_correct_observations_now_analyst_supplied": len(
                                    old_correct & supplied_here
                                ),
                                "new_wrong_accepted_observations": len(wrong - old_wrong),
                                "previous_wrong_now_correct_automatic": len(old_wrong & correct),
                                "previous_wrong_now_unresolved": len(
                                    old_wrong - accepted - supplied_here
                                ),
                                "wrong_signed_lobe_observations": len(wrong_lobes),
                                "median_accepted_timing_error_ns": float(
                                    np.median(
                                        [
                                            abs(
                                                float(samples[row])
                                                - initial[row]["reference_sample"]
                                            )
                                            for row in accepted
                                        ]
                                    )
                                )
                                * run["dt_ns"]
                                if accepted
                                else None,
                                "p95_accepted_timing_error_ns": float(
                                    np.percentile(
                                        [
                                            abs(
                                                float(samples[row])
                                                - initial[row]["reference_sample"]
                                            )
                                            for row in accepted
                                        ],
                                        95,
                                    )
                                )
                                * run["dt_ns"]
                                if accepted
                                else None,
                                "wrong_signed_lobe_observed_runs": sum(
                                    i == 0 or row != wrong_lobes[i - 1] + 1
                                    for i, row in enumerate(wrong_lobes)
                                ),
                                "longest_contiguous_wrong_accepted_observed_span_m": max(
                                    (len(g) * dx for g in groups), default=0
                                ),
                                "scored_observation_footprint_m": n * dx,
                                "correct_automatic_footprint_m": len(correct) * dx,
                                "incorrect_automatic_footprint_m": len(wrong) * dx,
                                "runtime_s": step["runtime_s"],
                                "runtime_environment_note": runtime_note(path),
                                "process_peak_memory_bytes": step["process_peak_memory_bytes"],
                                "action_outcomes": {
                                    name: sum(a.get("action") == name for a in layer_actions)
                                    for name in sorted(
                                        {a.get("action", "unknown") for a in layer_actions}
                                    )
                                },
                            }
                        )
                        previous[state_key] = correct
                        previous_wrong[state_key] = wrong
    document = {
        "schema": "seeded-interaction-metrics-v1",
        "claim": "Processed interpretation development; observed replay steps only",
        "manifest": str(Path(manifest)),
        "manifest_sha256": sha(manifest),
        "denominator": (
            "Initial reviewed nonseed cohort remains fixed. Revealed observations are separately "
            "analyst supplied, receive no automatic credit, and remain in the denominator. "
            "No missing-label interpolation."
        ),
        "switch_contract": (
            "Wrong signed-lobe observations/runs are interpretation errors, not independently "
            "established semantic reflector-switch events."
        ),
        "provenance": provenance,
        "comparability_checks": {
            case_name: {
                field + "_identical": len(
                    {
                        json.dumps(item[field], sort_keys=True)
                        for item in provenance
                        if item["case"] == case_name
                    }
                )
                == 1
                for field in (
                    "initial_anchors",
                    "initial_pulse_samples",
                    "backend_sha256",
                    "configuration_sha256",
                    "script_sha256",
                )
            }
            for case_name in sorted({item["case"] for item in provenance})
        },
        "results": reports,
    }
    write(output, document)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    for case_name in sorted({r["case"] for r in reports}):
        case_rows = [
            r
            for r in reports
            if r["case"] == case_name and r["segment"] == "whole" and r["layer"] in (2, 3)
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), squeeze=False, sharex=True)
        for layer_order, axes_row in zip((2, 3), axes, strict=True):
            for artifact in sorted({r["artifact"] for r in case_rows}):
                chosen = [
                    r for r in case_rows if r["artifact"] == artifact and r["layer"] == layer_order
                ]
                if not chosen:
                    continue
                label = Path(artifact).stem
                for ax, key in zip(
                    axes_row,
                    (
                        "correct_automatic_coverage_initial_pool",
                        "incorrect_automatic_coverage_initial_pool",
                    ),
                    strict=True,
                ):
                    ax.plot(
                        [r["requests_all_layers"] for r in chosen],
                        [100 * r[key] for r in chosen],
                        marker="o",
                        label=label,
                    )
                    ax.set(
                        xlabel="Additional requests across enabled deep layers",
                        ylabel="% of fixed initial reviewed observations",
                    )
                    ax.grid(alpha=0.2)
            for ax, key in zip(
                axes_row,
                (
                    "correct_automatic_coverage_initial_pool",
                    "incorrect_automatic_coverage_initial_pool",
                ),
                strict=True,
            ):
                values = [
                    100 * r[key]
                    for r in case_rows
                    if r["layer"] == layer_order and r[key] is not None
                ]
                ax.set_ylim(0, max(max(values, default=0) * 1.12, 1))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            axes_row[0].set_title(
                f"{'Base' if layer_order == 2 else 'Subbase'}: correct automatic coverage"
            )
            axes_row[1].set_title(
                f"{'Base' if layer_order == 2 else 'Subbase'}: incorrect automatic coverage"
            )
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{case_name}: measured seed–request–retrack replay; development only")
        fig.tight_layout()
        plot = output.with_name(f"{output.stem}-{case_name}.png")
        fig.savefig(plot, dpi=150)
        plt.close(fig)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    for command in ("run", "worker"):
        runner = sub.add_parser(command)
        runner.add_argument("--case", required=True)
        runner.add_argument("--manifest", type=Path, default=MANIFEST)
        runner.add_argument("--source", type=Path, default=BASELINE)
        runner.add_argument("--method", default="seed_hybrid")
        runner.add_argument("--config", type=Path)
        runner.add_argument("--stride", type=int)
        runner.add_argument("--output", type=Path, required=True)
    for command in ("summarize", "overlay", "interaction"):
        summary = sub.add_parser(command)
        summary.add_argument("paths", type=Path, nargs="+")
        summary.add_argument("--output", type=Path, required=True)
        if command in ("interaction", "overlay"):
            summary.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "summarize":
        summarize([p.resolve() for p in args.paths], args.output)
    elif args.command == "overlay":
        overlay(args.paths, args.output, args.manifest)
    elif args.command == "interaction":
        summarize_interaction([p.resolve() for p in args.paths], args.output, args.manifest)
    elif args.command == "worker":
        worker(args)
    else:
        if args.output.exists():
            raise ValueError("Output exists; preserve the completed run and choose a new path")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--case",
            args.case,
            "--source",
            str(args.source),
            "--manifest",
            str(args.manifest),
            "--method",
            args.method,
            "--output",
            str(args.output),
        ]
        if args.config:
            command += ["--config", str(args.config)]
        if args.stride:
            command += ["--stride", str(args.stride)]
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
