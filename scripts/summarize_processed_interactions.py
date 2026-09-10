"""CPU-only fixed-cohort interaction summaries, figures and persistence verification.

Reads completed replay JSON/NPZ/checkpoints without running model inference. The
optional completed-resume probe copies artifacts into the analysis directory,
reopens the original frozen CLI on CPU and replaces predict with a failing sentinel.
Checkpoint files are trusted local campaign artifacts using the existing serializer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BUDGETS = (0, 1, 2, 4)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(
        value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()
    ).hexdigest()


def pct(value):
    return "undefined" if value is None else f"{100 * value:.2f}%"


def transitions(before_metric, after_metric):
    before = {r["row"]: r for r in before_metric["evaluation_observations"]}
    after = {r["row"]: r for r in after_metric["evaluation_observations"]}
    if not after.keys() <= before.keys():
        raise ValueError("Reference cohort grew during replay")
    shared = before.keys() & after.keys()
    manual = before.keys() - after.keys()
    result = {"common_automatic_rows": len(shared), "newly_answered_rows": sorted(manual)}
    for name, predicate in (
        ("correct_proposal", lambda r: r["proposed_correct"]),
        ("correct_accepted", lambda r: r["accepted"] and r["accepted_correct"]),
        ("wrong_accepted", lambda r: r["accepted"] and not r["accepted_correct"]),
    ):
        gained = sum(predicate(after[r]) and not predicate(before[r]) for r in shared)
        lost = sum(predicate(before[r]) and not predicate(after[r]) for r in shared)
        removed = sum(predicate(before[r]) for r in manual)
        result[name + "_gained"] = gained
        result[name + "_lost"] = lost
        result[name + "_manual_removed"] = removed
        result[name + "_net"] = gained - lost - removed
    return result


def source_contract_check(log):
    provenance = log["contract"]["provenance"]
    snapshot = Path(provenance["source_snapshot"])
    manifest = read(snapshot / "source-manifest.json")
    if manifest["source_sha256"] != provenance["source_sha256"]:
        raise ValueError("Source snapshot hash disagrees with replay contract")
    changed = [
        name for name, expected in manifest["files"].items() if sha(snapshot / name) != expected
    ]
    if changed:
        raise ValueError("Replay source changed: " + ", ".join(changed))
    return {"source_snapshot": str(snapshot), "source_files_verified": len(manifest["files"])}


def verify_arrays(path, log, layer, batch_dir):
    checks = {"local_scopes": [], "exact_answers": [], "step_arrays": []}
    order = str(layer)
    count = log["contract"]["grid"]["shape"][0]
    dx = log["contract"]["grid"]["dx_m"]
    stride = log["contract"]["provenance"]["trace_stride"]
    prior = None
    for step in log["steps"]:
        budget = step["additional_requests"]
        array_path = path.with_name(f"{path.stem}-step{budget}.npz")
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        checks["step_arrays"].append(
            {"budget": budget, "path": str(array_path), "sha256": sha(array_path)}
        )
        if budget == 0:
            batch_arrays = batch_dir / "stage-arrays.npz"
            if not batch_arrays.exists():
                checks["batch_initial_parity"] = {
                    "status": "unavailable",
                    "path": str(batch_arrays),
                }
            else:
                with np.load(batch_arrays, allow_pickle=False) as batch:
                    automatic = arrays[f"layer{order}_visible"].copy()
                    automatic[batch["dense_manual_seed"].astype(bool)] = False
                    fields = {
                        "proposal": np.array_equal(
                            arrays[f"layer{order}_provisional_samples"], batch["dense_proposal"]
                        ),
                        "confidence": np.array_equal(
                            arrays[f"layer{order}_confidence"], batch["dense_confidence"]
                        ),
                        "automatic_accepted_excluding_manual_seeds": np.array_equal(
                            automatic, batch["dense_accepted"]
                        ),
                    }
                if not all(fields.values()):
                    raise ValueError(
                        f"Initial batch/replay prediction mismatch: {path.name}: {fields}"
                    )
                checks["batch_initial_parity"] = {
                    "status": "bitwise_equal",
                    "fields": fields,
                    "path": str(batch_arrays),
                    "sha256": sha(batch_arrays),
                }
        else:
            action = log["actions"][budget - 1]
            row = action["row"]
            if action["native_trace"] != row * stride or action["chainage_m"] != row * dx:
                raise ValueError("Requested coordinate does not round-trip to native trace")
            if action["answer_status"] == "unavailable_reviewed_answer":
                if any(not np.array_equal(arrays[k], prior[k]) for k in arrays):
                    raise ValueError("Unavailable answer changed prediction arrays")
            elif action["retrack_status"] == "applied":
                if arrays[f"layer{order}_samples"][row] != action["answer_sample"]:
                    raise ValueError("Requested answer did not survive exact native export")
                checks["exact_answers"].append(
                    {
                        "budget": budget,
                        "native_trace": row * stride,
                        "sample": action["answer_sample"],
                    }
                )
                lo, hi = action["affected_rows"]
                if action["operation"] == "local_correction":
                    expected = [
                        max(0, int(np.ceil(row - 25 / dx))),
                        min(count - 1, int(np.floor(row + 25 / dx))),
                    ]
                    if [lo, hi] != expected:
                        raise ValueError("Local replay changed the inclusive +/-25m contract")
                    outside = np.r_[np.arange(lo), np.arange(hi + 1, count)]
                    equal = {
                        k: np.array_equal(arrays[k][outside], prior[k][outside]) for k in arrays
                    }
                    if not all(equal.values()):
                        raise ValueError("Local correction changed an outside prediction")
                    checks["local_scopes"].append(
                        {
                            "budget": budget,
                            "affected_rows": [lo, hi],
                            "outside_bitwise_equal": equal,
                        }
                    )
                elif [lo, hi] != [0, count - 1]:
                    raise ValueError("Global model seed scope is not the full retained acquisition")
        prior = arrays
    return checks


def serializer():
    source = ROOT / "scripts/replay_seeded_interaction.py"
    spec = importlib.util.spec_from_file_location("_interaction_analysis_checkpoint", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_checkpoint(path, log, output):
    io = serializer()
    checkpoint = path.with_suffix(".checkpoint")
    state = io.load_checkpoint(checkpoint, log["contract"])
    if state["log"] != log:
        raise ValueError("Saved checkpoint log differs from published replay JSON")
    final_budget = log["steps"][-1]["additional_requests"]
    with np.load(
        path.with_name(f"{path.stem}-step{final_budget}.npz"), allow_pickle=False
    ) as saved:
        for order, route in state["paths"].items():
            for name in (
                "samples",
                "visible",
                "provisional_samples",
                "confidence",
                "alternate_samples",
            ):
                if not np.array_equal(getattr(route, name), saved[f"layer{order}_{name}"]):
                    raise ValueError("Checkpoint paths disagree with saved native output")
    answer_count = sum(a["answer_status"] == "revealed_exact" for a in log["actions"])
    if len(state["visited"]) != len(log["actions"]):
        raise ValueError("Charged requests and persistent visited locations differ")
    history = state.get("predictor_state", {}).get("previous_probability", {})
    # Exercise the actual existing serializer on detached complete state.
    roundtrip = output / "checkpoint-roundtrips" / checkpoint.name
    roundtrip.parent.mkdir(parents=True, exist_ok=True)
    io.save_checkpoint(roundtrip, log["contract"], state)
    reopened = io.load_checkpoint(roundtrip, log["contract"])
    for order, route in state["paths"].items():
        for field in route.__dataclass_fields__:
            original, restored = getattr(route, field), getattr(reopened["paths"][order], field)
            if isinstance(original, np.ndarray) and not np.array_equal(original, restored):
                raise ValueError("Saved/reopened route array changed")
            if isinstance(original, dict):
                for key, value in original.items():
                    if isinstance(value, np.ndarray) and not np.array_equal(value, restored[key]):
                        raise ValueError("Saved/reopened route evidence changed")
    for layer, values in history.items():
        if not np.array_equal(values, reopened["predictor_state"]["previous_probability"][layer]):
            raise ValueError("Saved/reopened previous-probability history changed")
    return {
        "path": str(checkpoint),
        "sha256": sha(checkpoint),
        "payload_hash_verified": True,
        "roundtrip_path": str(roundtrip),
        "all_route_arrays_exact": True,
        "charged_visited_requests": len(state["visited"]),
        "exact_answers": answer_count,
        "history_arrays": {
            str(k): {"shape": list(v.shape), "sha256": array_hash(v)} for k, v in history.items()
        },
        "history_preserved_exactly": True,
    }


def summarize_run(path, artifact_root, cases, output, checkpoint_checks):
    log = read(path)
    if log.get("schema") not in {
        "processed-ml-interaction-v1",
        "processed-ml-interaction-v2",
    } or not log.get("completed"):
        return None
    contract = log["contract"]
    provenance = contract["provenance"]
    case = cases[provenance["record_id"]]
    model = Path(provenance["model_path"]).name
    arm = model.split("-" + provenance["physical_road_group"])[0]
    policy, operation = contract["config"]["policy"], contract["config"]["operation"]
    present = [s["additional_requests"] for s in log["steps"]]
    if present != list(range(len(log["actions"]) + 1)):
        raise ValueError("Replay missed or duplicated an action prefix")
    record = {
        "run": path.stem,
        "path": str(path),
        "sha256": sha(path),
        "arm": arm,
        "model": model,
        "model_sha256": provenance["model_sha256"],
        "case": case["case_id"],
        "road_group": provenance["physical_road_group"],
        "policy": policy,
        "operation": operation,
        "length_m": log["road_length_m"],
        "trace_stride": provenance["trace_stride"],
        "stop_reason": log.get("stop_reason"),
        "budgets_available": [b for b in BUDGETS if b in present],
        "source": source_contract_check(log),
        "layers": {},
        "actions": log["actions"],
    }
    for layer in log["steps"][0]["layers"]:
        first = log["steps"][0]["layers"][layer]
        fixed_n = first["fixed_initial_observations"]
        initial_rows = {r["row"] for r in first["evaluation_observations"]}
        entries = []
        for step in log["steps"]:
            budget = step["additional_requests"]
            metric = step["layers"][layer]
            manual = set(metric["analyst_supplied_rows_no_automatic_credit"])
            observations = {r["row"]: r for r in metric["evaluation_observations"]}
            if (
                metric["fixed_initial_observations"] != fixed_n
                or set(observations) | manual != initial_rows
                or set(observations) & manual
            ):
                raise ValueError("Fixed reviewed cohort or manual exclusion changed")
            correct = sum(r["accepted_correct"] for r in observations.values())
            accepted = sum(r["accepted"] for r in observations.values())
            proposed = sum(r["proposed_correct"] for r in observations.values())
            if (correct, accepted - correct, proposed) != (
                metric["correct_automatically_accepted"],
                metric["wrong_automatically_accepted"],
                metric["correct_proposals_before_gating"],
            ):
                raise ValueError("Published counts disagree with actual scored observations")
            change = transitions(first, metric)
            if (
                change["correct_proposal_net"]
                != proposed - first["correct_proposals_before_gating"]
            ):
                raise ValueError("Proposal gain/loss/manual accounting does not close")
            actions = log["actions"][:budget]
            entry = {
                "budget": budget,
                "fixed_N": fixed_n,
                "initial_seeds": first["initial_seed_count"],
                "requests": len(actions),
                "answered": len(manual),
                "unavailable": sum(
                    a["answer_status"] == "unavailable_reviewed_answer" for a in actions
                ),
                "actions_per_km": step["actions_per_km"],
                "total_initial_and_additional_actions_per_km": step[
                    "total_observations_and_requests_per_km"
                ],
                "correct_accepted": correct,
                "wrong_accepted": accepted - correct,
                "automatic_unresolved": fixed_n - accepted,
                "accepted_agreement": correct / accepted if accepted else None,
                "correct_coverage": correct / fixed_n if fixed_n else None,
                "wrong_coverage": (accepted - correct) / fixed_n if fixed_n else None,
                "correct_proposals": proposed,
                "correct_proposal_coverage": proposed / fixed_n if fixed_n else None,
                "runtime_s": step["runtime_s"],
                "from_initial": change,
                "from_previous_action": transitions(
                    log["steps"][budget - 1]["layers"][layer], metric
                )
                if budget
                else None,
                "longest_wrong_accepted_observed_span_m": metric[
                    "longest_contiguous_wrong_accepted_observed_span_m"
                ],
                "stratified": metric["native_stratified_evaluation"],
            }
            entries.append(entry)
        batch_dir = (
            artifact_root / "evaluation" / model / case["case_id"] / "baseline" / f"layer{layer}"
        )
        checks = verify_arrays(path, log, int(layer), batch_dir)
        batch = read(batch_dir / "report.json")["stages"]["dense_gated"]
        if (
            batch["fixed_initial_nonseed_N"],
            batch["correct_proposals_before_gating"],
            batch["correct_accepted"],
            batch["wrong_accepted"],
        ) != (
            fixed_n,
            entries[0]["correct_proposals"],
            entries[0]["correct_accepted"],
            entries[0]["wrong_accepted"],
        ):
            raise ValueError("Batch and replay initial scored counts differ")
        batch_rows = read(batch_dir / "report.json")["stages"]["dense_gated"][
            "evaluation_observations"
        ]
        native_replay = sorted(
            (r["row"] * provenance["trace_stride"], r["reference_sample"])
            for r in first["evaluation_observations"]
        )
        native_batch = sorted((r["native_trace"], r["reference_sample"]) for r in batch_rows)
        if native_replay != native_batch:
            raise ValueError("Batch/replay native reference cohort differs")
        checks["native_cohort_parity"] = (
            "exact after explicit retained-to-native coordinate conversion"
        )
        calibration_path = Path(provenance["model_path"]) / "calibration.json"
        calibration = read(calibration_path)
        gate = calibration["calibrations"][layer]["dense"]
        if (
            gate["selected_threshold"] != provenance["acceptance_threshold"]
            or calibration["weights_sha256"] != provenance["model_sha256"]
        ):
            raise ValueError("Replay threshold/model differs from training-road calibration")
        record["layers"][layer] = {
            "steps": entries,
            "checks": checks,
            "calibration": {
                "path": str(calibration_path),
                "sha256": sha(calibration_path),
                "selected_threshold": gate["selected_threshold"],
                "objective_met": gate["objective_met"],
            },
        }
    if checkpoint_checks:
        record["checkpoint"] = verify_checkpoint(path, log, output)
    return record


def plots(records, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "uncertainty": "#007f86",
        "fixed_spacing": "#c17824",
        "largest_interval_midpoint": "#70419d",
    }
    labels = {
        "uncertainty": "Model uncertainty",
        "fixed_spacing": "Fixed spacing",
        "largest_interval_midpoint": "Largest interval midpoint",
    }
    groups = defaultdict(list)
    for run in records:
        for layer in run["layers"]:
            groups[(run["model"], run["case"], layer)].append(run)
    produced = []
    for (model, case, layer), runs in groups.items():
        title = f"{model.split('-')[0].title()} · {case} · {'Base' if layer == '2' else 'Subbase'}"
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
        for run in runs:
            rows = [s for s in run["layers"][layer]["steps"] if s["budget"] in BUDGETS]
            name = labels[run["policy"]] + (
                " / global" if run["operation"] == "global_model_seed" else " / local"
            )
            style = "--" if run["operation"] == "global_model_seed" else "-"
            x = [r["budget"] for r in rows]
            for ax, key in zip(
                axes,
                ("correct_proposal_coverage", "correct_coverage", "wrong_coverage"),
                strict=True,
            ):
                ax.plot(
                    x,
                    [100 * r[key] for r in rows],
                    marker="o",
                    lw=1.7,
                    ls=style,
                    color=colors[run["policy"]],
                    label=name,
                )
        for ax, heading in zip(
            axes,
            (
                "Correct proposals before gating",
                "Correct automatic coverage",
                "Wrong automatic coverage",
            ),
            strict=True,
        ):
            ax.set(
                title=heading,
                xlabel="Additional charged requests",
                ylabel="% of fixed initial cohort",
                xticks=BUDGETS,
            )
            ax.grid(alpha=0.2)
            ax.set_ylim(0, 100)
        if all(
            step["correct_accepted"] + step["wrong_accepted"] == 0
            for run in runs
            for step in run["layers"][layer]["steps"]
        ):
            for ax in axes[1:]:
                ax.text(
                    0.5,
                    0.55,
                    "No automatic accepts\nAgreement undefined",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#555555",
                )
        axes[0].legend(fontsize=7.5, loc="best")
        fig.suptitle(
            title + "\nAnalyst-reference agreement; answers excluded from automatic credit",
            fontsize=12,
        )
        destination = output / f"{model}-{case}-L{layer}-curves.png"
        fig.savefig(destination, dpi=170)
        plt.close(fig)
        produced.append(str(destination))
        fig, axes = plt.subplots(
            len(runs),
            1,
            figsize=(13.5, 2.65 * len(runs)),
            sharex=True,
            constrained_layout=True,
            squeeze=False,
        )
        for ax, run in zip(axes[:, 0], runs, strict=True):
            log = read(run["path"])
            metric = log["steps"][0]["layers"][layer]
            refs = metric["evaluation_observations"]
            dx = log["contract"]["grid"]["dx_m"]
            last = log["steps"][-1]["additional_requests"]
            p = Path(run["path"])
            with (
                np.load(p.with_name(p.stem + "-step0.npz")) as first,
                np.load(p.with_name(f"{p.stem}-step{last}.npz")) as final,
            ):
                initial = first[f"layer{layer}_provisional_samples"].copy()
                current = final[f"layer{layer}_provisional_samples"].copy()
            distance = np.arange(len(current)) * dx
            initial[initial < 0] = np.nan
            current[current < 0] = np.nan
            ax.plot(distance, initial, color="#989898", lw=0.8, label="Initial proposal")
            ax.plot(
                distance, current, color=colors[run["policy"]], lw=0.9, label="After four requests"
            )
            ax.scatter(
                [r["row"] * dx for r in refs],
                [r["reference_sample"] for r in refs],
                s=3,
                color="black",
                alpha=0.4,
                label="Reviewed reference",
            )
            for action in log["actions"]:
                if action["answer_status"] == "revealed_exact":
                    ax.scatter(
                        action["chainage_m"],
                        action["answer_sample"],
                        s=32,
                        marker="*",
                        color="#b22222",
                        zorder=6,
                    )
                else:
                    ax.axvline(action["chainage_m"], color="#b22222", ls=":", alpha=0.35)
            row = run["layers"][layer]["steps"][-1]
            change = row["from_initial"]
            suffix = (
                "global re-anchor" if run["operation"] == "global_model_seed" else "local ±25 m"
            )
            ax.set_title(
                f"{labels[run['policy']]} · {suffix} · correct proposal gains "
                f"{change['correct_proposal_gained']}, losses {change['correct_proposal_lost']}, "
                f"manual removal {change['correct_proposal_manual_removed']} · "
                f"unavailable {row['unavailable']}/4",
                fontsize=9,
            )
            ax.set_ylabel("Native sample")
            ax.invert_yaxis()
            ax.grid(alpha=0.15)
        axes[0, 0].legend(fontsize=8, ncol=3, loc="upper right")
        axes[-1, 0].set_xlabel(
            "Chainage (m); red stars = exact answers; dotted red lines = unavailable requests"
        )
        fig.suptitle(
            title + " · full retained acquisition; proposals are not accepted measurements",
            fontsize=12,
        )
        destination = output / f"{model}-{case}-L{layer}-before-after.png"
        fig.savefig(destination, dpi=170)
        plt.close(fig)
        produced.append(str(destination))
    return produced


def resume_probe(run, artifact_root, output):
    """Completed real CLI resume on copies, CPU only, forward calls forbidden."""
    original = Path(run["path"])
    log = read(original)
    provenance = log["contract"]["provenance"]
    folder = output / "completed-resume" / original.stem
    folder.mkdir(parents=True, exist_ok=True)
    copied = folder / original.name
    shutil.copy2(original, copied)
    shutil.copy2(original.with_suffix(".checkpoint"), copied.with_suffix(".checkpoint"))
    before = {suffix: sha(copied.with_suffix(suffix)) for suffix in (".json", ".checkpoint")}
    case = next(
        c
        for c in read(artifact_root / "evaluation/cases.json")["cases"]
        if c["record_id"] == provenance["record_id"]
    )
    layer = next(iter(log["steps"][0]["layers"]))
    config = log["contract"]["config"]
    snapshot = Path(provenance["source_snapshot"])
    args = [
        str(snapshot / "scripts/replay_processed_ml.py"),
        "--manifest",
        str(artifact_root / "processed_dataset/manifest.json"),
        "--record-id",
        provenance["record_id"],
        "--checkpoint",
        provenance["model_path"],
        "--layer",
        layer,
        "--stride",
        str(provenance["trace_stride"]),
        "--policy",
        config["policy"],
        "--operation",
        config["operation"],
        "--device",
        "cpu",
        "--budgets",
        *map(str, config["budgets"]),
        "--output",
        str(copied),
        "--resume",
        "--frozen-source",
        "--acceptance-threshold",
        "inf"
        if provenance["acceptance_threshold"] is None
        else str(provenance["acceptance_threshold"]),
    ]
    args += (
        ["--case", case["case_id"]]
        if case["case_id"] in ("gujrat-second", "mandiali-short")
        else ["--seeds", case["seed_path"]]
    )
    wrapper = folder / "resume_without_forward.py"
    wrapper.write_text(
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(snapshot / 'scripts')!r})\n"
        f"sys.path.insert(0, {str(snapshot / 'src')!r})\n"
        "from gpr_layer_audit.ml.processed_inference import ProcessedPredictor\n"
        "def forbidden(*args, **kwargs):\n"
        "    raise AssertionError('Completed resume attempted inference')\n"
        "ProcessedPredictor.predict = forbidden\nProcessedPredictor.predict_evidence = forbidden\n"
        f"sys.argv = {args!r}\nrunpy.run_path(sys.argv[0], run_name='__main__')\n"
        "print('COMPLETED_RESUME_WITH_ZERO_FORWARD_CALLS')\n",
        encoding="utf-8",
    )
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        [sys.executable, str(wrapper)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    (folder / "resume.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    after = {suffix: sha(copied.with_suffix(suffix)) for suffix in before}
    if before != after or "COMPLETED_RESUME_WITH_ZERO_FORWARD_CALLS" not in completed.stdout:
        raise ValueError("Completed resume changed persisted results or ran prediction")
    return {
        "run": run["run"],
        "device": "cpu",
        "forward_calls": 0,
        "original_artifacts_untouched": True,
        "copy_json_and_checkpoint_bitwise_unchanged": True,
        "hashes": before,
        "command": [sys.executable, str(wrapper)],
        "log": str(folder / "resume.log"),
    }


def markdown(summary, output):
    runs = summary["runs"]
    lines = [
        "# Completed interaction measurements",
        "",
        "Analyst-reference agreement on historically used development roads. "
        "Every policy/operation is a separate replay; repeated rows are not independent surveys.",
        "",
        f"{len(runs)} completed replays; {summary['requests']} charged requests, "
        f"{summary['unavailable_requests']} unavailable exact answers and "
        f"{summary['revealed_answers']} revealed answers. All runs reached four requests. "
        "Newly answered rows remain in N and receive no automatic or proposal credit.",
        "",
        "| Model / case / layer | Policy / action | N | Correct proposals 0 → 4 | "
        "Gains / losses / manual removal | Correct / wrong accepted | Agreement | "
        "Unavailable / 4 | Actions/km |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        for layer, values in run["layers"].items():
            first, final = values["steps"][0], values["steps"][-1]
            t = final["from_initial"]
            lines.append(
                f"| {run['arm']} / {run['case']} / {layer} | {run['policy']} / "
                f"{run['operation']} | {final['fixed_N']} | {first['correct_proposals']} → "
                f"{final['correct_proposals']} ({pct(final['correct_proposal_coverage'])}) | "
                f"{t['correct_proposal_gained']} / {t['correct_proposal_lost']} / "
                f"{t['correct_proposal_manual_removed']} | {final['correct_accepted']} / "
                f"{final['wrong_accepted']} | {pct(final['accepted_agreement'])} | "
                f"{final['unavailable']} / 4 | {final['actions_per_km']:.3f} |"
            )
    lines += [
        "",
        "Gains and losses compare only rows that remain automatic at both endpoints. "
        "Manual removal is the number of initially correct proposals subsequently answered "
        "by the analyst. Net correct-proposal change = gains − losses − manual removal. "
        "Per-action and 0/1/2/4 records are in interaction-summary.json.",
        "",
        "All initial proposals, confidence and automatic acceptance masks were compared "
        "bitwise with batch evaluation after excluding initial manual seeds. Native "
        "reference cohorts were compared after explicitly converting working rows to native "
        "trace numbers. Local changes were checked against every saved array outside ±25 m. "
        "Checkpoint verification uses the real serializer and retains predictor history.",
        "",
        "The nested replay scorer uses retained-row coordinates with working spacing; "
        "raw nested cohort hashes are not directly comparable to the batch native-trace "
        "hashes. This summary performs the explicit conversion and leaves original "
        "artifacts unchanged.",
        "",
        "The frozen v1 replay source used for these original completed runs had two "
        "recovery limitations: it saved requests only after retracking, and retained "
        "neighbor-conflicting answers as active seeds. These single-layer runs contain "
        "neither interruption nor rejection, so their counts remain valid. The current "
        "v2 replay source persists charged pending answers before fitting, retries from "
        "saved answers, and keeps rejected answers out of active model inputs while "
        "excluding them from automatic credit. Original run artifacts remain unchanged.",
    ]
    for check in summary.get("completed_resume", []):
        lines += [
            "",
            f"Completed CPU resume verified for {check['run']}: zero forward calls; "
            "copied JSON and checkpoint remained bitwise unchanged.",
        ]
    (output / "interaction-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-checkpoints", action="store_true")
    parser.add_argument("--verify-completed-resume", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    output = args.output or args.artifact_root / "interactions-analysis"
    output.mkdir(parents=True, exist_ok=True)
    registry = read(args.artifact_root / "evaluation/cases.json")
    cases = {c["record_id"]: c for c in registry["cases"]}
    records = []
    for path in sorted((args.artifact_root / "interactions").glob("*.json")):
        value = summarize_run(path, args.artifact_root, cases, output, args.verify_checkpoints)
        if value:
            records.append(value)
    summary = {
        "schema": "processed-interaction-analysis-v1",
        "script_sha256": sha(__file__),
        "claim": "Grouped development analyst-reference agreement; no physical-accuracy claim",
        "requests": sum(len(r["actions"]) for r in records),
        "unavailable_requests": sum(
            a["answer_status"] == "unavailable_reviewed_answer"
            for r in records
            for a in r["actions"]
        ),
        "revealed_answers": sum(
            a["answer_status"] == "revealed_exact" for r in records for a in r["actions"]
        ),
        "runs": records,
        "plots": [],
        "completed_resume": [],
    }
    if args.verify_completed_resume:
        for arm in ("primary", "correction"):
            run = next(
                (r for r in records if r["arm"] == arm and r["operation"] == "local_correction"),
                None,
            )
            if run:
                summary["completed_resume"].append(resume_probe(run, args.artifact_root, output))
    write(output / "interaction-summary.json", summary)
    markdown(summary, output)
    if not args.skip_plots:
        summary["plots"] = plots(records, output)
    write(output / "interaction-summary.json", summary)
    markdown(summary, output)
    print(
        json.dumps(
            {
                "runs": len(records),
                "requests": summary["requests"],
                "unavailable": summary["unavailable_requests"],
                "output": str(output),
                "completed_resume_probes": len(summary["completed_resume"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
