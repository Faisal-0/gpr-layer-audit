"""Report completed immutable campaign jobs; never rescore or count queued work."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

COUNT_KEYS = (
    "fixed_initial_nonseed_N",
    "correct_accepted",
    "wrong_accepted",
    "unresolved",
    "correct_proposals_before_gating",
)
STAGES = (
    "direct_ungated",
    "dense_ungated",
    "seed_interpolation_no_radar",
    "direct_gated",
    "dense_gated",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percent(value):
    return "undefined" if value is None else f"{100 * value:.2f}%"


def preserve_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        backup = (
            path.parent / "history" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") / path.name
        )
        backup.parent.mkdir(parents=True)
        shutil.copy2(path, backup)
    path.write_text(content, encoding="utf-8")


def pooled(rows):
    counts = {key: sum(r[key] for r in rows) for key in COUNT_KEYS}
    n, correct, wrong = (counts[k] for k in COUNT_KEYS[:3])
    accepted = correct + wrong
    if accepted + counts["unresolved"] != n:
        raise ValueError("Saved counts do not preserve the initial nonseed cohort")
    return {
        **counts,
        "accepted": accepted,
        "accepted_agreement": correct / accepted if accepted else None,
        "correct_coverage": correct / n if n else None,
        "wrong_coverage": wrong / n if n else None,
        "correct_proposal_coverage": counts["correct_proposals_before_gating"] / n if n else None,
    }


def average(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def collect(root):
    models, sources, rows, details, controls, supplements = [], {}, [], [], [], []
    manifest_path = root / "processed_dataset/manifest.json"
    records = {r["record_id"]: r for r in read(manifest_path)["records"]}
    sources[str(manifest_path)] = sha(manifest_path)
    for path in sorted((root / "models").glob("*/model.json")):
        m = read(path)
        ep = root / "evaluation" / path.parent.name
        if m.get("status") != "completed" or not (ep / "run.json").exists():
            continue
        run = read(ep / "run.json")
        if run.get("status") != "completed":
            continue
        if (
            run["weights_sha256"] != m["weights_sha256"]
            or sha(ep / "summary.json") != run["summary_sha256"]
        ):
            raise ValueError(f"Completed model/evaluation fingerprint mismatch: {ep}")
        calibration_path = path.parent / "calibration.json"
        calibration = read(calibration_path)
        for p in (path, ep / "run.json", ep / "summary.json", calibration_path):
            sources[str(p)] = sha(p)
        support = []
        for audit in m["support_and_inner_split_audit"]:
            record = records[audit["record_id"]]
            for layer, counts in audit["layers"].items():
                total = record["label_counts"].get(layer, 0)
                support.append(
                    {
                        "record_id": audit["record_id"],
                        "road_group": audit["physical_road_group"],
                        "layer": int(layer),
                        "available_coordinate_valid_labels": total,
                        **counts,
                        "outside_buffered_splits": total
                        - counts["train_observations"]
                        - counts["inner_validation_observations"],
                        "source_sha256": record["source_sha256"],
                        "label_sha256": record["label_sha256"],
                        "source_path": record["path"],
                        "label_source_path": record["label_source_path"],
                        "context_buffer_m": audit["context_buffer_m"],
                        "validation_blocks": audit["validation_blocks"],
                    }
                )
        models.append(
            {
                "run": path.parent.name,
                "model": str(path),
                "evaluation": str(ep),
                "held_out_group": m["held_out_group"],
                "config": m["config"],
                "steps": m["completed_steps"],
                "best_step": m["best_step"],
                "training_seconds": m["elapsed_seconds"],
                "parameter_count": m["parameter_count"],
                "weights_sha256": m["weights_sha256"],
                "model_sha256": sha(path),
                "training_road_groups": m["training_road_groups"],
                "training_record_ids": m["training_record_ids"],
                "excluded_training_records": m["excluded_training_records"],
                "fitting_support": support,
                "held_out_records": [
                    {"record_id": r["record_id"], "label_counts": r["label_counts"]}
                    for r in records.values()
                    if r["physical_road_group"] == m["held_out_group"]
                ],
                "calibration_path": str(calibration_path),
                "calibration": {
                    layer: {
                        stage: {
                            k: gate.get(k)
                            for k in (
                                "selected_threshold",
                                "objective_met",
                                "reason",
                                "training_groups",
                                "observed_episodes",
                            )
                        }
                        for stage, gate in stages.items()
                    }
                    for layer, stages in calibration["calibrations"].items()
                },
                "prediction_seconds": run["prediction_seconds"],
                "evaluation_seconds": run["total_seconds"],
                "peak_cuda_allocated_bytes": m.get("peak_cuda_allocated_bytes"),
                "case_ids": run["cases"],
            }
        )
        rows.extend(
            {
                "run": path.parent.name,
                "evaluation_run": path.parent.name,
                "road_group": m["held_out_group"],
                **r,
            }
            for r in read(ep / "summary.json")["per_case_layer"]
        )
        for rp in sorted(ep.glob("*/baseline/layer*/report.json")):
            report = read(rp)
            sources[str(rp)] = sha(rp)
            details.append(
                {
                    "run": path.parent.name,
                    "path": str(rp),
                    **{k: v for k, v in report.items() if k != "stages"},
                    "stages": {
                        stage: {
                            k: v
                            for k, v in values.items()
                            if k not in ("evaluation_observations", "precision_coverage_curve")
                        }
                        for stage, values in report["stages"].items()
                    },
                }
            )
    by_weights = {m["weights_sha256"]: m for m in models}
    for rp in sorted((root / "evaluation").glob("*/run.json")):
        run = read(rp)
        if run.get("status") != "completed" or run.get("weights_sha256") not in by_weights:
            continue
        m = by_weights[run["weights_sha256"]]
        if rp.parent.name == m["run"]:
            continue
        sp = rp.with_name("summary.json")
        if sha(sp) != run["summary_sha256"]:
            raise ValueError("Completed supplemental summary fingerprint mismatch")
        sources[str(rp)], sources[str(sp)] = sha(rp), sha(sp)
        extra_rows = [
            {
                "run": m["run"],
                "evaluation_run": rp.parent.name,
                "road_group": m["held_out_group"],
                **r,
            }
            for r in read(sp)["per_case_layer"]
        ]
        if any(c != "baseline" for c in run["controls"]):
            for p in sorted(rp.parent.glob("*/*/layer*/report.json")):
                report, evidence = read(p), read(p.with_name("evidence.json"))
                sources[str(p)], sources[str(p.with_name("evidence.json"))] = (
                    sha(p),
                    sha(p.with_name("evidence.json")),
                )
                controls.append(
                    {
                        "run": m["run"],
                        "evaluation_run": rp.parent.name,
                        "case_id": report["case_id"],
                        "layer": report["layer"],
                        "control": report["control"],
                        "path": str(p),
                        "conditioning_change": evidence.get("conditioning_change"),
                        "original_initial_seeds": evidence["original_initial_seeds"],
                        "inference_seeds": evidence["inference_seeds"],
                        "control_description": evidence["control"],
                        "score_interpretation": report["control_score_interpretation"],
                        "stages": {
                            stage: {
                                **pooled([v]),
                                "cohort_sha256": v["cohort_sha256"],
                                "manual_answers_in_initial_cohort": v.get(
                                    "manual_answers_in_initial_cohort", 0
                                ),
                            }
                            for stage, v in report["stages"].items()
                        },
                    }
                )
        else:
            supplements.append(
                {
                    "evaluation_run": rp.parent.name,
                    "rows": extra_rows,
                    "source": str(rp),
                    "registered_main_results_replaced": False,
                }
            )
    return models, rows, details, controls, supplements, sources


def table(lines, headers, rows):
    lines += ["", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)


def summarize(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    models, rows, details, controls, supplements, sources = collect(root)
    aggregates = []
    for m in models:
        for layer in sorted({r["layer"] for r in rows if r["run"] == m["run"]}):
            for stage in STAGES:
                members = [
                    r
                    for r in rows
                    if r["run"] == m["run"]
                    and r["layer"] == layer
                    and r["stage"] == stage
                    and r["control"] == "baseline"
                ]
                aggregates.append(
                    {
                        "run": m["run"],
                        "road_group": m["held_out_group"],
                        "layer": layer,
                        "stage": stage,
                        **pooled(members),
                    }
                )
    primary_names = {
        m["run"] for m in models if m["run"].startswith("primary-") and m["config"]["seed"] == 42
    }
    primary, comparisons = {}, []
    for layer in (1, 2, 3):
        stages = {}
        for stage in STAGES:
            members = [
                r
                for r in aggregates
                if r["run"] in primary_names and r["layer"] == layer and r["stage"] == stage
            ]
            stages[stage] = {
                "roads": {r["road_group"]: r for r in members},
                "pooled": pooled(members),
                "physical_road_count": len(members),
                "roads_with_defined_agreement": sum(
                    r["accepted_agreement"] is not None for r in members
                ),
                "macro_road": {
                    k: average([r[k] for r in members])
                    for k in ("accepted_agreement", "correct_coverage", "wrong_coverage")
                },
            }
        primary[str(layer)] = {
            "stages": stages,
            "macro_road_correct_proposal_coverage": stages["dense_ungated"]["macro_road"][
                "correct_coverage"
            ],
        }
    for road in ("gujrat", "mandiali", "jamshoro"):
        for layer in (1, 2, 3):
            values = {
                family: {
                    r["stage"]: r
                    for r in aggregates
                    if r["run"] == f"{family}-{road}-bf16-s42" and r["layer"] == layer
                }
                for family in ("primary", "layer-only")
            }
            if all(values.values()):
                p, u = values["primary"], values["layer-only"]
                if (
                    p["direct_ungated"]["fixed_initial_nonseed_N"]
                    != u["direct_ungated"]["fixed_initial_nonseed_N"]
                ):
                    raise ValueError("Primary/layer-only cohort mismatch")
                comparisons.append(
                    {
                        "road_group": road,
                        "layer": layer,
                        "primary": p,
                        "layer_only": u,
                        "direct_coverage_difference": p["direct_ungated"]["correct_coverage"]
                        - u["direct_ungated"]["correct_coverage"],
                    }
                )
    deep = [
        g
        for m in models
        for layer_key, stages in m["calibration"].items()
        if int(layer_key) in (2, 3)
        for g in stages.values()
    ]
    facts = {
        "completed_models": len(models),
        "deep_gate_count": len(deep),
        "null_deep_gate_count": sum(g["selected_threshold"] is None for g in deep),
        "all_deep_operating_gates_null": bool(deep)
        and all(g["selected_threshold"] is None for g in deep),
    }
    lines = [
        "# Completed processed-learning campaign",
        "",
        "Grouped development agreement with analyst interpretations; not physical thickness "
        "accuracy or untouched-road reliability. "
        "Outer roads were excluded from model fitting and threshold selection. Acquisitions "
        "are pooled within each physical road "
        "before equal-road macro averaging; adjacent observations are not independent samples.",
        "",
        f"**{len(models)} trained and evaluated models; "
        f"{facts['null_deep_gate_count']} of {len(deep)} deep-layer gates are undefined.** "
        "Undefined means the TRAIN objective was not met and automatic acceptance is withheld. "
        "Proposal gains do not constitute accepted coverage gains.",
        "",
        "## Completed models",
    ]
    table(
        lines,
        ["Run", "Outer road", "Steps / best", "Training seconds", "Parameters", "Weights SHA256"],
        [
            [
                m["run"],
                m["held_out_group"],
                f"{m['steps']} / {m['best_step']}",
                f"{m['training_seconds']:.1f}",
                m["parameter_count"],
                f"`{m['weights_sha256']}`",
            ]
            for m in models
        ],
    )
    lines += [
        "",
        "## Main registered per-acquisition results",
        "",
        "All percentages in the proposal columns are ungated correct observations divided by "
        "the initial nonseed N. "
        "Only automatic columns use TRAIN-selected gates. Supplied answers receive no "
        "automatic credit and never shrink N. "
        "Unknown labels remain unscored and break observed spans. Zero accepted agreement is "
        "undefined.",
    ]
    table(
        lines,
        [
            "Run / case / layer",
            "N",
            "Direct correct %",
            "Dense correct %",
            "Interpolation correct %",
            "Correct / wrong automatic",
            "Unresolved",
            "Accepted agreement",
            "Correct / wrong coverage",
        ],
        [
            [
                f"{d['run']} / {d['case_id']} / {d['layer']}",
                d["stages"]["dense_gated"]["fixed_initial_nonseed_N"],
                *[percent(d["stages"][s]["correct_coverage"]) for s in STAGES[:3]],
                f"{d['stages']['dense_gated']['correct_accepted']} / "
                f"{d['stages']['dense_gated']['wrong_accepted']}",
                d["stages"]["dense_gated"]["unresolved"],
                percent(d["stages"]["dense_gated"]["accepted_agreement"]),
                f"{percent(d['stages']['dense_gated']['correct_coverage'])} / "
                f"{percent(d['stages']['dense_gated']['wrong_coverage'])}",
            ]
            for d in details
        ],
    )
    lines += [
        "",
        "## Equal-road primary aggregate",
        "",
        "Base/subbase include all three outer folds. Jamshoro asphalt was not enabled by the "
        "frozen case registry; asphalt has two scored roads.",
    ]
    table(
        lines,
        [
            "Layer / stage",
            "Roads",
            "N pooled",
            "Correct / wrong",
            "Pooled agreement",
            "Pooled correct %",
            "Macro correct %",
        ],
        [
            [
                f"{layer_key} / {s}",
                v["physical_road_count"],
                v["pooled"]["fixed_initial_nonseed_N"],
                f"{v['pooled']['correct_accepted']} / {v['pooled']['wrong_accepted']}",
                percent(v["pooled"]["accepted_agreement"]),
                percent(v["pooled"]["correct_coverage"]),
                percent(v["macro_road"]["correct_coverage"]),
            ]
            for layer_key, a in primary.items()
            for s, v in a["stages"].items()
        ],
    )
    lines += ["", "## Primary versus layer-only"]
    table(
        lines,
        [
            "Road / layer",
            "Primary direct %",
            "Layer-only direct %",
            "Difference pp",
            "Primary dense %",
            "Layer-only dense %",
        ],
        [
            [
                f"{c['road_group']} / {c['layer']}",
                percent(c["primary"]["direct_ungated"]["correct_coverage"]),
                percent(c["layer_only"]["direct_ungated"]["correct_coverage"]),
                f"{c['direct_coverage_difference'] * 100:+.2f}",
                percent(c["primary"]["dense_ungated"]["correct_coverage"]),
                percent(c["layer_only"]["dense_ungated"]["correct_coverage"]),
            ]
            for c in comparisons
        ],
    )
    lines += [
        "",
        "## TRAIN operating gates",
        "",
        "Objective: at least 95% observed agreement on each contributing TRAIN road and at "
        "least 20 accepted episode observations per road. "
        "This is not a confidence bound. Outer precision/coverage curves are diagnostic and "
        "cannot select thresholds.",
    ]
    table(
        lines,
        ["Run / layer", "Direct threshold", "Dense threshold"],
        [
            [
                f"{m['run']} / {layer_key}",
                *[
                    "undefined → abstain"
                    if s[k]["selected_threshold"] is None
                    else f"{s[k]['selected_threshold']:.9g}"
                    for k in ("direct", "dense")
                ],
            ]
            for m in models
            for layer_key, s in m["calibration"].items()
        ],
    )
    lines += [
        "",
        "The registered Mandiali short specimen uses stride 1 while its initial calibration "
        "uses stride 4, so its registered gate is withheld. "
        "Matching stride-1 calibration/results are separately named supplements below; the "
        "main registered result is unchanged.",
        "",
        "## Exact initial analyst workload",
    ]
    table(
        lines,
        [
            "Run / case / layer",
            "Initial observations",
            "Additional actions",
            "Length m",
            "Initial actions/km",
        ],
        [
            [
                f"{d['run']} / {d['case_id']} / {d['layer']}",
                d["initial_workload"],
                d["additional_actions"],
                f"{d['retained_distance_span_m']:.3f}",
                f"{d['initial_actions_per_km']:.3f}",
            ]
            for d in details
        ],
    )
    lines += [
        "",
        "## Seed and context controls",
        "",
        "Original cohort and tolerance are fixed. Changed-target uses alternate reviewed "
        "interface seeds with layer conditioning held fixed; "
        "its original-target score measures change, not successful tracking of the alternate "
        "target. Altered input distributions have no calibrated gate. "
        "Seed-order permutation is the set-conditioning control.",
    ]
    control_rows = []
    for c in sorted(controls, key=lambda v: (v["layer"], v["control"])):
        change, direct = c["conditioning_change"], c["stages"]["direct_ungated"]
        control_rows.append(
            [
                f"{c['layer']} / {c['control']}",
                direct["fixed_initial_nonseed_N"],
                percent(direct["correct_coverage"]),
                percent(c["stages"]["dense_ungated"]["correct_coverage"]),
                percent(change["native_depth_argmax_changed_fraction"]) if change else "baseline",
                f"{change['mean_absolute_argmax_shift_samples']:.3f}" if change else "baseline",
                direct["manual_answers_in_initial_cohort"],
            ]
        )
    table(
        lines,
        [
            "Layer / control",
            "N",
            "Direct original-target correct %",
            "Dense correct %",
            "Argmax changed %",
            "Mean shift samples",
            "Manual rows without credit",
        ],
        control_rows,
    )
    lines += [
        "",
        "## Exact fitting support and exclusions",
        "",
        "Counts describe coordinate-valid stored interpretations, not independent human "
        "judgments. Source annotation production history is partly unknown. "
        "The JSON report includes each source/label hash and path, buffered split count, and "
        "excluded record.",
    ]
    support_groups = defaultdict(list)
    for m in models:
        key = hashlib.sha256(
            json.dumps(
                [m["fitting_support"], m["excluded_training_records"]], sort_keys=True
            ).encode()
        ).hexdigest()
        support_groups[key].append(m)
    for members in support_groups.values():
        m = members[0]
        lines += [
            "",
            f"### Outer road: {m['held_out_group']}",
            "",
            "Applies to: " + ", ".join(f"`{v['run']}`" for v in members) + ".",
            "",
            "Training groups: " + ", ".join(m["training_road_groups"]) + ".",
        ]
        table(
            lines,
            [
                "Record / layer",
                "Available labels",
                "Fitting",
                "Inner validation",
                "Outside buffered splits",
            ],
            [
                [
                    f"{s['record_id']} / {s['layer']}",
                    s["available_coordinate_valid_labels"],
                    s["train_observations"],
                    s["inner_validation_observations"],
                    s["outside_buffered_splits"],
                ]
                for s in m["fitting_support"]
            ],
        )
        lines += ["", "Additional exclusions beyond the entire outer road:"]
        table(
            lines,
            ["Record", "Asphalt / base / subbase labels excluded", "Reason"],
            [
                [
                    e["record_id"],
                    " / ".join(
                        str(e["label_counts"].get(str(layer_key), 0)) for layer_key in (1, 2, 3)
                    ),
                    e["reason"],
                ]
                for e in m["excluded_training_records"]
            ],
        )
    if supplements:
        lines += [
            "",
            "## Separately named supplements",
            "",
            "These do not replace registered main results.",
        ]
        for extra in supplements:
            lines += ["", f"### {extra['evaluation_run']}"]
            table(
                lines,
                [
                    "Case / layer / stage",
                    "N",
                    "Correct / wrong accepted",
                    "Agreement",
                    "Correct coverage",
                ],
                [
                    [
                        f"{r['case_id']} / {r['layer']} / {r['stage']}",
                        r["fixed_initial_nonseed_N"],
                        f"{r['correct_accepted']} / {r['wrong_accepted']}",
                        percent(r["accepted_agreement"]),
                        percent(r["correct_coverage"]),
                    ]
                    for r in extra["rows"]
                ],
            )
    lines += [
        "",
        "## Evidence limits",
        "",
        "The deep proposal gain is measurable, but every deep operating gate failed the TRAIN "
        "objective. Reliable automatic deep coverage is not demonstrated. "
        "Native timing errors, bracket/tail and seed-distance strata, and longest wrong "
        "observed spans remain in the referenced case details. "
        "These initial prediction results do not substitute for separately reported correction "
        "replay. No source/model/data/scoring code was changed to produce this report.",
        "",
        f"Artifact root: `{root}`. Exact commands/exit codes: `ledger.jsonl`. "
        "Reproduction requires each job's immutable source snapshot and recorded model hashes.",
        "",
    ]
    numerical_path = root / "report/calibration-precision-comparison.json"
    numerical = read(numerical_path) if numerical_path.exists() else []
    if numerical:
        sources[str(numerical_path)] = sha(numerical_path)
        facts["fp32_threshold_comparisons"] = len(numerical)
        facts["fp32_changed_thresholds"] = sum(
            r["previous_threshold"] != r["fp32_threshold"] for r in numerical
        )
        facts["fp32_all_cohorts_identical"] = all(r["same_cohort"] for r in numerical)
        facts["fp32_null_deep_gates"] = sum(
            r["layer"] in (2, 3) and r["fp32_threshold"] is None for r in numerical
        )
        lines += ["", "## Arithmetic-consistent calibration supplement", "",
                  "Initial calibration used FP16 inference while the frozen road predictor used "
                  "FP32. Twelve separately saved FP32 calibrations reused identical TRAIN episode "
                  f"cohorts. Of {len(numerical)} threshold comparisons, "
                  f"{facts['fp32_changed_thresholds']} changed; "
                  f"{facts['fp32_null_deep_gates']} deep entries remained undefined. "
                  "The recorded automatic acceptance outcomes therefore remain unchanged. "
                  "Original evidence is retained; new numerical-consistency evidence is "
                  "calibration-precision-comparison.json.", ""]
    document = {
        "schema": "processed-campaign-completed-report-v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "facts": facts,
        "models": models,
        "rows": rows,
        "run_aggregates": aggregates,
        "primary_aggregates": primary,
        "primary_layer_only_comparisons": comparisons,
        "per_case_details": details,
        "seed_controls": controls,
        "supplements": supplements,
        "input_artifact_sha256": sources,
        "reporter_sha256": sha(__file__),
        "production_promotion": False,
        "numerical_calibration_consistency": numerical,
    }
    for p, expected in sources.items():
        if sha(p) != expected:
            raise ValueError(f"Completed artifact changed during reporting: {p}")
    preserve_write(
        output / "completed-results.json", json.dumps(document, indent=2, allow_nan=False)
    )
    preserve_write(output / "completed-results.md", "\n".join(lines))
    print(
        json.dumps(
            {
                **facts,
                "seed_controls": len(controls),
                "supplements": len(supplements),
                "report": str(output / "completed-results.md"),
            }
        )
    )
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)
