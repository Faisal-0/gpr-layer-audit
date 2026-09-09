"""Copy compact, completed development evidence into a reviewable package.

Original exports are never changed. The index records source and packaged file
hashes separately when a large JSON is reduced to its summary fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = "exports/seeded-tracker/"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_lf_sha(path: Path) -> str:
    """Permit only CRLF-to-LF conversion, preserving every other source byte."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def endpoint_summary(value: dict) -> dict:
    result = {k: v for k, v in value.items() if k not in ("endpoint2", "distributed3")}
    for side in ("endpoint2", "distributed3"):
        result[side] = {
            k: v
            for k, v in value[side].items()
            if k not in ("common_three_seed_scoring", "operational_pulse_scoring")
        }
    result["packaged_summary_contract"] = (
        "All comparison cells retained unchanged. Per-observation repetitions and each arm's "
        "separate operational scoring remain in the hashed original."
    )
    return result


def workflow_summary(value: dict) -> dict:
    result = {k: v for k, v in value.items() if k != "metrics"}
    result["metrics"] = []
    for step in value["metrics"]:
        reduced = {k: v for k, v in step.items() if k not in ("layers", "application_layers")}
        for name in ("layers", "application_layers"):
            reduced[name] = {
                layer: {k: v for k, v in metrics.items() if k != "evaluation_observations"}
                for layer, metrics in step[name].items()
            }
        result["metrics"].append(reduced)
    result["packaged_summary_contract"] = (
        "Per-step summaries are unchanged; per-observation rows remain in the hashed original. "
        "application_layers scores finite application output. The legacy reflector_switches "
        "field counts wrong signed-lobe observations, not independent physical switch events. "
        "Additional observations and GUI warning confirmations are separate recorded actions."
    )
    return result


def results_summary(value: dict) -> dict:
    """Hoist repeated provenance without dropping any result or metric cell."""
    metadata = (
        "artifact_sha256",
        "method",
        "physical_road_group",
        "source_dzt_sha256",
        "backend_source_sha256",
        "source_dzx_sha256",
        "seeded_evaluation",
        "config",
        "config_canonical_sha256",
        "working_step_m",
        "dt_ns",
        "runtime_environment_note",
    )
    result = {k: v for k, v in value.items() if k != "results"}
    provenance = {}
    for row in value["results"]:
        artifact = row["artifact"]
        if artifact in provenance:
            continue
        group = [other for other in value["results"] if other["artifact"] == artifact]
        provenance[artifact] = {
            key: row[key]
            for key in metadata
            if key in row and all(key in other and other[key] == row[key] for other in group)
        }
    result["shared_result_provenance"] = provenance
    result["results"] = [
        {key: item for key, item in row.items() if key not in provenance[row["artifact"]]}
        for row in value["results"]
    ]
    restored = [{**provenance[row["artifact"]], **row} for row in result["results"]]
    if restored != value["results"]:
        raise ValueError("Result provenance compaction changed a comparison cell")
    result["packaged_summary_contract"] = (
        "Every original result cell is retained. Repeated identical metadata is hoisted to "
        "shared_result_provenance by artifact; merging it into each result restores the original."
    )
    return result


def interruption_summary(value: dict) -> dict:
    return {
        "schema": "seeded-interrupted-evidence-v1",
        "supplemental_control": value["interrupted_supplemental_control"],
        "other_excluded_runs": value["incomplete_excluded"],
        "claim": "Excluded incomplete runs have no scored result or completion claim.",
    }


def recipes() -> list[tuple[str, str, str]]:
    items = [
        ("benchmarks/seeded-mechanism-ledger.json", "mechanisms.json", "copy"),
        ("benchmarks/seeded-evaluation-inputs.json", "original-input-manifest.json", "copy"),
        ("benchmarks/seeded-label-confirmation.json", "layer-confirmation.json", "copy"),
        ("benchmarks/seeded-mechanism-ledger.json", "interrupted-runs.json", "interruptions"),
    ]
    exports = [
        ("runtime-environment-current.json", "integration-runtime-environment.json", "copy"),
        ("workflow-latency-diagnosis-v1.json", "workflow-latency-diagnosis.json", "copy"),
        (
            "additional-inputs/supplemental-evaluation-inputs-v1.json",
            "supplemental-input-manifest.json",
            "copy",
        ),
        (
            "additional-baseline/bahawalpur-complete-metrics.json",
            "bahawalpur-controls.json",
            "results",
        ),
        (
            "additional-baseline/mandiali-long-metrics.json",
            "mandiali-long-control.json",
            "results",
        ),
        (
            "additional-compiled-control-v1/gujrat-first-metrics.json",
            "gujrat-first-compiled-control.json",
            "results",
        ),
        (
            "additional-compiled-control-v1/gujrat-first-radar.png",
            "gujrat-first-compiled-control.png",
            "copy",
        ),
        ("additional-compiled-control-v1/job.json", "gujrat-first-compiled-job.json", "copy"),
        (
            "additional-compiled-source-v1/provenance.json",
            "gujrat-first-compiled-source.json",
            "copy",
        ),
        (
            "additional-baseline/runtime-environment-note.json",
            "runtime-environment-note.json",
            "copy",
        ),
        (
            "additional-baseline/wrapper-provenance-note.json",
            "wrapper-provenance-note.json",
            "copy",
        ),
        ("additional-baseline/mandiali-long-radar.png", "mandiali-long-control.png", "copy"),
        ("additional-baseline/bahawalpur-first-radar.png", "bahawalpur-first-control.png", "copy"),
        (
            "additional-baseline/bahawalpur-second-radar.png",
            "bahawalpur-second-control.png",
            "copy",
        ),
        ("endpoint-common/mandiali-common-score-v1.json", "endpoint-mandiali.json", "endpoint"),
        ("endpoint-common/gujrat-common-score-v1.json", "endpoint-gujrat.json", "endpoint"),
        ("endpoint-common/durable-script-parity-v1.json", "endpoint-script-parity.json", "copy"),
        (
            "endpoint-common/endpoint2-versus-distributed3-v1.json",
            "endpoint-figure-provenance.json",
            "copy",
        ),
        ("endpoint-common/endpoint2-versus-distributed3-v1.png", "endpoint-seeding.png", "copy"),
        (
            "interaction-prospective/mandiali-four-action-comparison.json",
            "mandiali-interaction.json",
            "results",
        ),
        (
            "interaction-prospective/mandiali-four-action-comparison-mandiali-short.png",
            "mandiali-action-curve.png",
            "copy",
        ),
        (
            "interaction-prospective/mandiali-fixed-policy-before-after.png",
            "mandiali-before-after.png",
            "copy",
        ),
        (
            "interaction-prospective/gujrat-four-action-comparison.json",
            "gujrat-interaction.json",
            "results",
        ),
        (
            "interaction-prospective/gujrat-four-action-comparison-gujrat-second.png",
            "gujrat-action-curve.png",
            "copy",
        ),
        (
            "interaction-prospective/gujrat-fixed-policy-before-after.png",
            "gujrat-before-after.png",
            "copy",
        ),
        (
            "interaction-prospective/scope-comparison/one-answer-scope-metrics.json",
            "mandiali-scope.json",
            "results",
        ),
        (
            "interaction/prior-observed-steps-fixed-denominator.json",
            "prior-interaction-failures.json",
            "results",
        ),
        ("interaction/prior-active-before-after.png", "prior-interaction-failure.png", "copy"),
        (
            "identity/gujrat-geometry-failure-atlas.json",
            "gujrat-geometry-failure-atlas.json",
            "copy",
        ),
        ("identity/gujrat-control-vs-geometry.png", "rejected-geometry-radar.png", "copy"),
        ("learning/verified/result.json", "rejected-learning.json", "copy"),
        ("learning/DECISION.md", "rejected-learning-decision.md", "copy"),
        ("edge-gate-loss/decision.json", "rejected-lobe-mutual.json", "copy"),
        ("edge-gate-loss/DECISION.md", "rejected-lobe-mutual-decision.md", "copy"),
        ("edge-gate-loss/comparison.json", "rejected-lobe-mutual-metrics.json", "results"),
        ("scoped-queries/readout.md", "rejected-scoped-query-readout.md", "copy"),
        ("scoped-queries/verdict.json", "rejected-scoped-query-verdict.json", "copy"),
        ("scoped-queries/gujrat-comparison.json", "rejected-scoped-query-metrics.json", "results"),
        (
            "scoped-queries/gujrat-comparison-gujrat-second.png",
            "rejected-scoped-query-curve.png",
            "copy",
        ),
        ("scoped-queries/regression.json", "scoped-query-regression.json", "copy"),
        (
            "scoped-queries/array-contract-verification.json",
            "scoped-query-array-verification.json",
            "copy",
        ),
        ("scoped-queries/actions.json", "scoped-query-actions.json", "copy"),
        ("timing-identity/FINDINGS.md", "timing-identity-findings.md", "copy"),
        ("timing-identity/summary.json", "timing-identity-summary.json", "copy"),
        ("timing-identity/extremum-basins.json", "timing-extremum-basins.json", "copy"),
        ("timing-identity/missing-peak-cap-loss.json", "timing-missing-peak-cap-loss.json", "copy"),
        ("dtw-acceleration/application-parity.json", "compiled-application-parity.json", "copy"),
        ("dtw-acceleration/gujrat-control-parity.json", "compiled-gujrat-parity.json", "copy"),
        ("dtw-acceleration/benchmark.json", "compiled-packet-benchmark.json", "copy"),
        ("dtw-acceleration/DECISION.md", "compiled-runtime-decision.md", "copy"),
        ("dtw-acceleration/NUMBA-LICENSE.txt", "NUMBA-LICENSE.txt", "copy"),
        ("block-uncertainty/paired-block-report.json", "paired-block-uncertainty.json", "copy"),
        ("block-uncertainty/verification.json", "paired-block-verification.json", "copy"),
        (
            "block-uncertainty/paired-block-report-v3.json",
            "paired-block-uncertainty-v3.json",
            "copy",
        ),
        ("block-uncertainty/verification-v3.json", "paired-block-verification-v3.json", "copy"),
        (
            "block-uncertainty/legacy-source/seeded_eval_blocks.py",
            "legacy-block-generator.py.txt",
            "copy",
        ),
        ("extremum-states/FINDINGS.md", "rejected-extremum-findings.md", "copy"),
        ("extremum-states/self-test.json", "extremum-numerical-contracts.json", "copy"),
        (
            "extremum-states/mandiali-short/comparison.json",
            "rejected-extremum-comparison.json",
            "copy",
        ),
        (
            "extremum-states/mandiali-short/objective-audit.json",
            "extremum-objective-audit.json",
            "copy",
        ),
        ("extremum-states/artifact-manifest.json", "extremum-artifact-manifest.json", "copy"),
        (
            "workflow-verification-action-history-v2/verification.json",
            "final-gui-workflow.json",
            "workflow",
        ),
        (
            "workflow-verification-action-history-v2/coverage-versus-intervention.png",
            "final-gui-action-curve.png",
            "copy",
        ),
        (
            "workflow-verification-action-history-v2/before-after-deep-radar.png",
            "final-gui-before-after.png",
            "copy",
        ),
        ("workflow-action-history-parity-v2.json", "final-workflow-path-parity.json", "copy"),
        ("final-integration-v1.json", "final-integration.json", "copy"),
        (
            "workflow-verification-compiled-v2/verification.json",
            "compiled-gui-workflow.json",
            "workflow",
        ),
        ("workflow-verification-20260909-v3/verification.json", "gui-workflow.json", "workflow"),
        (
            "workflow-verification-20260909-v3/coverage-versus-intervention.png",
            "gui-action-curve.png",
            "copy",
        ),
        (
            "workflow-verification-20260909-v3/before-after-deep-radar.png",
            "gui-before-after.png",
            "copy",
        ),
    ]
    items.extend(
        (EXPORTS + source, destination, transform) for source, destination, transform in exports
    )
    return items


def verify_package(output: Path, *, verify_originals: bool = True) -> dict:
    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    failures = []
    names = {"index.json", ".gitattributes"}
    for entry in index["files"]:
        name = Path(entry["file"])
        if name.is_absolute() or len(name.parts) != 1 or str(name) in names:
            failures.append(f"Invalid or duplicate package filename: {entry['file']}")
            continue
        names.add(str(name))
        packaged = output / name
        if packaged.resolve().parent != output.resolve():
            failures.append(f"Package file resolves outside package: {name}")
            continue
        checks = [(packaged, entry["sha256"])]
        if verify_originals:
            checks.append((ROOT / entry["original"], entry["original_sha256"]))
        for path, expected in checks:
            if not path.is_file() or sha(path) != expected:
                failures.append(str(path))
        if packaged.is_file() and packaged.stat().st_size != entry["bytes"]:
            failures.append(f"Size mismatch: {packaged}")
    attributes = output / ".gitattributes"
    if not attributes.is_file() or sha(attributes) != index["attributes_sha256"]:
        failures.append("Package byte-preservation attributes changed")
    script_check = "exact_bytes"
    if sha(Path(__file__)) != index["packaging_script_sha256"]:
        if not verify_originals and normalized_lf_sha(Path(__file__)) == index.get(
            "packaging_script_normalized_lf_sha256"
        ):
            script_check = "recorded_crlf_to_lf_companion"
        else:
            failures.append("Packaging script has changed")
    if failures:
        raise ValueError(f"Package verification failed: {failures}")
    return {
        "verified": True,
        "mode": "package_and_originals" if verify_originals else "package_only",
        "files": len(index["files"]),
        "output": str(output),
        "packaged_sizes_hashes_and_attributes_verified": True,
        "packaging_script_hash_check": script_check,
        "original_file_hashes_verified": verify_originals,
        "inference_reproduced": False,
        "independent_provenance_validation": False,
        "limitation": (
            "Original file hashes checked; inference and embedded provenance were not "
            "independently reproduced or audited."
            if verify_originals
            else "Only package bytes and the verifier match the index. Original sources, "
            "inference, and embedded provenance remain unverified."
        ),
    }


def test_package_only(output: Path) -> dict:
    """Check a source-free copy, line-ending portability, and adversarial edits."""
    verify_package(output, verify_originals=False)
    isolated = Path(tempfile.mkdtemp(prefix="seeded-package-check-"))
    script = isolated / "scripts/package_seeded_results.py"
    script.parent.mkdir()
    shutil.copyfile(Path(__file__), script)
    package = isolated / "benchmarks/seeded-results"
    shutil.copytree(output, package)
    if (isolated / "exports").exists():
        raise ValueError("Portability test must have no original exports")

    def run(flag: str, succeeds: bool) -> dict:
        completed = subprocess.run(
            [sys.executable, str(script), "--output", str(package), flag],
            cwd=isolated,
            capture_output=True,
            text=True,
            check=False,
        )
        if (completed.returncode == 0) != succeeds:
            raise AssertionError(f"Unexpected {flag} outcome: {completed.stderr}")
        if succeeds:
            return json.loads(completed.stdout)
        if "Package verification failed" not in completed.stderr:
            raise AssertionError(f"Expected a hash-verification rejection: {completed.stderr}")
        return {"rejected": True}

    initial = run("--verify-package-only", True)
    if initial["original_file_hashes_verified"] or initial["inference_reproduced"]:
        raise AssertionError("Package-only mode must not claim original or inference verification")
    strict = run("--verify", False)
    metric = package / "gujrat-interaction.json"
    saved_metric = metric.read_bytes()
    corrupted = saved_metric.replace(b'"accepted": 547', b'"accepted": 548', 1)
    if corrupted == saved_metric or len(corrupted) != len(saved_metric):
        raise AssertionError("Expected a same-size adversarial metric edit")
    metric.write_bytes(corrupted)
    changed_metric = run("--verify-package-only", False)
    metric.write_bytes(saved_metric)
    saved_attributes = (package / ".gitattributes").read_bytes()
    (package / ".gitattributes").write_bytes(saved_attributes + b"# changed\n")
    changed_attributes = run("--verify-package-only", False)
    (package / ".gitattributes").write_bytes(saved_attributes)
    source_lf = script.read_bytes().replace(b"\r\n", b"\n")
    script.write_bytes(source_lf)
    lf = run("--verify-package-only", True)
    script.write_bytes(source_lf.replace(b"\n", b"\r\n"))
    crlf = run("--verify-package-only", True)
    script.write_bytes(source_lf + b"\n# Unrecorded verifier edit\n")
    changed_script = run("--verify-package-only", False)
    shutil.copyfile(Path(__file__), script)
    run("--verify-package-only", True)
    return {
        "passed": True,
        "isolated_directory": str(isolated),
        "original_exports_present": False,
        "source_free_package_verification": initial,
        "strict_mode_without_originals": strict,
        "same_size_corrupted_metric": changed_metric,
        "changed_attributes": changed_attributes,
        "lf_script": lf["packaging_script_hash_check"],
        "crlf_script": crlf["packaging_script_hash_check"],
        "arbitrary_script_edit": changed_script,
        "note": "Isolated test copy retained for inspection; no original evidence was changed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/seeded-results")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Report missing inputs without writing")
    mode.add_argument("--verify", action="store_true", help="Verify all packaged and source hashes")
    mode.add_argument(
        "--verify-package-only",
        action="store_true",
        help="Verify package bytes only; original sources and inference remain unverified",
    )
    mode.add_argument(
        "--self-test-package-only",
        action="store_true",
        help="Check a source-free copy and reject corrupted metrics, attributes, or verifier edits",
    )
    args = parser.parse_args()
    if args.self_test_package_only:
        print(json.dumps(test_package_only(args.output), indent=2))
        return
    if args.verify or args.verify_package_only:
        print(json.dumps(verify_package(args.output, verify_originals=args.verify), indent=2))
        return
    items = recipes()
    missing = [source for source, _, _ in items if not (ROOT / source).is_file()]
    if args.check:
        print(json.dumps({"ready": not missing, "missing": missing}, indent=2))
        return
    if missing:
        raise ValueError(f"Missing completed source artifacts: {missing}")
    if args.output.exists():
        raise ValueError("Preserve an existing package; choose a new output directory")
    ledger = json.loads((ROOT / "benchmarks/seeded-mechanism-ledger.json").read_text())
    interruption = ledger["interrupted_supplemental_control"]
    for source in interruption["missing_completion_artifacts"]:
        if (ROOT / source).exists():
            raise ValueError(f"Interruption record needs review; artifact now exists: {source}")
    for relative in (
        "interaction-prospective/mandiali-four-action-comparison.json",
        "interaction-prospective/gujrat-four-action-comparison.json",
        "scoped-queries/gujrat-comparison.json",
    ):
        summary = json.loads((ROOT / EXPORTS / relative).read_text())
        if any(
            p["stop_reason"] != "Requested action budget completed" for p in summary["provenance"]
        ):
            raise ValueError(f"Action budget did not complete: {relative}")
    for relative in (
        "workflow-verification-action-history-v2/verification.json",
        "workflow-verification-compiled-v2/verification.json",
        "dtw-acceleration/application-parity.json",
        "dtw-acceleration/gujrat-control-parity.json",
    ):
        if json.loads((ROOT / EXPORTS / relative).read_text())["success"] is not True:
            raise ValueError(f"Verification did not pass: {relative}")
    entries = []
    args.output.mkdir(parents=True)
    attributes = args.output / ".gitattributes"
    attributes.write_bytes(
        b"# Preserve evidence bytes and CRLF line terminators across Git checkouts.\n"
        b"* -text whitespace=cr-at-eol\n"
    )
    for source, destination, transform in items:
        source_path = ROOT / source
        destination_path = args.output / destination
        source_hash = sha(source_path)
        if transform == "copy":
            shutil.copyfile(source_path, destination_path)
        else:
            value = json.loads(source_path.read_text())
            reduced = {
                "endpoint": endpoint_summary,
                "workflow": workflow_summary,
                "results": results_summary,
                "interruptions": interruption_summary,
            }[transform](value)
            destination_path.write_text(
                json.dumps(reduced, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        if sha(source_path) != source_hash:
            raise ValueError(f"Source changed during packaging: {source}")
        entries.append(
            {
                "file": destination,
                "bytes": destination_path.stat().st_size,
                "sha256": sha(destination_path),
                "original": source,
                "original_sha256": source_hash,
                "transformation": transform,
            }
        )
    index = {
        "schema": "seeded-results-evidence-package-v1",
        "claim": "Previously used processed-road interpretation development evidence only",
        "limitations": [
            "No untouched-road, independently verified raw-coordinate, "
            "or physical-thickness accuracy claim",
            "No tested configuration meets useful 95% deep-interface agreement across these roads",
            "Coverage fractions use eligible reviewed observations; unknown labels are unscored",
            "Neighboring observations are correlated; "
            "no pointwise IID uncertainty interval is claimed",
            "Replay reveals positive reviewed observations only; missing picks do not mean absent",
            "Runtime notes distinguish Windows suspension from compute and interactive latency",
            "Original exports and hashes preserve the full inference and scoring provenance",
            "Original scalar Gujrat-first launch was interrupted without result/arrays; "
            "the separately identified compiled control completed",
            "Compiled DTW preserves tested outputs; measured workflow latency "
            "remains tens of seconds",
            "Rejected lobe-mutual and learned/geometry experiments are retained without promotion",
            "Scoped-query ranking is rejected; three of four requests had no reviewed answer. "
            "Only the exact 25 m configuration is verified",
            "The implemented extremum experiment is rejected: five wrong deep acceptances "
            "remain and correct acceptance does not increase on the lobe-mutual comparator",
            "New correction snapshots preserve historical model-seed context; legacy projects "
            "without snapshots cannot reconstruct lost per-action observations",
        ],
        "packaging_script": "scripts/package_seeded_results.py",
        "packaging_script_sha256": sha(Path(__file__)),
        "packaging_script_normalized_lf_sha256": normalized_lf_sha(Path(__file__)),
        "packaging_script_hash_contract": (
            "Full --verify requires exact script bytes. --verify-package-only also permits "
            "the recorded companion hash after CRLF-to-LF conversion only; no other "
            "whitespace, encoding, or code changes are permitted."
        ),
        "attributes_sha256": sha(attributes),
        "attributes_contract": "The generated .gitattributes disables package newline conversion.",
        "files": entries,
    }
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verify_package(args.output), indent=2))


if __name__ == "__main__":
    main()
