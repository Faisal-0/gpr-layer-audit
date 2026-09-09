"""One controlled negative-sampling repair; architecture, seeds and inference stay fixed.

Replace three of six nearby timing negatives with three far candidates most
similar to the same immutable seed packet. All labels used here are Mandiali
training observations; inference receives only the resulting model and seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import shutil
from importlib.metadata import version
from pathlib import Path

import experiment_patchnet_dense as base
import numpy as np
from seeded_eval import source_manifest


def negative_indices(candidates, targets, reference, width, correlations):
    eligible = np.flatnonzero(targets[:, 0] == 0)
    nearby = eligible[abs(candidates[eligible] - reference) <= 4 * width]
    nearby = nearby[np.argsort(abs(candidates[nearby] - reference))[:3]]
    far = eligible[abs(candidates[eligible] - reference) > 4 * width]
    far = far[np.argsort(-correlations[far], kind="stable")[:3]]
    selected = np.r_[nearby, far]
    if len(selected) < 6:
        remaining = eligible[~np.isin(eligible, selected)]
        remaining = remaining[np.argsort(abs(candidates[remaining] - reference))]
        selected = np.r_[selected, remaining[: 6 - len(selected)]]
    return selected


def dependency_contract(workspace):
    package = source_manifest(workspace / "src")
    return {
        "package": {k: package[k] for k in ("sha256", "python_files")},
        "scripts": {
            name: base.sha(Path(base.__file__).with_name(name))
            for name in ("patchnet_model.py", "seeded_eval.py")
        },
        "runtime": {name: version(name) for name in ("numpy", "scipy", "torch")},
    }


def prepared_source():
    code = inspect.getsource(base.prepare)
    changes = [
        (
            "            tolerance = max(2, pulse.lobe_samples / 4)\n",
            "            tolerance = max(2, pulse.lobe_samples / 4)\n"
            "            seed_bank, _ = dense._packet_correlations(\n"
            "                data, valid, anchors, pulse.context_radius)\n",
        ),
        (
            "                negative = negative["
            "np.argsort(abs(candidates[negative] - point.sample))[:6]]\n",
            "                partner_seed = min(anchors, key=lambda r: (abs(r-point.trace), r))\n"
            "                negative = _negative_indices(candidates, target, point.sample,\n"
            "                    pulse.lobe_samples, "
            "seed_bank[partner_seed][point.trace, candidates])\n",
        ),
    ]
    for before, after in changes:
        if code.count(before) != 1:
            raise ValueError("Review changed training sampler before instrumentation")
        code = code.replace(before, after)
    return code


def run(mode, output, workspace):
    code = prepared_source()
    base.ROOT, base.OUT = workspace.resolve(), output.resolve()
    base.MANIFEST = base.ROOT / "benchmarks/seeded-evaluation-inputs.json"
    base.CONTRACT = base.CONTRACT | {
        "version": "native-patchnet-dense-v3-far-negatives",
        "negative_sampling": (
            "Three nearest timing misses plus three far (>4 initial lobe widths) "
            "candidates with highest signed NCC to the same nearest operating seed; "
            "fill any shortage from remaining nearest negatives"
        ),
        "selection": "Same fixed 12 epochs and architecture; one negative-coverage comparison",
        "base_runner_sha256": base.sha(base.__file__),
        "prepare_function_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "sampling_runner_sha256": base.sha(__file__),
        "previous_use": "Jamshoro diagnosis informed this training change; development only",
        "dependencies": dependency_contract(base.ROOT),
    }
    if mode == "prepare":
        namespace = dict(base.__dict__, _negative_indices=negative_indices, __file__=__file__)
        exec(compile(code, "<far-negative-training-prepare>", "exec"), namespace)
        namespace["prepare"]()
        source = base.OUT / "source"
        (source / "executed-prepare.py").write_text(code, encoding="utf-8")
        (source / "experiment_patchnet_dense.py").write_bytes(Path(base.__file__).read_bytes())
        shutil.copy2(Path(base.__file__).with_name("seeded_eval.py"), source / "seeded_eval.py")
        for relative in base.CONTRACT["dependencies"]["package"]["python_files"]:
            destination = source / "package" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base.ROOT / "src" / relative, destination)
        base.write(
            base.OUT / "sampling-source.json",
            {
                "scripts": {p.name: base.sha(p) for p in source.iterdir() if p.is_file()},
                "package": base.CONTRACT["dependencies"]["package"],
                "base_commit": "c442b60ad5f519564893a8da4f824b9327a7f195",
                "training_only": True,
            },
        )
    else:
        contract = base.read(base.OUT / "contract.json")
        if contract != base.CONTRACT:
            raise ValueError("Preserve the prepared sampler/architecture contract")
        getattr(base, mode)()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "predict", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=base.ROOT)
    args = parser.parse_args()
    run(args.mode, args.output, args.workspace_root)
