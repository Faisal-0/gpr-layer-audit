"""Reconstruct every saved local training label, coordinate and source partner."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_local_patch_pairs import eligible_pairs
from patchnet_model import patches
from seeded_eval import sha
from gpr_layer_audit.conventional import _same_lobe
from gpr_layer_audit.io.dzt import DZTFile
from gpr_layer_audit.io.dzx import read_dzx
from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask

folder = Path(__file__).parent
meta = json.loads((folder / "training-data.json").read_text())
assert sha(folder / "training.npz") == meta["training_sha256"]
train = np.load(folder / "training.npz")
coords = np.load(folder / "training-coordinates.npz")["pairs"]
y, partners, sources = train["y"], train["partner"], train["seeds"]
offset = 0
reports = []
for i, report in enumerate(meta["reports"]):
    entry = report["input"]
    assert entry["road"] == "mandiali"
    radar = DZTFile(entry["dzt"])
    data = np.asarray(radar.channel(), np.float32)
    valid = processed_boundary_mask(data)
    layer = next(l for l in read_dzx(entry["dzx"]).layers if l.number + 1 == report["layer"])
    truth = {p.trace: p.sample for p in layer.picks if p.channel == 0}
    operating = {int(r): s for r, s in entry["seeds"][str(report["layer"])].items()}
    lag = report["lag_traces"]
    eligible = eligible_pairs(truth, operating, lag)
    if len(eligible) > 3000:
        eligible = eligible[np.linspace(0, len(eligible)-1, 3000, dtype=int)]
    use = np.flatnonzero(coords[:, 0] == i)
    rr, ss, prior = coords[use, 1:].T
    assert np.all(rr-prior == lag)
    assert not set(rr) & set(operating)
    assert all(r in truth and p in truth for r, p in zip(rr, prior))
    np.testing.assert_array_equal(eligible[partners[use]-offset], rr)
    yy = np.array([[abs(s-truth[r]) <= report["tolerance_samples"] and _same_lobe(data[r], s, truth[r]),
                    _same_lobe(data[r], s, truth[r])] for r,s in zip(rr,ss)], np.float32)
    np.testing.assert_array_equal(yy, y[use])
    expected, usable = patches(data, valid, eligible-lag,
                              np.array([truth[r-lag] for r in eligible]), entry["dt_ns"], entry["dx_m"])
    np.testing.assert_array_equal(expected, sources[offset:offset+len(eligible)])
    assert usable[partners[use]-offset].all()
    offset += len(eligible)
    reports.append({"layer":report["layer"], "file":entry["dzx"], "pairs_verified":len(use),
                    "positive_timing":int(yy[:,0].sum()), "source_patches_verified":len(eligible)})
assert offset == len(sources)
result = {"all_labels_coordinates_and_source_partners_exact":True, "pairs":len(y),
          "source_patches":len(sources), "reports":reports, "audit_source_sha256":sha(__file__),
          "claim":"Training integrity only; no tracking accuracy established"}
(folder / "training-audit.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
