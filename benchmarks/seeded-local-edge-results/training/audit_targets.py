"""Verify the complete target-patch tensor against recorded native coordinates."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from patchnet_model import patches
from seeded_eval import sha
from gpr_layer_audit.io.dzt import DZTFile
from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask
folder = Path(__file__).parent
meta = json.loads((folder / "training-data.json").read_text())
assert sha(folder / "training.npz") == meta["training_sha256"]
x = np.load(folder / "training.npz")["x"]
coordinates = np.load(folder / "training-coordinates.npz")["pairs"]
verified = 0
for index, report in enumerate(meta["reports"]):
    entry = report["input"]
    assert sha(entry["dzt"]) == entry["dzt_sha256"]
    data = np.asarray(DZTFile(entry["dzt"]).channel(), np.float32)
    valid = processed_boundary_mask(data)
    chosen = np.flatnonzero(coordinates[:, 0] == index)
    for start in range(0, len(chosen), 4096):
        use = chosen[start:start+4096]
        expected, usable = patches(data, valid, coordinates[use, 1], coordinates[use, 2],
                                   entry["dt_ns"], entry["dx_m"])
        assert usable.all()
        np.testing.assert_array_equal(expected, x[use])
        verified += len(use)
assert verified == len(x)
result = {"target_patches_exact": verified, "script_sha256":sha(__file__),
          "claim":"Training tensor integrity only; no tracker performance claim"}
(folder / "target-patch-audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
