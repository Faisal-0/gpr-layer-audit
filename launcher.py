# ruff: noqa: I001
from __future__ import annotations

import os
import sys
from pathlib import Path


_qt_dll_directory = None
if getattr(sys, "frozen", False) and sys.platform == "win32":
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    qt_runtime = bundle_root / "PySide6"
    if qt_runtime.is_dir():
        # Keep the handle alive for the entire process. This makes QtCore load
        # its matching Qt 6.11 DLLs instead of a conflicting copy on PATH.
        _qt_dll_directory = os.add_dll_directory(str(qt_runtime))
        os.environ["PATH"] = str(qt_runtime) + os.pathsep + os.environ.get("PATH", "")

from gpr_layer_audit.ui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
