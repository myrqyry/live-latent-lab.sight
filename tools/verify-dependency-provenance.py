#!/usr/bin/env python3
"""Verify reviewed vendored dependency bytes against the recorded receipt."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
receipt = json.loads((root / "docs/qualification/dependency-provenance-20261126.json").read_text())
files = receipt["onnxruntime"]["files"] | receipt["libfacedetection"]["files"]
for name, expected in files.items():
    path = root / name
    if not path.is_file():
        raise SystemExit(f"Missing reviewed file: {name}")
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise SystemExit(f"Dependency drift: {name}; re-review provenance")
print(f"Verified {len(files)} reviewed vendor files; not redistribution approval")
