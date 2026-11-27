#!/usr/bin/env python3
"""Convert a verified Linux install stage to the version-specific OBS layout."""
from pathlib import Path
import sys

def stage(package, layout):
    directory = package / "bin/64bit"
    module = directory / "live-latent-lab-sight.so"
    if not module.is_file() or not module.stat().st_size:
        raise SystemExit("Missing staged Sight module")
    entries = list(directory.iterdir())
    if directory.is_symlink() or directory.parent.is_symlink() or any(p.is_symlink() or not p.is_file() for p in entries):
        raise SystemExit("Unexpected non-regular runtime file in install stage")
    if layout == "legacy":
        return
    if layout != "modern":
        raise SystemExit("Unknown OBS module layout")
    if set(directory.parent.iterdir()) != {directory} or any((package / p.name).exists() or (package / p.name).is_symlink() for p in entries):
        raise SystemExit("Unexpected staged files or modern-layout collision")
    # Keep runtime libraries beside the module so its $ORIGIN remains valid.
    for p in entries:
        p.rename(package / p.name)
    directory.rmdir()
    directory.parent.rmdir()

if __name__ == "__main__":
    stage(Path(sys.argv[1]), sys.argv[2])
