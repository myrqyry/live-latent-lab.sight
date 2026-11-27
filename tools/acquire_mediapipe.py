#!/usr/bin/env python3
"""Extract the pinned publisher MediaPipe wheel locally; never install into OBS."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

MEMBER = "mediapipe/tasks/c/libmediapipe.so"
WHEEL_SHA256 = "1fcda646cc7b96f73453acc416ce31d0fba287d0f1fc095865272a7ea6986497"
ELF_SHA256 = "7ce8b23159d3fd4e8b727e35afbfb35f3d45d6bde1ec11cc01c316b3bf469f63"
ELF_SIZE = 127908512


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def extract_verified(wheel, output, wheel_digest=WHEEL_SHA256,
                     elf_digest=ELF_SHA256, elf_size=ELF_SIZE):
    wheel, output = Path(wheel), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if sha256(wheel) != wheel_digest:
        raise ValueError("wheel SHA-256 mismatch")
    # Only the single expected member is ever read; no zip.extractall.
    with zipfile.ZipFile(wheel) as archive:
        members = [m for m in archive.infolist() if m.filename == MEMBER]
        if len(members) != 1 or members[0].file_size != elf_size:
            raise ValueError("unexpected runtime member or size")
        with tempfile.TemporaryDirectory(prefix="sight-mediapipe-", dir=output.parent) as temp:
            candidate = Path(temp) / "runtime"
            with archive.open(members[0]) as source, candidate.open("xb") as target:
                shutil.copyfileobj(source, target)
            if candidate.stat().st_size != elf_size or sha256(candidate) != elf_digest:
                raise ValueError("runtime SHA-256 or size mismatch")
            # Publish without overwriting even if another process wins the race.
            os.link(candidate, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="locally downloaded official 1.2.3 Linux x86_64 wheel")
    parser.add_argument("output", type=Path, help="new file in an existing user-owned directory outside OBS")
    args = parser.parse_args()
    try:
        extract_verified(args.wheel, args.output)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"MediaPipe extraction failed: {exc}\n")
    print(f"Verified local runtime: {args.output}\nSelect this file in Sight's runtime picker. Keep the publisher wheel and notices.")

if __name__ == "__main__":
    main()
