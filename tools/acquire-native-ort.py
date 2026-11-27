#!/usr/bin/env python3
"""Acquire exactly the reviewed native SDK files from Microsoft NuGet."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def extract(archive, dest, target, receipt):
    if dest.exists() or dest.is_symlink():
        raise ValueError("Refusing existing SDK destination")
    if sha256(archive) != receipt["archive_sha256"]:
        raise ValueError("ONNX Runtime archive SHA256 mismatch")
    files=receipt["targets"][target]
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sight-ort-",dir=dest.parent) as work:
        stage=Path(work)/"sdk"
        stage.mkdir()
        with zipfile.ZipFile(archive) as z:
            for relative,entry in files.items():
                path=PurePosixPath(relative)
                if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
                    raise ValueError("Unsafe output path")
                matches=[i for i in z.infolist() if i.filename==entry["member"]]
                if len(matches)!=1 or matches[0].is_dir():
                    raise ValueError(f"Missing or duplicate SDK member: {entry['member']}")
                out=stage.joinpath(*path.parts)
                out.parent.mkdir(parents=True,exist_ok=True)
                with z.open(matches[0]) as source,out.open("xb") as dest_file:
                    shutil.copyfileobj(source,dest_file)
                if sha256(out)!=entry["sha256"]:
                    raise ValueError(f"ONNX Runtime member SHA256 mismatch: {relative}")
        if dest.exists() or dest.is_symlink():
            raise ValueError("SDK destination appeared while extracting")
        stage.rename(dest)
    return dest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("target")
    p.add_argument("destination",type=Path)
    p.add_argument("--archive",type=Path,help="Use already downloaded pinned package")
    p.add_argument("--variant",choices=("cpu","cuda-linux"),default="cpu",help="Pinned publisher SDK family")
    args=p.parse_args()
    name="onnxruntime-cuda-sdk.json" if args.variant=="cuda-linux" else "onnxruntime-native-sdk.json"
    receipt=json.loads((ROOT/"docs/qualification"/name).read_text())
    if args.target not in receipt["targets"]:
        p.error("Unsupported SDK target")
    with tempfile.TemporaryDirectory(prefix="sight-ort-download-") as tmp:
        archive=args.archive
        if archive is None:
            archive=Path(tmp)/"ort.nupkg"
            with urllib.request.urlopen(receipt["url"],timeout=120) as response,archive.open("xb") as out:
                shutil.copyfileobj(response,out)
        result=extract(archive,args.destination,args.target,receipt)
        print(f"Verified {len(receipt['targets'][args.target])} pinned files in {result}")
if __name__=="__main__":main()
