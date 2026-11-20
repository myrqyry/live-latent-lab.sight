#!/usr/bin/env bash
# Synthetic, non-sensitive fixture; no Python image libraries required.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
python3 - "$repo/tests/fixtures/ocr-text.gray.gz" "$work" <<'PY'
import gzip, pathlib, struct, sys, zlib
pixels = gzip.decompress(pathlib.Path(sys.argv[1]).read_bytes())
out = pathlib.Path(sys.argv[2])
w,h = 800,400
assert len(pixels)==w*h
def chunk(kind,data):
    return struct.pack(">I",len(data))+kind+data+struct.pack(">I",zlib.crc32(kind+data)&0xffffffff)
def png(path,gray):
    raw=b"".join(b"\x00"+gray[y*w:(y+1)*w] for y in range(h))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,0,0,0,0))+chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b""))
(out/"text.gray").write_bytes(pixels)
png(out/"text.png",pixels)
png(out/"blank.png",bytes([255])*(w*h))
PY
if [[ "$1" == core ]]; then
    "$2" "$3" "$work/text.gray"
elif [[ "$1" == obs || "$1" == bridge ]]; then
    if [[ "$1" == bridge ]]; then
        if [[ $# != 7 ]]; then
            echo "Usage: $0 bridge SMOKE SIGHT_MODULE TESSDATA IMAGE_MODULE HYPERSHADER_MODULE HYPERSHADER_DATA" >&2
            exit 2
        fi
        export SIGHT_SMOKE_HYPERSHADER_MODULE="$6"
        export SIGHT_SMOKE_HYPERSHADER_DATA="$7"
        export SIGHT_SMOKE_HYPERSHADER_EFFECT="$7/shaders/tests/sight_text_geometry_probe.effect"
        export SIGHT_SMOKE_TEXT_PRESET="$7/shaders/root/filter.sight_text_scramble.effect"
    fi
    SIGHT_SMOKE_EFFECT=sight_text_obfuscation_filter SIGHT_SMOKE_WIDTH=800 SIGHT_SMOKE_HEIGHT=400 \
    SIGHT_SMOKE_IMAGE_PLUGIN="$5" SIGHT_SMOKE_IMAGE_FILE="$work/text.png" \
    SIGHT_SMOKE_OCR_BLANK="$work/blank.png" SIGHT_SMOKE_OCR_DATA="$4" \
    "$repo/tests/run_obs_sight_render.sh" "$2" "$3" "$4/eng.traineddata" "$repo/data" > "$work/obs.log" 2>&1 || { cat "$work/obs.log"; exit 1; }
    cat "$work/obs.log"
    grep -Fq "SIGHT_TEXT_OBFUSCATION_GPU_20261117" "$work/obs.log"
    grep -Fq "SIGHT_TEXT_OCR_20261117: local model unavailable" "$work/obs.log"
    if [[ "$1" == bridge ]]; then
        grep -Fq "SIGHT_TEXT_GEOMETRY_20261117" "$work/obs.log"
        grep -Fq "/nonexistent/sight-bridge-tessdata/eng.traineddata" "$work/obs.log"
    fi
else
    exit 2
fi
