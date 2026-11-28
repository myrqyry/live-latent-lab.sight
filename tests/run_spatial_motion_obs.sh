#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# == 5 ]]; then
    export SIGHT_SMOKE_HYPERSHADER_MODULE="$4"
    export SIGHT_SMOKE_HYPERSHADER_DATA="$5"
    export SIGHT_SMOKE_FLOW_EFFECT="$5/shaders/root/filter.sight_flow_warp.effect"
    export SIGHT_SMOKE_FLOW_ECHO_EFFECT="$5/shaders/root/filter.sight_flow_echo.effect"
    export SIGHT_SMOKE_FLOW_SPARKS_EFFECT="$5/shaders/root/filter.sight_flow_sparks.effect"
    test -f "$SIGHT_SMOKE_FLOW_ECHO_EFFECT" && test -f "$SIGHT_SMOKE_FLOW_SPARKS_EFFECT"
elif [[ $# != 3 ]]; then
    exit 2
fi
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
python3 - "$work" <<'PY'
import pathlib,random,struct,sys,zlib
w,h=320,180
out=pathlib.Path(sys.argv[1])
rng=random.Random(42)
noise=[rng.randrange(256) for _ in range(w*h)]
def at(x,y):return noise[min(h-1,max(0,y))*w+min(w-1,max(0,x))]
a=bytes(sum(at(x+i,y+j) for i in range(-1,2) for j in range(-1,2))//9 for y in range(h) for x in range(w))
b=bytes(a[min(h-1,y+3)*w+max(0,x-5)] for y in range(h) for x in range(w))
def chunk(k,d):return struct.pack(">I",len(d))+k+d+struct.pack(">I",zlib.crc32(k+d)&0xffffffff)
def png(path,p):
    raw=b"".join(b"\x00"+p[y*w:(y+1)*w] for y in range(h))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,0,0,0,0))+chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b""))
png(out/"a.png",a);png(out/"b.png",b);png(out/"flat.png",bytes([128])*(w*h))
PY
SIGHT_SMOKE_EFFECT=sight_spatial_motion_filter SIGHT_SMOKE_WIDTH=320 SIGHT_SMOKE_HEIGHT=180 \
SIGHT_SMOKE_IMAGE_PLUGIN="$3" SIGHT_SMOKE_IMAGE_FILE="$work/a.png" \
SIGHT_SMOKE_FLOW_SHIFT="$work/b.png" SIGHT_SMOKE_FLOW_FLAT="$work/flat.png" \
"$repo/tests/run_obs_sight_render.sh" "$1" "$2" "$work/a.png" "$repo/data" > "$work/log" 2>&1 || { cat "$work/log"; exit 1; }
cat "$work/log"
if grep -Eq "Parameter '(camera_offset|camera_margin)' set to invalid size" "$work/log"; then
    echo 'FAIL: unbound spatial-motion diagnostic uniforms' >&2
    exit 1
fi
grep -Fq 'SIGHT_DENSE_FLOW_20261117' "$work/log"
grep -Fq 'SIGHT_FLOW_REUSE_20261127' "$work/log"
grep -Fq 'SIGHT_CAUSAL_STABILIZER_20261117' "$work/log"
if [[ $# == 5 ]]; then
    grep -Fq 'SIGHT_FLOW_BRIDGE_20261117' "$work/log"
    # Cached native shaders may omit compile/load logs. Require the actual
    # successful graph consumption marker, not an incidental loader message.
    grep -Fq 'SIGHT_FLOW_GRAPH_STEP_20261118' "$work/log"
    grep -Fq 'SIGHT_FLOW_ECHO_BRIDGE seeded=1 trail=1' "$work/log"
    grep -Fq 'SIGHT_FLOW_SPARKS_BRIDGE quiet=1 emitted=1' "$work/log"
fi
