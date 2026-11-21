# ONNX Runtime dependency (Sight Linux candidate)

Sight's validated Linux build **requires ONNX Runtime** for ONNX-backed
portrait, segmentation and depth inference. Do not remove its headers or
shared library to fix documentation links.

## Version and files in this candidate

The checked-in `external/onnxruntime` path is a symlink to
`external/onnxruntime-linux-x64-1.17.1/`, containing:

- `include/` — C/C++ API headers required at build time
- `lib/libonnxruntime.so.1.17.1` — the Linux x86_64 shared library
- `LICENSE` and `ThirdPartyNotices.txt` — notices staged with the plugin
- `VERSION_NUMBER` and `GIT_COMMIT_ID` — the package's version markers

The local library's SHA-256 at this checkpoint is
`a34d8c226994dd4af2b2250442fc236ebba97aefc7ec451c35a86216588d2f69`.
This is a **local artifact fingerprint**, not proof of correspondence to an
official upstream release.

The source bundle formerly included upstream's generic `README.md` and
`Privacy.md`, but omitted much of the upstream repository they referenced.
Those two copies were removed because they caused dead links and were not
used by the plugin or installation. Their removal does **not** remove any
runtime component, license or notice. Consult the
[ONNX Runtime upstream repository](https://github.com/microsoft/onnxruntime)
and [official documentation](https://onnxruntime.ai/docs/) for general
documentation and upstream privacy details.

## Building and staging

The validated effects CMake module uses `SIGHT_ORT_ROOT`, defaulting to
`external/onnxruntime`. It expects the `include/` and `lib/` directories
and the two notice files above. To supply another **compatible, separately
verified** installation:

```sh
cmake -S . -B build-verified -DCMAKE_BUILD_TYPE=Release \
  -DSIGHT_ORT_ROOT=/absolute/path/to/onnxruntime \
  -DCMAKE_INSTALL_PREFIX="$HOME/.config/obs-studio/plugins/live-latent-lab-sight"
cmake --build build-verified -j2
ctest --test-dir build-verified --output-on-failure
tools/verify-release-stage.sh build-verified
```

The staging script copies the shared library next to the Sight plugin,
checks `$ORIGIN` linking and retained license/third-party notices, and
rejects bundled model weights. It does **not** modify a running OBS
installation or establish public binary redistribution approval.

Changing ONNX Runtime versions, providers or platforms needs an actual
build, model-backed regression coverage and licensing review. Sight has
only been qualified with the pinned Linux candidate described here; see
[release readiness](release-readiness.md) for remaining gates.
