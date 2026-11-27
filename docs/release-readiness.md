# Sight public-release readiness

Status: **developer preview; public redistribution not yet cleared**.

This document defines a candidate release boundary, not permission to
publish binaries or open a repository. The validated Linux effects must stay
separate from the experimental all-effects build.

## Candidate first-release scope

- **Target:** Linux x86_64 and OBS Studio/libobs API 32.x. The working
  desktop reference is OBS 32.2.0. Windows, macOS, and OBS 33 have not been
  certified by the current build or tests.
- **Holistic tracking:** 1.2.3-only filter has isolated OBS acceptance
  for three synchronized channels, opt-in same-source overlays (All,
  Face, Body, Hands), passthrough when disabled, live setting changes,
  and display-only stability controls with confidence/fade options.
  An optional, default-off person mask from the *same* Holistic graph
  is copied before result teardown and used to reject likely off-person
  elbows/wrists/hands in the preview. This does not remove the background
  or guarantee correct subject identification. Unmodified raw tracking
  channels remain available to consumers.
  Models and native runtime remain separately acquired. Production
  webcam performance and full GUI acceptance remain unverified.
- **Default included effects:** Portrait Sketch, Subject Cutout, Depth
  Camera, Tracking (face/body/hand/holistic/object and optional RTMW), Privacy V2, Tracking Signal
  Receiver, Value Router, and Native VRM Avatar. The source registrations
  in src/plugin-main.cpp are authoritative.
- **Experimental effects excluded:** Stable Diffusion, LiteRT, and
  unvalidated legacy effects. They exist in the repository but are behind
  SIGHT_ENABLE_UNVERIFIED_EFFECTS=ON.
- **Optional Live2D source:** retained behind the existing local SDK build option.
  See [native Live2D development and publication boundary](live2d-native.md).
  The default candidate excludes Cubism; an enabled binary is not cleared.
- **No separately acquired model files included:** users select compatible ONNX or
  MediaPipe Tasks models themselves. The vendored BSD face detector has
  embedded learned arrays; its source provenance is reviewed separately.
  Missing/unsupported models must remain
  unavailable, not return fabricated inference results.
- **Privacy V2 is experimental:** face concealment is not a guarantee of
  anonymity, identity protection, or automatic text redaction.

## Technical checks (never install into a running OBS)

Use the default Linux validated build, an explicit user-owned prefix,
and the dependency/model flags described in README.md:

    cmake -S . -B build-verified -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$HOME/.config/obs-studio/plugins/live-latent-lab-sight"
    cmake --build build-verified -j2
    python3 tests/test_release_safety.py
    ctest --test-dir build-verified --output-on-failure
    tools/verify-release-stage.sh build-verified

The last command exercises a genuine CMake install into a new private
temporary directory. It checks the plugin, ONNX Runtime shared library,
shaders/locales, third-party notices, absence of bundled model weights, and
relocatable $ORIGIN linking. It deletes only its own staging directory.
It never overwrites OBS plugin binaries or user configuration.

**CTest registration depends on optional models and fixtures.** A fresh
clone without licensed test images, ONNX model weights, or the pinned native
MediaPipe Tasks library will run a smaller test matrix. Do not equate a
green reduced suite with the full model-backed and OpenGL regressions.

The OBS image-source module is discovered from common Linux paths;
SIGHT_IMAGE_SOURCE_MODULE can override that path. The privacy runner
takes it through SIGHT_SMOKE_IMAGE_PLUGIN and uses the built module's
directory for dynamic libraries, not a personal OBS installation.

**Object Tracking qualification (November 13, 2026):**
An optional user-supplied EfficientDet-Lite0 model ran real native
MediaPipe 1.2.3 inference and isolated OBS render/metadata tests,
including no-overlay mode; the enlarged local suite passed 75/75.
This does not prove actual webcam frame rate, persistent re-identification,
camera follow/zoom, public model redistribution rights, or second-machine
installation. The TFLite weight file is excluded from stage contents.
See docs/object-tracking-mediapipe.md.

## Still blocking public release

1. **Plugin code license:** choose and add a top-level LICENSE for Sight's
   original code after checking copyright ownership and all source origins.
   The licenses of vendored dependencies do not grant rights to Sight itself.
   Do not select MIT, Apache, GPL, or another license on the user's behalf.
2. **Binary redistribution:** review provenance and redistribution terms
   for the tested legacy and official [MediaPipe 1.2.3](mediapipe-1.2.3.md)
   C runtimes. Matching hashes establish binary compatibility, not permission
   to redistribute either runtime.
   Third-party notices are staged, but notices alone do not grant every
   redistribution right.
3. **Publication-tree audit:** the curated
   [Sight source repository](https://github.com/myrqyry/live-latent-lab.sight)
   has its own initial commit and is separate from the private development
   history. Before changing repository visibility, audit **every reachable
   commit and ref in this release repository** for private files, credentials,
   local paths, model weights and imported source whose redistribution terms
   are unclear. Never merge or push development history wholesale into this
   release repository. Removing a file from the tip does not erase it from
   earlier reachable commits.
4. **Clean installation:** build and test from a fresh clone on a second
   Linux system without relying on files already installed in OBS. Establish
   a reproducible CI job for the validated path before claiming broad
   compatibility.
5. **Manual model downloads and first-run usability:** Sight will not
   bundle, mirror or automatically download model weights. See
   [manual downloads](model-downloads.md) for publisher links and known
   compatible files. Re-verify links during release checks; respect
   user-side model terms (especially PortraitSketch's commercial-use
   ambiguity), and keep licensed test-media provenance separate.
   Do not advertise turnkey inference while model selection is manual.
6. **Ordinary OBS acceptance:** exercise each exposed filter with model
   pickers and missing-model behavior; measure real webcam FPS and recording
   reliability. Isolated Xvfb regressions are necessary but not sufficient.
7. **Distribution review:** inspect the final archive and all vendor notices
   (ONNX Runtime, libfacedetection, tinygltf, stb, optional MediaPipe), and
   confirm version/architecture compatibility with the package's users.

## Safeguards in this development slice

- System-wide CMake prefixes /usr/local, /usr, and / are rejected before
  any install files are written. A safe explicit --prefix remains available.
- The real privacy OBS test no longer assumes /usr/local for image-source
  nor reads an older runtime out of the developer's installed Sight plugin.
- The staged module contains an explicit shader/preset/locale manifest and third-party notices.
  Test-only MediaPipe runtime staging is off unless explicitly requested for local use.
  The stage verifies no ONNX or Tasks model weights were accidentally bundled.
- Sight's versioned Tracking Signal Bus reader can share real face/body/hand
  results with HyperShaderFX in the same OBS process without second inference;
  it does not provide network, cross-process or shared depth APIs.

No push, publication, tag, release, model upload, or system installation
is authorized by this document.
