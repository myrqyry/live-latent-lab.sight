# Candidate status

Initial Linux candidate: fresh Release configure/build and 90/90 configured
CTest checks passed on OBS 32.2 headers and local model fixtures. Native
Live2D and LiteRT GPU whole-body paths were not configured. Real model and
isolated graphics coverage is not a live webcam or second-machine test.
Staged-install verification passed (18 files); MediaPipe runtime was included
in that local test stage only. Redistribution rights remain unresolved.

Next: confirm code license, audit dependency distribution rights, provide
portable runtime acquisition and clean-machine CI, qualify any additional
features independently. Do not ship private model weights or test fixtures.

## Expanded qualification

The existing implementation was re-qualified without changing C/C++ code:
93/93 tests passed in the non-RTMW configuration with the existing local
model fixtures and optional LiteRT tests configured. Then the RTMW GPU
option was explicitly enabled and rebuilt: all eight focused integration
and GPU checks passed, including actual YOLOX + RTMW, persistent native
inference, receiver/router controls and the real six-mode Unified Tracking
switch test. The separate GPU configuration is not a claim that all 98
tests were rerun in that configuration. Native Live2D remained excluded.
External runtime acquisition and redistribution are still separate gates.

## Consolidated Tracking and Hand Ripples ownership

The expanded public candidate now exposes ten source/filter registrations
(including optional OCR, spatial motion and native VRM). Face, body, hand,
holistic, object and optional RTMW remain modes of LLL.sight Tracking; the
six redundant development-only registration types and compatibility
branches have been removed. The mode-specific model browser remembers its
selected model directory. Receiver/router/channel discovery and older
integration fixtures now use the unified source.

Hand Ripples moved into HyperShaderFX (local commit e0286f6). Sight no
longer registers or ships Touch Field; its shader/consumer tests moved
with the effect. The native-wide hand snapshot producer test remains.
Final serial GPU-profile CTest: 92/92 passed, including real six-mode
switching and native-wide hand receiver. An earlier parallel 98-test pass
had one spatial-motion failure; the isolated retry and final serial suite
passed, but the earlier variability was not diagnosed. Release staging
passed (20 files, 78,686,935 bytes); licensing clearance is unchanged.
The full HyperShaderFX Sight interop script passed with both rebuilt
modules, including the real 640x360 hand model -> snapshot -> ripple
shader path: 90 active frames, 89 evolving, and passthrough before signal
and after expiry. The isolated HyperShaderFX test logs a pre-existing
double-destroy/UI-dispatch warning, also observed with its older binary
and Hand Highlights. Teardown warning is unresolved, not a clean-runtime
claim. Physical webcam and normal interactive OBS UI were not tested.
No installation or publication.

## Shared plugin settings (2026-11-25)

Added a native Settings → Plugins → LLL.sight tab to the validated build
when Qt6 Widgets is available (SIGHT_SETTINGS_PAGE=OFF disables it).
Uses the identical v1 widget convention from Sound e6add08; no dependency
on Sound or Aftermarket. Separate plugin-owned preferences for model
and runtime browsing roots, resource links from existing model guides,
no change to auto-discovery, selected models, or active pipelines.
Existing valid model-folder preference remains ahead of the global
browse root for ONNX and MediaPipe property paths that already implement
that behavior. RED absent bridge; GREEN 31/31 focused unit, release
safety, shared settings and real libobs picker tests. Actual isolated OBS
32.2.0 Fluffy Dark shows both independently loaded Sight and Sound tabs
under a single Plugins category. Full inference suite not rerun. Current
installed Sight includes native Live2D, absent from this public build;
do not replace it with the reduced candidate. No push or local plugin
replacement.

## Live2D development and publication boundary (2026-11-25)

Agreed to retain the existing native Live2D integration source and local
functionality, with official user-acquired SDK outside Git and the default
preview build excluding Cubism. Added docs/live2d-native.md, README and
release-checklist clarification, official SDK acquisition/publication
links in the Sight settings tab. Live2D's official terms require separate
review for expandable avatar applications; an open-source adapter or
external SDK does not establish release permission. Top-level Sight
license and full source-origin review remain open.

Checked reachable Git object paths and exact blob IDs of local Core library,
Core header and Haru sample manifest; none found. This is a targeted check,
not a complete provenance audit. Fresh separate Release build with the
existing official Native SDK 5 R5 and Core 6.0.1, Qt settings, RTMW and
optional system dependencies: 5/5 focused CTests passed (native OBS
Live2D render with local Haru sample, animation, intake, shared settings,
actual libobs picker defaults). Disposable install stage contained no
Cubism-named files or .moc3/.model3.json/.motion3.json assets. Default
build cache and installed Sight binary untouched: installed development
version includes newer OCR functionality absent from this curated checkout.
No private development history imported, no SDK/weights added, no push or
publication. Full combined inference suite not rerun.

## Integrated vision release candidate (2026-11-26)

Ported the maintained cutout, backend diagnostics, shared motion, semantic
regions, local OCR and Value Router changes while preserving the ten-source
curated registration boundary. Full configured 115-case Linux suite:
114 passed initially, including RTMW GPU, real six-mode tracking, OCR
routing/editor and all cutout/flow model tests. The VRM face easing smoke
failed an arbitrary first-frame pixel-count ratio despite actual continued
motion and correct hold/return. The corrected test checks continued change
and retains hold and return checks; real OBS rerun passed. This is combined
qualification of an unchanged module and rebuilt test, not a single green
115-case rerun. Release safety tests and technical install passed.
Default stage: 18 files, no test-only MediaPipe library or untracked data
canary; explicit ABI-verified local MediaPipe staging remains opt-in.

Nine pre-integration reachable commits and 64 pending files were scanned
for common private-key/API-key patterns and developer absolute paths, with
no matches. This is a targeted scan, not exhaustive security or legal
clearance. The two binary blobs are the pinned ONNX Runtime ELF and a
compressed OCR fixture; libfacedetection includes embedded CNN arrays.
Verified EnTT v3.16.0 header against upstream and restored its MIT notice.
The original ONNX Runtime v1.17.1 upstream archive could not be fetched
(HTTP 404); exact origin remains a release gate. Original plugin license,
source provenance, clean-machine qualification and experimental feature
promotion remain open. See [audit receipt](docs/qualification/curated-release-audit-20261126.json).
No visibility change or publication.
