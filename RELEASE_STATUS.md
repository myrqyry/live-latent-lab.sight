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
(HTTP 404); exact origin remains a release gate. Original plugin license was subsequently selected as GPL-3.0-only;
source provenance, clean-machine qualification and experimental feature
promotion remain open. See [audit receipt](docs/qualification/curated-release-audit-20261126.json).
No visibility change or publication.

## Optional native diffusion integration (2026-11-26)

Imported only maintained native generator/live-filter source, related unit
and integration tests, shaders, and benchmark receipt from development.
Updated the independent stable-diffusion.cpp gitlink to the verified
public upstream revision 532d2c1; no private history or model assets copied.
Both options default off; the curated tracking/vision registration remains
unchanged when they are disabled. Native diffusion requires OpenCV.
Configured and rebuilt with the existing locally qualified CUDA 12.9 SM75
engine, CPU ONNX runtime and the curated vision build. All 20 selected
CTest cases passed, including all 14 native diffusion cases, actual CUDA
motion/temporal inference, live Q8/default restoration and recursive export,
plus privacy, VRM, portrait, cutout, OCR and unified tracking checks.
Optional technical stage: 25 files / 146109348 bytes, no MediaPipe test
runtime or separately acquired model weights. Not a clean-machine source
build of SD.cpp or redistribution approval. The older unvalidated legacy
build and native neural-rendering worker are not promoted; neural model
provenance must be resolved separately. See docs/native-diffusion.md.

## Original-code licensing

The project owner selected GPL-3.0-only. The full license and scope notice
are included in source and local staged packages. Existing third-party
licenses are unchanged; this does not clear dependency or model distribution.

## Dependency and source-preview qualification (2026-11-26)

- All 16 retained ONNX Runtime 1.17.1 files match the pinned independent
  archive byte-for-byte; publisher original archive unavailable. Embedded
  libfacedetection weights match upstream after trailing-whitespace
  normalization, with equivalent explicit result cap and stride. Retain
  the original BSD notices and reviewed vendor files. Actual official
  MediaPipe 1.2.3 wheel and ELF match both recorded hashes; verified local
  extraction helper rejects bad inputs and existing outputs. Default
  package excludes this runtime. See dependency-provenance receipt.
- Optional neural rendering now includes only Sight's own integration,
  worker and CUDA source, with the original MIT reference notice. Both
  external graph pins and MIT notices were verified. No external graph,
  checkpoint, proprietary binaries or extraction scripts copied. Actual
  curated combined CUDA/depth/temporal test passed. Of 11 integration
  cases, 10 passed initially; custom-draw frame locking exceeded its
  startup test deadline during concurrent compilation. Corrected deadline
  and isolated rerun passed with old/new scene MAE 3.73/108.31. Stage 38
  files, no model assets or test-only MediaPipe library. External checkpoint
  rights and portability remain unapproved.
- Fresh clone plus explicit source-only pending changes, independent build
  directory and isolated home, no local model flags, 44/44 tests passed
  after moving the pure OCR output test outside the optional Tesseract
  condition. Default model-free package stage 17 files. Host lacks system
  Tesseract headers; real OCR remains separately qualified in the expanded
  build. New Linux pinned-OBS source qualification workflow publishes no
  artifacts; it has not run on hosted infrastructure. No claim of second
  machine, Windows, macOS or high-resolution webcam certification.
- Receipt: docs/qualification/source-readiness-20261126.json. Previous
  statements excluding the neural worker are historical and superseded.

## CUDA staging safeguard

The curated 1.30.0 local split-SDK GPU build passed Longpipe XS matte and
flow CUDA sessions, shared matte ABI and real moving/restored OBS pixels.
It now cannot be staged with the older SDK notices: CMake rejects mixed
header/library roots before any install write. The red/green test and all
five release-safety cases passed; the actual split CUDA install was rejected
and the ordinary optional stage still passes. A matched portable GPU
SDK/runtime distribution remains a separate qualification task.

## Native and matched CUDA qualification follow-up

Hosted Linux qualification at c820a2e passed both OBS 32.2.0 and 33.0.3
(44/44 each), with 17-file package layout, exports and archive verification.
Native baseline port now shares the maintained source list, uses verified
native ONNX SDKs, and blocks Linux-only MediaPipe ABI loading on other hosts.
Local Linux native baseline: four tests, including actual Unicode-path
inference and VRM parsing, passed. Extended Linux model-free gate: 46/46;
Windows/macOS runners must still establish their own qualification. Native
optional OCR, dense flow, Qt settings, tracking SDKs, diffusion, Live2D and
neural worker are not enabled in this initial native baseline.

Publisher NuGet ONNX Runtime GPU Linux 1.30.0 acquired with matching headers,
notices and CUDA provider; 21 output files verified, including loader/linker
names for the same library. Actual staged CUDA-only MatMul passed with CPU
fallback disabled and separately supplied CUDA 13 libraries. Stage content
verification passed (19 files). No NVIDIA binaries or models bundled. The
current host requires explicit CUDA runtime library directories. No claim of
portable GPU binary release or Windows/macOS GPU qualification. Public CI
binary uploads remain disabled.

## Lowercase plugin branding (2026-11-27)

Visible plugin/module, source/filter, settings tab and OCR dialog
branding and current documentation use lll.sight. Existing source
IDs, plugin filename, configuration/model paths, setting keys and
runtime contracts are unchanged. Actual main-branch Linux validated
module registry: nine Sight types and module name passed lowercase
check (standalone tests/verify-display-names.cpp). Optional build
labels updated in source; not all optional binaries rebuilt here.
No installation or push in this pass.

## Clean-history source convergence (2026-11-27)

The maintained source repository is intended to carry the useful
development implementation, not an intentionally reduced edition.
Transfer selected tracked source and tests without importing the
old repository's commit graph, assets, local configuration or SDKs.

- Spatial Motion now recycles CPU staging frames and grayscale buffers.
  Padded-stride ownership/reuse, dense motion, stabilizer and actual
  isolated libobs motion-render tests passed. The render test reported
  392 flow samples, endpoint error 0.0212124, and stabilization MAE
  24.0785 -> 9.91514 on its synthetic fixture.
- Native Preprocessors now registers with OpenCV on Linux, including
  conventional image transforms and Depth Anything V2, TEED, PiDiNet,
  HED and scribble modes. Uses existing public ONNX session implementation.
  Default C++ and opt-in offline Rust paths both passed real model and
  libobs render tests using separately acquired local checkpoints.
  Expected HED/PiDiNet/TEED reference hashes matched in both builds.
  Missing-model tests verify passthrough without fabricated results.
- Default isolated model-free suite 54/54; optional Rust build has an
  additional ABI test. Native interface mutation tests, offline Cargo
  tests, and technical staged install checks passed; no weights staged.
  New required cases enforced by existing Linux CI report validator.
- No Windows/macOS preprocessor build qualification, production camera
  benchmark, redistribution approval, remote CI, installation or push
  in this pass. Legacy individual tracking types and standalone Touch
  Field remain distinct from the consolidated tracking/hypershader
  workflows; compatibility and newer dev-side changes still need review.

### Final development handoff (2026-11-27)

Imported the last completed development change (non-waiting Portrait
Sketch handoffs) as source only. The sketch filter and enhancement
processor match the final development versions byte-for-byte. The
contention regression failed before the change, then passed; actual
320x180 OBS / 256-detail AnimeGAN rendering reached six completed
frames with eight capacity growth requests at warm-up and eight at
completion. Isolated Linux C++ model-free suite: 55/55 passed; the
handoff regression is now required by CI.

Use this clean-history repository's main as the primary development
checkout. The historical checkout is preserved independently, not
merged or rewritten. Do not reintroduce obsolete standalone tracking
filters or Touch Field; keep unified tracking and hypershader.
Retain public portable runtime/path discovery and packaging rather
than overwriting with older Linux-only development code. The older
unverified comprehensive build's latent-engine/style-transfer
implementation still differs; it was not part of the final verified
Portrait Sketch changes. Native platform and optional GPU package
qualification are separate from this verified Linux handoff. No
installation or push in this pass.

## CUDA/OpenGL texture interoperability (2026-11-27)

Independently implemented the useful GPU texture-handoff mechanism
identified in OBS RTX SuperResolution without its Maxine models.
Optional real Portrait Sketch path uses the existing ONNX model, native
CUDA driver API, persistent GL registrations, non-waiting capture fence
polling, worker-owned stream/device buffers and reusable GPU I/O
binding. Image normalization/padding and output repacking use CUDA
kernels. No application-side host pixel readback in GPU mode. Explicit
CUDA request fails closed; standard portable staged path retained.

Actual NVIDIA hardware/staged module passed six GPU uploads with two
registrations, zero readbacks/failures, maximum handoff 453 us; resize
produced nine uploads/four registrations while retaining one model
load. Live staged/GPU switch passed, RGB max error 1/255 and mean
0.0000115741. Separate full-model feasibility probe median 15.17 ms
warm, CPU reference max error 1/255; CPU graph nodes were shape-only
(max 32 byte activations/outputs). Driver mapping may still synchronize.
Normal CPU suite 55/55 and technical GPU package staging passed.
Qualification helper and documentation in tools/verify-cuda-texture.sh
and docs/cuda-texture-interop.md. No Maxine dependency, model change,
installation or push. Other effects/platforms and contention/device
loss remain unqualified.

## Subject Cutout reusable frame exchange (2026-11-27)

Extended the texture-inference optimization work to the existing matte
pipeline without replacing its CPU RGB/matte ABI, history, motion or
Longpipe flow. Persistent pixel workspaces, fused normalization/resize,
non-waiting render handoffs, upload outside publication mutex, generation
checks, optional identical-size ordinary-source draw reuse (off by
default). Actual CPU U2Net buffer regression passed. CUDA U2NetP, MODNet,
Longpipe XS all passed eight-frame reuse and byte-identical shared matte
snapshots against previous build (512x512 portrait, ABI 320x320).
Six/eight/eight counted capacity growth requests stable after warm-up.
Opt-in Longpipe flow run: 36 completions, 35 source draws avoided,
ten stable capacity growth requests, motion/restore/toggle/ABI passed.
Downscaled reuse was rejected after measurable U2NetP matte deviation.
All 56 model-free tests passed; new contention regression required by CI.
No claim of GPU-resident Cutout, full allocation-free operation, or
measured end-to-end speedup. Depth and Preprocessors still pending.
No install or push.

## Depth Camera reusable frame exchange (2026-11-28)

Reusable capture/RGB/depth/history/upload buffers; non-waiting render
handoffs and deferred geometry invalidation; generation-tagged textures,
smoothing outside the control mutex. CPU and actual CUDA Depth Anything
V2 tests passed: eight initial growth requests stable across eight
completions, edge change to 336 stable at sixteen, one model load total,
source resize immediate passthrough then recovery. CUDA shared depth
snapshot range/metadata/known sequence/ABI/disable/expiry/recovery
passed. 57 model-free tests passed. Original normalization and history
algorithm retained. Host staging and CPU depth ABI retained; no new
GPU-resident depth inference or end-to-end performance claim.
No install or push.

## Preprocessors reusable frame exchange (2026-11-28)

Non-waiting input/output exchange, graphics-owned upload vector swapped
from publication slot without extra full-frame copy, generation/age
checks, worker-owned shared OpenCV and gray storage, persistent
transactional TEED/PiDiNet/HED output buffers. Actual CPU and CUDA OBS
tests passed four native modes and four neural modes, safe/scribble
and mode changes. Stable counted pixel capacity per mode after warm-up,
78 CPU/81 CUDA uploads. Repeated output exact except CUDA Depth Anything
(max 1/255; final total absolute error 27 over 245760 bytes); narrowly
bounded CUDA depth tolerance in the reuse test. New stats ABI documented
and checked, 57/57 model-free cases passed, including missing-model
passthrough. Host staging remains, no GPU-native I/O binding or measured
end-to-end speedup. No install or push.

## Depth Anything CUDA device tensor I/O (2026-11-28)

Optional Linux CUDA build with embedded PTX, reusable device tensors,
pinned RGB input, GPU normalization/reduction/quantization, bounded
byte-map download; shared by Depth Camera and Depth Anything
Preprocessors. Final actual model benchmark on RTX 2070 SUPER:
252x140 6.725->5.417 ms, 336x196 9.905->7.787 ms,
518x294 17.525->16.851 ms medians; max difference 1/255.
Boundary payload 3N + N+4 bytes instead of nominal 16N (not physical
PCIe instrumentation). Reuse/resize/reload/CPU conflict tested.
Actual OBS device-mode depth and preprocessor output, history,
resize and snapshot lifecycle passed. Ordinary CPU regression
57/57. CPU capture, temporal and snapshot consumers retained; other
models and cross-platform builds not qualified. No install/push.

## HED CUDA multiscale tensor I/O (2026-11-28)

Optional CUDA-bound five-scale HED outputs and GPU interpolation,
averaging, sigmoid and safe-step quantization. Existing model,
persistent buffers, pinned compact RGB input, worker-owned CUDA
stream and embedded PTX. Actual same-provider benchmark on RTX 2070
SUPER: 192x128 4.200->3.350 ms, 320x192 9.177->8.367 ms,
512x320 16.453->15.051 ms. Soft and safe maps byte-identical at all
three sizes; logical model-boundary payload about 77% smaller.
Reuse, resize, reload, invalid input and CPU/device conflict passed.
Forced-device OBS Preprocessors HED rendering and mode lifecycle
passed; ordinary CPU suite 57/57 and original HED model parity
passed. Host capture, CPU scribble NMS and final upload remain.
TEED, PiDiNet and Cutout tensor I/O are still pending. No install/push.
