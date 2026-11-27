# Shared CI qualification

The shared OBS32.2.0/33.0.3 qualification matrix uses obs-plugin-ci e9eb704.
The bootstrap reads its literal immutable pin from the workflow.
All 46 required Linux model-free cases must appear once in the report and pass,
even with Python optimization enabled. Extra tests are also required to pass.

Private repository Actions artifacts are enabled for development testing.
If visibility changes to public, uploads automatically stop until a
separate redistribution review changes the policy. The source-only
workflow stays intact. No automatic release, model weights, or local
MediaPipe runtime is published. Corresponding source is the exact
private repository commit accessible to the artifact recipients.

Prior local clean build: 44/44, 17-file package; hosted runs were prevented
from starting by the Actions budget. No Windows/macOS or live-model
qualification is implied. The local package name is
live-latent-lab-sight-linux-x86_64-obs32.2.0-test.tar.gz.

OBS33 uses an explicit 33.0 API series and modern module layout. The
nested release-safety configure tests inherit the selected SDK API, and
no explicit SDK lookup falls back to a system installation. Local
OBS33 SDK configure was blocked by the host FFmpeg 6.1 versus required
8.0; no system dependencies were changed. OBS33 remains a hosted
qualification candidate, not a verified build.

Hosted qualification run 45938738362 reached actual compilation on OBS32
and configuration on OBS33. Both pinned SDK builds succeeded. Sight now
installs and explicitly requires nlohmann JSON in both Linux workflows,
and reads zero-valued OBS API minors correctly. OBS33 still requires a
complete hosted build/test pass; header parsing alone is not qualification.

## Native Windows and macOS baseline

The shared caller now configures Windows x64, macOS Intel and macOS arm64
against their pinned native OBS SDKs for both OBS API series. All hosts
compile the maintained baseline source inventory. Native CPU ONNX Runtime
1.17.1 is acquired from the publisher package with matching architecture,
headers and notices; the JSON header and MIT license are also pinned.
The native package includes only verified stage contents and the correct
DLL/Mach-O runtime layout. Native tests require real Unicode-path ONNX
inference, VRM asset parsing, MediaPipe host-boundary policy, and the
system-install guard. Native graphics tests and Linux-only optional
integrations (MediaPipe Tasks, OCR, OpenCV flow, Qt settings, LiteRT,
diffusion, Live2D and neural rendering) are not covered by this baseline.

The native baseline and its four tests passed locally on Linux using the
publisher SDK. Actual Windows/macOS compilation and packaging must pass on
their respective hosted runners before being called qualified. The prior
Linux 32/33 hosted runs at c820a2e both passed all 44 required cases and
verified archive layout; the extended Linux gate adds two portability tests.
