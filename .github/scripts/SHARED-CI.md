# Shared CI qualification

The shared Linux OBS32.2.0/33.0.3 model-free gate uses obs-plugin-ci e9eb704.
The bootstrap reads its literal immutable pin from the workflow.
All 44 required native cases must appear once in the report and pass,
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
