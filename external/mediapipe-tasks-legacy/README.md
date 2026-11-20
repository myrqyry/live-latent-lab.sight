# MediaPipe Tasks C API headers (legacy ABI snapshot)

These C headers come from the Apache-2.0 licensed `google-ai-edge/mediapipe`
MediaPipe Tasks C API. Original copyright/license headers remain intact.

`core/base_options.h` is **deliberately adapted** to the ABI of the
locally installed `libmediapipe.so` used by the Move Transition plugin on
2026-10-10 (SHA256
`7ba887d5e09d87440b3713816ec02897d6a849eadf2325c9aabe3707cbdac86b`).
The newer upstream version inserts `file_descriptor` into the struct and
adds `app_id`/`app_version`, which changes the layout: mixing the newer header
with this older library causes `Unsupported running mode: unknown mode`.

**Do not link these legacy headers to arbitrary MediaPipe builds.** Sight
uses a separate, independently tested options layout for the official
MediaPipe 1.2.3 Linux x86_64 library. The runtime loader checks the
known ABI size and symbols; the install packager verifies full SHA-256.
See [1.2.3 runtime](../../docs/mediapipe-1.2.3.md). No library binaries or
model weights are committed here.

Face model SHA256:
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
Pose model SHA256:
`64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b`
