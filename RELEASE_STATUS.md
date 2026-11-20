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
