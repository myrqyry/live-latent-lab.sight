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
