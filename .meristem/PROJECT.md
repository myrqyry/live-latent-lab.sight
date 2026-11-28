<!-- meristem-template:v1 -->
# Project

## Purpose

lll.sight is a native OBS Studio plugin for live computer vision, visual effects, and signal-driven interactions. This separately rooted release-candidate repository is the maintained development line; the historical checkout remains separate. See the [README](../README.md) and the final development handoff in [RELEASE_STATUS.md](../RELEASE_STATUS.md#final-development-handoff-2026-11-27).

## Invariants

- Keep the lowercase `lll.sight` branding, consolidated Tracking, and hypershader-based Hand Ripples. Do not restore the retired standalone trackers or Touch Field as part of a source port.
- Preserve portable runtime discovery and packaging. Optional acceleration must not displace the ordinary staged/CPU path. Native Preprocessors does not require ComfyUI or a Python service; unavailable model-backed modes do not fabricate output.
- The original code is GPL-3.0-only. Keep separately acquired models and SDKs out of release inputs unless the documented packaging and licensing rules allow them. Technical staging does not grant redistribution clearance.

## Architecture and boundaries

- [`CMakeLists.txt`](../CMakeLists.txt) selects the validated Linux effects build, the portable native CPU/ONNX baseline, or the explicitly enabled legacy experimental build. [`src/plugin-main.cpp`](../src/plugin-main.cpp) owns OBS registrations; `src/core/` contains processor implementations. The optional Rust core supplements the default C++ Preprocessors path; see the [native interface contract](../docs/native-preprocessor-interface.md).
- The [README](../README.md) owns product orientation. [RELEASE_STATUS.md](../RELEASE_STATUS.md) holds prior implementation decisions, handoffs, and reusable qualification evidence; feature contracts and qualification artifacts live under `docs/`. Meristem points to those records rather than copying them. [AGENTS.md](../AGENTS.md) is the agent entry point; `ACTIVE_WORK.md` carries an implementation handoff only when ongoing work warrants one.
- [Release readiness](../docs/release-readiness.md), [LICENSING.md](../LICENSING.md), and [model downloads](../docs/model-downloads.md) own publication gates, third-party boundaries, and user-provisioned model guidance, respectively.

## Known-good workflows

- Follow [Build and verify](../README.md#build-and-verify) for the Linux OBS 32 CMake build, CTest, and staged install verification. Use [shared CI](../.github/scripts/SHARED-CI.md) for native platform qualification and the [CUDA runtime guide](../docs/onnx-cuda.md) for the optional matching GPU SDK.

## Verification

- Check the configured suite with `ctest --test-dir build-public --output-on-failure` and packaging with `./tools/verify-release-stage.sh build-public`. Use [tests/README.md](../tests/README.md) for model and fixture setup and how to list the cases enabled for a build.
- Distinguish model-free regression, real-model parity, isolated OBS rendering, and package qualification. Identify the build, provider, and fixtures behind each result; a historical pass does not qualify a changed tree or another platform.

## Open structural uncertainty

- Technical qualification is narrower than public release clearance. Track platform support and redistribution gaps in [release readiness](../docs/release-readiness.md) and [release status](../RELEASE_STATUS.md).
