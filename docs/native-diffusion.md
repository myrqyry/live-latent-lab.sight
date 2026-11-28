# Optional native diffusion

The validated Linux build can additionally expose **lll.sight Generator
(Experimental)** and **lll.sight Live Diffusion (Experimental)**. The
optional generator and filter are independent of the legacy all-effects
build. Both are disabled by default. They run locally; no cloud fallback.

## Build

Install a compatible OBS/libobs 32.x development environment, CMake, C++20,
OpenCV core/imgproc/video and (for CUDA) a compatible NVIDIA driver and
CUDA toolkit. Initialize the public upstream dependency at the pinned Git
revision `532d2c15c430a785a712b2d94281f04e733008a4`:

```sh
git submodule update --init --recursive libs/stable-diffusion.cpp
cmake -S . -B build-diffusion -DCMAKE_BUILD_TYPE=Release \
  -DSIGHT_ENABLE_NATIVE_SD_GENERATOR=ON \
  -DSIGHT_ENABLE_NATIVE_SD_LIVE_FILTER=ON \
  -DSD_CUDA=ON
cmake --build build-diffusion --parallel 4
ctest --test-dir build-diffusion --output-on-failure
tools/verify-release-stage.sh build-diffusion
```

For Turing, the separately built and tested local engine used CUDA 12.9,
SM75, graphs off and flash attention off. A reproducible source build and
portable binary package on a second machine are still outstanding. The
optional `SIGHT_NATIVE_SD_ROOT` points at an existing checkout of the same
revision; `SIGHT_NATIVE_SD_PREBUILT_LIBRARY` is a trusted local shared
library of that exact API, not a download mechanism or redistribution
approval. The tested curated build reused that existing local engine.

## Use and test

Select a compatible, legitimately acquired SD1.x LCM checkpoint and matching
TAESD via the OBS model pickers. No model files are supplied. Start with
CUDA, one requested LCM step, guidance 1.0, and a 256-pixel inference long
edge. Low denoise retains more of the input. The generation rate limit
controls diffusion keyframes rather than OBS rendering. Q8_0 optionally
reduces weight memory; it changes output and was slightly slower in the
controlled comparison, so checkpoint-default precision remains the default.
The generator supports finite and recursive generation, preview motion,
exposure adjustment and frame export. Real-time performance, temporal
quality and usable resolutions depend on the checkpoint and hardware.

Explicitly opt into real CUDA tests using existing local files:

```sh
cmake -S . -B build-diffusion \
  -DSIGHT_SD_TEST_MODEL=/absolute/path/to/checkpoint.safetensors \
  -DSIGHT_SD_TEST_TAE=/absolute/path/to/taesd.safetensors
ctest --test-dir build-diffusion --output-on-failure -R 'sight_native_sd_'
```

Test with OBS closed and sufficient free VRAM; the CUDA guard prevents
silent CPU fallback or admission with insufficient memory. The model-free
smoke tests alone do not establish inference. The local curated test
reused the qualified CUDA engine and passed all 14 native diffusion cases
plus six cross-feature integration cases (20/20). These include real
live motion/temporal rendering, Q8/default switching and an eight-frame
recursive export with final pixel equality. See the [SM75 benchmark
receipt](qualification/native-sd-sm75-20261126.json) for the earlier
full-development engine measurements and limitations. The optional local
stage passed without bundling model files or the MediaPipe test runtime.

The SD.cpp, ggml, utf8proc and oniguruma notices are installed with the
optional engine. The source pin is publicly reachable upstream. External
checkpoint terms, the runtime/dependency distribution review and original
Sight source license remain separate release gates.
