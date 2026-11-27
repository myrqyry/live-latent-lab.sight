# Optional neural rendering (experimental, Linux/CUDA)

The **LLL.sight Neural Rendering (Experimental)** filter is an opt-in
integration with a trusted local CUDA worker. It is not NVIDIA's official
NGX/DLSS runtime. Sight's source includes the worker and bridge, not the
external graph or learned parameters. It does not obtain or extract any
proprietary weights. Use only a compatible checkpoint that you have the
right to use. Model rights, extraction rights and binary redistribution are
not established by the upstream MIT source notices.

## Build and local setup

Configure the validated build with `-DSIGHT_ENABLE_NEURAL_RENDERING=ON`.
This does not require enabling native diffusion or the old all-effects
build. OpenCV enables motion compensation; the worker runs inference off
the OBS render thread. The default build leaves this filter disabled.

Provide a trusted local Python/CUDA/PyTorch environment, graph checkout and
compatible Small31 checkpoint. The local JSON configuration is
`neural-rendering.json` in Sight's OBS module configuration directory.
The trusted environment variable `SIGHT_NR_CONFIG` can point to an absolute
configuration path for isolated tests. Do not load configuration, Python
code, or checkpoints from untrusted sources.

Example configuration (replace placeholders with your existing local paths):

```json
{
  "python": "/absolute/path/to/venv/bin/python",
  "graph_repo": "/absolute/path/to/graph-source",
  "checkpoint": "/absolute/path/to/compatible-small31.pt",
  "cuda_graph": false
}
```

The graph Python package `dlssnr` must match
[DylanWaken/DLSS5-Pytorch](https://github.com/DylanWaken/DLSS5-Pytorch/tree/09befd1ef418c8d23be8db8d33cb8fa1c2b009db)
revision `09befd1ef418c8d23be8db8d33cb8fa1c2b009db`. The frontend semantics
reference [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR/tree/9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1)
revision `9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1`. Both pins and matching MIT
notices were checked on 2026-11-26. See the bundled upstream notice in
`data/neural-rendering/UPSTREAM-NOTICE.txt`. Graph code and models are
external; neither needs to be copied into the plugin data directory.

Optional `fused_so` is a trusted absolute path to the locally compiled
`data/neural-rendering/fused.cu` pointwise library; `worker` is a trusted
absolute development override. Normally omit both; the built-in worker is
resolved from the installed module data. Install CUDA/PyTorch dependencies
from their publishers for your actual driver/GPU. No cloud fallback.

## Controls and qualification

Choose live, motion-compensated or frame-locked synchronization; control
resolution, correction precision, strength, temporal history and edge/depth
guidance. The source keeps rendering while its bounded worker processes
one request. Missing/invalid local configuration and expired corrections
pass through rather than synthesizing a result. The filter is experimental;
1080p/4K sustained performance and arbitrary webcam quality are not proven.

For real local qualification set `SIGHT_NR_TEST_CONFIG` to a trusted local
configuration and optionally `SIGHT_DEPTH_MODEL` to a compatible local
depth model, then run `ctest --test-dir build-verified -R 'sight_neural_' -V`.
Registration-only tests do not imply model-backed inference. The development
qualification used actual CUDA Small31 on RTX 2070 SUPER; no weight or
runtime distribution is authorized by that test.
