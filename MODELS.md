# Model references

This file lists model families used by the plugin. Download behavior is
implemented in the source code and must not be inferred from this reference
file.

For the default validated Sight filters, see
[manual publisher model downloads](docs/model-downloads.md).
Sight will not bundle or mirror model weights.

## Stable Diffusion models

The Stable Diffusion backend accepts the model formats supported by the
checked-in `stable-diffusion.cpp` submodule. Place checkpoints, LoRAs,
ControlNet files, VAEs, and TAESD files below the module configuration path.
The exact paths are created by `src/plugin-main.cpp`.

Use the model's own license and distribution terms when downloading it. The
links below are references, not bundled assets or verified release artifacts.

- [Z-Image-Turbo GGUF](https://huggingface.co/leejet/Z-Image-Turbo-GGUF)
- [ControlNet v1.1 GGUF](https://huggingface.co/leejet/ControlNet-v1.1-GGUF)
- [TAESD](https://huggingface.co/madebyollin/taesd)
- [TAESD-XL](https://huggingface.co/madebyollin/taesdxl)
- [LCM LoRA SD1.5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
- [LCM LoRA SDXL](https://huggingface.co/latent-consistency/lcm-lora-sdxl)

## LiteRT catalog

The built-in LiteRT catalog currently defines these model IDs in
`src/core/litert_model_catalog.cpp`:

- `depth-anything-3-small`
- `6drepnet-head-pose`
- `blaze-face-full-range`
- `sam2-tiny-image-encoder`
- `sam2-tiny-mask-decoder`
- `ormbg`
- `realesrgan-x4v3`
- `nafnet-sidd`
- `fast-style-candy`
- `rtmw-wholebody`

Catalog entries include a source URL, filename, and SHA-256 digest. The model
manager downloads to a temporary file, verifies the digest, and atomically
installs the result below the configured model directory. NSFW conversion is
deferred and has no LiteRT catalog entry.

## Related code

- `src/core/litert_runtime.*` provides shared LiteRT execution.
- `src/core/litert_model_manager.*` handles downloads and verification.
- `src/core/litert_model_catalog.*` defines built-in model metadata.
