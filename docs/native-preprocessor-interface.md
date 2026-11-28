# Sight native Preprocessors interface

These interfaces are a **native C/C++ ABI** and per-source libobs
procedure contract, not an HTTP API or a ComfyUI integration. The reviewed,
machine-checkable signature inventory is in
[native-preprocessor-abi.json](native-preprocessor-abi.json). Its checker
compares the inventory with the actual Rust exports, C prototypes and
OBS procedure registration/callback outputs.

## OBS source-specific procedures

All four procedures live on each individual Sight Preprocessors filter
instance. Call through that instance's obs_source_get_proc_handler(), not
the global OBS procedure handler. They are read-only cumulative counters.

| Procedure | Output fields (all out int) | Meaning |
| --- | --- | --- |
| sight_preprocessor_stats_v1 | captures, completed, skipped, failed, depth_completed | Captured/processed/skipped/failed frames, successful Depth Anything V2 processing |
| sight_preprocessor_teed_stats_v1 | teed_completed | Successful local TEED processing |
| sight_preprocessor_pidinet_stats_v1 | pidinet_completed | Successful local PiDiNet processing (including Scribble) |
| sight_preprocessor_hed_stats_v1 | hed_completed | Successful local five-scale HED model processing (including Scribble HED) |

Consumers must check proc_handler_call() for success and read each
output through calldata_int(). Counters do not guarantee particular
pixels are currently visible, model availability, real-time FPS or latency.
Preserve the versioned source-side procedure names; introduce a new
version when an incompatible parameter or behavior change is necessary.

## Rust/C FFI symbols

The dependency-free Rust static library is linked into the native
Sight module only when SIGHT_ENABLE_RUST_PREPROCESSOR_CORE=ON. With
this option OFF, Sight's C++ implementations are retained.

| Function | Native model preparation or output processing |
| --- | --- |
| sight_normalize_rgb_to_chw_f32_v1 | RGB8 packed to planar CHW float32, with per-channel mean/std normalization |
| sight_rgb8_to_chw_f32_v2 | RGB8 or BGR8 packed to planar CHW float32, with caller-selected input scaling and channel mean/std |
| sight_teed_edges_to_gray_v1 | Four TEED float32 logit maps through mean, sigmoid and Safe Steps into grayscale |
| sight_pidinet_probs_to_gray_v1 | PiDiNet float32 sigmoid-probability map into grayscale, optionally applying Safe Edges |

The C header is src/core/rust_frame_core.h and the implementation is
rust/sight-frame-core/src/lib.rs. The JSON contract stores **ordered
parameter types and names**. Types map between C and Rust (e.g., size_t
to usize, const float* to *const f32). Rust local argument aliases can
differ from public C parameter names; those aliases are explicitly
reviewed in the contract.

The caller owns every passed buffer. Input/output allocations must
remain accessible, non-overlapping and correctly aligned for the
declared element counts. The functions retain no pointers, and return
int32_t 1 on success or 0 on rejected input. They do not load a model,
return an owned buffer, or call OBS/ComfyUI.

## Deterministic verification

For a direct check, run:

    python3 tools/check_native_interface_contract.py

For mutation/regression tests, run:

    python3 -m unittest discover -s tests -p test_native_interface_contract.py

CTest also runs these when a Python 3 interpreter is available. The
checker fails for missing/unreviewed Rust exports, C declarations,
type/order/name drift, missing or altered OBS procedures, mismatched
callback output names, and undocumented public symbols.

**Limit:** A static declaration checker is not a compiler, symbol
inspection, lifetime/aliasing verifier, real-model parity test,
interactive OBS UI test or binary compatibility tool. Continue running
native Rust/C++ builds and real libobs rendering tests when changing
behavior. The original ControlNet Aux preprocessor family remains open.
