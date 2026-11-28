#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Versioned ABI: packed RGB8 (not RGBA), no row padding. Output is planar
// float32 CHW (a single 3-channel frame). Caller owns all buffers.
// Returns 1 on success, or 0 without modifying output on invalid arguments.
// Pointers must refer to their declared valid and non-overlapping allocations.
int32_t sight_normalize_rgb_to_chw_f32_v1(
    const uint8_t *rgb, size_t rgb_len, uint32_t width, uint32_t height,
    const float *mean3, const float *std3, float *output, size_t output_len);

// General RGB8 -> planar CHW model input, explicit scale (1.0 for TEED,
// 1/255 for normalized inputs). Caller owns all input/output memory.
int32_t sight_rgb8_to_chw_f32_v2(
    const uint8_t *rgb, size_t rgb_len, uint32_t width, uint32_t height,
    float scale, const float *mean3, const float *std3,
    float *output, size_t output_len);

// TEED four float32 logit maps to one 8-bit grayscale map, matching the
// reference reduction/sigmoid/safe_step contract.
int32_t sight_teed_edges_to_gray_v1(
    const float *edge0, const float *edge1, const float *edge2,
    const float *edge3, size_t pixels, uint32_t safe_steps,
    uint8_t *output, size_t output_len);

// PiDiNet single-channel sigmoid probabilities with optional
// reference safe_step(2) -> uint8 output, disjoint caller-owned buffers.
int32_t sight_pidinet_probs_to_gray_v1(
    const float *probabilities, size_t pixels, int32_t safe,
    uint8_t *output, size_t output_len);

#ifdef __cplusplus
}
#endif
