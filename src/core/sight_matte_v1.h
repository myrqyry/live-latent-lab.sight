#pragma once
#include <stdint.h>

/* Read-only caller-owned R8 foreground snapshot, normalized to the producer
 * input. Stabilized inference matte; GPU refinement/calibration not included.
 * No external texture pointer, no model invocation. The caller initializes
 * abi_version and struct_size (exact sizeof). Read via the producer's OBS
 * sight_matte_read_v1 proc: max_age_ms in [1,5000], known_sequence = currently
 * uploaded sequence or 0. On available=true metadata is refreshed; pixel
 * bytes are copied only if known_sequence differs. Sequence is process-wide
 * and nonzero. Age is measured from input capture, not inference completion.
 * Never use fields if available=false. */
#define SIGHT_MATTE_ABI_V1 1u
#define SIGHT_MATTE_MAX_EDGE 320u
#define SIGHT_MATTE_MAX_PIXELS (SIGHT_MATTE_MAX_EDGE * SIGHT_MATTE_MAX_EDGE)
typedef struct sight_matte_v1 {
    uint32_t abi_version, struct_size;
    uint32_t width, height, source_width, source_height;
    uint64_t sequence;
    uint8_t pixels[SIGHT_MATTE_MAX_PIXELS];
} sight_matte_v1;
