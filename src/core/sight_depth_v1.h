#pragma once
#include <stdint.h>
/* Caller-owned normalized relative inverse-depth R8, higher = nearer.
 * Per-inference normalization, optional temporal smoothing; not metric.
 * Initialize exact ABI and size. A fresh capture is required even if the
 * standalone visualization is configured to hold its last result forever.
 * Known sequence may omit the pixel copy. No borrowed graphics resources. */
#define SIGHT_DEPTH_ABI_V1 1u
#define SIGHT_DEPTH_MAX_EDGE 518u
#define SIGHT_DEPTH_MAX_PIXELS (SIGHT_DEPTH_MAX_EDGE*SIGHT_DEPTH_MAX_EDGE)
typedef struct sight_depth_v1 {
    uint32_t abi_version,struct_size,width,height,source_width,source_height;
    uint64_t sequence;
    uint8_t pixels[SIGHT_DEPTH_MAX_PIXELS];
} sight_depth_v1;
