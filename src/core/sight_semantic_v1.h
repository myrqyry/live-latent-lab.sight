#pragma once
#include <stdint.h>
/* Caller-owned copy, exact version and size. Normalized source rectangles.
 * No borrowed objects or textures. Sequence is process-wide. */
#define SIGHT_SEMANTIC_ABI_V1 1u
#define SIGHT_SEMANTIC_MAX_REGIONS 256u
typedef struct sight_semantic_region_v1 {
    float x0,y0,x1,y1,confidence;
    uint32_t category,held,reserved;
    uint64_t id;
} sight_semantic_region_v1;
typedef struct sight_semantic_v1 {
    uint32_t abi_version,struct_size,source_width,source_height;
    uint64_t sequence;
    uint32_t count,reserved;
    sight_semantic_region_v1 regions[SIGHT_SEMANTIC_MAX_REGIONS];
} sight_semantic_v1;
