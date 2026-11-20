#pragma once
#include <stdint.h>
#define SIGHT_FLOW_ABI_V1 1u
#define SIGHT_FLOW_WIDTH 32u
#define SIGHT_FLOW_HEIGHT 18u
#define SIGHT_FLOW_CELLS (SIGHT_FLOW_WIDTH*SIGHT_FLOW_HEIGHT)
/* At current-frame cell centers: backward displacement in normalized
 * source UV (current UV + displacement samples previous capture).
 * Confidence is 0..1; invalid cells are zero. Capture interval seconds
 * is separate from displacement. No borrowed memory or textures. */
typedef struct sight_flow_cell_v1 {float dx,dy,confidence,reserved;} sight_flow_cell_v1;
typedef struct sight_flow_v1 {
    uint32_t abi_version,struct_size,source_width,source_height;
    uint64_t sequence;
    float interval_seconds,coverage;
    sight_flow_cell_v1 cells[SIGHT_FLOW_CELLS];
} sight_flow_v1;
