#pragma once
#include <stdint.h>
/* Source-local worker capture motion. Caller initializes ABI/version+size.
 * Read sight_motion_read_v1(max_age_ms,snapshot,available). Energy is mean
 * absolute luma delta across the whole input, normalized [0,1], not speed.
 * A static pair is valid zero; first frame, excessive gap and stale = absent.
 * No pointer ownership or changes to the matte/landmark ABI. */
#define SIGHT_MOTION_ABI_V1 1u
typedef struct sight_motion_v1 {
    uint32_t abi_version, struct_size, source_width, source_height;
    uint64_t sequence;
    float energy;
    uint32_t reserved;
} sight_motion_v1;
