#pragma once
#include <stdint.h>
/* Read with sight_text_read_v1(max_age_ms,snapshot,available).
 * Caller initializes exact ABI/version and sizeof. Source-normalized
 * padded line rectangles used for the OCR effect, confidence 0..1.
 * A valid empty result has count=0. No strings, borrowed textures or
 * disk output. Capture age, not inference completion age. */
#define SIGHT_TEXT_ABI_V1 1u
#define SIGHT_TEXT_MAX_LINES 512u
typedef struct sight_text_line_v1 {
    float x0,y0,x1,y1,confidence;
} sight_text_line_v1;
typedef struct sight_text_v1 {
    uint32_t abi_version,struct_size,source_width,source_height;
    uint64_t sequence;
    uint32_t count,reserved;
    sight_text_line_v1 lines[SIGHT_TEXT_MAX_LINES];
} sight_text_v1;
