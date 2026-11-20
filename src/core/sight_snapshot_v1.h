#pragma once

/* Read-only OBS-process interoperability contract.
 * The caller owns this complete buffer; the producer never retains pointers.
 * Coordinates are normalized to the Sight producer's untransformed video.
 * Version/size MUST be checked before reading or writing. */
#include <stdint.h>

#define SIGHT_SNAPSHOT_ABI_V1 1u
#define SIGHT_SNAPSHOT_MAX_POINTS 478u
#define SIGHT_SNAPSHOT_MAX_HANDS 2u
#define SIGHT_SNAPSHOT_HAND_POINTS 21u
#define SIGHT_SNAPSHOT_MAX_BLENDSHAPES 64u

#define SIGHT_SNAPSHOT_FACE 1u
#define SIGHT_SNAPSHOT_BODY 2u
#define SIGHT_SNAPSHOT_HAND 3u
/* COCO WholeBody 133-point 2D topology; not MediaPipe Face/Body/Hand. */
#define SIGHT_SNAPSHOT_WHOLEBODY 4u

typedef struct sight_snapshot_point_v1 {
    float x;
    float y;
    float z;
    float confidence;
} sight_snapshot_point_v1;

typedef struct sight_snapshot_v1 {
    uint32_t abi_version;          /* initialized to SIGHT_SNAPSHOT_ABI_V1 by caller */
    uint32_t struct_size;          /* initialized to sizeof(sight_snapshot_v1) by caller */
    uint32_t kind;                 /* FACE/BODY/HAND/WHOLEBODY, 0 on failed read */
    uint32_t source_width;         /* native Sight producer dimensions */
    uint32_t source_height;
    uint32_t point_count;          /* face 478, body 33, hand first 21, wholebody 133, or 0 */
    uint32_t hands_count;          /* up to 2, 0 for other kinds */
    uint32_t blendshape_count;     /* up to 64, face only */
    uint32_t has_head_transform;   /* 0 or 1, face only */
    uint32_t reserved;
    uint64_t sequence;             /* producer bus revision, monotonic */
    sight_snapshot_point_v1 points[SIGHT_SNAPSHOT_MAX_POINTS];
    sight_snapshot_point_v1 hands[SIGHT_SNAPSHOT_MAX_HANDS][SIGHT_SNAPSHOT_HAND_POINTS];
    float pinch[SIGHT_SNAPSHOT_MAX_HANDS];
    float gesture_score[SIGHT_SNAPSHOT_MAX_HANDS];
    char gesture[SIGHT_SNAPSHOT_MAX_HANDS][32];
    char handedness[SIGHT_SNAPSHOT_MAX_HANDS][16];
    float blendshapes[SIGHT_SNAPSHOT_MAX_BLENDSHAPES];
    float head_transform[16];
} sight_snapshot_v1;
