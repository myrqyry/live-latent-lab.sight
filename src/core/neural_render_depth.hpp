#pragma once
#include "sight_depth_v1.h"
#include <callback/proc.h>
namespace sight::nr {
// Producer owns its locks and capture-age validation; caller owns pixels.
// The consumer never retains a borrowed GPU texture from the producer.
inline bool read_depth_snapshot(proc_handler_t* proc,sight_depth_v1& s,
                                uint32_t w,uint32_t h,uint64_t known){
    if(!proc||!w||!h)return false;
    s.abi_version=SIGHT_DEPTH_ABI_V1;s.struct_size=sizeof(s);s.sequence=0;
    calldata_t args;calldata_init(&args);
    calldata_set_ptr(&args,"snapshot",&s);
    calldata_set_int(&args,"max_age_ms",150);
    calldata_set_int(&args,"known_sequence",known);
    bool available=false;
    const bool called=proc_handler_call(proc,"sight_depth_read_v1",&args)&&
        calldata_get_bool(&args,"available",&available);
    calldata_free(&args);
    return called&&available&&s.abi_version==SIGHT_DEPTH_ABI_V1&&s.struct_size==sizeof(s)&&
        s.sequence&&s.width&&s.height&&s.width<=SIGHT_DEPTH_MAX_EDGE&&s.height<=SIGHT_DEPTH_MAX_EDGE&&
        s.source_width==w&&s.source_height==h;
}
}
