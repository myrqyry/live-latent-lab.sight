#include <obs.h>
#include <obs-module.h>
#include "core/sight_snapshot_v1.h"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv)
{
    if (argc != 3 || !obs_startup("en-US", nullptr, nullptr)) return 2;
    obs_video_info video{};
    video.graphics_module = "libobs-opengl";
    video.fps_num=30; video.fps_den=1;
    video.base_width=128; video.base_height=128;
    video.output_width=128; video.output_height=128;
    video.output_format=VIDEO_FORMAT_RGBA;
    video.scale_type=OBS_SCALE_DISABLE;
    if (obs_reset_video(&video) != OBS_VIDEO_SUCCESS) { obs_shutdown(); return 3; }
    obs_module_t* module = nullptr;
    const int status=obs_open_module(&module, argv[1], argv[2]);
    if (status != MODULE_SUCCESS || !obs_init_module(module)) { obs_shutdown(); return 4; }
    auto* reader=obs_source_create_private("sight_tracking_receiver_filter", "Sight bridge test", nullptr);
    if (!reader) { obs_shutdown(); return 5; }
    sight_snapshot_v1 snap{};
    snap.abi_version=SIGHT_SNAPSHOT_ABI_V1;
    snap.struct_size=sizeof(snap);
    calldata_t cd;
    calldata_init(&cd);
    calldata_set_string(&cd,"channel","missing");
    calldata_set_int(&cd,"max_age_ms",750);
    calldata_set_ptr(&cd,"snapshot",&snap);
    const bool called=proc_handler_call(obs_source_get_proc_handler(reader),"sight_tracking_read_v1",&cd);
    bool available=true;
    const bool read=calldata_get_bool(&cd,"available",&available);
    calldata_free(&cd);
    obs_source_release(reader);
    obs_shutdown();
    if (!called || !read || available || snap.abi_version != SIGHT_SNAPSHOT_ABI_V1 ||
        snap.point_count != 0 || snap.hands_count != 0) return 6;
    puts("SIGHT_SNAPSHOT_PROC_PASS=1");
    return 0;
}
