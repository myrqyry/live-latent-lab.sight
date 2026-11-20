#include "core/sight_snapshot_export.hpp"
#include <cstdio>
#include <chrono>

int main()
{
    sight::TrackingSignalBus::clear_for_tests();
    auto pub = sight::TrackingSignalBus::new_publisher_id();
    sight::TrackingData face;
    face.landmarks.resize(478);
    face.landmarks[0] = {.25f, .75f, .125f, .9f};
    face.blendshapes = {.2f, .8f};
    face.head_transform[12] = 2.f;
    face.has_head_transform = true;
    if (!sight::TrackingSignalBus::publish("face", pub, sight::TrackingKind::Face, face, 1280, 720)) return 1;
    sight_snapshot_v1 out = {};
    out.abi_version = SIGHT_SNAPSHOT_ABI_V1;
    out.struct_size = sizeof(out);
    if (!sight::export_tracking_snapshot("face", 750, &out)) return 2;
    if (out.kind != SIGHT_SNAPSHOT_FACE || out.point_count != 478 || out.hands_count != 0 ||
        out.points[0].x != .25f || out.points[0].confidence != .9f ||
        out.blendshape_count != 2 || out.blendshapes[1] != .8f ||
        out.head_transform[12] != 2.f || out.has_head_transform != 1 ||
        out.source_width != 1280 || out.source_height != 720 || !out.sequence) return 3;
    sight::TrackingData hand;
    hand.landmarks.resize(21);
    hand.hands.resize(2);
    hand.hands[0].landmarks.resize(21);
    hand.hands[1].landmarks.resize(21);
    hand.hands[0].landmarks[8].x = .4f;
    hand.hands[1].landmarks[8].x = .6f;
    hand.hands[0].pinch = .3f;
    hand.hands[1].pinch = .8f;
    if (!sight::TrackingSignalBus::publish("hand", pub + 1, sight::TrackingKind::Hand, hand, 640, 360)) return 4;
    if (!sight::export_tracking_snapshot("hand", 750, &out)) return 5;
    if (out.kind != SIGHT_SNAPSHOT_HAND || out.hands_count != 2 ||
        out.hands[0][8].x != .4f || out.hands[1][8].x != .6f ||
        out.pinch[0] != .3f || out.pinch[1] != .8f) return 6;
    // A new empty result is valid and must NOT reuse previous points.
    if (!sight::TrackingSignalBus::publish("hand", pub + 1, sight::TrackingKind::Hand, {}, 640, 360)) return 7;
    if (!sight::export_tracking_snapshot("hand", 750, &out)) return 8;
    if (out.point_count || out.hands_count || out.points[0].x != 0) return 9;
    out.abi_version = 31337;
    if (sight::export_tracking_snapshot("hand", 750, &out)) return 10;
    out.abi_version = SIGHT_SNAPSHOT_ABI_V1;
    out.struct_size = sizeof(out) - 8;
    if (sight::export_tracking_snapshot("hand", 750, &out)) return 11;
    out.struct_size = sizeof(out);
    if (sight::export_tracking_snapshot("missing", 750, &out)) return 12;
    if (out.point_count || out.hands_count) return 13;
    sight::TrackingData whole;
    whole.landmarks.resize(133);
    whole.landmarks[9]={.3f,.7f,0,.8f};
    if(!sight::TrackingSignalBus::publish("wholebody",pub+2,sight::TrackingKind::WholeBody,whole,512,512))return 14;
    if(!sight::export_tracking_snapshot("wholebody",750,&out)||out.kind!=SIGHT_SNAPSHOT_WHOLEBODY||
       out.point_count!=133||out.hands_count||out.blendshape_count||out.has_head_transform||
       out.points[9].x!=.3f||out.points[9].z!=0)return 15;
    if(!sight::TrackingSignalBus::publish("wholebody",pub+2,sight::TrackingKind::WholeBody,{},512,512)||
       !sight::export_tracking_snapshot("wholebody",750,&out)||out.point_count||out.kind!=SIGHT_SNAPSHOT_WHOLEBODY)return 16;
    puts("SIGHT_SNAPSHOT_ABI_PASS=1");
    sight::TrackingSignalBus::clear_for_tests();
    return 0;
}
