#include "core/holistic_signal_adapter.hpp"
#include "core/tracking_signal_bus.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {
std::vector<MpNormalizedLandmark> normalized(size_t n,float bias=0.f) {
    std::vector<MpNormalizedLandmark> out(n);
    for(size_t i=0;i<n;++i) {
        out[i].x=bias+static_cast<float>(i)/2000;
        out[i].y=.12f+static_cast<float>(i)/3000;
        out[i].z=-.002f*static_cast<float>(i);
        out[i].has_visibility=true;
        out[i].visibility=.9f;
        out[i].has_presence=true;
        out[i].presence=.8f;
    }
    return out;
}
std::vector<MpLandmark> world(size_t n,float bias=0.f) {
    std::vector<MpLandmark> out(n);
    for(size_t i=0;i<n;++i) {
        out[i].x=bias+.01f*static_cast<float>(i);
        out[i].y=.02f*static_cast<float>(i);
        out[i].z=-.03f*static_cast<float>(i);
    }
    return out;
}
}
int main() {
    auto face=normalized(478), pose=normalized(33);
    auto left=normalized(21,.1f),right=normalized(21,.4f);
    // Separate world coords must stay attached to correct hand.
    auto pose_world=world(33,.2f),left_world=world(21,.7f),right_world=world(21,-.8f);
    std::vector<MpCategory> blends(52);
    const char* blink="eyeBlinkLeft";
    blends[0].category_name=const_cast<char*>(blink);
    blends[0].score=.75f;
    for(size_t i=1;i<blends.size();++i) blends[i].score=.5f;
    MpHolisticLandmarkerResult raw{};
    raw.face_landmarks={face.data(),478};
    raw.pose_landmarks={pose.data(),33};
    raw.pose_world_landmarks={pose_world.data(),33};
    raw.left_hand_landmarks={left.data(),21};
    raw.left_hand_world_landmarks={left_world.data(),21};
    raw.right_hand_landmarks={right.data(),21};
    raw.right_hand_world_landmarks={right_world.data(),21};
    raw.face_blendshapes={blends.data(),52};
    const auto bundle=sight::holistic_to_tracking(raw);
    assert(bundle.face.landmarks.size()==478);
    assert(bundle.face.blendshapes.size()==52);
    assert(bundle.face.blendshape_names[0]=="eyeBlinkLeft");
    assert(std::fabs(bundle.face.blendshapes[0]-.75f)<.0001f);
    assert(!bundle.face.has_head_transform); // not returned by Holistic
    assert(bundle.body.landmarks.size()==33 && bundle.body.world_landmarks.size()==33);
    assert(std::fabs(bundle.body.world_landmarks[0].x-.2f)<.0001f);
    assert(bundle.hand.hands.size()==2 && bundle.hand.landmarks.size()==21);
    assert(bundle.hand.hands[0].handedness=="Left");
    assert(bundle.hand.hands[1].handedness=="Right");
    assert(bundle.hand.hands[0].world_landmarks.size()==21);
    assert(bundle.hand.hands[1].world_landmarks.size()==21);
    assert(std::fabs(bundle.hand.hands[0].world_landmarks[0].x-.7f)<.0001f);
    assert(std::fabs(bundle.hand.hands[1].world_landmarks[0].x+.8f)<.0001f);
    assert(bundle.hand.hands[0].gesture.empty() && bundle.hand.hands[0].gesture_score==0);
    assert(bundle.hand.hands[1].gesture.empty() && bundle.hand.hands[1].gesture_score==0);
    for(auto &h:bundle.hand.hands) assert(std::isfinite(h.pinch) && h.pinch>=0 && h.pinch<=1);

    // Same physical producer may publish synchronized modality snapshots to
    // separate Sight channels without pretending gestures were classified.
    sight::TrackingSignalBus::clear_for_tests();
    auto publisher=sight::TrackingSignalBus::new_publisher_id();
    assert(sight::TrackingSignalBus::publish("holistic/face",publisher,
       sight::TrackingKind::Face,bundle.face,640,480));
    assert(sight::TrackingSignalBus::publish("holistic/body",publisher,
       sight::TrackingKind::Body,bundle.body,640,480));
    assert(sight::TrackingSignalBus::publish("holistic/hand",publisher,
       sight::TrackingKind::Hand,bundle.hand,640,480));
    sight::TrackingSignal received;
    assert(sight::TrackingSignalBus::read("holistic/hand",std::chrono::seconds(2),received));
    assert(received.data.hands.size()==2 && received.data.hands[1].handedness=="Right");
    assert(received.source_width==640 && received.source_height==480);

    // Incomplete modalities must clear only that modality, not silently
    // reuse earlier frames or let partial pose/hand data masquerade as valid.
    face[12].x=std::numeric_limits<float>::quiet_NaN();
    pose_world[4].z=std::numeric_limits<float>::quiet_NaN();
    right[9].z=std::numeric_limits<float>::infinity();
    auto broken=sight::holistic_to_tracking(raw);
    assert(broken.face.landmarks.empty() && broken.face.blendshapes.empty());
    assert(broken.body.landmarks.size()==33 && broken.body.world_landmarks.empty());
    assert(broken.hand.hands.size()==1);
    assert(broken.hand.hands[0].handedness=="Left");
    assert(broken.hand.landmarks.size()==21);
    assert(broken.hand.landmarks[0].x==broken.hand.hands[0].landmarks[0].x);
    // Missing detections must yield empty outputs rather than stale landmarks.
    MpHolisticLandmarkerResult empty{};
    auto missing=sight::holistic_to_tracking(empty);
    assert(missing.face.landmarks.empty() && missing.body.landmarks.empty());
    assert(missing.hand.hands.empty());
    std::cout << "SIGHT_HOLISTIC_SIGNALS_ADAPTER_PASS=1\n";
}
