#pragma once

// Pure conversion of a *single* MediaPipe 1.2.3 Holistic result into three
// existing Sight tracking payloads. No model ownership, OBS calls or process
// side effects here; the future producer owns publishing/expiry.
#include "tasks_tracker.hpp"
#include "tasks_world_landmarks.hpp"
#include "hand_signals.hpp"
#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker_result.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace sight {

struct HolisticSignals {
    TrackingData face;
    TrackingData body;
    TrackingData hand;
};

// Reject partial or invalid observations as an entire modality. Zero points
// means no detection; callers must publish an empty result to clear old state.
inline std::vector<TrackedPoint> copy_holistic_image_points(
    const MpNormalizedLandmarks& raw, size_t expected) {
    if (!raw.landmarks || raw.landmarks_count != expected ||
        (expected != 21 && expected != 33 && expected != 478))
        return {};
    std::vector<TrackedPoint> points;
    points.reserve(expected);
    for(size_t i=0;i<expected;++i) {
        const auto& p=raw.landmarks[i];
        if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)||
           (p.has_visibility && !std::isfinite(p.visibility)) ||
           (p.has_presence && !std::isfinite(p.presence)))
            return {};
        float confidence=1.f;
        if(p.has_visibility)
            confidence=std::min(confidence,std::clamp(p.visibility,0.f,1.f));
        if(p.has_presence)
            confidence=std::min(confidence,std::clamp(p.presence,0.f,1.f));
        points.push_back({p.x,p.y,p.z,confidence});
    }
    return points;
}

inline HolisticSignals holistic_to_tracking(
    const MpHolisticLandmarkerResult& raw) {
    HolisticSignals bundle;

    bundle.face.landmarks=copy_holistic_image_points(raw.face_landmarks,478);
    if(!bundle.face.landmarks.empty() && raw.face_blendshapes.categories &&
       raw.face_blendshapes.categories_count==52) {
        std::vector<float> scores;
        std::vector<std::string> names;
        scores.reserve(52);names.reserve(52);
        bool valid=true;
        for(size_t i=0;i<52;++i) {
            const auto& c=raw.face_blendshapes.categories[i];
            if(!std::isfinite(c.score)){valid=false;break;}
            scores.push_back(std::clamp(c.score,0.f,1.f));
            names.emplace_back(c.category_name?c.category_name:"");
        }
        if(valid) {
            bundle.face.blendshapes=std::move(scores);
            bundle.face.blendshape_names=std::move(names);
        }
    }
    // Holistic does not emit a face transformation matrix. Never fabricate
    // one from landmarks: downstream VRM consumers must see it as missing.

    bundle.body.landmarks=copy_holistic_image_points(raw.pose_landmarks,33);
    if(!bundle.body.landmarks.empty())
        bundle.body.world_landmarks=copy_world_landmarks(
            &raw.pose_world_landmarks,33);

    const auto add_hand=[&](const MpNormalizedLandmarks& image,
                            const MpLandmarks& world,
                            const char* side) {
        auto landmarks=copy_holistic_image_points(image,21);
        if(landmarks.empty())return;
        HandData hand;
        hand.handedness=side;
        hand.landmarks=std::move(landmarks);
        hand.world_landmarks=copy_world_landmarks(&world,21);
        hand.pinch=normalized_pinch(hand.landmarks);
        // Holistic DOES NOT classify gestures. A "Thumb_Up" label here
        // would be fabricated even when the input visually resembles one.
        bundle.hand.hands.push_back(std::move(hand));
    };
    // These are semantic subject labels from the task result, not indices
    // assigned according to detection order or a mirrored OBS preview.
    add_hand(raw.left_hand_landmarks,raw.left_hand_world_landmarks,"Left");
    add_hand(raw.right_hand_landmarks,raw.right_hand_world_landmarks,"Right");
    if(!bundle.hand.hands.empty())
        bundle.hand.landmarks=bundle.hand.hands[0].landmarks;

    // The optional segmentation mask is an MpImagePtr owned by the result.
    // Never forward this pointer across the TrackingSignalBus or after the
    // MediaPipe result is closed. A mask export needs an explicit copy/API.
    return bundle;
}

} // namespace sight
