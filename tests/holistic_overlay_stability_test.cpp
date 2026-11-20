#include "core/holistic_overlay_stability.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

namespace {
using sight::HolisticSignals;
using sight::TrackedPoint;
using sight::TrackingData;

TrackedPoint point(float x,float y,float confidence=1.f) {
    return TrackedPoint{x,y,0.f,confidence};
}
HolisticSignals human(float confidence=1.f) {
    HolisticSignals raw;
    raw.face.landmarks.resize(478,point(.5f,.2f,1.f));
    raw.body.landmarks.resize(33,point(.5f,.5f,confidence));
    auto& pose=raw.body.landmarks;
    pose[11]=point(.39f,.30f,confidence);
    pose[12]=point(.61f,.30f,confidence);
    pose[23]=point(.43f,.59f,confidence);
    pose[24]=point(.57f,.59f,confidence);
    pose[13]=point(.32f,.42f,confidence);
    pose[14]=point(.68f,.42f,confidence);
    pose[15]=point(.27f,.55f,confidence);
    pose[16]=point(.73f,.55f,confidence);
    sight::HandData left;left.handedness="Left";
    left.landmarks.resize(21,point(.27f,.55f,1.f));
    for(unsigned i=0;i<21;++i) {
        left.landmarks[i].x+=.004f*float(i%5);
        left.landmarks[i].y+=.004f*float(i/5);
    }
    raw.hand.hands.push_back(left);
    raw.hand.landmarks=left.landmarks;
    return raw;
}
bool visible(const TrackedPoint& p) {
    return std::isfinite(p.x)&&p.x>=0&&p.x<=1&&p.confidence>.01f;
}
}
int main() {
    using namespace sight;
    HolisticStabilitySettings config;
    assert(config.mode==HolisticStabilityMode::Balanced);
    assert(std::fabs(config.minimum_confidence-.55f)<.001f);
    assert(config.fade_lost);

    const auto raw=human();
    HolisticOverlayStabilizer filter;
    const auto first=filter.update(raw,config);
    assert(first.face.landmarks.size()==478);
    assert(first.face.landmarks[0].x==raw.face.landmarks[0].x);
    assert(first.body.landmarks.size()==33);
    assert(!visible(first.body.landmarks[15])); // new limb needs confirmation
    assert(first.hand.hands.empty());             // same for hands
    const auto second=filter.update(raw,config);
    assert(visible(second.body.landmarks[15]));
    assert(second.hand.hands.size()==1);
    assert(second.hand.hands[0].handedness=="Left");
    assert(second.hand.hands[0].landmarks.size()==21);
    // Never change the input or raw semantic data used by signal publishing.
    assert(raw.body.landmarks[15].x==.27f);
    assert(raw.hand.hands.size()==1 && raw.hand.hands[0].landmarks[0].x==.27f);

    // Small real arm movement is allowed but softened in Balanced mode,
    // rather than snapping to every minor confidence/noise fluctuation.
    auto wobble=raw;
    wobble.body.landmarks[15]=point(.30f,.55f);
    const auto eased=filter.update(wobble,config);
    assert(eased.body.landmarks[15].x>.27f);
    assert(eased.body.landmarks[15].x<.30f);
    filter.reset();
    filter.update(raw,config);
    filter.update(raw,config);

    // Isolated, implausible wrist jump must not draw a phantom hand/arm.
    auto phantom=raw;
    phantom.body.landmarks[15]=point(.97f,.05f,1.f);
    phantom.hand.hands[0].landmarks[0]=point(.96f,.05f);
    auto rejected=filter.update(phantom,config);
    assert(rejected.body.landmarks[15].x<.40f);
    assert(rejected.hand.hands.size()<=1);
    if(!rejected.hand.hands.empty())
        assert(rejected.hand.hands[0].landmarks[0].x<.40f);
    // Returning to plausible coordinates retains original pose identity.
    auto restored=filter.update(raw,config);
    assert(visible(restored.body.landmarks[15]));
    assert(restored.body.landmarks[15].x<.40f);

    // A real sustained arm movement must eventually be accepted even
    // if it initially crosses the jump gate. Never permanently lock an
    // arm to an old pose after confirming a plausible new position.
    auto move=raw;
    move.body.landmarks[15]=point(.52f,.55f);
    auto held=filter.update(move,config);
    assert(held.body.landmarks[15].x<.40f);
    auto adopted=filter.update(move,config);
    assert(visible(adopted.body.landmarks[15]));
    assert(std::fabs(adopted.body.landmarks[15].x-.52f)<.02f);
    // Resume stable baseline before the low-confidence fade test.
    filter.reset();
    filter.update(raw,config);
    restored=filter.update(raw,config);

    // Low-confidence wrist: fade briefly and expire, even when stale
    // coordinates still exist in the model result.
    auto faint=raw;
    faint.body.landmarks[15].confidence=.20f;
    const auto faded=filter.update(faint,config);
    assert(visible(faded.body.landmarks[15]));
    assert(faded.body.landmarks[15].confidence <
           restored.body.landmarks[15].confidence);
    filter.update(faint,config);
    filter.update(faint,config);
    auto expired=filter.update(faint,config);
    assert(!visible(expired.body.landmarks[15]));

    // Anatomical constraint: a consistent impossibly long forearm is not
    // promoted merely because it repeats.
    filter.reset();
    auto impossible=raw;
    impossible.body.landmarks[15]=point(.99f,.01f);
    for(int i=0;i<6;++i)
        expired=filter.update(impossible,config);
    assert(!visible(expired.body.landmarks[15]));

    // Detached hand away from credible pose wrist is rejected even if
    // stable for many frames (background phantom).
    filter.reset();
    auto background=raw;
    for(auto& p:background.hand.hands[0].landmarks) {
        p.x=.90f;p.y=.05f;
    }
    for(int i=0;i<6;++i)
        expired=filter.update(background,config);
    assert(expired.hand.hands.empty());

    // Ambiguous duplicate handedness must not turn into two phantom
    // hands or borrow another side's temporal state.
    filter.reset();
    auto duplicates=raw;
    duplicates.hand.hands.push_back(duplicates.hand.hands[0]);
    filter.update(duplicates,config);
    auto ambiguous=filter.update(duplicates,config);
    assert(ambiguous.hand.hands.empty());

    // Right-side hand must preserve its own subject identity, independently
    // from the left. No first-hand/result-index identity assumption.
    filter.reset();
    auto right=raw;
    right.hand.hands[0].handedness="Right";
    for(auto& p:right.hand.hands[0].landmarks) {
        p.x+=.46f;
    }
    filter.update(right,config);
    auto right_view=filter.update(right,config);
    assert(right_view.hand.hands.size()==1);
    assert(right_view.hand.hands[0].handedness=="Right");
    assert(right_view.hand.hands[0].landmarks[0].x>.70f);

    // If body is outside the frame, a consistent hand remains possible.
    filter.reset();
    auto hand_only=raw;
    hand_only.body.landmarks.clear();
    auto hfirst=filter.update(hand_only,config);
    assert(hfirst.hand.hands.empty());
    auto hsecond=filter.update(hand_only,config);
    assert(hsecond.hand.hands.size()==1);

    // A dropped hand fades temporarily, then clears. Turning fade off
    // hides it on the first missing frame.
    hand_only.hand.hands.clear();
    auto missing=filter.update(hand_only,config);
    assert(missing.hand.hands.size()==1);
    assert(missing.hand.hands[0].landmarks[0].confidence <
           hsecond.hand.hands[0].landmarks[0].confidence);
    for(int i=0;i<4;++i)missing=filter.update(hand_only,config);
    assert(missing.hand.hands.empty());
    filter.reset();
    config.fade_lost=false;
    filter.update(raw,config);
    filter.update(raw,config);
    auto immediate=filter.update(hand_only,config);
    assert(immediate.hand.hands.empty());

    // Off is a reversible, exact preview bypass; no pose rewrite.
    filter.reset();
    config.mode=HolisticStabilityMode::Off;
    auto bypass=filter.update(phantom,config);
    assert(bypass.body.landmarks[15].x==phantom.body.landmarks[15].x);
    assert(bypass.hand.hands[0].landmarks[0].x==
           phantom.hand.hands[0].landmarks[0].x);

    // An out-of-range confidence or non-finite point is never displayed.
    filter.reset();
    config.mode=HolisticStabilityMode::Balanced;
    auto nan=raw;
    nan.body.landmarks[15].x=std::numeric_limits<float>::quiet_NaN();
    filter.update(nan,config);
    auto safe=filter.update(nan,config);
    assert(!visible(safe.body.landmarks[15]));

    // Stable requires three consistent observations. Responsive accepts
    // immediately; strictness is a behavioral distinction, not just a label.
    filter.reset();
    config.mode=HolisticStabilityMode::Stable;
    assert(!visible(filter.update(raw,config).body.landmarks[15]));
    assert(!visible(filter.update(raw,config).body.landmarks[15]));
    assert(visible(filter.update(raw,config).body.landmarks[15]));
    filter.reset();
    config.mode=HolisticStabilityMode::Responsive;
    assert(visible(filter.update(raw,config).body.landmarks[15]));
    std::cout<<"HOLISTIC_STABILITY_PROOF_PASS=YES\n";
}
