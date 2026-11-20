#include "core/holistic_overlay_stability.hpp"
#include "core/holistic_person_mask.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

using namespace sight;
static TrackedPoint point(float x,float y) {
    return {x,y,0.f,1.f};
}
static HolisticSignals human() {
    HolisticSignals raw;
    raw.body.landmarks.assign(33,point(.50f,.51f));
    auto& b=raw.body.landmarks;
    b[11]=point(.38f,.30f);b[12]=point(.62f,.30f);
    b[23]=point(.43f,.60f);b[24]=point(.57f,.60f);
    b[13]=point(.28f,.44f);b[14]=point(.72f,.44f);
    b[15]=point(.22f,.56f);b[16]=point(.78f,.56f);
    HandData hand; hand.handedness="Left";
    hand.landmarks.assign(21,point(.22f,.56f));
    raw.hand.hands.push_back(hand);
    return raw;
}
static bool visible(const TrackedPoint& p) {
    return std::isfinite(p.x)&&p.x>=0.f&&p.confidence>.02f;
}
int main() {
    HolisticStabilitySettings settings;
    auto raw=human();
    HolisticPersonMask m;
    assert(!m.valid());
    assert(!m.has_credible_person(raw.body));
    // Explicit dimensions, a privately owned copy, and no raw modifications.
    m.width=80;m.height=80;m.values.assign(80*80,0.f);
    for(int y=12;y<70;++y)for(int x=20;x<61;++x)
        m.values[size_t(y)*80+x]=.95f;
    assert(m.valid());
    assert(m.has_credible_person(raw.body));
    assert(m.foreground_near(.50f,.50f,.04f));
    assert(!m.foreground_near(.96f,.04f,.055f));
    assert(m.foreground_near(.20f,.56f,.085f)); // margin protects thin limbs
    assert(m.foreground_near(.50f,.50f,.02f));
    assert(!m.foreground_near(std::nanf(""),.5f,.04f));
    auto outside=raw;
    // Geometrically plausible background arm, but entirely outside
    // the confidently segmented person silhouette.
    outside.body.landmarks[13]=point(.16f,.32f);
    outside.body.landmarks[15]=point(.08f,.45f);
    for(auto& p:outside.hand.hands[0].landmarks)
        p=point(.08f,.45f);
    // Geometry and mask both reject implausible background elbow/wrist.
    HolisticOverlayStabilizer stabilizer;
    for(int i=0;i<3;++i) {
        auto out=stabilizer.update(outside,settings,&m);
        assert(!visible(out.body.landmarks[13]));
        assert(!visible(out.body.landmarks[15]));
        assert(outside.body.landmarks[15].x==.08f);
    }
    // Valid center body retained; published RAW points unaffected.
    stabilizer.reset();
    for(int i=0;i<3;++i) {
        auto out=stabilizer.update(raw,settings,&m);
        if(i==2)assert(visible(out.body.landmarks[11]));
    }
    // Corrupt and missing masks do not silently remove landmarks.
    HolisticPersonMask invalid=m;
    invalid.values.resize(1);
    assert(!invalid.valid());
    stabilizer.reset();
    auto out=stabilizer.update(raw,settings,&invalid);
    out=stabilizer.update(raw,settings,&invalid);
    assert(visible(out.body.landmarks[15]));
    // All-zero/all-foreground masks fail closed to no gating.
    HolisticPersonMask empty=m,solid=m;
    empty.values.assign(6400,0.f);
    solid.values.assign(6400,1.f);
    assert(!empty.has_credible_person(raw.body));
    assert(!solid.has_credible_person(raw.body));
    stabilizer.reset();
    stabilizer.update(raw,settings,&empty);
    out=stabilizer.update(raw,settings,&empty);
    assert(visible(out.body.landmarks[15]));
    // Setting Off bypasses all masking, even with valid person mask.
    settings.mode=HolisticStabilityMode::Off;
    out=stabilizer.update(outside,settings,&m);
    assert(out.body.landmarks[15].x==outside.body.landmarks[15].x);
    std::cout<<"HOLISTIC_PERSON_MASK_VALIDATION_PASS=YES\n";
}
