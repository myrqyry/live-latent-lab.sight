#include "core/touch_field_events.hpp"
#include <cmath>
#include <iostream>

int main() {
    sight::TouchRippleTrail trail;
    sight::TouchMotionMeter meter;
    sight::TouchFields hands;
    hands.count=1;
    hands.fingers[0]={.2f,.45f,.25f,1.f};
    const float aspect=16.f/9.f;
    if(!trail.consume(hands,1,0.f,aspect))return 1;
    if(trail.consume(hands,1,.1f,aspect))return 2;
    if(trail.active(.1f,1.f)[0].amplitude<1.f)return 3;
    unsigned launches=1;
    for(unsigned i=1;i<=80;++i) {
        hands.fingers[0].x=.2f+.001f*i;
        meter.observe(hands,i,aspect);
        if(trail.consume(hands,i+1,.02f*i,aspect))++launches;
    }
    // Each individual step is <0.004, yet the total spans 154px at 1080p;
    // the previous diagnostics incorrectly logged finger_motion=0.
    if(launches<4 || meter.travel<.12f || meter.max_delta>=.004f)
        return 4;
    if(meter.moved_snapshots<70)return 5;
    const auto live=trail.active(1.6f,1.f);
    unsigned active=0;
    for(const auto& r:live) if(r.amplitude>0)++active;
    if(active<2)return 6;
    const auto expired=trail.active(5.f,1.f);
    for(const auto& r:expired)if(r.amplitude>0)return 7;
    hands.count=0;
    if(trail.consume(hands,999,5.f,aspect))return 8;
    hands.count=1;
    if(!trail.consume(hands,1000,5.01f,aspect))return 9;
    trail.reset();
    if(!trail.consume(hands,1000,6.f,aspect))return 10;
    meter.clear_interval();
    if(meter.travel!=0||meter.moved_snapshots!=0)return 11;
    std::cout<<"PASS: slow cumulative motion emits multiple ripples, unique frames, expiry and reentry\n";
}
