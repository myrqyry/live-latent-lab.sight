#include "core/holistic_tracker.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=4) {
        std::cerr<<"usage: libmediapipe.so holistic.task frame-rgba-512.raw\n";
        return 2;
    }
    std::ifstream file(argv[3],std::ios::binary);
    std::vector<unsigned char> rgba((std::istreambuf_iterator<char>(file)),{});
    if(rgba.size()!=512*512*4) {std::cerr<<"wrong RGBA fixture";return 2;}
    sight::HolisticTracker tracker;
    sight::HolisticSignals preloaded;
    assert(!tracker.detect(rgba.data(),512,512,0,preloaded));
    assert(tracker.load(argv[1],argv[2]));
    for(int i=0;i<3;++i) {
        sight::HolisticSignals data;
        if(!tracker.detect(rgba.data(),512,512,33*(i+1),data)) {
            std::cerr<<"Real Holistic failed: "<<tracker.last_error()<<"\n"; return 1;
        }
        assert(data.face.landmarks.size()==478 && data.face.blendshapes.size()==52);
        assert(data.body.landmarks.size()==33 && data.body.world_landmarks.size()==33);
        assert(!data.face.has_head_transform);
        assert(data.hand.hands.size()<=2);
        for(auto& hand:data.hand.hands) {
            assert(hand.landmarks.size()==21 && hand.world_landmarks.size()==21);
            assert(hand.gesture.empty());
        }
        std::cout<<"HOLISTIC_REAL frame="<<i<<" face="<<data.face.landmarks.size()
                 <<" pose="<<data.body.landmarks.size()
                 <<" hands="<<data.hand.hands.size()<<"\n";
    }
    sight::HolisticSignals stale;
    assert(!tracker.detect(rgba.data(),512,512,99,stale)); // nonmonotonic
    tracker.unload();
    assert(!tracker.ready());
    // Separate mode enables the official pose mask; it must be copied
    // while the native MpHolisticLandmarkerResult is still alive.
    assert(tracker.load(argv[1],argv[2],true));
    sight::HolisticSignals masked_signals;
    sight::HolisticPersonMask owned;
    assert(tracker.detect(rgba.data(),512,512,133,masked_signals,&owned));
    assert(owned.valid());
    assert(owned.width>0 && owned.height>0);
    assert(!owned.values.empty());
    size_t foreground=0;
    for(float v:owned.values) {
        assert(std::isfinite(v) && v>=0.f && v<=1.f);
        if(v>.5f)++foreground;
    }
    const bool credible=owned.has_credible_person(masked_signals.body);
    assert(credible); // this real reference frame must reach mask validation
    std::cout<<"HOLISTIC_REAL_MASK size="<<owned.width<<"x"<<owned.height
             <<" foreground="<<foreground<<" pixels="<<owned.values.size()
             <<" credible_person="<<credible<<"\n";
    assert(foreground>0 && foreground<owned.values.size());
    tracker.unload();
    assert(owned.valid()); // deep copy survives native result and graph teardown

    std::cout<<"HOLISTIC_REAL_TRACKER_PASS=YES\n";
}
