#include "core/tracking_signal_bus.hpp"
#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
using namespace sight;
int main() {
    TrackingSignalBus::clear_for_tests();
    const uint64_t writer=TrackingSignalBus::new_publisher_id();
    const uint64_t intruder=TrackingSignalBus::new_publisher_id();
    const std::array<std::string,3> channels{"holistic/face","holistic/body","holistic/hand"};
    std::array<TrackingData,3> data{};
    data[0].landmarks.resize(478);
    data[1].landmarks.resize(33);
    data[2].hands.resize(2);
    for(auto& hand:data[2].hands)hand.landmarks.resize(21);
    data[2].landmarks=data[2].hands.front().landmarks;
    assert(TrackingSignalBus::publish_group(channels,writer,data,640,480));
    std::array<TrackingSignal,3> signals;
    for(size_t i=0;i<3;++i)
        assert(TrackingSignalBus::read(channels[i],std::chrono::seconds(1),signals[i]));
    assert(signals[0].kind==TrackingKind::Face);
    assert(signals[1].kind==TrackingKind::Body);
    assert(signals[2].kind==TrackingKind::Hand);
    assert(signals[1].sequence==signals[0].sequence+1);
    assert(signals[2].sequence==signals[1].sequence+1);
    assert(signals[0].updated==signals[1].updated);
    assert(signals[1].updated==signals[2].updated);
    assert(signals[0].publisher==writer && signals[2].publisher==writer);

    // A collision on ONE modality must not partially publish the other two.
    TrackingSignalBus::withdraw(channels[1],writer);
    TrackingData occupied{};
    assert(TrackingSignalBus::publish(channels[1],intruder,TrackingKind::Body,
                                      occupied,640,480));
    auto face_prev=signals[0].sequence, hand_prev=signals[2].sequence;
    assert(!TrackingSignalBus::publish_group(channels,writer,data,640,480));
    assert(TrackingSignalBus::read(channels[0],std::chrono::seconds(1),signals[0]));
    assert(TrackingSignalBus::read(channels[2],std::chrono::seconds(1),signals[2]));
    assert(signals[0].sequence==face_prev && signals[2].sequence==hand_prev);
    TrackingSignalBus::withdraw(channels[1],intruder);

    // Distinct non-empty channel names are mandatory.
    auto duplicates=channels; duplicates[1]=duplicates[0];
    assert(!TrackingSignalBus::publish_group(duplicates,writer,data,640,480));
    auto missing=channels; missing[0]="";
    assert(!TrackingSignalBus::publish_group(missing,writer,data,640,480));

    // A fresh empty set must clear stale detections rather than keeping them.
    std::array<TrackingData,3> empty{};
    assert(TrackingSignalBus::publish_group(channels,writer,empty,640,480));
    for(size_t i=0;i<3;++i) {
        assert(TrackingSignalBus::read(channels[i],std::chrono::seconds(1),signals[i]));
        assert(signals[i].data.landmarks.empty());
        assert(signals[i].data.hands.empty());
    }
    for(const auto& channel:channels)TrackingSignalBus::withdraw(channel,writer);
    for(const auto& channel:channels)
        assert(!TrackingSignalBus::read(channel,std::chrono::seconds(1),signals[0]));
    std::cout<<"HOLISTIC_BATCH_ATOMIC_PASS=YES\n";
}
