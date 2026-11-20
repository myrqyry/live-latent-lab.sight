#pragma once

#include "touch_field_signals.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace sight {

// Graphics-thread-only state: time-limited wave launches in response to
// cumulative fingertip movement. No inference, textures, or heap allocations.
struct TouchRipple {
    float x=0, y=0, age=0, amplitude=0;
};
class TouchRippleTrail {
public:
    static constexpr size_t Capacity=8;
    static constexpr float Lifetime=1.6f;
    static constexpr float MovementThreshold=.022f;

    bool consume(const TouchFields& fingers, uint64_t sequence,
                 float seconds, float aspect) {
        if(have_sequence_ && sequence==last_sequence_)return false;
        have_sequence_=true;
        last_sequence_=sequence;
        bool launched=false;
        for(size_t i=0;i<anchor_.size();++i) {
            const bool present=i<fingers.count && fingers.fingers[i].active>.5f;
            if(!present) { anchored_[i]=false;continue; }
            const auto point=fingers.fingers[i];
            const auto last=anchor_[i];
            if(!anchored_[i]) {
                emit(point,seconds);
                anchor_[i]=point;
                anchored_[i]=true;
                launched=true;
                continue;
            }
            const float dx=(point.x-last.x)*aspect;
            const float dy=point.y-last.y;
            const float distance=std::sqrt(dx*dx+dy*dy);
            if(distance>=MovementThreshold) {
                // A large tracked jump generates several coherent fronts
                // along the path, rather than jumping the sole ripple center.
                const int steps=std::clamp(int(distance/MovementThreshold),1,4);
                for(int j=1;j<=steps;++j) {
                    const float t=float(j)/float(steps);
                    TouchPoint mid=point;
                    mid.x=last.x+(point.x-last.x)*t;
                    mid.y=last.y+(point.y-last.y)*t;
                    emit(mid,seconds);
                }
                anchor_[i]=point;
                launched=true;
            }
        }
        return launched;
    }

    std::array<TouchRipple,Capacity> active(float seconds,float pinch_boost) const {
        std::array<TouchRipple,Capacity> out{};
        for(size_t i=0;i<Capacity;++i) {
            const auto& item=events_[i];
            const float age=seconds-item.time;
            if(item.valid && age>=0 && age<Lifetime) {
                out[i]={item.x,item.y,age,1.f+item.pinch*pinch_boost};
            }
        }
        return out;
    }
    void reset() {
        anchored_={};
        have_sequence_=false;
        events_={};
    }

private:
    struct Event {
        float x=0,y=0,time=0,pinch=0;
        bool valid=false;
    };
    void emit(const TouchPoint& point,float seconds) {
        events_[next_]={point.x,point.y,seconds,point.pinch,true};
        next_=(next_+1)%Capacity;
    }
    std::array<Event,Capacity> events_{};
    std::array<TouchPoint,2> anchor_{};
    std::array<bool,2> anchored_{};
    size_t next_=0;
    uint64_t last_sequence_=0;
    bool have_sequence_=false;
};

// Unlike a per-frame >0.004 delta counter, this records slow movements
// cumulatively across unique producer snapshots.
struct TouchMotionMeter {
    TouchPoint last{};
    bool have_last=false;
    uint64_t last_sequence=0;
    bool have_sequence=false;
    float travel=0;
    float max_delta=0;
    unsigned moved_snapshots=0;

    void observe(const TouchFields& fields,uint64_t seq,float aspect) {
        if(have_sequence&&seq==last_sequence)return;
        have_sequence=true;
        last_sequence=seq;
        if(!fields.count||fields.fingers[0].active<.5f) {
            have_last=false;return;
        }
        const auto& point=fields.fingers[0];
        if(have_last) {
            const float dx=(point.x-last.x)*aspect;
            const float dy=point.y-last.y;
            const float delta=std::sqrt(dx*dx+dy*dy);
            travel+=delta;
            max_delta=std::max(max_delta,delta);
            if(delta>.0001f)++moved_snapshots;
        }
        last=point;
        have_last=true;
    }
    void clear_interval() {
        travel=0;max_delta=0;moved_snapshots=0;
    }
};
}
