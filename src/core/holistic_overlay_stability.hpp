#pragma once

// Display-only temporal stabilization for Holistic. One worker owns this
// object. TrackingSignalBus always receives original model observations.
// Neither inferred pose nor faded hands are allowed into raw signal outputs.

#include "holistic_signal_adapter.hpp"
#include "holistic_person_mask.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace sight {

enum class HolisticStabilityMode : int { Off=0, Responsive=1, Balanced=2, Stable=3 };

struct HolisticStabilitySettings {
    HolisticStabilityMode mode=HolisticStabilityMode::Balanced;
    float minimum_confidence=.55f;
    bool fade_lost=true;
};

namespace stability_detail {
inline bool valid(const TrackedPoint& p,float confidence,bool wrist=false) {
    const float margin=wrist?0.f:.18f;
    return std::isfinite(p.x) && std::isfinite(p.y) &&
           std::isfinite(p.z) && std::isfinite(p.confidence) &&
           p.x>=-margin && p.x<=1.f+margin &&
           p.y>=-margin && p.y<=1.f+margin &&
           p.confidence>=confidence;
}
inline float distance(const TrackedPoint& a,const TrackedPoint& b) {
    return std::hypot(a.x-b.x,a.y-b.y);
}
inline TrackedPoint hidden() {return TrackedPoint{-1.f,-1.f,0.f,0.f};}
inline int confirmations(HolisticStabilityMode mode) {
    return mode==HolisticStabilityMode::Stable?3:
           mode==HolisticStabilityMode::Balanced?2:1;
}
inline int lost_limit(HolisticStabilityMode mode) {
    return mode==HolisticStabilityMode::Stable?3:
           mode==HolisticStabilityMode::Balanced?2:1;
}
inline float max_jump(HolisticStabilityMode mode) {
    return mode==HolisticStabilityMode::Stable?.13f:
           mode==HolisticStabilityMode::Balanced?.20f:.32f;
}
inline float smoothing_alpha(HolisticStabilityMode mode) {
    return mode==HolisticStabilityMode::Stable?.38f:
           mode==HolisticStabilityMode::Balanced?.60f:1.f;
}
inline TrackedPoint mix(const TrackedPoint& previous,const TrackedPoint& next,
                        float alpha) {
    auto out=next;
    out.x=previous.x*(1.f-alpha)+next.x*alpha;
    out.y=previous.y*(1.f-alpha)+next.y*alpha;
    out.z=previous.z*(1.f-alpha)+next.z*alpha;
    return out;
}
struct Joint {
    TrackedPoint last{};
    TrackedPoint candidate{};
    unsigned candidate_hits=0;
    unsigned missed=0;
    bool active=false;

    void clear(){*this=Joint{};}

    TrackedPoint update(const TrackedPoint* observed,
                        const HolisticStabilitySettings& opts) {
        const float minimum=std::clamp(opts.minimum_confidence,0.f,1.f);
        const bool plausible=observed&&valid(*observed,minimum);
        const int required=confirmations(opts.mode);
        const float gate=max_jump(opts.mode);
        // New joints and large jumps require consecutive consistent samples.
        const bool matches=plausible && (!active ||
                             distance(last,*observed)<=gate);
        if(plausible && active && matches) {
            last=mix(last,*observed,smoothing_alpha(opts.mode));
            missed=0;candidate_hits=0;
            return last;
        }
        if(plausible) {
            const float proposal_gate=std::min(.13f,gate*.65f);
            if(candidate_hits && distance(candidate,*observed)<=proposal_gate)
                ++candidate_hits;
            else candidate_hits=1;
            candidate=*observed;
            if(candidate_hits>=unsigned(required)) {
                last=*observed;
                active=true;missed=0;candidate_hits=0;
                return last;
            }
        } else candidate_hits=0;

        if(!active)return hidden();
        ++missed;
        if(missed>unsigned(lost_limit(opts.mode))) {
            active=false;last={};
            return hidden();
        }
        if(!opts.fade_lost)return hidden();
        auto fading=last;
        fading.confidence*=std::pow(.70f,float(missed));
        return fading;
    }
};
struct Hand {
    HandData last{};
    HandData candidate{};
    unsigned candidate_hits=0,missed=0;
    bool active=false;
    void clear(){*this=Hand{};}

    static bool usable(const HandData* data,float minimum) {
        if(!data || data->landmarks.size()!=21 ||
           !valid(data->landmarks[0],minimum,true))return false;
        return std::all_of(data->landmarks.begin(),data->landmarks.end(),
          [minimum](const TrackedPoint& p){return valid(p,minimum);});
    }
    bool update(const HandData* observed,
                const HolisticStabilitySettings& opts,HandData& out) {
        const float minimum=std::clamp(opts.minimum_confidence,0.f,1.f);
        const bool plausible=usable(observed,minimum);
        const int required=confirmations(opts.mode);
        const float gate=max_jump(opts.mode);
        const bool matches=plausible && active &&
            distance(last.landmarks[0],observed->landmarks[0])<=gate;
        if(matches) {
            const float alpha=smoothing_alpha(opts.mode);
            // Reuse the original hand identity; smooth only its image
            // coordinates for the overlay, never body-world metrics.
            auto next=*observed;
            for(size_t i=0;i<21;++i)
                next.landmarks[i]=mix(last.landmarks[i],
                                      observed->landmarks[i],alpha);
            last=std::move(next);
            missed=0;candidate_hits=0;
            out=last;return true;
        }
        if(plausible) {
            if(candidate_hits &&
               distance(candidate.landmarks[0],
                        observed->landmarks[0])<=.12f)
                ++candidate_hits;
            else candidate_hits=1;
            candidate=*observed;
            if(candidate_hits>=unsigned(required)) {
                last=*observed;active=true;missed=0;
                candidate_hits=0;
                out=last;return true;
            }
        } else candidate_hits=0;
        if(!active)return false;
        ++missed;
        if(missed>unsigned(lost_limit(opts.mode))) {
            active=false;last={};return false;
        }
        if(!opts.fade_lost)return false;
        out=last;
        for(auto& point:out.landmarks)
            point.confidence*=std::pow(.70f,float(missed));
        return true;
    }
};
} // namespace stability_detail

class HolisticOverlayStabilizer {
public:
    void reset() {
        for(auto& joint:pose_)joint.clear();
        for(auto& hand:hands_)hand.clear();
    }
    HolisticSignals update(const HolisticSignals& input,
                          HolisticStabilitySettings options,
                          const HolisticPersonMask* mask=nullptr) {
        // Explicit Off is an exact reversible bypass.
        if(options.mode==HolisticStabilityMode::Off) {
            reset();return input;
        }
        options.minimum_confidence=
            std::clamp(options.minimum_confidence,0.f,1.f);
        HolisticSignals output;
        // The original face path has no temporal confidence model and is
        // unchanged; pose/hand association is what helps phantom limbs.
        output.face=input.face;
        const auto& raw_pose=input.body.landmarks;
        output.body.landmarks.resize(33,stability_detail::hidden());
        // The mask may be absent, blank, stale/corrupt, or omit an arm.
        // Only gate when torso anchors independently support a person.
        const bool mask_credible=mask && mask->has_credible_person(input.body);

        float torso=.18f;
        if(raw_pose.size()==33 &&
           stability_detail::valid(raw_pose[11],options.minimum_confidence) &&
           stability_detail::valid(raw_pose[23],options.minimum_confidence))
            torso=std::max(torso,stability_detail::distance(raw_pose[11],raw_pose[23]));
        if(raw_pose.size()==33 &&
           stability_detail::valid(raw_pose[12],options.minimum_confidence) &&
           stability_detail::valid(raw_pose[24],options.minimum_confidence))
            torso=std::max(torso,stability_detail::distance(raw_pose[12],raw_pose[24]));

        for(size_t i=0;i<pose_.size();++i) {
            const TrackedPoint* observed=raw_pose.size()==33?&raw_pose[i]:nullptr;
            // Conservative local limb checks. The arm endpoints can be
            // partially offscreen; only reject truly unreasonable geometry.
            const int parent = i==13?11:i==15?13:i==14?12:i==16?14:-1;
            if(parent>=0 && observed && raw_pose.size()==33 &&
               stability_detail::valid(*observed,options.minimum_confidence) &&
               stability_detail::valid(raw_pose[size_t(parent)],options.minimum_confidence) &&
               stability_detail::distance(*observed,raw_pose[size_t(parent)]) >
                     std::max(.30f,torso*1.65f))
                observed=nullptr;
            if(mask_credible && observed && (i==13||i==14||i==15||i==16) &&
               !mask->foreground_near(observed->x,observed->y,.085f))
                observed=nullptr;
            output.body.landmarks[i]=pose_[i].update(observed,options);
        }

        // Prevent isolated elbow/wrist dots floating without their
        // supporting visible arm chain. The raw bus is unaffected.
        for(const auto [child,parent]:std::array<std::pair<int,int>,4>{{
             {13,11},{15,13},{14,12},{16,14}
        }}) {
            if(!stability_detail::valid(output.body.landmarks[size_t(parent)],.02f))
                output.body.landmarks[size_t(child)]=stability_detail::hidden();
        }

        // Holistic explicitly labels LEFT and RIGHT in subject coordinates.
        // Never switch a stored hand over to the other side by result index.
        std::array<const HandData*,2> input_hands{};
        std::array<unsigned,2> side_counts{};
        for(const auto& hand:input.hand.hands) {
            const int side=hand.handedness=="Left"?0:
                            hand.handedness=="Right"?1:-1;
            if(side>=0) {
                ++side_counts[size_t(side)];
                input_hands[size_t(side)]=&hand;
            }
        }
        for(size_t side=0;side<2;++side) {
            const auto* hand=side_counts[side]==1?input_hands[side]:nullptr;
            if(hand && raw_pose.size()==33) {
                const size_t wrist=side==0?15:16;
                const auto& body_wrist=raw_pose[wrist];
                if(stability_detail::valid(body_wrist,options.minimum_confidence,true) &&
                   hand->landmarks.size()==21 &&
                   stability_detail::distance(
                       hand->landmarks[0],body_wrist)>.26f)
                    hand=nullptr; // Reject clearly detached background hands.
            }
            if(mask_credible && hand && hand->landmarks.size()==21 &&
               !mask->foreground_near(hand->landmarks[0].x,
                                     hand->landmarks[0].y,.095f)) {
                // Segmentation can miss outstretched hands. Only reject a
                // background root when it also lacks wrist confirmation.
                bool wrist_agrees=false;
                const auto wrist=side==0?15u:16u;
                if(raw_pose.size()==33 &&
                   stability_detail::valid(raw_pose[wrist],options.minimum_confidence,true))
                    wrist_agrees=stability_detail::distance(
                        raw_pose[wrist],hand->landmarks[0])<.13f &&
                        mask->foreground_near(raw_pose[wrist].x,raw_pose[wrist].y,.085f);
                if(!wrist_agrees)hand=nullptr;
            }
            HandData displayed;
            if(hands_[side].update(hand,options,displayed))
                output.hand.hands.push_back(std::move(displayed));
        }
        if(!output.hand.hands.empty())
            output.hand.landmarks=output.hand.hands[0].landmarks;
        return output;
    }
private:
    std::array<stability_detail::Joint,33> pose_{};
    std::array<stability_detail::Hand,2> hands_{};
};
} // namespace sight
