#include "value_modulation.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace sight {
namespace {
constexpr double TwoPi=6.283185307179586476925286766559;
double fract(double x) { return x - std::floor(x); }
}
double WaveOscillator::random_value() {
    uint64_t x=state_;
    if(!x)x=0x9e3779b97f4a7c15ULL;
    x^=x>>12;x^=x<<25;x^=x>>27;
    state_=x;
    const auto number=x*2685821657736338717ULL;
    return 2.*double(number>>11)*(1./9007199254740992.)-1.;
}
WaveOscillator::WaveOscillator(uint64_t seed):state_(seed?seed:1) {
    last_random_=random_value();
    next_random_=random_value();
}
void WaveOscillator::advance(double seconds,double frequency) {
    if(!std::isfinite(seconds)||!std::isfinite(frequency)||
       seconds<=0||frequency<=0)return;
    // OBS may deliver a delayed tick. Advance the phase correctly rather
    // than accumulating unbounded state or only handling one wrap.
    const double cycles=std::min(seconds,1.0)*std::clamp(frequency,.01,20.);
    const auto wraps=static_cast<unsigned>(std::floor(phase_+cycles));
    phase_=fract(phase_+cycles);
    for(unsigned i=0;i<std::min(20u,wraps);i++){
        last_random_=next_random_;
        next_random_=random_value();
    }
}
double WaveOscillator::sample(const WaveOptions& opts) const {
    const double t=fract(phase_+opts.phase_offset);
    const double angle=TwoPi*t;
    double wave=0;
    switch(opts.waveform){
    case Waveform::Sine: wave=std::sin(angle);break;
    case Waveform::Cosine: wave=std::cos(angle);break;
    case Waveform::Square: wave=t<std::clamp(opts.pulse_width,.01,.99)?1:-1;break;
    case Waveform::Pulse:
        // Unlike bipolar square, unipolar pulse rests at the midpoint
        // of the requested output range and rises to the high endpoint.
        wave=t<std::clamp(opts.pulse_width,.01,.99)?1:0;
        break;
    case Waveform::SawUp: wave=2*t-1;break;
    case Waveform::SawDown: wave=1-2*t;break;
    case Waveform::Triangle: wave=1-4*std::abs(t-.5);break;
    case Waveform::AbsSine: wave=2*std::abs(std::sin(angle))-1;break;
    case Waveform::Step: {
        const double steps=std::clamp(opts.steps,2u,64u);
        wave=2*std::floor(t*steps)/(steps-1)-1;
        break;
    }
    case Waveform::RandomHold: wave=last_random_;break;
    case Waveform::SmoothRandom: {
        const double eased=t*t*(3-2*t);
        wave=last_random_*(1-eased)+next_random_*eased;break;
    }
    }
    if(opts.invert)wave=-wave;
    return std::clamp(.5+.5*wave*std::clamp(opts.amplitude,0.,2.),0.,1.);
}
std::optional<double> ThresholdGate::sample(std::optional<double> value,
                                              double activate,double release) {
    if(!std::isfinite(activate)||!std::isfinite(release)||
       release<0||activate>1||release>=activate) {
        reset();
        return std::nullopt;
    }
    if(!value||!std::isfinite(*value)) {
        active_=false;
        return seen_?std::optional<double>(0.):std::nullopt;
    }
    seen_=true;
    if(*value>=activate)active_=true;
    else if(*value<=release)active_=false;
    return active_?1.:0.;
}
std::optional<double> read_signal_value(const TrackingSignal& signal,
                                        SignalFeature feature,unsigned index,
                                        unsigned hand_index,HandGesture selected) {
    if(feature==SignalFeature::WholeBodyLandmarkX||feature==SignalFeature::WholeBodyLandmarkY) {
        if(signal.kind!=TrackingKind::WholeBody||signal.data.landmarks.size()!=133||index>=133)
            return std::nullopt;
        const auto& p=signal.data.landmarks[index];
        const float v=feature==SignalFeature::WholeBodyLandmarkX?p.x:p.y;
        if(!std::isfinite(p.confidence)||p.confidence<.45f||!std::isfinite(v))
            return std::nullopt;
        return std::clamp(double(v),0.,1.);
    }
    if(feature>=SignalFeature::HandLandmarkX) {
        if(signal.kind!=TrackingKind::Hand || hand_index>=signal.data.hands.size())
            return std::nullopt;
        const auto& hand=signal.data.hands[hand_index];
        if(feature==SignalFeature::HandPinch)
            return std::clamp(double(hand.pinch),0.,1.);
        if(feature==SignalFeature::HandGesture)
            return std::clamp(double(gesture_match(hand,selected)),0.,1.);
        if(index>=hand.landmarks.size())return std::nullopt;
        const auto& pt=hand.landmarks[index];
        const double v=feature==SignalFeature::HandLandmarkX?pt.x:pt.y;
        if(!std::isfinite(v))return std::nullopt;
        return std::clamp(v,0.,1.);
    }
    const bool face=feature==SignalFeature::FaceLandmarkX||
                    feature==SignalFeature::FaceLandmarkY||
                    feature==SignalFeature::FaceBlendshape;
    if(signal.kind!=(face?TrackingKind::Face:TrackingKind::Body))
        return std::nullopt;
    double v=0;
    if (feature==SignalFeature::FaceBlendshape) {
        if(index>=signal.data.blendshapes.size())return std::nullopt;
        v=signal.data.blendshapes[index];
    } else {
        if(index>=signal.data.landmarks.size())return std::nullopt;
        const auto& p=signal.data.landmarks[index];
        if(p.confidence<.45f)return std::nullopt;
        const bool use_x=feature==SignalFeature::FaceLandmarkX||
                         feature==SignalFeature::BodyLandmarkX;
        v=use_x?p.x:p.y;
    }
    if(!std::isfinite(v)) return std::nullopt;
    return std::clamp(v,0.,1.);
}
double range_map(double t,double min,double max) {
    if(!std::isfinite(t)||!std::isfinite(min)||!std::isfinite(max)||
       max<min) return min;
    return min+(max-min)*std::clamp(t,0.,1.);
}
}
