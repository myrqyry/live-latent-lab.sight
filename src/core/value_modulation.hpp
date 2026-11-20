#pragma once
#include "tracking_signal_bus.hpp"
#include "hand_signals.hpp"
#include <cstdint>
#include <optional>

namespace sight {
enum class Waveform : int { Sine, Cosine, Square, Pulse, SawUp, SawDown,
    Triangle, AbsSine, Step, RandomHold, SmoothRandom };
enum class SignalFeature : int {
    FaceLandmarkX, FaceLandmarkY, FaceBlendshape,
    BodyLandmarkX, BodyLandmarkY,
    HandLandmarkX, HandLandmarkY, HandPinch, HandGesture,
    WholeBodyLandmarkX, WholeBodyLandmarkY
};
struct WaveOptions {
    Waveform waveform = Waveform::Sine;
    double frequency = 1;
    double phase_offset = 0;
    double amplitude = 1;
    double pulse_width = .5;
    unsigned steps = 4;
    bool invert = false;
};

class WaveOscillator {
public:
    // Deterministic, local RNG; never uses global rand() state.
    explicit WaveOscillator(uint64_t seed=1);
    void advance(double seconds, double frequency);
    double sample(const WaveOptions& o) const; // 0..1
    double phase() const { return phase_; }
private:
    uint64_t state_;
    double phase_ = 0;
    double last_random_ = 0;
    double next_random_ = 0;
    double random_value();
};

// A two-threshold interaction gate. Never writes before a valid sample;
// after acquisition, missing input releases to zero rather than freezing.
class ThresholdGate {
public:
    std::optional<double> sample(std::optional<double> value,
                                 double activate, double release);
    void reset() { seen_=false; active_=false; }
private:
    bool seen_=false;
    bool active_=false;
};

std::optional<double> read_signal_value(const TrackingSignal& signal,
                                        SignalFeature feature, unsigned index,
                                        unsigned hand_index=0,
                                        HandGesture selected=HandGesture::OpenPalm);
double range_map(double normalized, double minimum, double maximum);
}
