#include "core/value_modulation.hpp"
#include <cmath>
#include <iostream>
using namespace sight;
bool equal(double a,double b,double tolerance=1e-6) {return std::abs(a-b)<=tolerance;}
int main(){
    WaveOscillator oscillator(123);
    WaveOptions p{};
    if(!equal(oscillator.sample(p),.5))return 1;
    oscillator.advance(.25,1);
    if(!equal(oscillator.sample(p),1))return 2;
    oscillator.advance(.25,1);
    if(!equal(oscillator.sample(p),.5))return 3;
    oscillator.advance(.5,1);
    if(!equal(oscillator.sample(p),.5))return 4;
    p.waveform=Waveform::Square;
    if(!equal(oscillator.sample(p),1))return 5;
    p.waveform=Waveform::Pulse;
    if(!equal(oscillator.sample(p),1))return 16;
    oscillator.advance(.75,1);
    if(!equal(oscillator.sample(p),.5))return 17;
    oscillator.advance(.25,1);
    p.waveform=Waveform::Triangle;
    if(!equal(oscillator.sample(p),0))return 6;
    p.waveform=Waveform::SawUp;
    if(!equal(oscillator.sample(p),0))return 7;
    p.waveform=Waveform::Cosine;
    if(!equal(oscillator.sample(p),1))return 8;
    p.waveform=Waveform::Sine;p.amplitude=0;
    if(!equal(oscillator.sample(p),.5))return 9;
    p.amplitude=1;p.invert=true;
    oscillator.advance(.25,1);
    if(!equal(oscillator.sample(p),0))return 10;
    p.invert=false;
    p.waveform=Waveform::SmoothRandom;
    if(oscillator.sample(p)<0||oscillator.sample(p)>1)return 11;
    p.waveform=Waveform::Step;p.steps=4;
    if(!equal(oscillator.sample(p),1./3.))return 12;
    TrackingSignal signal;
    signal.kind=TrackingKind::Face;
    signal.data.landmarks.resize(478);
    signal.data.landmarks[4].x=.2f;
    signal.data.landmarks[4].y=.7f;
    signal.data.blendshapes.resize(52);
    signal.data.blendshapes[25]=.9f;
    auto x=read_signal_value(signal,SignalFeature::FaceLandmarkX,4);
    auto mouth=read_signal_value(signal,SignalFeature::FaceBlendshape,25);
    if(!x||!mouth||!equal(*x,.2,.001)||!equal(*mouth,.9,.001))return 13;
    if(read_signal_value(signal,SignalFeature::BodyLandmarkX,4) ||
       read_signal_value(signal,SignalFeature::FaceBlendshape,99))return 14;
    signal.data.landmarks.clear();
    if(read_signal_value(signal,SignalFeature::FaceLandmarkX,4))return 15;
    if(!equal(range_map(.2,10,110),30))return 16;
    sight::TrackingSignal hand;
    hand.kind=TrackingKind::Hand;
    hand.data.hands.resize(2);
    hand.data.hands[0].landmarks.resize(21);
    hand.data.hands[1].landmarks.resize(21);
    hand.data.hands[0].landmarks[8].x=.77f;
    hand.data.hands[1].landmarks[8].x=.23f;
    hand.data.hands[0].gesture="Thumb_Up";
    hand.data.hands[0].gesture_score=.8f;
    hand.data.hands[0].pinch=.9f;
    auto thumb=read_signal_value(hand,SignalFeature::HandGesture,0,0,HandGesture::ThumbUp);
    auto index=read_signal_value(hand,SignalFeature::HandLandmarkX,8,1);
    auto pinch=read_signal_value(hand,SignalFeature::HandPinch,0,0);
    if(!thumb||!index||!pinch||!equal(*thumb,.8,.001)||
       !equal(*index,.23,.001)||!equal(*pinch,.9,.001))return 18;
    if(read_signal_value(hand,SignalFeature::HandLandmarkX,21,0) ||
       read_signal_value(hand,SignalFeature::HandPinch,0,2) ||
       read_signal_value(signal,SignalFeature::HandPinch,0,0))return 19;
    if(!equal(read_signal_value(hand,SignalFeature::HandGesture,0,0,
                   HandGesture::Victory).value_or(-1),0))return 20;
    std::vector<TrackedPoint> points(21);
    points[0].x=.1f;points[0].y=.1f;
    points[9].x=.1f;points[9].y=.3f;
    points[4].x=.3f;points[4].y=.3f;
    points[8].x=.3f;points[8].y=.3f;
    if(normalized_pinch(points)<.95f)return 21;
    points[8].x=.7f;
    if(normalized_pinch(points)>.1f)return 22;
    TrackingSignal whole;
    whole.kind=TrackingKind::WholeBody;
    whole.data.landmarks.resize(133);
    whole.data.landmarks[9]={.23f,.77f,0,.8f};
    if(!equal(read_signal_value(whole,SignalFeature::WholeBodyLandmarkX,9).value_or(-1),.23,.001))return 37;
    if(!equal(read_signal_value(whole,SignalFeature::WholeBodyLandmarkY,9).value_or(-1),.77,.001))return 38;
    if(read_signal_value(whole,SignalFeature::BodyLandmarkX,9)||
       read_signal_value(whole,SignalFeature::FaceLandmarkX,9)||
       read_signal_value(whole,SignalFeature::HandLandmarkX,9)||
       read_signal_value(hand,SignalFeature::WholeBodyLandmarkX,9)||
       read_signal_value(whole,SignalFeature::WholeBodyLandmarkX,133))return 39;
    whole.data.landmarks[9].confidence=.1f;
    if(read_signal_value(whole,SignalFeature::WholeBodyLandmarkY,9))return 40;
    whole.data.landmarks.clear();
    if(read_signal_value(whole,SignalFeature::WholeBodyLandmarkX,9))return 41;
    ThresholdGate gate;
    // No data before first observation must not touch the destination.
    if(gate.sample(std::nullopt,.65,.45))return 23;
    if(gate.sample(.50,.65,.45)!=0.)return 24;
    if(gate.sample(.65,.65,.45)!=1.)return 25;
    if(gate.sample(.55,.65,.45)!=1.)return 26;
    if(gate.sample(.45,.65,.45)!=0.)return 27;
    if(gate.sample(.55,.65,.45)!=0.)return 28;
    if(gate.sample(.90,.65,.45)!=1.)return 29;
    // Missing tracking releases the effect; a mid-band recovery stays off.
    if(gate.sample(std::nullopt,.65,.45)!=0.)return 30;
    if(gate.sample(.55,.65,.45)!=0.)return 31;
    if(gate.sample(.90,.65,.45)!=1.)return 32;
    gate.reset();
    if(gate.sample(std::nullopt,.65,.45))return 33;
    if(gate.sample(.9,.4,.6))return 34;
    if(gate.sample(.9,.65,.65))return 35;
    if(gate.sample(.9,1.2,.45))return 36;
    std::cout<<"PASS: wave shapes, phase progression, output mapping, typed signal selection\n";
}
