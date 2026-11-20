#include "core/rtmw_tracker.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
int main(int argc,char** argv){
    if(argc!=5&&argc!=6)return 2;
    const bool yolo=argc==6;
    const size_t expected_compiles=yolo?2:1;
    std::ifstream f(argv[4],std::ios::binary);
    constexpr unsigned w=512,h=512;
    std::vector<uint8_t> pixels(w*h*4);
    if(!f.read(reinterpret_cast<char*>(pixels.data()),pixels.size()))return 2;
    sight::RtmwTracker tracker;
    if(!tracker.load(argv[1],argv[2],argv[3],yolo?sight::RtmwPersonBackend::YoloX:sight::RtmwPersonBackend::EfficientDet)){
        std::cerr<<tracker.last_error()<<"\n";return 3;
    }
    sight::TrackingData result;
    auto run=[&](const std::vector<uint8_t>& source,int frame){
        if(!tracker.detect(source.data(),w,h,33*frame,result)){
            std::cerr<<tracker.last_error()<<"\n";return false;
        }
        return true;
    };
    if(!run(pixels,1)||result.landmarks.size()!=133)return 4;
    const auto original=result.landmarks;
    unsigned confident=0;
    for(auto& p:original){
        if(!std::isfinite(p.x)||!std::isfinite(p.y)||p.confidence<0||p.confidence>1)return 5;
        if(p.confidence>=.45f)++confident;
    }
    if(confident<10)return 6;
    // Real transformed image, person is moved sideways within the frame.
    auto shifted=std::vector<uint8_t>(pixels.size(),0);
    for(unsigned y=0;y<h;++y)
        std::memcpy(shifted.data()+(y*w+32)*4,pixels.data()+y*w*4,(w-32)*4);
    if(!run(shifted,2)||result.landmarks.size()!=133)return 7;
    float sum=0;unsigned count=0;
    for(unsigned i=0;i<17;++i)
        if(original[i].confidence>=.45f&&result.landmarks[i].confidence>=.45f){
            sum+=result.landmarks[i].x-original[i].x;++count;
        }
    if(count<4||sum/count<.025f||sum/count>.12f)return 8;
    // No person: genuine detector output must clear the published points.
    std::vector<uint8_t> empty(pixels.size(),0);
    if(!run(empty,3)||!result.landmarks.empty())return 9;
    if(!run(pixels,4)||result.landmarks.size()!=133)return 10;
    if(tracker.compilation_count()!=expected_compiles)return 11;
    // The same 512px real photo in a wider 768x512 capture.
    constexpr unsigned wide=768,offset=128;
    std::vector<uint8_t> padded(wide*h*4,0);
    for(unsigned y=0;y<h;++y)std::memcpy(padded.data()+(size_t(y)*wide+offset)*4,pixels.data()+size_t(y)*w*4,w*4);

    std::vector<double> times;
    for(int i=5;i<20;++i){
        auto start=std::chrono::steady_clock::now();
        if(!run(i%2?pixels:shifted,i)||result.landmarks.size()!=133)return 12;
        times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    if(!tracker.detect(padded.data(),wide,h,33*20,result)||result.landmarks.size()!=133)return 13;
    float error=0;unsigned compared=0;
    for(unsigned i=0;i<17;++i)
        if(original[i].confidence>=.45f&&result.landmarks[i].confidence>=.45f){
            const auto expected=(offset+original[i].x*w)/wide;
            error+=std::hypot(result.landmarks[i].x-expected,result.landmarks[i].y-original[i].y);++compared;
        }
    if(compared<4||error/compared>.055f){
        std::cerr<<"wide letterbox error "<<(compared?error/compared:1)<<" count "<<compared<<"\n";return 14;
    }
    if(tracker.compilation_count()!=expected_compiles)return 15;
    std::sort(times.begin(),times.end());
    std::cout<<"SIGHT_RTMW_PERSON_GPU_NATIVE_20261118_R62 points=133 confident="<<confident
             <<" shift="<<sum/count<<" loss=1 recovery=1 compile_count="<<tracker.compilation_count()
             <<" wide_error="<<error/compared<<" detector="<<(yolo?"yolox-gpu":"efficientdet-cpu")
             <<" pipeline_p50_ms="<<times[times.size()/2]<<" pipeline_p95_ms="<<times[times.size()-1]<<"\n";
}
