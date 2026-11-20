#include "core/face_quality.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
using namespace sight;
int main(int argc,char**argv){
    if(argc!=4)return 2;
    std::ifstream in(argv[3],std::ios::binary);
    std::vector<uint8_t> rgba((std::istreambuf_iterator<char>(in)),{});
    constexpr unsigned W=512,H=512;
    if(rgba.size()!=size_t(W)*H*4)return 3;
    TasksTracker tracker;
    if(!tracker.load(argv[1],argv[2],TrackingKind::Face))return 4;
    TrackingData raw;
    if(!tracker.detect(rgba.data(),W,H,1,raw)||raw.landmarks.size()!=478)return 5;
    FaceQuality q;auto refined=raw;
    const auto start=std::chrono::steady_clock::now();
    if(!q.apply(refined,{rgba.data(),W,H,rgba.size()},1,FaceQualityMode::Precision,0))return 6;
    unsigned moved=0;float maximum=0;
    constexpr int lids[]={173,157,158,159,160,161,246,398,384,385,386,387,388,466};
    for(size_t i=0;i<478;++i){
        auto a=raw.landmarks[i],b=refined.landmarks[i];
        float delta=std::hypot((a.x-b.x)*W,(a.y-b.y)*H);
        bool lid=std::find(std::begin(lids),std::end(lids),int(i))!=std::end(lids);
        if(delta>0){++moved;if(!lid)return 7;}
        maximum=std::max(maximum,delta);
        if(delta>5.01f||a.z!=b.z||a.confidence!=b.confidence)return 8;
    }
    if(raw.blendshapes!=refined.blendshapes||raw.blendshape_names!=refined.blendshape_names||
       raw.head_transform!=refined.head_transform||raw.has_head_transform!=refined.has_head_transform)return 9;
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    if(moved==0)return 10;
    std::cout<<"FACE_QUALITY_REAL landmarks=478 refined="<<moved<<" max_displacement_px="<<maximum<<" processing_ms="<<ms<<" metadata_preserved=1\n";
    return 0;
}
