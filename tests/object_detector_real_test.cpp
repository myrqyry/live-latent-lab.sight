#include "core/object_detector.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <vector>
int main(int argc,char** argv){
    if(argc!=6){
        std::cerr<<"usage: test native.so efficientdet.tflite rgba.raw width height\n";
        return 2;
    }
    const int w=std::stoi(argv[4]),h=std::stoi(argv[5]);
    std::ifstream in(argv[3],std::ios::binary);
    std::vector<uint8_t> rgba((std::istreambuf_iterator<char>(in)),{});
    if(w<16||h<16||rgba.size()!=size_t(w)*h*4)return 2;
    sight::ObjectDetector detector;
    std::vector<sight::ObjectDetection> predictions;
    assert(!detector.detect(rgba.data(),w,h,1,predictions));
    assert(!detector.load(argv[1],"/tmp/sight-object-model-does-not-exist.tflite"));
    assert(detector.load(argv[1],argv[2]));
    assert(detector.ready());
    std::set<std::string> labels;
    for(int i=1;i<=3;++i){
        predictions.clear();
        if(!detector.detect(rgba.data(),w,h,i*50,predictions)){
            std::cerr<<"native detect error: "<<detector.last_error()<<"\n";
            return 1;
        }
        for(const auto& p:predictions){
            assert(sight::valid_object(p));
            labels.insert(p.label);
            std::cout<<"REAL_OBJECT label="<<p.label<<" score="<<p.score
                     <<" box="<<p.x<<","<<p.y<<","<<p.w<<","<<p.h<<"\n";
        }
    }
    assert(!predictions.empty());
    // The sample image really contains recognizable animals, rather
    // than a mock detector that fabricates a generic "object" box.
    assert(labels.count("cat") || labels.count("dog"));
    assert(!detector.detect(rgba.data(),w,h,149,predictions));
    detector.unload();
    assert(!detector.ready());
    std::cout<<"SIGHT_NATIVE_OBJECT_DETECTOR_REAL_PASS=YES\n";
}
