#include "core/semantic_detector.hpp"
#include <fstream>
#include <chrono>
#include <iostream>
int main(int argc,char** argv) {
 if(argc!=3&&argc!=4)return 2;
 const unsigned classes=argc==4?unsigned(std::stoul(argv[3])):14;
 std::ifstream in(argv[2],std::ios::binary);
 std::vector<uint8_t> frame((std::istreambuf_iterator<char>(in)),{});
 if(frame.size()!=512*512*4)return 3;
 sight::semantic::Detector d;
 if(!d.load(argv[1],0)){std::cerr<<d.error()<<"\n";return 4;}
 if(d.class_count()!=classes)return 9;
 std::vector<sight::privacy::Rect> regions;
 using clock=std::chrono::steady_clock;
 std::vector<double> samples;
 for(int i=0;i<36;++i) {
  auto t=clock::now();
  if(!d.detect(frame.data(),512,512,0,{1},.2f,.15f,regions)||regions.empty()){
   std::cerr<<"Expected actual face region: "<<d.error()<<"\n";return 5;
  }
  const double ms=std::chrono::duration<double,std::milli>(clock::now()-t).count();
  if(i>=6)samples.push_back(ms);
  if(i==0)std::cout<<"first inference ms="<<ms<<"\n";
 }
 std::sort(samples.begin(),samples.end());
 std::cout<<"classes="<<d.class_count()<<" warm median ms="<<samples[15]<<" p95 ms="<<samples[28]<<" regions="<<regions.size()<<"\n";
 if(!d.detect(frame.data(),512,512,0,{1},1.f,0.f,regions)||!regions.empty())return 6;
 if(d.detect(frame.data(),512,512,0,{int(classes)},.2f,0.f,regions))return 7;
 if(d.load(argv[1],classes+1))return 8;
 std::cout<<"PASS: actual model inference, confidence gate, invalid classes and shape rejection\n";
}
