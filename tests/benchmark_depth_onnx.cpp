// Standalone real-ONNX benchmark for Sight Depth Camera, no mocked inference.
#include "core/onnx_depth_processor.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv) {
 if(argc!=7){std::cerr<<"Usage: model.onnx image.rgb width height repeats out_mask.gray\n";return 2;}
 const std::string model=argv[1],image=argv[2],out=argv[6];
 const unsigned w=std::stoul(argv[3]),h=std::stoul(argv[4]),count=std::stoul(argv[5]);
 std::vector<uint8_t> raw(size_t(w)*h*3);
 std::ifstream input(image,std::ios::binary);
 if(!input.read(reinterpret_cast<char*>(raw.data()),raw.size()))return 3;
 lll::OnnxDepthProcessor depth;
 const auto load0=Clock::now();
 if(!depth.load(model)){std::cerr<<depth.last_error()<<"\n";return 4;}
 const auto load1=Clock::now();
 std::vector<uint8_t> mask;
 std::vector<double> ms;
 for(unsigned i=0;i<count+1;i++){
   auto t0=Clock::now();
   if(!depth.estimate(raw.data(),w,h,mask)){std::cerr<<depth.last_error()<<"\n";return 5;}
   auto t1=Clock::now();
   if(i>0)ms.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
 }
 if(ms.empty())return 2;
 std::sort(ms.begin(),ms.end());
 std::ofstream output(out,std::ios::binary);
 output.write(reinterpret_cast<char*>(mask.data()),mask.size());
 auto [minimum,maximum]=std::minmax_element(mask.begin(),mask.end());
 std::cout<<"ONNX_V2_CPU width="<<w<<" height="<<h<<" iterations="<<ms.size()
          <<" load_ms="<<std::chrono::duration<double,std::milli>(load1-load0).count()
          <<" infer_median_ms="<<ms[ms.size()/2]<<" infer_min_ms="<<ms.front()
          <<" infer_max_ms="<<ms.back()
          <<" depth_min="<<int(*minimum)<<" depth_max="<<int(*maximum)
          <<" output_bytes="<<mask.size()<<"\n";
 return *minimum < *maximum && mask.size()==size_t(w)*h ? 0:6;
}
