#include "core/onnx_ivme_denoiser.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
int main(int argc,char**argv) {
 if(argc!=2&&argc!=3)return 2;
 if(argc==3 && std::string(argv[2])=="--native-720p"){
  lll::OnnxIvmeDenoiser large;
  if(!large.load(argv[1]))return 20;
  const uint32_t w=1280,h=720;
  std::vector<uint8_t> rgb(size_t(w)*h*3,125),out;
  const auto start=std::chrono::steady_clock::now();
  if(!large.denoise(rgb.data(),w,h,25.f,out)){
   std::cerr<<"NATIVE_720P_FAILED "<<large.last_error()<<"\n";return 21;
  }
  const auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now()-start).count();
  if(out.size()!=rgb.size())return 22;
  // Geometry must be rejected BEFORE reading the caller's small input.
  if(large.denoise(rgb.data(),1920,1080,25.f,out))return 23;
  if(large.denoise(rgb.data(),1280,1024,25.f,out))return 24;
  std::cout<<"REAL_IVME_NATIVE_720P width="<<w<<" height="<<h
           <<" ms="<<ms<<"\n";
  return 0;
 }
 lll::OnnxIvmeDenoiser backend;
 if(backend.load("/dev/null"))return 3;
 if(!backend.load(argv[1])){std::cerr<<"ONNX_LOAD_FAILED: "<<backend.last_error()<<"\n";return 4;}
 if(!backend.ready())return 5;
 constexpr uint32_t w=256,h=256;
 std::vector<uint8_t> original(size_t(w)*h*3),noisy(original.size()),clean;
 uint32_t random=0x12345678;
 auto rand01=[&](){random^=random<<13;random^=random>>17;random^=random<<5;
    return (double(random%65536)/65536.0);};
 double before=0;
 for(uint32_t y=0;y<h;++y)for(uint32_t x=0;x<w;++x)for(size_t c=0;c<3;++c){
   size_t i=(size_t(y)*w+x)*3+c;
   uint8_t p=static_cast<uint8_t>(45+(x>80&&x<174?65:0)+(y>110?50:0));
   original[i]=p;
   int n=std::clamp(int(std::round(p+(rand01()-.5)*90)),0,255);
   noisy[i]=uint8_t(n);before+=double(n-p)*(n-p);
 }
 if(!backend.denoise(noisy.data(),w,h,25.0f,clean)){
   std::cerr<<"ONNX_INFER_FAILED: "<<backend.last_error()<<"\n";return 6;
 }
 if(clean.size()!=noisy.size())return 7;
 double after=0;
 for(size_t i=0;i<clean.size();++i)after+=std::pow(double(clean[i])-original[i],2);
 before/=clean.size();after/=clean.size();
 std::cout<<"REAL_IVME_ONNX_DENOISE input_mse="<<before
          <<" output_mse="<<after<<" width="<<w<<" height="<<h<<"\n";
 if(!(after<before*.65))return 8;
 if(backend.denoise(noisy.data(),0,h,25,clean))return 9;
 if(backend.denoise(noisy.data(),w,h,0,clean))return 10;
 if(!backend.denoise(noisy.data(),w,h,25.0f,clean))return 11;
 std::cout<<"PASS: real ONNX checkpoint reduces known noise\n";
 return 0;
}
