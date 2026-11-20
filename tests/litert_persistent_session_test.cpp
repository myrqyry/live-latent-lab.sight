#include "core/litert_runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    std::ifstream f(argv[2],std::ios::binary);
    std::vector<float> input(3*256*192);
    if(!f.read(reinterpret_cast<char*>(input.data()),input.size()*sizeof(float)))return 2;
    auto& rt=lll::LiteRtRuntime::instance();
    if(!rt.initialize())return 3;
    const lll::LiteRtRunOptions strict{false};
    std::vector<std::vector<float>> first,out;
    std::vector<double> times;
    auto run=[&](){
        if(rt.run_multi(argv[1],lll::LiteRtDevice::Gpu,input,out,strict))return true;
        std::cerr<<rt.last_error()<<"\n";return false;
    };
    if(!run())return 4;
    if(out.size()!=2||out[0].size()!=133*384||out[1].size()!=133*512)return 5;
    first=out;
    for(unsigned i=0;i<25;++i){
        const auto start=std::chrono::steady_clock::now();
        if(!run())return 6;
        if(i>=5)times.push_back(std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-start).count());
        if(out!=first)return 7;
    }
    if(rt.compilation_count()!=1 || rt.cached_session_count()!=1)return 8;
    // A malformed input cannot run stale values from a previously valid frame.
    input.pop_back();
    if(run())return 9;
    input.push_back(0);
    if(!run()||rt.compilation_count()!=1)return 10;
    std::sort(times.begin(),times.end());
    std::cout<<"SIGHT_LITERT_PERSISTENT_NATIVE_20261118_K94 compilations="
             <<rt.compilation_count()<<" p50_ms="<<times[times.size()/2]
             <<" p95_ms="<<times[18]<<" repetitions="<<times.size()<<"\n";
    rt.shutdown();
    if(rt.cached_session_count()!=0||rt.ready())return 11;
    if(!rt.initialize()||!run()||rt.compilation_count()!=1)return 12;
    rt.shutdown();
}
