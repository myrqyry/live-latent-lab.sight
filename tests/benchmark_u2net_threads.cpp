#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
int main(int argc, char **argv) {
    if (argc!=3) {
        std::cerr << "Usage: sight_u2net_benchmark /path/to/u2net.onnx THREAD_COUNT\n";
        return 2;
    }
    const int threads = std::stoi(argv[2]);
    if (threads < 1 || threads > 32)
        return 2;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sight_benchmark");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(threads);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    auto start=std::chrono::steady_clock::now();
    Ort::Session session(env, argv[1], opts);
    auto init=std::chrono::steady_clock::now();
    std::vector<float> input(3 * 320 * 320, 0.0f);
    for (size_t i=0; i<input.size(); ++i)
        input[i] = ((i*37u)%256u) / 128.0f - 1.0f;
    const int64_t dims[] = {1, 3, 320, 320};
    auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor=Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), dims, 4);
    Ort::AllocatorWithDefaultOptions alloc;
    auto in=session.GetInputNameAllocated(0,alloc);
    auto out=session.GetOutputNameAllocated(0,alloc);
    const char *names[]={in.get()}, *outs[]={out.get()};
    std::vector<double> latencies;
    for (int run=0;run<5;++run) {
        auto t0=std::chrono::steady_clock::now();
        auto result=session.Run(Ort::RunOptions{nullptr}, names, &tensor, 1, outs, 1);
        auto t1=std::chrono::steady_clock::now();
        if(result.empty()) return 3;
        auto ms=std::chrono::duration<double,std::milli>(t1-t0).count();
        latencies.push_back(ms);
    }
    std::cout<<"threads="<<threads
             <<" load_ms="<<std::chrono::duration<double,std::milli>(init-start).count()
             <<" first_ms="<<latencies[0]
             <<" steady_mean_ms="<<std::accumulate(latencies.begin()+1,latencies.end(),0.0)/4
             <<" steady_best_ms="<<*std::min_element(latencies.begin()+1,latencies.end())
             <<" all=";
    for(double val:latencies) std::cout<<val<<",";
    std::cout<<std::endl;
}
