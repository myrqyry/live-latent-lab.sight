#pragma once
// Worker-only CUDA driver/ORT I/O binding, no graphics dependencies.
#include <cuda.h>
#include <cstring>
#include "sight-edge-cuda-ptx.hpp"
#include "edge_tensor_io_stats.hpp"
#include <array>

namespace lll {
namespace edge_cuda_detail {
inline void edge_cuda_check(CUresult r,const char* expr){
    if(r==CUDA_SUCCESS)return;
    const char* msg=nullptr;cuGetErrorString(r,&msg);
    throw std::runtime_error(std::string(expr)+": "+(msg?msg:"CUDA error"));
}
#define EDGE_CUDA(expr) ::lll::edge_cuda_detail::edge_cuda_check((expr),#expr)
struct EdgeCurrentContext {
    explicit EdgeCurrentContext(CUcontext c){EDGE_CUDA(cuCtxPushCurrent(c));}
    EdgeCurrentContext(const EdgeCurrentContext&)=delete;
    ~EdgeCurrentContext(){CUcontext old;cuCtxPopCurrent(&old);}
};
}
class EdgeDeviceIO {
public:
    explicit EdgeDeviceIO(bool probability):probability_(probability){}
    EdgeDeviceIO(const EdgeDeviceIO&)=delete;
private:
    const bool probability_;
public:
    CUcontext context=nullptr;
    CUstream stream=nullptr;
    CUmodule module=nullptr;
    CUfunction pack=nullptr,combine=nullptr;
    CUdeviceptr rgb=0,input=0,mask=0,invalid=0;
    std::array<CUdeviceptr,4> maps{};
    void* pinned=nullptr;
    std::unique_ptr<Ort::Value> in_value;
    std::array<std::unique_ptr<Ort::Value>,4> out_values;
    std::unique_ptr<Ort::IoBinding> binding;
    unsigned w=0,h=0;
    std::vector<uint8_t> host_mask;
    EdgeTensorIOStats stats;
    void init(){
        EDGE_CUDA(cuInit(0));EDGE_CUDA(cuDevicePrimaryCtxRetain(&context,0));
        edge_cuda_detail::EdgeCurrentContext current(context);
        EDGE_CUDA(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
        EDGE_CUDA(cuModuleLoadData(&module,sight_edge_ptx));
        EDGE_CUDA(cuModuleGetFunction(&pack,module,"edge_pack"));
        EDGE_CUDA(cuModuleGetFunction(&combine,module,"edge_combine"));
    }
    void release_buffers() noexcept{
        binding.reset();in_value.reset();
        for(auto& v:out_values)v.reset();
        for(auto& p:maps){if(p)cuMemFree(p);p=0;}
        for(auto* p:{&rgb,&input,&mask,&invalid}){if(*p)cuMemFree(*p);*p=0;}
        if(pinned)cuMemFreeHost(pinned);
        pinned=nullptr;w=h=0;
    }
    ~EdgeDeviceIO(){
        if(!context)return;
        try{
            edge_cuda_detail::EdgeCurrentContext current(context);
            if(stream)cuStreamSynchronize(stream);
            release_buffers();
            if(module)cuModuleUnload(module);
            if(stream)cuStreamDestroy(stream);
        }catch(...){}
        cuDevicePrimaryCtxRelease(0);
    }
    void prepare(Ort::Session& session,unsigned width,unsigned height){
        if(w==width&&h==height&&binding)return;
        EDGE_CUDA(cuStreamSynchronize(stream));release_buffers();
        size_t n=size_t(width)*height;
        EDGE_CUDA(cuMemAlloc(&rgb,n*3));EDGE_CUDA(cuMemAllocHost(&pinned,n*3));
        EDGE_CUDA(cuMemAlloc(&input,n*3*sizeof(float)));EDGE_CUDA(cuMemAlloc(&mask,n));
        EDGE_CUDA(cuMemAlloc(&invalid,sizeof(unsigned)));
        host_mask.resize(n);
        Ort::MemoryInfo mem("Cuda",OrtDeviceAllocator,0,OrtMemTypeDefault);
        const int64_t shape[]={1,3,height,width};
        in_value=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(mem,
            reinterpret_cast<float*>(input),n*3,shape,4));
        binding=std::make_unique<Ort::IoBinding>(session);
        binding->BindInput(probability_?"bgr":"rgb",*in_value);
        for(unsigned i=0;i<(probability_?1u:4u);++i){
            const int64_t dims[]={1,1,height,width};
            size_t count=n;
            EDGE_CUDA(cuMemAlloc(&maps[i],count*sizeof(float)));
            out_values[i]=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(mem,
                reinterpret_cast<float*>(maps[i]),count,dims,4));
            const auto name=probability_?std::string("edge"):"edge"+std::to_string(i);
            binding->BindOutput(name.c_str(),*out_values[i]);
        }
        w=width;h=height;++stats.prepares;
    }
    void run(Ort::Session& session,const uint8_t* pixels,unsigned width,unsigned height,unsigned steps,std::vector<uint8_t>&gray){
        edge_cuda_detail::EdgeCurrentContext current(context);prepare(session,width,height);
        unsigned n=width*height,grid=(n+255)/256,probability=probability_?1u:0u;
        std::memcpy(pinned,pixels,size_t(n)*3);
        EDGE_CUDA(cuMemcpyHtoDAsync(rgb,pinned,size_t(n)*3,stream));
        void* p[]={&rgb,&input,&n,&probability};
        EDGE_CUDA(cuLaunchKernel(pack,grid,1,1,256,1,1,0,stream,p,nullptr));
        EDGE_CUDA(cuStreamSynchronize(stream));
        binding->SynchronizeInputs();session.Run(Ort::RunOptions{nullptr},*binding);binding->SynchronizeOutputs();
        EDGE_CUDA(cuMemsetD32Async(invalid,0,1,stream));
        void* args[]={&maps[0],&maps[1],&maps[2],&maps[3],&mask,&invalid,&n,&steps,&probability};
        EDGE_CUDA(cuLaunchKernel(combine,grid,1,1,256,1,1,0,stream,args,nullptr));
        EDGE_CUDA(cuStreamSynchronize(stream));
        unsigned bad=0;
        EDGE_CUDA(cuMemcpyDtoH(&bad,invalid,sizeof(bad)));
        EDGE_CUDA(cuMemcpyDtoH(host_mask.data(),mask,n));
        stats.upload_bytes+=size_t(n)*3;stats.download_bytes+=n+sizeof(bad);
        if(bad)throw std::runtime_error("CUDA edge model returned nonfinite values");
        gray.swap(host_mask);host_mask.resize(n);++stats.frames;
    }
};
#undef EDGE_CUDA
}
