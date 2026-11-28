// Worker-only CUDA driver/ORT I/O binding, no graphics dependencies.
#include <cuda.h>
#include <cstring>
#include "sight-hed-cuda-ptx.hpp"

namespace lll {
namespace {
void hed_cuda_check(CUresult r,const char* expr){
    if(r==CUDA_SUCCESS)return;
    const char* msg=nullptr;cuGetErrorString(r,&msg);
    throw std::runtime_error(std::string(expr)+": "+(msg?msg:"CUDA error"));
}
#define HED_CUDA(expr) hed_cuda_check((expr),#expr)
struct HedCurrentContext {
    explicit HedCurrentContext(CUcontext c){HED_CUDA(cuCtxPushCurrent(c));}
    HedCurrentContext(const HedCurrentContext&)=delete;
    ~HedCurrentContext(){CUcontext old;cuCtxPopCurrent(&old);}
};
}
struct OnnxHedProcessor::DeviceIO {
    CUcontext context=nullptr;
    CUstream stream=nullptr;
    CUmodule module=nullptr;
    CUfunction pack=nullptr,combine=nullptr;
    CUdeviceptr rgb=0,input=0,mask=0,invalid=0;
    std::array<CUdeviceptr,5> maps{};
    void* pinned=nullptr;
    std::unique_ptr<Ort::Value> in_value;
    std::array<std::unique_ptr<Ort::Value>,5> out_values;
    std::unique_ptr<Ort::IoBinding> binding;
    unsigned w=0,h=0;
    std::vector<uint8_t> host_mask;
    HedTensorIOStats stats;
    void init(){
        HED_CUDA(cuInit(0));HED_CUDA(cuDevicePrimaryCtxRetain(&context,0));
        HedCurrentContext current(context);
        HED_CUDA(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
        HED_CUDA(cuModuleLoadData(&module,sight_hed_ptx));
        HED_CUDA(cuModuleGetFunction(&pack,module,"hed_pack"));
        HED_CUDA(cuModuleGetFunction(&combine,module,"hed_combine"));
    }
    void release_buffers() noexcept{
        binding.reset();in_value.reset();
        for(auto& v:out_values)v.reset();
        for(auto& p:maps){if(p)cuMemFree(p);p=0;}
        for(auto* p:{&rgb,&input,&mask,&invalid}){if(*p)cuMemFree(*p);*p=0;}
        if(pinned)cuMemFreeHost(pinned);
        pinned=nullptr;w=h=0;
    }
    ~DeviceIO(){
        if(!context)return;
        try{
            HedCurrentContext current(context);
            if(stream)cuStreamSynchronize(stream);
            release_buffers();
            if(module)cuModuleUnload(module);
            if(stream)cuStreamDestroy(stream);
        }catch(...){}
        cuDevicePrimaryCtxRelease(0);
    }
    void prepare(Ort::Session& session,unsigned width,unsigned height){
        if(w==width&&h==height&&binding)return;
        HED_CUDA(cuStreamSynchronize(stream));release_buffers();
        size_t n=size_t(width)*height;
        HED_CUDA(cuMemAlloc(&rgb,n*3));HED_CUDA(cuMemAllocHost(&pinned,n*3));
        HED_CUDA(cuMemAlloc(&input,n*3*sizeof(float)));HED_CUDA(cuMemAlloc(&mask,n));
        HED_CUDA(cuMemAlloc(&invalid,sizeof(unsigned)));
        host_mask.resize(n);
        Ort::MemoryInfo mem("Cuda",OrtDeviceAllocator,0,OrtMemTypeDefault);
        const int64_t shape[]={1,3,height,width};
        in_value=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(mem,
            reinterpret_cast<float*>(input),n*3,shape,4));
        binding=std::make_unique<Ort::IoBinding>(session);
        binding->BindInput("rgb",*in_value);
        for(unsigned i=0;i<5;++i){
            const int64_t dims[]={1,1,height>>i,width>>i};
            size_t count=size_t(width>>i)*(height>>i);
            HED_CUDA(cuMemAlloc(&maps[i],count*sizeof(float)));
            out_values[i]=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(mem,
                reinterpret_cast<float*>(maps[i]),count,dims,4));
            const auto name="scale"+std::to_string(i+1);
            binding->BindOutput(name.c_str(),*out_values[i]);
        }
        w=width;h=height;++stats.prepares;
    }
    void run(Ort::Session& session,const uint8_t* pixels,unsigned width,unsigned height,bool safe,std::vector<uint8_t>&gray){
        HedCurrentContext current(context);prepare(session,width,height);
        unsigned n=width*height,grid=(n+255)/256,step=safe?1u:0u;
        std::memcpy(pinned,pixels,size_t(n)*3);
        HED_CUDA(cuMemcpyHtoDAsync(rgb,pinned,size_t(n)*3,stream));
        void* p[]={&rgb,&input,&n};
        HED_CUDA(cuLaunchKernel(pack,grid,1,1,256,1,1,0,stream,p,nullptr));
        HED_CUDA(cuStreamSynchronize(stream));
        binding->SynchronizeInputs();session.Run(Ort::RunOptions{nullptr},*binding);binding->SynchronizeOutputs();
        HED_CUDA(cuMemsetD32Async(invalid,0,1,stream));
        void* args[]={&maps[0],&maps[1],&maps[2],&maps[3],&maps[4],&mask,&invalid,&width,&height,&step};
        HED_CUDA(cuLaunchKernel(combine,grid,1,1,256,1,1,0,stream,args,nullptr));
        HED_CUDA(cuStreamSynchronize(stream));
        unsigned bad=0;
        HED_CUDA(cuMemcpyDtoH(&bad,invalid,sizeof(bad)));
        HED_CUDA(cuMemcpyDtoH(host_mask.data(),mask,n));
        stats.upload_bytes+=size_t(n)*3;stats.download_bytes+=n+sizeof(bad);
        if(bad)throw std::runtime_error("CUDA HED returned nonfinite logits");
        gray.swap(host_mask);host_mask.resize(n);++stats.frames;
    }
};
#undef HED_CUDA
}
