// Included by onnx_depth_processor.cpp only when explicitly built with
// CUDA device I/O. No OpenGL dependency, and no render-thread operations.
#include <cuda.h>
#include <cstring>
#include <cstddef>
#include "sight-depth-cuda-ptx.hpp"

namespace lll {
namespace {
void depth_cuda_check(CUresult result,const char* expression){
    if(result==CUDA_SUCCESS)return;
    const char* message=nullptr;cuGetErrorString(result,&message);
    throw std::runtime_error(std::string(expression)+": "+(message?message:"CUDA error"));
}
#define DEPTH_CUDA(expr) depth_cuda_check((expr),#expr)
struct DepthRangeLayout {float lo,hi;unsigned invalid;};
static_assert(sizeof(DepthRangeLayout)==12&&offsetof(DepthRangeLayout,invalid)==8);
struct CurrentDepthContext {
    explicit CurrentDepthContext(CUcontext context){DEPTH_CUDA(cuCtxPushCurrent(context));}
    CurrentDepthContext(const CurrentDepthContext&)=delete;
    ~CurrentDepthContext(){CUcontext previous;cuCtxPopCurrent(&previous);}
};
}
struct OnnxDepthProcessor::DeviceIO {
    CUcontext context=nullptr;
    CUstream stream=nullptr;
    CUmodule module=nullptr;
    CUfunction pack=nullptr,partial=nullptr,range=nullptr,unpack=nullptr;
    CUdeviceptr rgb=0,input=0,output=0,blocks=0,total=0,mask=0;
    void* pinned_rgb=nullptr;
    std::unique_ptr<Ort::Value> in_value,out_value;
    std::unique_ptr<Ort::IoBinding> binding;
    uint32_t w=0,h=0;
    std::vector<uint8_t> host_mask;
    DepthTensorIOStats stats;

    void init(){
        DEPTH_CUDA(cuInit(0));
        DEPTH_CUDA(cuDevicePrimaryCtxRetain(&context,0));
        CurrentDepthContext current(context);
        DEPTH_CUDA(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
        DEPTH_CUDA(cuModuleLoadData(&module,sight_depth_ptx));
        DEPTH_CUDA(cuModuleGetFunction(&pack,module,"depth_pack"));
        DEPTH_CUDA(cuModuleGetFunction(&partial,module,"depth_partial"));
        DEPTH_CUDA(cuModuleGetFunction(&range,module,"depth_range"));
        DEPTH_CUDA(cuModuleGetFunction(&unpack,module,"depth_unpack"));
    }
    void release_buffers() noexcept{
        binding.reset();in_value.reset();out_value.reset();
        for(auto* p:{&rgb,&input,&output,&blocks,&total,&mask}){
            if(*p)cuMemFree(*p);
            *p=0;
        }
        if(pinned_rgb)cuMemFreeHost(pinned_rgb);
        pinned_rgb=nullptr;w=h=0;
    }
    ~DeviceIO(){
        if(!context)return;
        try{
            CurrentDepthContext current(context);
            if(stream)cuStreamSynchronize(stream);
            release_buffers();
            if(module)cuModuleUnload(module);
            if(stream)cuStreamDestroy(stream);
        }catch(...){}
        cuDevicePrimaryCtxRelease(0);
    }
    void prepare(Ort::Session& session,const std::string& in,const std::string& out,uint32_t width,uint32_t height){
        if(w==width&&h==height&&binding)return;
        DEPTH_CUDA(cuStreamSynchronize(stream));release_buffers();
        const size_t n=size_t(width)*height;
        DEPTH_CUDA(cuMemAlloc(&rgb,n*3));
        DEPTH_CUDA(cuMemAllocHost(&pinned_rgb,n*3));
        DEPTH_CUDA(cuMemAlloc(&input,n*3*sizeof(float)));
        DEPTH_CUDA(cuMemAlloc(&output,n*sizeof(float)));
        DEPTH_CUDA(cuMemAlloc(&blocks,((n+255)/256)*sizeof(DepthRangeLayout)));
        DEPTH_CUDA(cuMemAlloc(&total,sizeof(DepthRangeLayout)));
        DEPTH_CUDA(cuMemAlloc(&mask,n));
        host_mask.resize(n);
        const int64_t in_shape[]={1,3,height,width},out_shape[]={1,height,width};
        Ort::MemoryInfo memory("Cuda",OrtDeviceAllocator,0,OrtMemTypeDefault);
        in_value=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(memory,
            reinterpret_cast<float*>(input),n*3,in_shape,4));
        out_value=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(memory,
            reinterpret_cast<float*>(output),n,out_shape,3));
        binding=std::make_unique<Ort::IoBinding>(session);
        binding->BindInput(in.c_str(),*in_value);binding->BindOutput(out.c_str(),*out_value);
        w=width;h=height;++stats.prepares;
    }
    void run(Ort::Session& session,const std::string& in,const std::string& out,const uint8_t* pixels,
             uint32_t width,uint32_t height,std::vector<uint8_t>& depth){
        CurrentDepthContext current(context);
        prepare(session,in,out,width,height);
        unsigned n=width*height,count=(n+255)/256;
        // Pageable CPU RGB is still needed by depth history. Transfer only
        // its compact bytes, not the 3-channel float tensor.
        std::memcpy(pinned_rgb,pixels,size_t(n)*3);
        // The transfer and pack kernel must be ordered on the same stream.
        // Reuse pinned storage rather than relying on pageable default-stream
        // copy completion semantics with a non-blocking CUDA stream.
        DEPTH_CUDA(cuMemcpyHtoDAsync(rgb,pinned_rgb,size_t(n)*3,stream));
        void* p[]={&rgb,&input,&n};
        DEPTH_CUDA(cuLaunchKernel(pack,count,1,1,256,1,1,0,stream,p,nullptr));
        // ORT owns its provider stream; explicit ordering at the boundary.
        DEPTH_CUDA(cuStreamSynchronize(stream));
        binding->SynchronizeInputs();
        session.Run(Ort::RunOptions{nullptr},*binding);
        binding->SynchronizeOutputs();
        void* a[]={&output,&blocks,&n};
        DEPTH_CUDA(cuLaunchKernel(partial,count,1,1,256,1,1,0,stream,a,nullptr));
        void* b[]={&blocks,&total,&count};
        DEPTH_CUDA(cuLaunchKernel(range,1,1,1,256,1,1,0,stream,b,nullptr));
        void* c[]={&output,&total,&mask,&n};
        DEPTH_CUDA(cuLaunchKernel(unpack,count,1,1,256,1,1,0,stream,c,nullptr));
        DEPTH_CUDA(cuStreamSynchronize(stream));
        unsigned invalid=0;
        DEPTH_CUDA(cuMemcpyDtoH(&invalid,total+offsetof(DepthRangeLayout,invalid),sizeof(invalid)));
        DEPTH_CUDA(cuMemcpyDtoH(host_mask.data(),mask,n));
        stats.upload_bytes+=size_t(n)*3;stats.download_bytes+=n+sizeof(invalid);
        if(invalid)throw std::runtime_error("CUDA depth returned nonfinite values");
        depth.swap(host_mask);host_mask.resize(n);++stats.frames;
    }
};
#undef DEPTH_CUDA
}
