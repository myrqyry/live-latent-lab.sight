#pragma once
// Worker-only CUDA driver/ORT I/O binding, no graphics dependencies.
#include <cuda.h>
#include <cstring>
#include "sight-cutout-cuda-ptx.hpp"
#include "edge_tensor_io_stats.hpp"
#include <array>

namespace lll {
namespace cutout_cuda_detail {
inline void cutout_cuda_check(CUresult r,const char* expr){
    if(r==CUDA_SUCCESS)return;
    const char* msg=nullptr;cuGetErrorString(r,&msg);
    throw std::runtime_error(std::string(expr)+": "+(msg?msg:"CUDA error"));
}
#define CUTOUT_CUDA(expr) ::lll::cutout_cuda_detail::cutout_cuda_check((expr),#expr)
struct CutoutCurrentContext {
    explicit CutoutCurrentContext(CUcontext c){CUTOUT_CUDA(cuCtxPushCurrent(c));}
    CutoutCurrentContext(const CutoutCurrentContext&)=delete;
    ~CutoutCurrentContext(){CUcontext old;cuCtxPopCurrent(&old);}
};
}
class CutoutDeviceIO {
public:
    CutoutDeviceIO()=default;
    CutoutDeviceIO(const CutoutDeviceIO&)=delete;
    CUcontext context=nullptr;
    CUstream stream=nullptr;
    CUmodule module=nullptr;
    CUfunction pack=nullptr,validate=nullptr,unpack=nullptr;
    CUdeviceptr rgb=0,input=0,mask=0,invalid=0;
    CUdeviceptr alpha=0;
    void* pinned=nullptr;
    std::unique_ptr<Ort::Value> in_value;
    std::unique_ptr<Ort::Value> out_value;
    std::unique_ptr<Ort::IoBinding> binding;
    unsigned w=0,h=0,iw=0,ih=0;
    std::vector<uint8_t> host_mask;
    EdgeTensorIOStats stats;
    void init(){
        CUTOUT_CUDA(cuInit(0));CUTOUT_CUDA(cuDevicePrimaryCtxRetain(&context,0));
        cutout_cuda_detail::CutoutCurrentContext current(context);
        CUTOUT_CUDA(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
        CUTOUT_CUDA(cuModuleLoadData(&module,sight_cutout_ptx));
        CUTOUT_CUDA(cuModuleGetFunction(&pack,module,"cutout_pack"));
        CUTOUT_CUDA(cuModuleGetFunction(&validate,module,"cutout_validate"));
        CUTOUT_CUDA(cuModuleGetFunction(&unpack,module,"cutout_unpack"));
    }
    void release_buffers() noexcept{
        binding.reset();in_value.reset();
        out_value.reset();
        for(auto* p:{&rgb,&input,&mask,&invalid,&alpha}){if(*p)cuMemFree(*p);*p=0;}
        if(pinned)cuMemFreeHost(pinned);
        pinned=nullptr;w=h=iw=ih=0;
    }
    ~CutoutDeviceIO(){
        if(!context)return;
        try{
            cutout_cuda_detail::CutoutCurrentContext current(context);
            if(stream)cuStreamSynchronize(stream);
            release_buffers();
            if(module)cuModuleUnload(module);
            if(stream)cuStreamDestroy(stream);
        }catch(...){}
        cuDevicePrimaryCtxRelease(0);
    }
    void prepare(Ort::Session& session,const char* in,const char* out,unsigned width,unsigned height,unsigned input_w,unsigned input_h){
        if(w==width&&h==height&&iw==input_w&&ih==input_h&&binding)return;
        CUTOUT_CUDA(cuStreamSynchronize(stream));release_buffers();
        const size_t n=size_t(width)*height,model_n=size_t(input_w)*input_h;
        CUTOUT_CUDA(cuMemAlloc(&rgb,n*3));CUTOUT_CUDA(cuMemAllocHost(&pinned,n*3));
        CUTOUT_CUDA(cuMemAlloc(&input,model_n*3*sizeof(float)));
        CUTOUT_CUDA(cuMemAlloc(&alpha,model_n*sizeof(float)));
        CUTOUT_CUDA(cuMemAlloc(&mask,n));CUTOUT_CUDA(cuMemAlloc(&invalid,sizeof(unsigned)));
        host_mask.resize(n);
        Ort::MemoryInfo mem("Cuda",OrtDeviceAllocator,0,OrtMemTypeDefault);
        const int64_t shape[]={1,3,input_h,input_w},dims[]={1,1,input_h,input_w};
        in_value=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(mem,
            reinterpret_cast<float*>(input),model_n*3,shape,4));
        out_value=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(mem,
            reinterpret_cast<float*>(alpha),model_n,dims,4));
        binding=std::make_unique<Ort::IoBinding>(session);
        binding->BindInput(in,*in_value);binding->BindOutput(out,*out_value);
        w=width;h=height;iw=input_w;ih=input_h;++stats.prepares;
    }
    void run(Ort::Session& session,const char* in,const char* out,const uint8_t* pixels,
        unsigned width,unsigned height,unsigned input_w,unsigned input_h,unsigned mode,uint8_t* gray){
        cutout_cuda_detail::CutoutCurrentContext current(context);
        prepare(session,in,out,width,height,input_w,input_h);
        unsigned n=w*h,m=iw*ih,grid=(n+255)/256,model_grid=(m+255)/256;
        std::memcpy(pinned,pixels,size_t(n)*3);
        CUTOUT_CUDA(cuMemcpyHtoDAsync(rgb,pinned,size_t(n)*3,stream));
        void* p[]={&rgb,&input,&w,&h,&iw,&ih,&mode};
        CUTOUT_CUDA(cuLaunchKernel(pack,model_grid,1,1,256,1,1,0,stream,p,nullptr));
        CUTOUT_CUDA(cuStreamSynchronize(stream));
        binding->SynchronizeInputs();session.Run(Ort::RunOptions{nullptr},*binding);binding->SynchronizeOutputs();
        CUTOUT_CUDA(cuMemsetD32Async(invalid,0,1,stream));
        void* v[]={&alpha,&m,&invalid};
        CUTOUT_CUDA(cuLaunchKernel(validate,model_grid,1,1,256,1,1,0,stream,v,nullptr));
        void* args[]={&alpha,&mask,&w,&h,&iw,&ih};
        CUTOUT_CUDA(cuLaunchKernel(unpack,grid,1,1,256,1,1,0,stream,args,nullptr));
        CUTOUT_CUDA(cuStreamSynchronize(stream));
        unsigned bad=0;
        CUTOUT_CUDA(cuMemcpyDtoH(&bad,invalid,sizeof(bad)));
        CUTOUT_CUDA(cuMemcpyDtoH(host_mask.data(),mask,n));
        stats.upload_bytes+=size_t(n)*3;stats.download_bytes+=n+sizeof(bad);
        if(bad)throw std::runtime_error("CUDA Cutout returned nonfinite alpha");
        std::memcpy(gray,host_mask.data(),n);++stats.frames;
    }
};
#undef CUTOUT_CUDA
}
