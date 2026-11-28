#pragma once
// Linux/NVIDIA optional texture interop. CUDA driver API only: the ONNX
// provider owns its matched CUDA runtime. No cudart linked here.
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <cuda.h>
#include <cudaGL.h>
#include <onnxruntime_cxx_api.h>
#include <obs-module.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace sight_cuda {
inline void check(CUresult r,const char* what){
    if(r==CUDA_SUCCESS)return;
    const char* message=nullptr;cuGetErrorString(r,&message);
    throw std::runtime_error(std::string(what)+": "+(message?message:"CUDA error"));
}
#define SIGHT_CUDA(call) ::sight_cuda::check((call),#call)
class Primary {
    CUcontext context_=nullptr;
public:
    Primary()=default;
    Primary(const Primary&)=delete;
    ~Primary(){if(context_)cuDevicePrimaryCtxRelease(0);}
    void init(){
        if(context_)return;
        SIGHT_CUDA(cuInit(0));SIGHT_CUDA(cuDevicePrimaryCtxRetain(&context_,0));
    }
    CUcontext get()const{return context_;}
    class Current {
        bool pushed_=false;
    public:
        explicit Current(CUcontext ctx){
            if(!ctx)throw std::runtime_error("CUDA context unavailable");
            SIGHT_CUDA(cuCtxPushCurrent(ctx));pushed_=true;
        }
        Current(const Current&)=delete;
        ~Current(){if(pushed_){CUcontext old;cuCtxPopCurrent(&old);}}
    };
};
class Mapped {
    CUgraphicsResource resource_;
    CUstream stream_;
    bool mapped_=false;
public:
    Mapped(CUgraphicsResource resource,CUstream stream):resource_(resource),stream_(stream){
        SIGHT_CUDA(cuGraphicsMapResources(1,&resource_,stream_));mapped_=true;
    }
    ~Mapped(){if(mapped_)cuGraphicsUnmapResources(1,&resource_,stream_);}
    CUarray array()const{
        CUarray a=nullptr;SIGHT_CUDA(cuGraphicsSubResourceGetMappedArray(&a,resource_,0,0));return a;
    }
    void unmap(){
        SIGHT_CUDA(cuGraphicsUnmapResources(1,&resource_,stream_));mapped_=false;
    }
};
// Only the inference worker allocates device storage, creates sessions,
// and runs inference. Graphics thread may copy a single owned slot
// when the caller's handoff gate permits it.
class SketchSession {
    Primary primary_;
    CUstream stream_=nullptr;
    CUmodule module_=nullptr;
    CUfunction pack_=nullptr,unpack_=nullptr;
    CUdeviceptr rgba_=0,input_=0,output_=0;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Value> input_value_,output_value_;
    std::unique_ptr<Ort::IoBinding> binding_;
    std::string input_name_,output_name_;
    unsigned w_=0,h_=0,pw_=0,ph_=0;
    void release_buffers(){
        binding_.reset();input_value_.reset();output_value_.reset();
        if(rgba_)cuMemFree(rgba_);if(input_)cuMemFree(input_);if(output_)cuMemFree(output_);
        rgba_=input_=output_=0;w_=h_=pw_=ph_=0;
    }
public:
    SketchSession()=default;
    SketchSession(const SketchSession&)=delete;
    ~SketchSession(){
        if(primary_.get()){
            try{
                Primary::Current current(primary_.get());
                if(stream_)cuStreamSynchronize(stream_);
                release_buffers();session_.reset();env_.reset();
                if(module_)cuModuleUnload(module_);
                if(stream_)cuStreamDestroy(stream_);
            }catch(const std::exception&e){blog(LOG_ERROR,"[Sight/CUDA texture] cleanup: %s",e.what());}
        }
    }
    void load(const std::string& path,const std::string& ptx){
        const char* request=std::getenv("SIGHT_ONNX_DEVICE");
        if(request&&std::string(request)=="cpu")throw std::runtime_error("CUDA disabled by SIGHT_ONNX_DEVICE=cpu");
        if(path.empty()||ptx.empty())throw std::runtime_error("Missing model or compiled CUDA kernels");
        primary_.init();Primary::Current current(primary_.get());
        SIGHT_CUDA(cuStreamCreate(&stream_,CU_STREAM_NON_BLOCKING));
        SIGHT_CUDA(cuModuleLoad(&module_,ptx.c_str()));
        SIGHT_CUDA(cuModuleGetFunction(&pack_,module_,"sketch_pack"));
        SIGHT_CUDA(cuModuleGetFunction(&unpack_,module_,"sketch_unpack"));
        env_=std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,"Sight CUDA texture");
        Ort::SessionOptions opts;opts.SetIntraOpNumThreads(2);opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        OrtCUDAProviderOptions cuda;
        cuda.device_id=0;cuda.cudnn_conv_algo_search=OrtCudnnConvAlgoSearchHeuristic;
        cuda.has_user_compute_stream=1;cuda.user_compute_stream=stream_;
        // CPU shape nodes are permitted; full image tensors stay device-bound.
        opts.AppendExecutionProvider_CUDA(cuda);
        session_=std::make_unique<Ort::Session>(*env_,path.c_str(),opts);
        if(session_->GetInputCount()!=1||session_->GetOutputCount()!=1)throw std::runtime_error("Expected one input/output");
        for(int i=0;i<2;++i){
            auto type=i?session_->GetOutputTypeInfo(0):session_->GetInputTypeInfo(0);
            auto t=type.GetTensorTypeAndShapeInfo();
            auto shape=t.GetShape();
            if(t.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||shape.size()!=4||shape[3]!=3)
                throw std::runtime_error("Expected float32 NHWC RGB model");
        }
        Ort::AllocatorWithDefaultOptions alloc;
        input_name_=session_->GetInputNameAllocated(0,alloc).get();
        output_name_=session_->GetOutputNameAllocated(0,alloc).get();
    }
    unsigned width()const{return w_;}unsigned height()const{return h_;}
    void prepare(unsigned w,unsigned h){
        if(!w||!h||w>1024||h>1024)throw std::runtime_error("Invalid CUDA texture dimensions");
        if(w==w_&&h==h_&&binding_)return;
        Primary::Current current(primary_.get());
        SIGHT_CUDA(cuStreamSynchronize(stream_));release_buffers();
        unsigned pw=(w+31u)&~31u,ph=(h+31u)&~31u;
        const size_t elements=size_t(pw)*ph*3,bytes=size_t(w)*h*4;
        SIGHT_CUDA(cuMemAlloc(&rgba_,bytes));
        SIGHT_CUDA(cuMemAlloc(&input_,elements*sizeof(float)));
        SIGHT_CUDA(cuMemAlloc(&output_,elements*sizeof(float)));
        std::array<int64_t,4> shape{1,ph,pw,3};
        Ort::MemoryInfo mem("Cuda",OrtDeviceAllocator,0,OrtMemTypeDefault);
        input_value_=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(mem,reinterpret_cast<float*>(input_),elements,shape.data(),shape.size()));
        output_value_=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(mem,reinterpret_cast<float*>(output_),elements,shape.data(),shape.size()));
        binding_=std::make_unique<Ort::IoBinding>(*session_);
        binding_->BindInput(input_name_.c_str(),*input_value_);binding_->BindOutput(output_name_.c_str(),*output_value_);
        w_=w;h_=h;pw_=pw;ph_=ph;
    }
    void capture(CUgraphicsResource resource){
        Primary::Current current(primary_.get());Mapped map(resource,stream_);
        CUDA_MEMCPY2D copy{};copy.srcMemoryType=CU_MEMORYTYPE_ARRAY;copy.srcArray=map.array();
        copy.dstMemoryType=CU_MEMORYTYPE_DEVICE;copy.dstDevice=rgba_;copy.dstPitch=size_t(w_)*4;
        copy.WidthInBytes=size_t(w_)*4;copy.Height=h_;
        SIGHT_CUDA(cuMemcpy2DAsync(&copy,stream_));map.unmap();
    }
    void run(){
        Primary::Current current(primary_.get());
        unsigned w=w_,h=h_,pw=pw_,ph=ph_;
        void* args[]={&rgba_,&input_,&w,&h,&pw,&ph};
        SIGHT_CUDA(cuLaunchKernel(pack_,(pw*ph+255)/256,1,1,256,1,1,0,stream_,args,nullptr));
        binding_->SynchronizeInputs();
        session_->Run(Ort::RunOptions{nullptr},*binding_);
        binding_->SynchronizeOutputs();
        void* out[]={&output_,&rgba_,&w,&h,&pw};
        SIGHT_CUDA(cuLaunchKernel(unpack_,(w*h+255)/256,1,1,256,1,1,0,stream_,out,nullptr));
        SIGHT_CUDA(cuStreamSynchronize(stream_));
    }
    void upload(CUgraphicsResource resource){
        Primary::Current current(primary_.get());Mapped map(resource,stream_);
        CUDA_MEMCPY2D copy{};copy.srcMemoryType=CU_MEMORYTYPE_DEVICE;copy.srcDevice=rgba_;copy.srcPitch=size_t(w_)*4;
        copy.dstMemoryType=CU_MEMORYTYPE_ARRAY;copy.dstArray=map.array();
        copy.WidthInBytes=size_t(w_)*4;copy.Height=h_;
        SIGHT_CUDA(cuMemcpy2DAsync(&copy,stream_));map.unmap();
    }
};
// Graphics-side resources live exclusively on the OBS graphics thread.
// Registration is persistent and removed before texture destruction.
struct Graphics {
    Primary primary;
    gs_texrender_t* capture=nullptr;
    gs_texture_t* output=nullptr;
    CUgraphicsResource capture_resource=nullptr,output_resource=nullptr;
    GLuint capture_id=0,output_id=0;
    GLsync fence=nullptr;
    unsigned w=0,h=0;
    uint64_t generation=0;
    bool valid_output=false;
    void init(){
        if(primary.get())return;
        if(gs_get_device_type()!=GS_DEVICE_OPENGL)throw std::runtime_error("CUDA texture path requires OpenGL");
        SIGHT_CUDA(cuInit(0));CUdevice device;unsigned count=0;
        SIGHT_CUDA(cuGLGetDevices(&count,&device,1,CU_GL_DEVICE_LIST_ALL));
        if(count!=1||device!=0)throw std::runtime_error("OpenGL must share CUDA device 0");
        primary.init();
    }
    void unregister(CUgraphicsResource& resource){
        if(!resource)return;
        Primary::Current current(primary.get());
        SIGHT_CUDA(cuGraphicsUnregisterResource(resource));resource=nullptr;
    }
    void reset(){
        if(fence){glDeleteSync(fence);fence=nullptr;}
        unregister(capture_resource);unregister(output_resource);
        if(capture)gs_texrender_destroy(capture);if(output)gs_texture_destroy(output);
        capture=nullptr;output=nullptr;capture_id=output_id=0;w=h=0;valid_output=false;
    }
    bool register_texture(gs_texture_t* tex,CUgraphicsResource& resource,GLuint& id,unsigned flags){
        if(!tex)throw std::runtime_error("Missing texture");
        auto* native=static_cast<GLuint*>(gs_texture_get_obj(tex));
        if(!native||!glIsTexture(*native))throw std::runtime_error("Invalid OpenGL texture");
        if(resource&&id==*native)return false;
        unregister(resource);Primary::Current current(primary.get());
        SIGHT_CUDA(cuGraphicsGLRegisterImage(&resource,*native,GL_TEXTURE_2D,flags));id=*native;return true;
    }
    bool fence_ready(){
        if(!fence)return false;
        GLenum result=glClientWaitSync(fence,0,0);
        if(result==GL_WAIT_FAILED)throw std::runtime_error("OpenGL capture fence failed");
        return result==GL_ALREADY_SIGNALED||result==GL_CONDITION_SATISFIED;
    }
    void consume_fence(){if(fence){glDeleteSync(fence);fence=nullptr;}}
};
#undef SIGHT_CUDA
}
