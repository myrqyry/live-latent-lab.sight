#include "onnx_ivme_denoiser.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
namespace lll {
bool OnnxIvmeDenoiser::load(const std::string& model) {
    session_.reset();
    last_error_.clear();
    if(model.empty()){last_error_="No model provided";return false;}
    try {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        opts.SetIntraOpNumThreads(4);
        auto session=std::make_unique<Ort::Session>(env_,model.c_str(),opts);
        if(session->GetInputCount()!=2||session->GetOutputCount()!=1)
            throw std::runtime_error("Expected two inputs (image,sigma) and one output");
        Ort::AllocatorWithDefaultOptions allocator;
        auto name0=session->GetInputNameAllocated(0,allocator);
        auto name1=session->GetInputNameAllocated(1,allocator);
        auto output=session->GetOutputNameAllocated(0,allocator);
        if(std::string(name0.get())!="image"||std::string(name1.get())!="sigma"||
           std::string(output.get())!="clean")
            throw std::runtime_error("Unexpected Ivme ONNX input/output names");
        auto type=session->GetInputTypeInfo(0);
        auto shape=type.GetTensorTypeAndShapeInfo();
        if(shape.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||
            shape.GetShape().size()!=4)
            throw std::runtime_error("Expected FP32 NCHW image");
        session_=std::move(session);
        return true;
    }catch(const std::exception& e){last_error_=e.what();return false;}
}
bool OnnxIvmeDenoiser::denoise(const uint8_t* rgb,uint32_t w,uint32_t h,float sigma,
                               std::vector<uint8_t>& cleaned) {
    last_error_.clear();
    // Model uses dynamic spatial shapes. Extending the former 1024 axis cap
    // to verified 1280x720 is opt-in; CPU ORT uses ~4 GB and ~8s per frame.
    // Keep at most the old 1024-square pixel budget, avoiding sudden OOM.
    constexpr uint64_t MAX_PIXELS=1024ULL*1024ULL;
    if(!session_||!rgb||w<32||h<32||w>1280||h>1280||
       uint64_t(w)*h>MAX_PIXELS||
       (w%4)||(h%4)||!std::isfinite(sigma)||sigma<1||sigma>80){
        last_error_="Invalid geometry/sigma, or model unavailable";
        return false;
    }
    try {
        const size_t plane=static_cast<size_t>(w)*h;
        input_.resize(plane*3);
        for(size_t k=0;k<plane;++k)
            for(size_t c=0;c<3;++c)
                input_[c*plane+k]=rgb[k*3+c]/255.f;
        float s=sigma;
        std::array<int64_t,4> shape{1,3,static_cast<int64_t>(h),static_cast<int64_t>(w)};
        std::array<int64_t,1> scalar_shape{1};
        auto frame=Ort::Value::CreateTensor<float>(memory_,input_.data(),input_.size(),
                                                    shape.data(),shape.size());
        auto noise=Ort::Value::CreateTensor<float>(memory_,&s,1,
                                                    scalar_shape.data(),scalar_shape.size());
        std::array<Ort::Value,2> inputs{std::move(frame),std::move(noise)};
        const char* names[]={"image","sigma"};
        const char* outputs[]={"clean"};
        auto result=session_->Run(Ort::RunOptions{nullptr},names,inputs.data(),
                                  inputs.size(),outputs,1);
        if(result.size()!=1||!result[0].IsTensor()||
           result[0].GetTensorTypeAndShapeInfo().GetElementCount()!=plane*3)
            throw std::runtime_error("Invalid result shape");
        const float* out=result[0].GetTensorData<float>();
        cleaned.resize(plane*3);
        for(size_t k=0;k<plane;++k)for(size_t c=0;c<3;++c){
            const float v=out[c*plane+k];
            if(!std::isfinite(v))throw std::runtime_error("Nonfinite model output");
            cleaned[k*3+c]=static_cast<uint8_t>(std::round(std::clamp(v,0.f,1.f)*255.f));
        }
        return true;
    }catch(const std::exception& e){last_error_=e.what();return false;}
}
}
