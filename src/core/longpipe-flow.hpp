#pragma once
#include "flow-matte.hpp"
#include "subject_matte_history.hpp"
#include <chrono>
#include <string>
#include <memory>
#include <stdexcept>
#if defined(USE_ONNX_MEDIAPIPE) && defined(SIGHT_ANIMEGAN_ONLY)
#include "onnx_session.hpp"
#endif

namespace sight {
// One optional auxiliary session on the existing matte worker. No render
// callback, extra capture, timer or independent queue.
class LongpipeFlow {
    using Clock=std::chrono::steady_clock;
    unsigned w_=0,h_=0,sw_=0,sh_=0,cw_=0,ch_=0,bw_=0,bh_=0,fw_=0,fh_=0;
    Clock::time_point previous_time_{};
    std::vector<float> previous_canvas_,frames_,flow_;
    std::vector<uint8_t> previous_rgb_,previous_alpha_;
#if defined(USE_ONNX_MEDIAPIPE) && defined(SIGHT_ANIMEGAN_ONLY)
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
#endif
    std::string status_,provider_;
    bool ready_=false,active_=false;
    uint64_t runs_=0;
public:
    void reset(){
        previous_canvas_.clear();previous_rgb_.clear();previous_alpha_.clear();
        w_=h_=sw_=sh_=0;previous_time_={};
        if(ready_){active_=false;status_="Flow ready — "+provider_;}
    }
    bool ready()const{return ready_;}
    uint64_t runs()const{return runs_;}
    const std::string&status()const{return status_;}
    void load(const std::string&path,unsigned cw,unsigned ch,bool prefer_gpu,bool require_cuda){
        reset();ready_=active_=false;runs_=0;status_.clear();provider_.clear();frames_.clear();flow_.clear();
#if defined(USE_ONNX_MEDIAPIPE) && defined(SIGHT_ANIMEGAN_ONLY)
        session_.reset();env_.reset();
        if(path.empty())return;
        try{
            // (landscape canvas width,height, base width,height, grid width,height)
            constexpr unsigned tiers[][6]={{384,240,128,80,64,40},{384,224,192,112,48,28},
                {512,320,256,160,64,40},{640,400,256,160,64,40},{1280,768,320,192,80,48}};
            cw_=cw;ch_=ch;bw_=bh_=fw_=fh_=0;
            for(const auto&t:tiers){
                if(cw==t[0]&&ch==t[1]){bw_=t[2];bh_=t[3];fw_=t[4];fh_=t[5];break;}
                if(cw==t[1]&&ch==t[0]){bw_=t[3];bh_=t[2];fw_=t[5];fh_=t[4];break;}
            }
            if(!bw_)throw std::runtime_error("Unknown Longpipe canvas");
            env_=std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,"LongpipeFlow");
            Ort::SessionOptions options;
            options.SetIntraOpNumThreads(2);options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            const char*provider=nullptr;
            session_=lll::open_onnx_session(*env_,path.c_str(),options,prefer_gpu,&provider,require_cuda);
            if(session_->GetInputCount()!=1||session_->GetOutputCount()!=1)
                throw std::runtime_error("Expected one frames input and one flow output");
            Ort::AllocatorWithDefaultOptions a;
            auto in_type=session_->GetInputTypeInfo(0),out_type=session_->GetOutputTypeInfo(0);
            const auto in=in_type.GetTensorTypeAndShapeInfo(),out=out_type.GetTensorTypeAndShapeInfo();
            if(in.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||
                out.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||
                in.GetShape()!=std::vector<int64_t>{1,6,ch_,cw_}||
                out.GetShape()!=std::vector<int64_t>{1,2,fh_,fw_}||
                std::string(session_->GetInputNameAllocated(0,a).get())!="frames"||
                std::string(session_->GetOutputNameAllocated(0,a).get())!="flow")
                throw std::runtime_error("Flow graph must match the selected Longpipe tier and orientation");
            ready_=true;provider_=provider?provider:"provider unknown";status_="Flow ready — "+provider_;
        }catch(const std::exception&e){
            session_.reset();env_.reset();status_="Flow unavailable; using ordinary stabilization. "+std::string(e.what()).substr(0,384);
        }
#else
        if(!path.empty())status_="Flow unavailable in this build; using ordinary stabilization.";
#endif
    }
    // Return changed alpha count. On inference failure keep fresh alpha,
    // release the auxiliary session, and fall back on later frames.
    size_t apply(std::vector<uint8_t>&alpha,const std::vector<uint8_t>&rgb,
        const std::vector<float>&canvas,unsigned w,unsigned h,unsigned sw,unsigned sh,
        Clock::time_point captured,float retention)
    {
        if(!ready_||!(retention>0)){reset();return 0;}
        const size_t n=size_t(w)*h,cn=size_t(cw_)*ch_*3;
        if(!w||!h||w>1280||h>1280||!sw||!sh||alpha.size()!=n||rgb.size()!=3*n||canvas.size()!=cn){
            reset();return 0;
        }
        size_t changed=0;
#if defined(USE_ONNX_MEDIAPIPE) && defined(SIGHT_ANIMEGAN_ONLY)
        const bool comparable=w==w_&&h==h_&&sw==sw_&&sh==sh_&&previous_canvas_.size()==cn&&
           captured>previous_time_&&captured-previous_time_<=std::chrono::milliseconds(500);
        if(!comparable)reset();
        if(comparable&&flow_stationary(previous_rgb_,rgb,w,h)){
            changed=blend_stationary_matte(alpha,rgb,previous_alpha_,previous_rgb_,retention);
            active_=false;
            status_="Flow ready — "+provider_+"; stationary stabilization";
        }else if(comparable){
            try{
                frames_.resize(2*cn);
                std::copy(previous_canvas_.begin(),previous_canvas_.end(),frames_.begin());
                std::copy(canvas.begin(),canvas.end(),frames_.begin()+cn);
                const int64_t shape[]={1,6,ch_,cw_};
                auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
                auto input=Ort::Value::CreateTensor<float>(mem,frames_.data(),frames_.size(),shape,4);
                const char*ins[]={"frames"};const char*outs[]={"flow"};
                auto result=session_->Run(Ort::RunOptions{nullptr},ins,&input,1,outs,1);
                if(result.size()!=1||!result[0].IsTensor())throw std::runtime_error("Missing flow tensor");
                auto info=result[0].GetTensorTypeAndShapeInfo();
                if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||
                   info.GetShape()!=std::vector<int64_t>{1,2,fh_,fw_})throw std::runtime_error("Unexpected flow shape");
                auto*p=result[0].GetTensorData<float>();
                flow_.assign(p,p+size_t(fw_)*fh_*2);
                if(!std::all_of(flow_.begin(),flow_.end(),[](float v){return std::isfinite(v);}))
                    throw std::runtime_error("Nonfinite flow");
                changed=warp_matte(alpha,rgb,previous_alpha_,previous_rgb_,w,h,flow_,fw_,fh_,bw_,bh_,retention);
                ++runs_;
                if(!active_){active_=true;status_="Flow active — "+provider_;}
            }catch(const std::exception&e){
                ready_=false;session_.reset();env_.reset();reset();frames_.clear();flow_.clear();
                status_="Flow failed; using ordinary stabilization. "+std::string(e.what()).substr(0,384);
                return 0;
            }
        }
#endif
        previous_canvas_=canvas;previous_rgb_=rgb;previous_alpha_=alpha;
        w_=w;h_=h;sw_=sw;sh_=sh;previous_time_=captured;
        return changed;
    }
};
}
