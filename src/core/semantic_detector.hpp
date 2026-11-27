#pragma once
#include "semantic_regions.hpp"
#include "onnx_session.hpp"
#include <stdexcept>
namespace sight::semantic {
class Detector {
 Ort::Env env_{ORT_LOGGING_LEVEL_WARNING,"SightSemantic"};
 std::unique_ptr<Ort::Session> session_;
 std::string input_,output_;
 unsigned iw_=0,ih_=0,classes_=0;
 std::string error_;
public:
 const std::string& error() const {return error_;}
 unsigned class_count() const {return classes_;}
 bool load(const std::string& path,unsigned classes) {
  session_.reset();classes_=0;error_.clear();
  try {
   Ort::SessionOptions options;
   options.SetIntraOpNumThreads(2);options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
   auto s=lll::open_onnx_session(env_,path.c_str(),options);
   if(s->GetInputCount()!=1||s->GetOutputCount()!=1)throw std::runtime_error("Expected one input and one raw YOLO output");
   Ort::AllocatorWithDefaultOptions allocator;
   auto input=s->GetInputNameAllocated(0,allocator);
   auto output=s->GetOutputNameAllocated(0,allocator);
   auto it=s->GetInputTypeInfo(0);auto ot=s->GetOutputTypeInfo(0);
   auto i=it.GetTensorTypeAndShapeInfo();auto o=ot.GetTensorTypeAndShapeInfo();
   auto is=i.GetShape(),os=o.GetShape();
   if(i.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||
      o.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||
      is.size()!=4||is[0]!=1||is[1]!=3||is[2]<32||is[3]<32||is[2]>1280||is[3]>1280||
      os.size()!=3||(os[0]!=1&&os[0]!=-1)||(os[1]!=-1&&(os[1]<5||os[1]>1028)))
    throw std::runtime_error("Expected float32 NCHW static input and [1,4+classes,N] pixel-box output");
   if(classes>1024||(classes&&os[1]!=-1&&os[1]!=int64_t(classes+4)))
    throw std::runtime_error("Configured class count does not match model output");
   if(!classes&&os[1]==-1)throw std::runtime_error("Dynamic class dimension requires an explicit count");
   classes_=classes?classes:unsigned(os[1]-4);
   input_=input.get();output_=output.get();ih_=unsigned(is[2]);iw_=unsigned(is[3]);
   session_=std::move(s);return true;
  }catch(const std::exception& e){error_=e.what();return false;}
 }
 bool detect_regions(const uint8_t* rgba,unsigned w,unsigned h,unsigned classes,
             const std::vector<int>& ids,float threshold,float padding,
             const std::vector<Threshold>& thresholds,std::vector<Region>& boxes) {
  boxes.clear();
  if(!session_)return false;
  if(classes&&classes!=classes_){error_="Class count changed; reload required";return false;}
  classes=classes_;
  try {
   auto transform=letterbox(w,h,iw_,ih_);
   auto values=prepare(rgba,w,h,transform);
   if(values.empty())throw std::runtime_error("Invalid frame");
   const int64_t dims[]={1,3,ih_,iw_};
   auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
   auto input=Ort::Value::CreateTensor<float>(memory,values.data(),values.size(),dims,4);
   const char* in=input_.c_str();const char* out=output_.c_str();
   auto result=session_->Run(Ort::RunOptions{nullptr},&in,&input,1,&out,1);
   auto info=result.at(0).GetTensorTypeAndShapeInfo();auto shape=info.GetShape();
   if(shape.size()!=3||shape[0]!=1||shape[1]!=int64_t(classes+4)||shape[2]<=0||
      shape[2]>100000||info.GetElementCount()!=size_t(shape[1])*shape[2]||
      !decode_regions(result[0].GetTensorData<float>(),size_t(shape[1]),size_t(shape[2]),transform,ids,threshold,padding,thresholds,boxes))
    throw std::runtime_error("Unsupported or invalid raw detector output/class selection");
   error_.clear();return true;
  }catch(const std::exception& e){boxes.clear();error_=e.what();return false;}
 }
 bool detect(const uint8_t* rgba,unsigned w,unsigned h,unsigned classes,
             const std::vector<int>& ids,float threshold,float padding,
             std::vector<privacy::Rect>& boxes) {
  std::vector<Region> regions;
  bool ok=detect_regions(rgba,w,h,classes,ids,threshold,padding,{},regions);
  boxes=ok?boxes_of(regions):std::vector<privacy::Rect>{};return ok;
 }
};
}
