#include "rtmw_tracker.hpp"
#include <algorithm>
#include <cmath>
namespace sight {
namespace {
constexpr unsigned W=192,H=256,Count=133,XBins=384,YBins=512;
constexpr unsigned DetectorEdge=416,DetectorRows=3549,DetectorColumns=85;
constexpr float Mean[]={123.675f,116.28f,103.53f};
constexpr float Std[]={58.395f,57.12f,57.375f};
bool valid_output(const std::vector<std::vector<float>>& out) {
    return out.size()==2 && out[0].size()==Count*XBins && out[1].size()==Count*YBins;
}
}
bool RtmwTracker::load(const std::string& model,const std::string& detector,
                       const std::string& mediapipe_runtime,RtmwPersonBackend backend,float person_confidence) {
    ready_=false;error_.clear();runtime_.shutdown();detector_.unload();
    backend_=backend;detector_model_=detector;person_confidence_=std::clamp(person_confidence,.25f,.95f);
    if(backend_==RtmwPersonBackend::EfficientDet&&!detector_.load(mediapipe_runtime,detector,person_confidence_)){
        error_=detector_.last_error();return false;
    }
    if(!runtime_.initialize()){error_=runtime_.last_error();return false;}
    input_.assign(W*H*3,0.f);
    model_=model;
    std::vector<std::vector<float>> output;
    if(!runtime_.run_multi(model_,lll::LiteRtDevice::Gpu,input_,output,{false})){
        error_=runtime_.last_error();return false;
    }
    if(!valid_output(output)){error_="RTMW-m expected 133x384 and 133x512 SimCC outputs";return false;}
    if(backend_==RtmwPersonBackend::YoloX){
        detector_input_.assign(DetectorEdge*DetectorEdge*3,114.f);
        if(!runtime_.run(detector_model_,lll::LiteRtDevice::Gpu,detector_input_,detector_output_,{false})){
            error_=runtime_.last_error();return false;
        }
        if(detector_output_.size()!=DetectorRows*DetectorColumns){
            error_="YOLOX-Nano expected 3549x85 raw output";return false;
        }
    } else {detector_input_.clear();detector_output_.clear();}
    return ready_=true;
}
// The converted Nano graph accepts NHWC BGR 0..255 at 416 square and
// returns raw anchor-major COCO heads, strides 8/16/32. The letterbox
// is top-left aligned with gray bottom/right padding. Decode only person
// candidates: the highest valid person wins, just like EfficientDet.
bool RtmwTracker::detect_people_gpu(const uint8_t* rgba,uint32_t w,uint32_t h,
                                    std::vector<ObjectDetection>& detections){
    detections.clear();
    const float ratio=std::min(float(DetectorEdge)/w,float(DetectorEdge)/h);
    const unsigned rw=std::clamp(unsigned(std::lround(w*ratio)),1u,DetectorEdge);
    const unsigned rh=std::clamp(unsigned(std::lround(h*ratio)),1u,DetectorEdge);
    // Use the actual rounded dimensions to invert resize coordinates.
    const float sx=float(w)/rw,sy=float(h)/rh;
    std::fill(detector_input_.begin(),detector_input_.end(),114.f);
    for(unsigned y=0;y<rh;++y){
        const float fy=std::clamp((y+.5f)*sy-.5f,0.f,float(h-1));
        const unsigned y0=unsigned(fy),y1=std::min(y0+1,h-1);const float dy=fy-y0;
        for(unsigned x=0;x<rw;++x){
            const float fx=std::clamp((x+.5f)*sx-.5f,0.f,float(w-1));
            const unsigned x0=unsigned(fx),x1=std::min(x0+1,w-1);const float dx=fx-x0;
            const auto* a=&rgba[(size_t(y0)*w+x0)*4];
            const auto* b=&rgba[(size_t(y0)*w+x1)*4];
            const auto* c=&rgba[(size_t(y1)*w+x0)*4];
            const auto* d=&rgba[(size_t(y1)*w+x1)*4];
            auto* out=&detector_input_[(size_t(y)*DetectorEdge+x)*3];
            for(unsigned k=0;k<3;++k){
                const unsigned ch=2-k;
                out[k]=(a[ch]+(b[ch]-a[ch])*dx)*(1-dy)+(c[ch]+(d[ch]-c[ch])*dx)*dy;
            }
        }
    }
    if(!runtime_.run(detector_model_,lll::LiteRtDevice::Gpu,detector_input_,detector_output_,{false})){
        error_=runtime_.last_error();return false;
    }
    if(detector_output_.size()!=DetectorRows*DetectorColumns){
        error_="YOLOX-Nano output shape changed";return false;
    }
    size_t row=0;
    for(unsigned stride:{8u,16u,32u}){
        const unsigned n=DetectorEdge/stride;
        for(unsigned gy=0;gy<n;++gy)for(unsigned gx=0;gx<n;++gx,++row){
            const float* p=&detector_output_[row*DetectorColumns];
            if(!std::all_of(p,p+DetectorColumns,[](float v){return std::isfinite(v);})){
                error_="Non-finite YOLOX-Nano output";detections.clear();return false;
            }
            if(p[4]<0||p[4]>1||p[5]<0||p[5]>1)continue;
            const float score=p[4]*p[5];
            if(score<person_confidence_||std::max_element(p+5,p+DetectorColumns)!=p+5)continue;
            // Ignore impossible/extreme box values before exponentiation.
            if(std::abs(p[2])>12||std::abs(p[3])>12)continue;
            const float cx=(p[0]+gx)*stride,cy=(p[1]+gy)*stride;
            const float bw=std::exp(p[2])*stride,bh=std::exp(p[3])*stride;
            const float left=std::clamp((cx-bw*.5f)/rw,0.f,1.f);
            const float top=std::clamp((cy-bh*.5f)/rh,0.f,1.f);
            const float right=std::clamp((cx+bw*.5f)/rw,0.f,1.f);
            const float bottom=std::clamp((cy+bh*.5f)/rh,0.f,1.f);
            // Reject candidates centered wholly in the letterbox padding.
            if(cx<0||cy<0||cx>=rw||cy>=rh)continue;
            ObjectDetection person{left,top,right-left,bottom-top,score,"person",0};
            if(valid_object(person))detections.push_back(std::move(person));
        }
    }
    return true;
}
bool RtmwTracker::detect(const uint8_t* rgba,uint32_t w,uint32_t h,
                          int64_t timestamp,TrackingData& result) {
    result={};error_.clear();
    if(!ready_||!rgba||w<8||h<8||w>4096||h>4096){
        error_="Invalid RTMW input or unloaded tracker";return false;
    }
    std::vector<ObjectDetection> detections;
    if(backend_==RtmwPersonBackend::YoloX){
        if(!detect_people_gpu(rgba,w,h,detections))return false;
    } else if(!detector_.detect(rgba,w,h,timestamp,detections)){
        error_=detector_.last_error();return false;
    }
    // Single-person selection; choose highest scored person, never a
    // nonperson box or a full-frame fallback on detection loss.
    const ObjectDetection* person=nullptr;
    for(const auto& detection:detections)
        if(detection.label=="person"&&(!person||detection.score>person->score))
            person=&detection;
    if(!person)return true;
    const float cx=(person->x+person->w*.5f)*w;
    const float cy=(person->y+person->h*.5f)*h;
    const float height=1.25f*std::max(person->h*h,person->w*w*float(H)/W);
    const float width=height*W/H;
    if(width<4||height<4)return true;
    const float left=cx-.5f*width,top=cy-.5f*height;
    // Top-down affine crop, aspect-ratio corrected and padded outside
    // the source. Normalize RGB, preserve source coordinates on decoding.
    const auto pixel=[&](int x,int y,unsigned c)->float{
        if(x<0||y<0||x>=int(w)||y>=int(h))return 0.f;
        return rgba[(size_t(y)*w+x)*4+c];
    };
    for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x){
        const float sx=left+(x+.5f)*width/W-.5f,sy=top+(y+.5f)*height/H-.5f;
        const int ix=int(std::floor(sx)),iy=int(std::floor(sy));
        const float fx=sx-ix,fy=sy-iy;
        for(unsigned c=0;c<3;++c){
            const float a=pixel(ix,iy,c)*(1-fx)+pixel(ix+1,iy,c)*fx;
            const float b=pixel(ix,iy+1,c)*(1-fx)+pixel(ix+1,iy+1,c)*fx;
            input_[(c*H+y)*W+x]=(a*(1-fy)+b*fy-Mean[c])/Std[c];
        }
    }
    std::vector<std::vector<float>> output;
    if(!runtime_.run_multi(model_,lll::LiteRtDevice::Gpu,input_,output,{false})){
        error_=runtime_.last_error();return false;
    }
    if(!valid_output(output)){error_="RTMW output shape changed";return false;}
    std::vector<TrackedPoint> points;
    points.reserve(Count);
    for(unsigned i=0;i<Count;++i){
        auto xb=output[0].begin()+i*XBins,yb=output[1].begin()+i*YBins;
        if(!std::all_of(xb,xb+XBins,[](float v){return std::isfinite(v);})||
           !std::all_of(yb,yb+YBins,[](float v){return std::isfinite(v);})){
            error_="Non-finite RTMW output";return false;
        }
        auto x=std::max_element(xb,xb+XBins),y=std::max_element(yb,yb+YBins);
        const float px=(left+float(x-xb)*.5f*width/W)/w;
        const float py=(top+float(y-yb)*.5f*height/H)/h;
        // Raw minimum SimCC maximum: do not sigmoid negative scores into
        // fake confidence. Out-of-frame points are unavailable.
        float score=std::clamp(std::min(*x,*y),0.f,1.f);
        if(px<0||px>1||py<0||py>1)score=0;
        points.push_back({std::clamp(px,0.f,1.f),std::clamp(py,0.f,1.f),0.f,score});
    }
    result.landmarks=std::move(points);
    return true;
}
}
