#pragma once
#include "sight_flow_v1.h"
#ifdef SIGHT_HAVE_DENSE_FLOW
#include "dense_motion.hpp"
#endif
#include <opencv2/imgproc.hpp>
#include <array>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace sight::nr {
using SyncClock=std::chrono::steady_clock;
struct MotionReference {
    uint64_t sequence=0;
    cv::Mat rgba,gray;
};
inline std::shared_ptr<const MotionReference> make_reference(
        uint64_t sequence,const std::vector<uint8_t>&rgb,int w,int h){
    auto ref=std::make_shared<MotionReference>();
    ref->sequence=sequence;
    const int edge=std::min(320,std::max(w,h));
    cv::Mat small;
    cv::resize(cv::Mat(h,w,CV_8UC3,const_cast<uint8_t*>(rgb.data())),small,
        {std::max(1,w*edge/std::max(w,h)),std::max(1,h*edge/std::max(w,h))},
        0,0,cv::INTER_AREA);
    cv::cvtColor(small,ref->rgba,cv::COLOR_RGB2RGBA);
    cv::cvtColor(small,ref->gray,cv::COLOR_RGB2GRAY);
    return ref;
}
struct MotionResult {
    uint64_t sequence=0;
    SyncClock::time_point captured{};
    sight_flow_v1 flow{};
};
class MotionSync {
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stop_=false,pending_=false;
    std::shared_ptr<const MotionReference> reference_;
    cv::Mat frame_;
    SyncClock::time_point captured_{};
    MotionResult result_;
    std::thread thread_;
    void run(){
#ifdef SIGHT_HAVE_DENSE_FLOW
        sight::DenseMotion engine;
#endif
        for(;;){
            cv::Mat frame;std::shared_ptr<const MotionReference> ref;
            SyncClock::time_point captured;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock,[&]{return stop_||pending_;});
                if(stop_)return;
                frame=std::move(frame_);ref=std::move(reference_);
                captured=captured_;pending_=false;
            }
            MotionResult result;
            result.sequence=ref->sequence;result.captured=captured;
#ifdef SIGHT_HAVE_DENSE_FLOW
            try{
                cv::Mat gray;cv::cvtColor(frame,gray,cv::COLOR_RGBA2GRAY);
                if(gray.size()!=ref->gray.size()){
                    cv::Mat resized;
                    cv::resize(gray,resized,ref->gray.size(),0,0,cv::INTER_AREA);
                    gray=std::move(resized);
                }
                engine.estimate(ref->gray,gray,result.flow);
            }catch(const cv::Exception&){result.flow={};}
#endif
            std::lock_guard lock(mutex_);
            if(!stop_)result_=result;
        }
    }
public:
    MotionSync():thread_([this]{run();}){}
    ~MotionSync(){
        {std::lock_guard lock(mutex_);stop_=true;}
        wake_.notify_one();thread_.join();
    }
    void submit(std::shared_ptr<const MotionReference>ref,cv::Mat frame,SyncClock::time_point captured){
        std::unique_lock lock(mutex_,std::try_to_lock);
        if(!lock||stop_||!ref)return;
        reference_=std::move(ref);frame_=std::move(frame);captured_=captured;pending_=true;
        wake_.notify_one();
    }
    bool latest(uint64_t sequence,MotionResult&out){
        std::unique_lock lock(mutex_,std::try_to_lock);
        if(!lock||result_.sequence!=sequence||
           SyncClock::now()-result_.captured>std::chrono::milliseconds(150))return false;
        out=result_;return true;
    }
};
}
