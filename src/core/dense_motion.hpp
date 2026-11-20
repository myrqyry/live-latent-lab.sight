#pragma once
#include "sight_flow_v1.h"
#include <opencv2/video/tracking.hpp>
namespace sight {
class DenseMotion {
    cv::Ptr<cv::DISOpticalFlow> engine_;
public:
    bool estimate(const cv::Mat& previous,const cv::Mat& current,sight_flow_v1& out);
};
}
