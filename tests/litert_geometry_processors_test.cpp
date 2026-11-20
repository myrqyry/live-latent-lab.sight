#include "core/depth_estimator.hpp"
#include "core/head_pose_estimator.hpp"
#include "fast-face-detector.hpp"

#include <cassert>
#include <cstdint>

int main()
{
    lll::DepthEstimator depth;
    assert(depth.get_input_size() == 518);
    assert(!depth.load(nullptr));
    assert(!depth.estimate(nullptr, 16, 16, nullptr));
    assert(!depth.estimate_u8(nullptr, 16, 16, nullptr));

    lll::HeadPoseEstimator pose;
    const auto invalid_pose = pose.estimate(nullptr, 0, 0);
    assert(invalid_pose.pitch == 0.0f);
    assert(invalid_pose.yaw == 0.0f);
    assert(invalid_pose.roll == 0.0f);
    assert(!pose.load(nullptr));

    FastFaceDetector faces;
    const uint8_t pixel[3] = {0, 0, 0};
    assert(faces.detect(nullptr, 1, 1, 3).empty());
    assert(faces.detect(pixel, 0, 1, 3).empty());
    assert(faces.detect(pixel, 1, 1, 2).empty());
    return 0;
}
